/* =========================================================
   Shared atoms and cards, ported from the pre-rewrite frontend.

   These are the originals' markup and behaviour, re-pointed at the endpoints
   that exist now. The class names, the structure and the choreography are not
   new work — only what they talk to is. What was genuinely bound to the deleted
   architecture (the task rows, the plan card's fenced-block approval flow, the
   demo login) is gone with it; everything here outlived its data source.
   ========================================================= */

/* The status dot. `live` pings, because a running thing should look running. */
function Dot({ kind, status, title }) {
  const tone = kind || toneFor(status);
  return <span className={"dot" + (tone ? " " + tone : "")} title={title || statusLabel(status)} />;
}

/* The old vocabulary was live / work / stop, and it is a better one than five
   flat colours: one thing is alive, one wants you, one has stopped. */
function toneFor(status) {
  if (status === "running") return "live";
  if (status === "awaiting_approval") return "work";
  if (status === "failed") return "stop";
  return "";
}

function Empty({ glyph, children }) {
  return (
    <div className="empty">
      {glyph && <span className="glyph">{glyph}</span>}
      {children}
    </div>
  );
}

function Spinner() {
  return <span className="spin" />;
}

/* The approval card, ported. It collapses itself on resolve rather than
   vanishing, and it carries the note field: answering a question and adding a
   sentence about why are the same gesture, so they are the same control.

   The old one took a workshopped plan; this one takes an `approval` row, which
   is the same thing the backend now calls a question. */
function ApprovalCard({ item, onResolve, onError }) {
  const [noteOpen, setNoteOpen] = useState(false);
  const [gone, setGone] = useState(false);
  const card = useRef(null);
  const note = useRef(null);

  const isAsk = item.kind === "ask";

  async function resolve(answer) {
    const extra = note.current ? note.current.value.trim() : "";
    const body = extra ? `${answer}\n\n${extra}` : answer;
    setGone(true);

    // Collapse in place: the card shrinking is what says "answered", and the
    // list closing over it is quieter than a row disappearing.
    if (card.current) {
      const height = card.current.offsetHeight;
      card.current.style.transition =
        "opacity .35s var(--ease), transform .35s var(--ease), margin .35s var(--ease), " +
        "max-height .35s var(--ease), padding .35s var(--ease), border-color .35s var(--ease)";
      card.current.style.maxHeight = height + "px";
      requestAnimationFrame(() => {
        if (!card.current) return;
        card.current.style.maxHeight = "0px";
        card.current.style.opacity = "0";
        card.current.style.paddingTop = "0px";
        card.current.style.paddingBottom = "0px";
        card.current.style.marginBottom = "-12px";
        card.current.style.borderColor = "transparent";
        card.current.style.transform = "translateY(-4px)";
      });
    }

    try {
      await api.answer(item.approval_id, body);
      setTimeout(onResolve, 360);
    } catch (e) {
      setGone(false);
      if (card.current) card.current.removeAttribute("style");
      onError(e);
    }
  }

  return (
    <div className="card approval" ref={card} style={{ overflow: "hidden" }}>
      <div className="top">
        <span className="src">
          <Dot kind="work" /> {item.session_title || "session"}
        </span>
        <span className="tag accent">{isAsk ? "answer" : "approve / decline"}</span>
      </div>
      <div className="title">{item.prompt}</div>
      {item.project_title && <div className="body">{item.project_title}</div>}

      {isAsk ? (
        <AskAnswer disabled={gone} onSend={resolve} />
      ) : (
        <>
          <textarea
            ref={note}
            className={"note" + (noteOpen ? " open" : "")}
            placeholder="add a note or a condition before approving…"
          />
          <div className="actions">
            <button className="btn ghost" onClick={() => setNoteOpen((o) => !o)}>
              {noteOpen ? "− note" : "+ note"}
            </button>
            <span className="grow" />
            <button className="btn" disabled={gone} onClick={() => resolve("no")}>
              decline
            </button>
            <button className="btn primary" disabled={gone} onClick={() => resolve("yes")}>
              approve →
            </button>
          </div>
        </>
      )}
    </div>
  );
}

/* An ask wants prose, not a verdict, so it gets the field rather than the pair
   of buttons. */
function AskAnswer({ disabled, onSend }) {
  const [text, setText] = useState("");
  return (
    <form
      className="actions"
      onSubmit={(e) => {
        e.preventDefault();
        if (text.trim()) onSend(text.trim());
      }}
    >
      <input
        className="answer"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="your answer…"
        disabled={disabled}
      />
      <button className="btn primary" type="submit" disabled={disabled || !text.trim()}>
        send →
      </button>
    </form>
  );
}

/* =========================================================
   the stream — one session's live transcript

   Shared by the chat view and the looking glass detail, because they are the
   same thing (D5): the same session, the same component, the only difference
   being whether a human is turn-taking with it.
   ========================================================= */

/* One reply is many `content` events — that is what streaming is — so a run of
   them is one message, not one paragraph each. Same for `reasoning`. */
function grouped(events) {
  const out = [];
  for (const event of events) {
    const last = out[out.length - 1];
    const streams = event.kind === "content" || event.kind === "reasoning";
    if (streams && last && last.kind === event.kind) {
      out[out.length - 1] = { ...last, text: (last.text || "") + (event.text || "") };
      continue;
    }
    out.push(event);
  }
  return out;
}

function useStream(sessionId, onError, onPulse) {
  const [session, setSession] = useState(null);
  const [events, setEvents] = useState([]);
  const [pending, setPending] = useState([]);
  const [questions, setQuestions] = useState([]);
  const [todo, setTodo] = useState(null);
  const [browserUrl, setBrowserUrl] = useState(null);
  // What the browser run last said about itself. Null until one announces
  // itself, so a lease-waiting status is never read as a page.
  const [browserLabel, setBrowserLabel] = useState(null);
  const seen = useRef(new Set());

  const refreshQuestions = useCallback(async () => {
    if (!sessionId) return;
    try {
      setQuestions(await api.attention({ session_id: sessionId }));
    } catch (e) {
      onError(e);
    }
  }, [sessionId, onError]);

  /* Snapshot first, then the stream from where the snapshot ended, so nothing
     between the two is missed: the replay starts at that seq. */
  useEffect(() => {
    if (!sessionId) return undefined;
    let source = null;
    let dead = false;

    (async () => {
      try {
        const snapshot = await api.session(sessionId);
        if (dead) return;
        setSession(snapshot);
        const recent = snapshot.recent_events || [];
        seen.current = new Set(recent.map((e) => e.seq));
        setEvents(recent);
        const plans = recent.filter((e) => e.kind === "todo");
        if (plans.length) setTodo(plans[plans.length - 1].items);

        /* A browser run that announced itself BEFORE this window opened. Frames
           are a side-channel and are never replayed, but the stream is still
           there to subscribe to — without this, reloading the page mid-run left
           the canvas saying no browser had run, and it stayed wrong until the
           next step happened to fire. Only while the session is still running,
           and only if nothing has finished since the announcement. */
        const announced = recent.filter((e) => e.kind === "status" && e.url);
        const ended = recent.filter((e) => e.kind === "done");
        const live = announced.length ? announced[announced.length - 1] : null;
        const over = ended.length ? ended[ended.length - 1].seq : -1;
        if (live && live.seq > over && snapshot.status === "running") {
          setBrowserUrl(live.url);
          const since = recent.filter((e) => e.kind === "status" && e.seq >= live.seq);
          setBrowserLabel(since[since.length - 1].label || "");
        }

        await refreshQuestions();

        const last = recent.length ? recent[recent.length - 1].seq : 0;
        source = api.stream(
          sessionId,
          last,
          (event) => {
            if (seen.current.has(event.seq)) return;
            seen.current.add(event.seq);
            setEvents((current) => current.concat([event]));

            if (event.kind === "user" && event.source === "human") {
              // The real one arrived; drop the local echo it replaces.
              setPending((current) => {
                const at = current.findIndex((p) => p.text === event.text);
                return at === -1 ? current : current.filter((_, i) => i !== at);
              });
            }
            if (event.kind === "todo") setTodo(event.items);
            if (event.kind === "status") {
              if (event.url) {
                setBrowserUrl(event.url);
                setBrowserLabel(event.label || "");
              } else {
                // Only once a run has announced itself: before that, a status
                // is somebody else's (a lease wait, a discarded edit).
                setBrowserLabel((current) => (current === null ? null : event.label || current));
              }
            }
            if (event.kind === "budget") {
              setSession((s) => (s ? { ...s, hops_used: event.hops_used, hops_max: event.hops_max } : s));
            }
            if (event.kind === "lifecycle") {
              setSession((s) => (s ? { ...s, status: event.to } : s));
              refreshQuestions();
              if (onPulse) onPulse();
            }
            if (event.kind === "done") {
              setBrowserUrl(null);
              setBrowserLabel(null);
              refreshQuestions();
              if (onPulse) onPulse();
            }
          },
          (failure) => onError(failure),
        );
      } catch (e) {
        if (!dead) onError(e);
      }
    })();

    return () => {
      dead = true;
      if (source) source.close();
    };
  }, [sessionId, refreshQuestions, onError, onPulse]);

  /* Optimistic: the words appear now and reconcile when the log echoes them.
     Waiting a round trip to see what you just typed is the one latency a chat
     cannot have. */
  const send = useCallback(
    async (body) => {
      const mine = { id: `local-${Date.now()}`, text: body };
      setPending((current) => current.concat([mine]));
      try {
        await api.send(sessionId, body);
      } catch (e) {
        setPending((current) => current.filter((p) => p.id !== mine.id));
        onError(e);
      }
    },
    [sessionId, onError],
  );

  return { session, events, pending, questions, todo, browserUrl, browserLabel, send, refreshQuestions };
}

/* One event, in the design's stream vocabulary. */
function StreamEvent({ event, questions, onAnswered, onError }) {
  const [openArgs, setOpenArgs] = useState(false);
  const [openResult, setOpenResult] = useState(false);
  const [openThinking, setOpenThinking] = useState(false);

  switch (event.kind) {
    case "user":
      return event.source === "system" ? (
        <div className="ev-block ev-assist">
          <span className="who">system</span>
          <p>{event.text}</p>
        </div>
      ) : (
        <div className="ev-block ev-user">
          <span className="who">you</span>
          <div className="said">{event.text}</div>
        </div>
      );

    case "content":
      return (
        <div className="ev-block ev-assist">
          <span className="who">ark</span>
          <p>{event.text}</p>
        </div>
      );

    case "reasoning":
      return (
        <div className="ev-block ev-reasoning">
          <button className="toggle" onClick={() => setOpenThinking((o) => !o)}>
            {openThinking ? "hide thinking" : "thinking…"}
          </button>
          {openThinking && <pre>{event.text}</pre>}
        </div>
      );

    case "tool_call": {
      const args = event.args && Object.keys(event.args).length ? JSON.stringify(event.args) : "";
      return (
        <div className="ev-block ev-tool">
          <div className="row1">
            <span className="arrow">→</span>
            <span className="name">{event.name}</span>
            <span className="kicker">tool</span>
          </div>
          {args && (
            <div className={"args" + (openArgs ? "" : " collapsed")} onClick={() => setOpenArgs((o) => !o)}>
              {args}
            </div>
          )}
        </div>
      );
    }

    case "tool_result": {
      const body = event.content || "";
      return (
        <div className="ev-block">
          <div className={"ev-result" + (event.ok ? "" : " error")}>
            <div className="rhead">
              <span className="src">
                <span className={"dot " + (event.ok ? "live" : "stop")} />
                {event.ok ? "result" : event.error_kind || "failed"}
              </span>
              <button className="expand" onClick={() => setOpenResult((o) => !o)}>
                {openResult ? "collapse" : "expand"}
              </button>
            </div>
            <pre className={openResult ? "open" : ""}>{body}</pre>
            {event.total_chars > body.length && (
              <div className="note">
                {event.total_chars} chars · showing first {body.length}
              </div>
            )}
          </div>
        </div>
      );
    }

    case "status":
      return (
        <div className="ev-block ev-status">
          <span className="spin" />
          {event.label}
        </div>
      );

    case "lifecycle":
      return (
        <div className="ev-block ev-lifecycle">
          {event.from} → {event.to}
          {event.reason ? " · " + event.reason : ""}
        </div>
      );

    case "view_transform":
      return (
        <div className="ev-block ev-transform">
          older results cleared to make room ({(event.dropped_refs || []).length})
        </div>
      );

    case "done":
      return <div className="ev-block ev-done">— {event.reason} —</div>;

    case "todo":
    case "budget":
      return null; // both live in the context panel, where they stay glanceable

    default:
      return <div className="ev-block ev-lifecycle">{event.kind}</div>;
  }
}

/* A parked session's open question, answered where it was asked. */
function AskBlock({ item, onAnswered, onError }) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);

  const answer = async (value) => {
    setBusy(true);
    try {
      await api.answer(item.approval_id, value);
      onAnswered();
    } catch (e) {
      setBusy(false);
      onError(e);
    }
  };

  return (
    <div className="ev-block ev-ask">
      <span className="who">ark — needs input</span>
      {item.prompt}
      {item.kind === "approval" ? (
        <div className="opts">
          <span className="opt" onClick={() => !busy && answer("yes")}>approve</span>
          <span className="opt" onClick={() => !busy && answer("no")}>decline</span>
        </div>
      ) : (
        <form
          className="answer"
          onSubmit={(e) => {
            e.preventDefault();
            if (text.trim()) answer(text.trim());
          }}
        >
          <input value={text} onChange={(e) => setText(e.target.value)} placeholder="your answer…" disabled={busy} />
          <button className="opt" type="submit" disabled={busy || !text.trim()}>send</button>
        </form>
      )}
    </div>
  );
}
