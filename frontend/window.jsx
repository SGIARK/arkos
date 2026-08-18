/* =========================================================
   The window: one session, live.

   A snapshot for the tail of the log, then one EventSource for everything
   after it. Strictly one-way — the human suggests, answers and cancels; there
   is no affordance anywhere in this file that executes a tool.
   ========================================================= */

/* Rendering, kind by kind. Store-shape is wire-shape, so an event arrives here
   exactly as it was appended and each renderer reads its own fields. */

function ToolCall({ event }) {
  const args = event.args && Object.keys(event.args).length ? JSON.stringify(event.args) : "";
  return (
    <div className="ev ev-call">
      <span className="ev-arrow">→</span>
      <span className="ev-name">{event.name}</span>
      {args && <span className="ev-args">{args.length > 160 ? args.slice(0, 160) + "…" : args}</span>}
    </div>
  );
}

function ToolResult({ event }) {
  const [open, setOpen] = useState(false);
  const body = event.content || "";
  const long = body.length > 280;
  return (
    <div className={"ev ev-result " + (event.ok ? "ok" : "bad")}>
      <div className="ev-result-head">
        <span>{event.ok ? "✓" : "✗ " + (event.error_kind || "failed")}</span>
        {long && (
          <button className="link" onClick={() => setOpen(!open)}>
            {open ? "less" : "more"}
          </button>
        )}
      </div>
      <pre>{open || !long ? body : body.slice(0, 280) + "…"}</pre>
      {event.ref && (
        <div className="ev-ref">
          {event.total_chars} chars stored{event.ref ? " · " + event.ref.slice(0, 8) : ""}
        </div>
      )}
    </div>
  );
}

function Reasoning({ event }) {
  // Collapsed by default: it is the model thinking, not the model speaking.
  const [open, setOpen] = useState(false);
  return (
    <div className="ev ev-reasoning">
      <button className="link" onClick={() => setOpen(!open)}>
        {open ? "hide thinking" : "thinking…"}
      </button>
      {open && <pre>{event.text}</pre>}
    </div>
  );
}

/* One reply is many `content` events — that is what streaming is — so a run of
   them is one message, not one paragraph each. The same goes for `reasoning`.
   Everything else stands alone and is left alone. */
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

function EventRow({ event }) {
  switch (event.kind) {
    case "user":
      return (
        <div className={"ev ev-user " + (event.source === "system" ? "from-system" : "")}>
          {event.source === "system" && <span className="tag">system</span>}
          <span>{event.text}</span>
        </div>
      );
    case "content":
      return <div className="ev ev-content">{event.text}</div>;
    case "reasoning":
      return <Reasoning event={event} />;
    case "tool_call":
      return <ToolCall event={event} />;
    case "tool_result":
      return <ToolResult event={event} />;
    case "status":
      return <div className="ev ev-status">{event.label}</div>;
    case "todo":
      return null; // pinned in the right panel, where it stays glanceable
    case "budget":
      return null; // the meter in the header carries this
    case "lifecycle":
      return (
        <div className="ev ev-lifecycle">
          {event.from} → {event.to}
          {event.reason ? " · " + event.reason : ""}
        </div>
      );
    case "view_transform":
      return (
        <div className="ev ev-transform">
          older results cleared to make room ({(event.dropped_refs || []).length})
        </div>
      );
    case "done":
      return <div className="ev ev-done">— {event.reason} —</div>;
    default:
      // A kind this build does not know renders as a row rather than crashing.
      return <div className="ev ev-unknown">{event.kind}</div>;
  }
}

function SessionWindow({ sessionId, onBack, onError, onActivity }) {
  const [session, setSession] = useState(null);
  const [events, setEvents] = useState([]);
  const [questions, setQuestions] = useState([]);
  const [text, setText] = useState("");
  // Sent, not yet echoed by the stream. Contracts: commands are optimistic —
  // apply locally, reconcile on the event. Waiting a round trip to see your own
  // words is the one latency a chat cannot have.
  const [pending, setPending] = useState([]);
  const [live, setLive] = useState(false);
  // The frame stream a `status` event announced, if the run is still going.
  const [canvas, setCanvas] = useState(null);
  // The latest plan, pinned in the panel rather than buried up the transcript.
  const [todo, setTodo] = useState(null);
  const tail = useRef(null);
  const seen = useRef(new Set());

  const refreshQuestions = useCallback(async () => {
    try {
      // Asked at session scope: the same rows the rail and the grid show, from
      // the one query, narrowed to this window.
      setQuestions(await api.attention({ session_id: sessionId }));
    } catch (e) {
      onError(e);
    }
  }, [sessionId, onError]);

  /* Snapshot, then stream from where the snapshot ended. Opened in this order
     so nothing between the two is missed: the stream replays from that seq. */
  useEffect(() => {
    let source = null;
    let cancelled = false;

    (async () => {
      try {
        const snapshot = await api.session(sessionId);
        if (cancelled) return;
        setSession(snapshot);
        const recent = snapshot.recent_events || [];
        seen.current = new Set(recent.map((e) => e.seq));
        setEvents(recent);
        // The panel shows the latest plan the snapshot already carries.
        const plans = recent.filter((e) => e.kind === "todo");
        if (plans.length) setTodo(plans[plans.length - 1].items);
        await refreshQuestions();

        const last = recent.length ? recent[recent.length - 1].seq : 0;
        source = api.stream(
          sessionId,
          last,
          (event) => {
            // The snapshot and the replay overlap by design; seq is the identity.
            if (seen.current.has(event.seq)) return;
            seen.current.add(event.seq);
            setEvents((current) => current.concat([event]));
            setLive(true);
            // The real one has arrived; drop the local echo it replaces.
            if (event.kind === "user" && event.source === "human") {
              setPending((current) => {
                const at = current.findIndex((p) => p.text === event.text);
                return at === -1 ? current : current.filter((_, i) => i !== at);
              });
            }
            if (event.kind === "lifecycle") {
              setSession((s) => (s ? { ...s, status: event.to } : s));
              refreshQuestions();
              // A status move is exactly what the rail's two sections are made
              // of, so it is told then and not on a timer.
              if (onActivity) onActivity();
            }
            if (event.kind === "status" && event.url) setCanvas(event.url);
            if (event.kind === "todo") setTodo(event.items);
            if (event.kind === "budget") {
              setSession((s) => (s ? { ...s, hops_used: event.hops_used, hops_max: event.hops_max } : s));
            }
            if (event.kind === "done") {
              // The run is over, so there is nothing left to look at.
              setCanvas(null);
              refreshQuestions();
              if (onActivity) onActivity();
            }
          },
          (failure) => onError(failure),
        );
      } catch (e) {
        if (!cancelled) onError(e);
      }
    })();

    return () => {
      cancelled = true;
      if (source) source.close();
    };
  }, [sessionId, refreshQuestions, onError, onActivity]);

  useEffect(() => {
    if (tail.current) tail.current.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [events.length, pending.length]);

  const suggest = async (e) => {
    e.preventDefault();
    const body = text.trim();
    if (!body) return;

    const mine = { id: `local-${Date.now()}`, text: body };
    setPending((current) => current.concat([mine]));
    setText("");
    try {
      await api.send(sessionId, body);
    } catch (err) {
      // It never landed, so take it back rather than leave a ghost that never
      // reconciles, and give them their words to try again with.
      setPending((current) => current.filter((p) => p.id !== mine.id));
      setText(body);
      onError(err);
    }
  };

  if (!session) return <Empty>opening…</Empty>;

  const ended = ["completed", "failed", "cancelled"].includes(session.status);

  return (
    <div className="window">
      <div className="window-head">
        {onBack && (
          <button className="link" onClick={onBack}>
            ← projects
          </button>
        )}
        {session.status === "running" ? <Spinner /> : <Dot status={session.status} />}
        <h2>{session.title || "untitled session"}</h2>
        <span className="window-meta">
          <span className="hops">
            hop {session.hops_used}/{session.hops_max}
          </span>
          {session.mode === "unattended" && <span className="tag">unattended</span>}
          {live && <span className="tag accent">live</span>}
        </span>
        {session.status === "running" && (
          <button className="ghost" onClick={() => api.cancel(sessionId).catch(onError)}>
            cancel
          </button>
        )}
        {session.mode === "attended" && session.status === "idle" && (
          <button className="primary" onClick={() => api.approve(sessionId).catch(onError)}>
            let it run →
          </button>
        )}
      </div>

      {questions.length > 0 && (
        <div className="stack questions">
          {questions.map((item) => (
            <ApprovalCard
              key={item.approval_id}
              item={item}
              onResolve={refreshQuestions}
              onError={onError}
            />
          ))}
        </div>
      )}

      <div className="window-body">
        <div className="transcript">
          {grouped(events).map((event) => (
            <EventRow key={event.seq} event={event} />
          ))}
          {pending.map((item) => (
            <div className="ev ev-user pending" key={item.id}>
              <span>{item.text}</span>
            </div>
          ))}
          <div ref={tail} />
        </div>

        <RightPanel
          projectId={session.project_id}
          todo={todo}
          browserUrl={canvas}
          onError={onError}
        />
      </div>

      <form className="composer" onSubmit={suggest}>
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder={ended ? "Say something to start it again…" : "Suggest something…"}
          aria-label="Suggest something"
        />
        <button className="primary" type="submit" disabled={!text.trim()}>
          send →
        </button>
      </form>
    </div>
  );
}
