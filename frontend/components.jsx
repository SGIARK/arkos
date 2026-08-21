/* =========================================================
   Shared atoms and cards, ported from the pre-rewrite frontend.

   These are the originals' markup and behaviour, re-pointed at the endpoints
   that exist now. The class names, the structure and the choreography are not
   new work — only what they talk to is. What was genuinely bound to the deleted
   architecture (the task rows, the plan card's fenced-block approval flow, the
   demo login) is gone with it; everything here outlived its data source.
   ========================================================= */

/* The status dot. `live` pings, because a running thing should look running. */
/* One dot, one prop API. `kind` names a tone directly ("work"), `status` names
   a lifecycle state and is translated. Both are kept because both are real
   questions — "show me the attention colour" and "show me what this session is
   doing" — but only one of them may be passed. */
function Dot({ kind, status, title }) {
  const tone = kind !== undefined ? kind : toneFor(status);
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

/* Escape closes the thing on top. Written four times across two files before
   11.7.5 — each an effect, a listener and a matching teardown, and each one a
   chance to forget the teardown. `active` gates it so a closed modal is not
   holding a listener for a key it would ignore. */
function useEscape(active, close) {
  useEffect(() => {
    if (!active) return undefined;
    const key = (e) => e.key === "Escape" && close();
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [active, close]);
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
  // A gated call is decided, not discussed: the API takes exactly approve or
  // decline, so no note is appended to it and the verbs are its own.
  const isCall = item.kind === "call";
  /* A pending plan, on the desk. It gets its steps and the two words here and
     nothing else: workshopping a plan is the session window's job, where the
     conversation that produced it is on screen. */
  const isPlan = item.kind === "plan";
  const plan = isPlan ? item.tool_args || {} : null;
  // A stopped run: resume or cancel, and the prose that would steer it belongs
  // in the session window's composer, not here.
  const isResume = item.kind === "resume";
  const decided = isCall || isPlan || isResume;

  async function resolve(answer) {
    const extra = note.current ? note.current.value.trim() : "";
    const body = decided || !extra ? answer : `${answer}\n\n${extra}`;
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
        <span className="tag accent">
          {isAsk ? "answer" : isPlan ? `plan v${item.version || 1}` : isResume ? "stopped" : "approve / decline"}
        </span>
      </div>
      <div className="title">{isCall ? `Run ${item.tool_name}?` : item.prompt}</div>
      {isCall && <pre className="args">{JSON.stringify(item.tool_args || {}, null, 2)}</pre>}
      {isPlan && (
        <div className="plan-brief">
          {plan.done_when && <div className="done">{plan.done_when}</div>}
          <ol>
            {(plan.steps || []).map((step, i) => (
              <li key={i}>{step}</li>
            ))}
          </ol>
          {(plan.missing || []).length > 0 && (
            <div className="missing">
              {(plan.missing || []).length} open question
              {(plan.missing || []).length === 1 ? "" : "s"} — answer them in the session
            </div>
          )}
        </div>
      )}
      {item.project_title && <div className="body">{item.project_title}</div>}

      {isAsk ? (
        <AskAnswer disabled={gone} onSend={resolve} />
      ) : (
        <>
          {!decided && (
            <textarea
              ref={note}
              className={"note" + (noteOpen ? " open" : "")}
              placeholder="add a note or a condition before approving…"
            />
          )}
          <div className="actions">
            {!decided && (
              <button className="btn ghost" onClick={() => setNoteOpen((o) => !o)}>
                {noteOpen ? "− note" : "+ note"}
              </button>
            )}
            <span className="grow" />
            <button className="btn" disabled={gone} onClick={() => resolve(decided ? "decline" : "no")}>
              {isResume ? "cancel the run" : "decline"}
            </button>
            <button className="btn primary" disabled={gone} onClick={() => resolve(decided ? "approve" : "yes")}>
              {isPlan ? "approve and run →" : isResume ? "resume →" : "approve →"}
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

   ONE component for every session, because there is only one kind (D5): the
   only difference between them is whether a human is turn-taking with it. It
   was "shared by the chat view and the looking glass detail" until 11.8 removed
   the chat view; what is left is the rule that outlived it.
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

  /* Re-read the session's own fields. `questions` covers what is OPEN; this
     covers what was decided — `plan.answer` is how the window knows a plan was
     approved, dismissed, or spent by a cancel, and none of those leave a row
     behind to notice. Only the fields are taken: `events` has its own stream and
     re-seeding it from the snapshot would duplicate everything since. */
  const refreshSession = useCallback(async () => {
    if (!sessionId) return;
    try {
      const snapshot = await api.session(sessionId);
      setSession((current) => (current ? { ...current, ...snapshot, recent_events: undefined } : current));
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
              /* Mode moves in exactly one place — approving a plan — and it
                 moves in the SAME conditional UPDATE as the status, so the
                 lifecycle event is where the window learns about it. A terminal
                 hands the session back attended, the other half of the same
                 rule. */
              setSession((s) =>
                s
                  ? {
                      ...s,
                      status: event.to,
                      mode:
                        event.reason === "plan_approved"
                          ? "unattended"
                          : ["completed", "failed", "cancelled"].includes(event.to)
                            ? "attended"
                            : s.mode,
                    }
                  : s,
              );
              refreshQuestions();
              refreshSession();
              if (onPulse) onPulse();
            }
            if (event.kind === "done") {
              setBrowserUrl(null);
              setBrowserLabel(null);
              refreshQuestions();
              refreshSession();
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
  }, [sessionId, refreshQuestions, refreshSession, onError, onPulse]);

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

  return {
    session,
    events,
    pending,
    questions,
    todo,
    browserUrl,
    browserLabel,
    send,
    refreshQuestions,
    refreshSession,
  };
}

/* One event, in the design's stream vocabulary. */
function StreamEvent({ event, questions, onAnswered, onError }) {
  const [openArgs, setOpenArgs] = useState(false);
  const [openResult, setOpenResult] = useState(false);
  const [openThinking, setOpenThinking] = useState(false);

  switch (event.kind) {
    case "user":
      /* `source: system` is the harness talking to the model — the play
         button's handoff, the continuation, the finish nudge. It is in the log
         because the fold needs it, and it is NOT rendered: nobody typed it, and
         showing it put the instructions we send in with the words the human
         said, which read as a leaked prompt rather than as a turn. */
      return event.source === "system" ? null : (
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

  /* A gated call is not a question. The session is parked with this exact call
     open in the transcript, so the card shows the call and its arguments — the
     thing that will actually run — and answering runs or closes it. Approving a
     description of an action was the old shape, and nothing bound the
     description to the action. */
  if (item.kind === "call") {
    return (
      <div className="ev-block ev-ask ev-gated">
        <span className="who">ark — wants to run this</span>
        <div className="call">
          <span className="nm">{item.tool_name}</span>
          <span className="age">{relTime(item.created_at)}</span>
        </div>
        <pre className="args">{JSON.stringify(item.tool_args || {}, null, 2)}</pre>
        <div className="opts">
          <span className="opt" onClick={() => !busy && answer("approve")}>approve</span>
          <span className="opt" onClick={() => !busy && answer("decline")}>decline</span>
        </div>
      </div>
    );
  }

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

/* =========================================================
   the plan card (11.8.5)

   An unattended run starts from an approved plan and no other way, so this is
   the surface that starts one. It is not a summary of a decision made
   elsewhere: the approvals row carries the plan itself, the card renders those
   args, and answering the card is what writes plan.md and flips the mode.

   Three answers, not two, which is the whole difference between this and the
   gated-call card. Approve runs it, ✕ closes the park, and anything typed into
   the field is a REPLY — the model reads it and proposes again. That is why the
   input sits on the card rather than in the composer: the composer refuses a
   plan-parked session with a 409, because "yes do that" typed down there would
   read as a reply rather than as consent.

   What this card deliberately does NOT do is track a conversation. There is no
   revising banner and no "changed since v{n-1}" diff: replying closes THIS card
   and the next plan arrives as a whole new one. The diff was unreadable by v3 —
   changes stack, and a list of every edit since the last version says less than
   the plan itself does. A reply is answered by a new plan, not by an annotation
   on the old one.
   ========================================================= */

function PlanCard({ item, onAnswered, onError }) {
  const [ask, setAsk] = useState("");
  const [busy, setBusy] = useState(false);

  const plan = (item && item.tool_args) || {};
  const steps = Array.isArray(plan.steps) ? plan.steps : [];
  const inputs = Array.isArray(plan.inputs) ? plan.inputs : [];
  const missing = Array.isArray(plan.missing) ? plan.missing : [];

  /* Every action is the same request with a different word in it, because on
     the wire they ARE the same request: the answer text is what decides. */
  const respond = async (answer, outcome) => {
    setBusy(true);
    try {
      await api.answer(item.approval_id, answer);
      onAnswered(outcome, answer);
      return true;
    } catch (e) {
      setBusy(false);
      onError(e);
      return false;
    }
  };

  /* The field is cleared only once the request has landed. Clearing it first
     lost the reply outright when the post failed — the one thing on this card
     the human actually typed. */
  const sendAsk = async () => {
    const said = ask.trim();
    if (!said) return;
    if (await respond(said, "replied")) setAsk("");
  };

  return (
    <div className="plan-card">
      <div className="pl-head">
        <span className="kicker">plan</span>
        <span className="ver">v{item.version || 1}</span>
        <span className="grow" />
        <span className="mute">nothing runs until you approve</span>
        {/* The way out, where a way out is looked for: on the card itself, not
            in a footer below the fold of a card that scrolls. */}
        <button className="pl-x" title="dismiss this plan" disabled={busy} onClick={() => respond("decline", "dismissed")}>
          ✕
        </button>
      </div>

      <div className="pl-body">
        <div className="pl-field">
          <span className="kicker">goal</span>
          <span className="goal">{plan.goal}</span>
        </div>
        {plan.done_when && (
          <div className="pl-field">
            <span className="kicker">done when</span>
            <span className="done">{plan.done_when}</span>
          </div>
        )}
        {steps.length > 0 && (
          <div className="pl-field">
            <span className="kicker">steps</span>
            <ol className="pl-steps">
              {steps.map((step, i) => (
                <li key={i}>{step}</li>
              ))}
            </ol>
          </div>
        )}
        {inputs.length > 0 && (
          <div className="pl-field">
            <span className="kicker">inputs i have</span>
            <div className="pl-inputs">
              {inputs.map((input, i) => (
                <span className="pl-input" key={i}>
                  <Dot kind="live" title={input.label} />
                  {input.label}
                  <span className="grow" />
                  <span className="note">{input.note}</span>
                </span>
              ))}
            </div>
          </div>
        )}
        {/* Insufficiency renders INSIDE the card, as named questions. An
            under-informed plan is still a plan, and this is what makes the card
            an intake form rather than a thing to reject. */}
        {missing.length > 0 && (
          <div className="pl-field">
            <span className="kicker">missing</span>
            <div className="pl-missing">
              {missing.map((question, i) => (
                <span className="q" key={i}>
                  <span className="mark">?</span>
                  <span>{question}</span>
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="pl-foot">
        <input
          value={ask}
          onChange={(e) => setAsk(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendAsk()}
          placeholder={missing.length ? "answer a question or ask for a change" : "ask for a change"}
          spellCheck={false}
          disabled={busy}
        />
        <span className="grow" />
        <button className="btn ghost" disabled={busy || !ask.trim()} onClick={sendAsk}>
          send
        </button>
        <button className="btn primary" disabled={busy} onClick={() => respond("approve", "approved")}>
          approve and run
        </button>
      </div>
    </div>
  );
}

/* =========================================================
   shared primitives

   These live here because components.jsx loads FIRST. `fileSize` was
   defined in lookingglass.jsx and called from views.jsx, which loads before
   it — working only because every call happens at render time, after all
   five scripts have evaluated. One top-level use would have thrown. The file
   tree is the same story in the other direction: defined in views.jsx,
   rendered by lookingglass.jsx.
   ========================================================= */

function PageHead({ title, accent, lede }) {
  return (
    <div className="head">
      <h1>
        {title}
        {accent && <span className="accent">{accent}</span>}
        <span className="caret" />
      </h1>
      {lede && <div className="lede">{lede}</div>}
    </div>
  );
}

function fileSize(n) {
  if (n === null || n === undefined) return "";
  if (n < 1024) return n + " b";
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " kb";
  return (n / 1024 / 1024).toFixed(1) + " mb";
}

/* Flat paths into a tree. The store keeps `arkos/.git/objects/pack/…` as one
   string per file, which is the right shape for a tree and the wrong shape for
   a list: a clone turns the panel into 182 rows of somebody else's repository. */
/* The zero-byte file that makes an empty folder durable. It is a real row, so
   it rides materialize and flush like anything else — which is why an empty
   folder survives a session at all — and it is never shown. */
const SENTINEL = ".keep";

function isSentinel(path) {
  return path === SENTINEL || path.endsWith("/" + SENTINEL);
}

function asTree(files) {
  const root = { dirs: new Map(), files: [] };
  const descend = (parts) => {
    let node = root;
    for (const dir of parts) {
      if (!node.dirs.has(dir)) node.dirs.set(dir, { dirs: new Map(), files: [] });
      node = node.dirs.get(dir);
    }
    return node;
  };
  for (const file of files) {
    const parts = file.path.split("/");
    // Descending is the whole point of the sentinel; listing it is not.
    const node = descend(parts.slice(0, -1));
    if (!isSentinel(file.path)) node.files.push({ ...file, name: parts[parts.length - 1] });
  }
  return root;
}

/* The directory a path sits in; "" is the project root. */
function dirOf(path) {
  return path && path.includes("/") ? path.split("/").slice(0, -1).join("/") : "";
}

function countFiles(node) {
  let n = node.files.length;
  for (const child of node.dirs.values()) n += countFiles(child);
  return n;
}

/* ONE FILE TREE, used by both surfaces (11.9).

   The Files tab and a session's working-files pane are two SCOPES on the same
   store — everything, and the folders one project links — so they are one
   component with one set of powers: open, drag to move, drop to upload,
   double-click to rename. They differed before because they grew separately,
   which is the only reason a file could be renamed in one pane and not the
   other, and dropped into a nested folder in one and not the other.

   What a caller still supplies is what genuinely differs: which rows to load,
   and what clicking one does. Everything below is behaviour, and behaviour is
   the same in both.

   `reveal` is a path to select and open the ancestors of — how the pane hands a
   file to the tab. `onFiles` reports the loaded rows back, for a header that
   counts them. */
function FileTree({ load, onOpen, onError, onFiles, reveal, onRevealed, header, zoneIdle, onDragState }) {
  const [files, setFiles] = useState(null);
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState(null);
  // Closed by default: what you came to look at is never the hundredth row.
  const [open, setOpen] = useState(() => new Set());
  // `target` is the directory a drop lands in; `moving` is the path being
  // dragged, which is what makes a drop a rearrange rather than an upload.
  const [dragging, setDragging] = useState(false);
  const [target, setTarget] = useState("");
  // `moving` is the path being dragged; `movingDir` is whether it is a
  // directory, which decides what the EDGE means: a directory dropped there
  // moves out to the top level and becomes a folder, a file cannot.
  const [moving, setMoving] = useState(null);
  const [movingDir, setMovingDir] = useState(false);
  // The row being renamed, and the text so far. One at a time.
  const [renaming, setRenaming] = useState(null);
  const [renameText, setRenameText] = useState("");
  const renameCancelled = useRef(false);
  // The row whose delete is ARMED — the first click — and the last delete, kept
  // so it can be taken back. One level, because a second undo is a history and
  // a history is a different feature; the bar goes as soon as it is used.
  const [arming, setArming] = useState(null);
  const [undone, setUndone] = useState(null);

  const refresh = useCallback(async () => {
    try {
      const rows = await load();
      setFiles(rows);
      if (onFiles) onFiles(rows);
    } catch (e) {
      onError(e);
    }
  }, [load, onFiles, onError]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    if (onDragState) onDragState({ dragging, target, moving, movingDir });
  }, [dragging, target, moving, movingDir, onDragState]);

  const toggle = (path) =>
    setOpen((current) => {
      const next = new Set(current);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });

  const expandTo = (path) =>
    setOpen((current) => {
      const next = new Set(current);
      const parts = path.split("/").slice(0, -1);
      for (let i = 0; i < parts.length; i++) next.add(parts.slice(0, i + 1).join("/"));
      return next;
    });

  /* Arrived from the other scope: the same store and the same path, so this is
     a scroll-to rather than a second listing to keep in step. */
  useEffect(() => {
    if (!reveal || files === null) return;
    const wanted = files.find((f) => f.path === reveal);
    if (onRevealed) onRevealed();
    if (!wanted) return;
    expandTo(wanted.path);
    setSelected(wanted.path);
    onOpen(wanted);
  }, [reveal, files]);

  /* Every file in the store is in a folder, so a drop needs one. Refused here
     rather than sent and refused, and it names the way to do what was meant. */
  const add = async (list, dir) => {
    if (!list || !list.length) return;
    if (!dir) {
      onError(
        new ApiError(
          "no_folder",
          "The top level holds folders, not files. Make one with + folder, then drop into it."
        )
      );
      return;
    }
    setBusy(true);
    try {
      for (const file of list) await api.upload(file, dir);
      await refresh();
      setOpen((current) => new Set(current).add(dir));
    } catch (e) {
      onError(e);
    } finally {
      setBusy(false);
      setDragging(false);
    }
  };

  /* The store moves in one transaction and every live box is corrected in the
     same request, so the only copy that can be behind is this one — which is
     why the tree is re-read rather than patched in place. */
  const move = async (from, into, isDir) => {
    /* Dropped on the EDGE rather than on a row. For a directory that means OUT:
       it leaves the folder it was in and becomes a top-level folder of its own,
       which is what "a folder is a top-level path segment" already says. For a
       file it means nothing there is to land on, because the top level holds
       folders. */
    if (!into && !isDir) {
      onError(
        new ApiError(
          "no_folder",
          "The top level holds folders, not files. Drop it into a folder, or make one with + folder."
        )
      );
      return;
    }
    const name = from.split("/").pop();
    const to = into ? `${into}/${name}` : name;
    // Into itself, or back where it already is: both are no-ops, and the first
    // would be a folder swallowing its own subtree.
    if (to === from || into === from || (into && into.startsWith(from + "/"))) return;
    setBusy(true);
    try {
      const result = await api.moveFile(from, to);
      await refresh();
      setOpen((current) => new Set(current).add(into));
      if (selected && (selected === from || selected.startsWith(from + "/"))) {
        setSelected(to + selected.slice(from.length));
      }
      warnStale(result, "Moved", onError);
    } catch (e) {
      onError(e);
    } finally {
      setBusy(false);
      setMoving(null);
      setTarget("");
    }
  };

  /* A rename changes what a thing is CALLED and not where it is, so what goes
     over the wire is a NAME. Renaming a top-level folder carries the projects
     that link it and the claims that mount it along with the paths, and the
     server refuses it while a run has that folder mounted — which arrives here
     as an ordinary error saying so. */
  const commitRename = async () => {
    if (renameCancelled.current) {
      renameCancelled.current = false;
      return;
    }
    const from = renaming;
    const name = renameText.trim();
    setRenaming(null);
    setRenameText("");
    if (!from || !name || name === from.split("/").pop()) return;
    setBusy(true);
    try {
      const result = await api.renameFile(from, name);
      await refresh();
      // What is open, what is expanded and what is aimed at all follow the new
      // name: a rename that collapsed the tree you were reading would look like
      // a different tree.
      const follow = (path) =>
        path === from || path.startsWith(from + "/") ? result.to + path.slice(from.length) : path;
      if (selected) setSelected(follow(selected));
      setOpen((current) => new Set([...current].map(follow)));
      setTarget((current) => follow(current));
      warnStale(result, "Renamed", onError);
    } catch (e) {
      onError(e);
    } finally {
      setBusy(false);
    }
  };

  /* Deleting takes the rows and leaves the BLOBS, which are content-addressed
     and never collected — so the `batch` the server hands back restores the
     same content under the same id rather than a copy of it. A delete that
     empties a folder takes the folder and the project links that named it, and
     they come back together, because one click removed them for one reason. */
  const destroy = async (path) => {
    setArming(null);
    setBusy(true);
    try {
      const gone = await api.deleteFile(path);
      await refresh();
      if (selected && (selected === path || selected.startsWith(path + "/"))) setSelected(null);
      setUndone({ path, batch: gone.batch, files: gone.files });
    } catch (e) {
      onError(e);
    } finally {
      setBusy(false);
    }
  };

  const undo = async () => {
    if (!undone) return;
    setBusy(true);
    try {
      await api.undoDelete(undone.batch);
      await refresh();
      setUndone(null);
    } catch (e) {
      onError(e);
    } finally {
      setBusy(false);
    }
  };

  const rename = {
    path: renaming,
    text: renameText,
    onText: setRenameText,
    onCommit: commitRename,
    onCancel: () => {
      renameCancelled.current = true;
      setRenaming(null);
      setRenameText("");
    },
    onStart: (path, name) => {
      renameCancelled.current = false;
      setRenaming(path);
      setRenameText(name);
    },
  };

  return (
    <div
      className="ft"
      onDragOver={(e) => {
        e.preventDefault();
        if (!dragging) setDragging(true);
      }}
      onDragLeave={(e) => {
        if (e.currentTarget.contains(e.relatedTarget)) return;
        setDragging(false);
        if (!moving) setTarget("");
      }}
      onDrop={(e) => {
        e.preventDefault();
        /* Read the drag before clearing it, and clear it HERE rather than in
           whatever runs next: both of those can decide there is nothing to do
           and return early, and an overlay that outlives the drop is the bug
           that makes. */
        const lifted = moving;
        const liftedDir = movingDir;
        const into = target;
        setDragging(false);
        setMoving(null);
        setMovingDir(false);
        setTarget("");
        // Lifted from the tree: a rearrange. From the desktop: an upload.
        if (lifted) move(lifted, into, liftedDir);
        else add(e.dataTransfer.files, into);
      }}
    >
      {header && header({ busy, files, add })}
      <div className="ft-rows">
        {files === null && <div className="cv-entry loading">reading…</div>}
        {files !== null && !files.length && <div className="cv-entry loading">{zoneIdle.empty}</div>}
        {files !== null && files.length > 0 && (
          <Branch
            node={asTree(files)}
            path=""
            depth={0}
            open={open}
            onToggle={toggle}
            onRead={(file) => {
              setSelected(file.path);
              onOpen(file);
            }}
            selected={selected}
            dropTarget={dragging ? target : null}
            onTarget={setTarget}
            onLift={(path, isDir) => {
              setMoving(path);
              setMovingDir(!!isDir);
              // dragend with nothing lifted means the drag was abandoned —
              // Escape, or a drop outside the panel — and no drop is coming.
              if (!path) {
                setDragging(false);
                setTarget("");
              }
            }}
            lifted={moving}
            rename={rename}
            remove={{ armed: arming, onArm: setArming, onConfirm: destroy }}
          />
        )}
        {/* Under the rows, where the thing that vanished was — not a toast in a
            corner that times out before it is read. It stays until it is used
            or the panel is left. */}
        {undone && (
          <div className="ft-undo">
            <span className="what">
              deleted {undone.path}
              {undone.files > 1 ? ` and ${undone.files - 1} more` : ""}
            </span>
            <button type="button" onClick={undo} disabled={busy}>
              undo
            </button>
          </div>
        )}
      </div>
      <label className={"dropzone" + (dragging ? " over" : "")}>
        {busy
          ? "working…"
          : dragging
            ? target
              ? `${moving ? "move" : "drop"} into ${target}/`
              : movingDir
                ? `move ${moving.split("/").pop()}/ out to the top level`
                : "aim at a folder — the top level holds folders, not files"
            : zoneIdle.label}
        <input
          type="file"
          multiple
          hidden
          onChange={(e) => {
            add(e.target.files, target || dirOf(selected));
            e.target.value = "";
          }}
        />
      </label>
    </div>
  );
}

/* A move or a rename corrects every live box in the same request, because flush
   commits what is on disk. A box that refused comes back named rather than
   swallowed — the store is right and that box is not. */
function warnStale(result, what, onError) {
  const stale = (result && result.stale_sessions) || [];
  if (!stale.length) return;
  onError(
    new ApiError(
      "stale_box",
      `${what} in the store, but ${stale.length} running session(s) still hold the old path on disk.`
    )
  );
}

/* `rename` is optional and the Files tab is the only caller: it is
   `{path, text, onText, onCommit, onCancel, onStart}`, where `path` is the row
   being edited. The working-files pane passes none and gets no rename, which is
   the design — that pane is a view of what a project works in, and what a thing
   is CALLED is a fact about the store. */
function Branch({
  node,
  path,
  depth,
  open,
  onToggle,
  onRead,
  selected,
  dropTarget,
  onTarget,
  onLift,
  lifted,
  rename,
  remove,
}) {
  const indent = (d) => ({ paddingLeft: 18 + d * 13 });
  const dirs = [...node.dirs.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  const files = [...node.files].sort((a, b) => a.name.localeCompare(b.name));

  /* The delete affordance: a `✕` that is invisible until the row is hovered,
     and ARMS on the first click into the word "delete?" before it will do
     anything. Two clicks rather than a modal, because a modal for every file
     is a modal nobody reads — and the second click is not a confirmation of a
     dialog, it is the same gesture again on the same row. */
  const trash = (path) => {
    const armed = remove.armed === path;
    return (
      <span
        className={"fsdel" + (armed ? " armed" : "")}
        title={armed ? "click again to delete" : "delete"}
        onClick={(e) => {
          e.stopPropagation();
          if (armed) remove.onConfirm(path);
          else remove.onArm(path);
        }}
      >
        {armed ? "delete?" : "✕"}
      </span>
    );
  };

  /* The inline editor, drawn in place of the name. Enter and blur commit,
     Escape cancels — the gesture people already have for renaming a file, and
     the same one the project card uses. */
  const editor = () => (
    <input
      className="cv-rename"
      value={rename.text}
      autoFocus
      spellCheck={false}
      onClick={(e) => e.stopPropagation()}
      onChange={(e) => rename.onText(e.target.value)}
      onBlur={rename.onCommit}
      onKeyDown={(e) => {
        if (e.key === "Enter") rename.onCommit();
        if (e.key === "Escape") rename.onCancel();
      }}
    />
  );

  return (
    <React.Fragment>
      {dirs.map(([name, child]) => {
        const full = path ? path + "/" + name : name;
        const isOpen = open.has(full);
        const editing = !!rename && rename.path === full;
        return (
          <React.Fragment key={full}>
            <div
              className={"cv-entry dir" + (dropTarget === full ? " drop" : "") + (lifted === full ? " lifted" : "")}
              style={indent(depth)}
              onClick={() => onToggle(full)}
              draggable={!editing}
              onDragStart={(e) => {
                e.stopPropagation();
                e.dataTransfer.effectAllowed = "move";
                // Firefox starts no drag without payload; the path is the payload.
                e.dataTransfer.setData("text/plain", full);
                // A directory, which is the one thing that may be dragged OUT
                // to the top level — there it becomes a folder of its own.
                onLift(full, true);
              }}
              onDragEnd={() => onLift(null)}
              onDragOver={(e) => {
                e.preventDefault();
                onTarget(full);
              }}
            >
              <span className="nm">
                <span className="g">{isOpen ? "▾" : "▸"}</span>
                {editing ? (
                  editor()
                ) : (
                  <span
                    className="lbl"
                    title={rename ? "double-click to rename" : full}
                    onDoubleClick={(e) => {
                      if (!rename) return;
                      e.stopPropagation();
                      rename.onStart(full, name);
                    }}
                  >
                    {name}
                  </span>
                )}
              </span>
              <span className="meta">
                <span className="sz">{countFiles(child)}</span>
                {remove && trash(full)}
              </span>
            </div>
            {isOpen && (
              <Branch
                node={child}
                path={full}
                depth={depth + 1}
                open={open}
                onToggle={onToggle}
                onRead={onRead}
                selected={selected}
                dropTarget={dropTarget}
                onTarget={onTarget}
                onLift={onLift}
                lifted={lifted}
                rename={rename}
                remove={remove}
              />
            )}
          </React.Fragment>
        );
      })}
      {files.map((file, i) => (
        <div
          className={
            "cv-entry" +
            (selected === file.path ? " sel" : "") +
            /* The bar sits under the LAST file of the directory a drop would
               land in, so it reads as "into this folder" rather than "onto
               this file" — dropping on a file means the folder holding it. */
            (dropTarget === path && i === files.length - 1 ? " drop" : "") +
            (lifted === file.path ? " lifted" : "")
          }
          key={file.file_id}
          style={indent(depth)}
          onClick={() => onRead(file)}
          draggable={!(rename && rename.path === file.path)}
          onDragStart={(e) => {
            e.stopPropagation();
            e.dataTransfer.effectAllowed = "move";
            e.dataTransfer.setData("text/plain", file.path);
            onLift(file.path, false);
          }}
          onDragEnd={() => onLift(null)}
          onDragOver={(e) => {
            e.preventDefault();
            onTarget(path);
          }}
          title={file.path}
        >
          <span className="nm">
            <span className="g">·</span>
            {rename && rename.path === file.path ? (
              editor()
            ) : (
              <span
                className="lbl"
                title={rename ? "double-click to rename" : file.path}
                onDoubleClick={(e) => {
                  if (!rename) return;
                  e.stopPropagation();
                  rename.onStart(file.path, file.name);
                }}
              >
                {file.name}
              </span>
            )}
          </span>
          <span className="meta">
            <span className="sz">{fileSize(file.size)}</span>
            {remove && trash(file.path)}
          </span>
        </div>
      ))}
    </React.Fragment>
  );
}
