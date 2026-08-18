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

function Todo({ event }) {
  const items = event.items || [];
  if (!items.length) return null;
  return (
    <div className="ev ev-todo">
      {items.map((item, i) => (
        <div key={i} className={"todo-item todo-" + (item.status || "pending")}>
          <span className="todo-mark">
            {item.status === "completed" ? "✓" : item.status === "in_progress" ? "▸" : "○"}
          </span>
          {item.text || item.title || String(item)}
        </div>
      ))}
    </div>
  );
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
      return <Todo event={event} />;
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

/* The one question a parked session is waiting on, answered here rather than
   somewhere else: the person watching is the person who answers. */
function Question({ item, onAnswered, onError }) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);

  const answer = async (value) => {
    setBusy(true);
    try {
      await api.answer(item.approval_id, value);
      setText("");
      onAnswered();
    } catch (e) {
      onError(e);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="question">
      <div className="question-prompt">
        <span className={"tag " + (item.kind === "ask" ? "tag-ask" : "tag-wait")}>{item.kind}</span>
        {item.prompt}
      </div>
      {item.kind === "approval" ? (
        <div className="question-actions">
          <button disabled={busy} onClick={() => answer("yes")}>
            Approve
          </button>
          <button className="ghost" disabled={busy} onClick={() => answer("no")}>
            Decline
          </button>
        </div>
      ) : (
        <form
          className="question-actions"
          onSubmit={(e) => {
            e.preventDefault();
            if (text.trim()) answer(text.trim());
          }}
        >
          <input value={text} onChange={(e) => setText(e.target.value)} placeholder="Your answer…" />
          <button type="submit" disabled={busy || !text.trim()}>
            Send
          </button>
        </form>
      )}
    </div>
  );
}

function SessionWindow({ sessionId, onBack, onError, onActivity }) {
  const [session, setSession] = useState(null);
  const [events, setEvents] = useState([]);
  const [questions, setQuestions] = useState([]);
  const [text, setText] = useState("");
  const [live, setLive] = useState(false);
  const tail = useRef(null);
  const seen = useRef(new Set());

  const refreshQuestions = useCallback(async () => {
    try {
      const open = await api.attention();
      setQuestions(open.filter((q) => q.session_id === sessionId));
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
            if (event.kind === "lifecycle") {
              setSession((s) => (s ? { ...s, status: event.to } : s));
              refreshQuestions();
              // A status move is exactly what the rail's two sections are made
              // of, so it is told then and not on a timer.
              if (onActivity) onActivity();
            }
            if (event.kind === "budget") {
              setSession((s) => (s ? { ...s, hops_used: event.hops_used, hops_max: event.hops_max } : s));
            }
            if (event.kind === "done") {
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
  }, [events.length]);

  const suggest = async (e) => {
    e.preventDefault();
    const body = text.trim();
    if (!body) return;
    setText("");
    try {
      await api.send(sessionId, body);
    } catch (err) {
      onError(err);
    }
  };

  if (!session) return <div className="window"><p className="empty">Opening…</p></div>;

  const ended = ["completed", "failed", "cancelled"].includes(session.status);

  return (
    <div className="window">
      <div className="window-head">
        {onBack && (
          <button className="link" onClick={onBack}>
            ← projects
          </button>
        )}
        <Dot status={session.status} />
        <h2>{session.title || "untitled session"}</h2>
        <span className="window-meta">
          <span className="hops">
            hop {session.hops_used}/{session.hops_max}
          </span>
          {session.mode === "unattended" && <span className="tag">unattended</span>}
          {live && <span className="tag tag-live">live</span>}
        </span>
        {session.status === "running" && (
          <button className="ghost" onClick={() => api.cancel(sessionId).catch(onError)}>
            Cancel
          </button>
        )}
        {session.mode === "attended" && session.status === "idle" && (
          <button onClick={() => api.approve(sessionId).catch(onError)}>Let it run</button>
        )}
      </div>

      {questions.map((item) => (
        <Question key={item.approval_id} item={item} onAnswered={refreshQuestions} onError={onError} />
      ))}

      <div className="transcript">
        {events.map((event) => (
          <EventRow key={event.seq} event={event} />
        ))}
        <div ref={tail} />
      </div>

      <form className="composer" onSubmit={suggest}>
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder={ended ? "Say something to start it again…" : "Suggest something…"}
          aria-label="Suggest something"
        />
        <button type="submit" disabled={!text.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}
