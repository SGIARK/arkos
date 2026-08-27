/* Shared atoms and cards. */

/* `kind` names a tone directly, `status` names a lifecycle state and is
   translated; exactly one of the two may be passed. */
function Dot({ kind, status, title }) {
  const tone = kind !== undefined ? kind : toneFor(status);
  return <span className={"dot" + (tone ? " " + tone : "")} title={title || statusLabel(status)} />;
}

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

function useEscape(active, close) {
  useEffect(() => {
    if (!active) return undefined;
    const key = (e) => e.key === "Escape" && close();
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [active, close]);
}

function ApprovalCard({ item, onResolve, onError }) {
  const [noteOpen, setNoteOpen] = useState(false);
  const [gone, setGone] = useState(false);
  const card = useRef(null);
  const note = useRef(null);

  const isAsk = item.kind === "ask";
  // The API takes exactly "approve" or "decline" for a gated call; no note is
  // appended to it.
  const isCall = item.kind === "call";
  const isPlan = item.kind === "plan";
  const plan = isPlan ? item.tool_args || {} : null;
  const decided = isCall || isPlan;

  async function resolve(answer) {
    const extra = note.current ? note.current.value.trim() : "";
    const body = decided || !extra ? answer : `${answer}\n\n${extra}`;
    setGone(true);

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
          {isAsk ? "answer" : isPlan ? `plan v${item.version || 1}` : "approve / decline"}
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
              decline
            </button>
            <button className="btn primary" disabled={gone} onClick={() => resolve(decided ? "approve" : "yes")}>
              {isPlan ? "approve and run →" : "approve →"}
            </button>
          </div>
        </>
      )}
    </div>
  );
}

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
   ========================================================= */

/* One reply arrives as many `content` events, so a consecutive run of them is
   one message rather than one paragraph each. Same for `reasoning`. */
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
  // Null until a browser run announces itself, so a lease-waiting status is
  // never read as a page.
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

  /* Fields only: `recent_events` is dropped because the stream owns events and
     re-seeding from the snapshot would duplicate everything since. */
  const refreshSession = useCallback(async () => {
    if (!sessionId) return;
    try {
      const snapshot = await api.session(sessionId);
      setSession((current) => (current ? { ...current, ...snapshot, recent_events: undefined } : current));
    } catch (e) {
      onError(e);
    }
  }, [sessionId, onError]);

  /* Snapshot first, then the stream replayed from that seq, so nothing between
     the two is missed. */
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

        /* Frames are a side-channel and are never replayed: re-attach to a run
           announced before this window opened, only while the session is still
           running and nothing has finished since. */
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
                // Before a run announces itself, a status is somebody else's
                // (a lease wait, a discarded edit).
                setBrowserLabel((current) => (current === null ? null : event.label || current));
              }
            }
            if (event.kind === "budget") {
              setSession((s) => (s ? { ...s, hops_used: event.hops_used, hops_max: event.hops_max } : s));
            }
            if (event.kind === "lifecycle") {
              /* Mode moves only here: the server flips it in the same UPDATE as
                 the status, on plan approval and on any terminal state. */
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

  /* Optimistic: the local echo is dropped when the log echoes the same text. */
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

function StreamEvent({ event, questions, onAnswered, onError }) {
  const [openArgs, setOpenArgs] = useState(false);
  const [openResult, setOpenResult] = useState(false);
  const [openThinking, setOpenThinking] = useState(false);

  switch (event.kind) {
    case "user":
      /* `source: system` is the harness talking to the model: in the log
         because the fold needs it, deliberately not rendered. */
      return event.source === "system" ? null : (
        <div className="ev-block ev-user">
          <span className="who">you</span>
          <div className="said">{event.text}</div>
        </div>
      );

    case "content":
      return (
        <div className="ev-block ev-assist">
          <span className="who">buddy</span>
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
      /* A retry is the run waiting, not working, so it gets no spinner. */
      if (String(event.label || "").startsWith(RETRY_LABEL)) {
        return (
          <div className="ev-block ev-status ev-retry">
            <span className="tag">waiting</span>
            {String(event.label).slice(RETRY_LABEL.length)}
          </div>
        );
      }
      /* Approvals are not an event kind, so an auto-approved one arrives as a
         status carrying this prefix. */
      if (String(event.label || "").startsWith(AUTO_BADGE)) {
        return (
          <div className="ev-block ev-status ev-auto">
            <span className="tag">auto</span>
            {String(event.label).slice(AUTO_BADGE.length)}
          </div>
        );
      }
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
      return null; // both live in the context panel

    default:
      return <div className="ev-block ev-lifecycle">{event.kind}</div>;
  }
}

/* Mirrors `_AUTO_BADGE` in harness_module/runner.py. */
const AUTO_BADGE = "auto-approved ";

/* Mirrors the label agent_module/loop.py builds while the client backs off. */
const RETRY_LABEL = "model busy ";

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

  /* Consent binds to the call, so the card shows the exact tool and args that
     answering will run. */
  if (item.kind === "call") {
    return (
      <div className="ev-block ev-ask ev-gated">
        <span className="who">buddy — wants to run this</span>
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
      <span className="who">buddy — needs input</span>
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
   the plan card — three answers: approve runs it, ✕ closes the park, and typed
   text is a REPLY that closes this card (the next plan arrives as a new one).
   The field lives here because the composer 409s a plan-parked session.
   ========================================================= */

function PlanCard({ item, onAnswered, onError }) {
  const [ask, setAsk] = useState("");
  const [busy, setBusy] = useState(false);

  const plan = (item && item.tool_args) || {};
  const steps = Array.isArray(plan.steps) ? plan.steps : [];
  const inputs = Array.isArray(plan.inputs) ? plan.inputs : [];
  const missing = Array.isArray(plan.missing) ? plan.missing : [];

  /* Approve, decline and reply are one request; the answer text decides. */
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

  /* Cleared only once the request lands, so a failed post keeps the reply. */
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
        {/* The way out sits on the card, not in a footer below its scroll. */}
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
        {/* Insufficiency renders inside the card as named questions, so an
            under-informed plan is an intake form rather than a rejection. */}
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
   shared primitives — this file loads FIRST, so anything the later scripts
   share at module scope belongs here.
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

/* The zero-byte file that makes an empty folder durable: a real row that rides
   materialize and flush like any other, and is never shown. */
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
    // A sentinel still creates its directory; it is just never listed.
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

/* One tree for both scopes (the Files tab and a session's working-files pane).
   `reveal` is a path to select and open the ancestors of; `onFiles` reports the
   loaded rows back to the caller. */
function FileTree({ load, onOpen, onError, onFiles, reveal, onRevealed, header, zoneIdle, onDragState }) {
  const [files, setFiles] = useState(null);
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState(null);
  const [open, setOpen] = useState(() => new Set());
  // `target` is the directory a drop lands in; a set `moving` makes that drop a
  // rearrange rather than an upload.
  const [dragging, setDragging] = useState(false);
  const [target, setTarget] = useState("");
  // `movingDir` decides what the EDGE means: a directory dropped there moves out
  // to the top level and becomes a folder, a file cannot.
  const [moving, setMoving] = useState(null);
  const [movingDir, setMovingDir] = useState(false);
  const [renaming, setRenaming] = useState(null);
  const [renameText, setRenameText] = useState("");
  const renameCancelled = useRef(false);
  // `arming` is the row whose delete the first click armed; `undone` keeps one
  // level of undo, discarded as soon as it is used.
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

  useEffect(() => {
    if (!reveal || files === null) return;
    const wanted = files.find((f) => f.path === reveal);
    if (onRevealed) onRevealed();
    if (!wanted) return;
    expandTo(wanted.path);
    setSelected(wanted.path);
    onOpen(wanted);
  }, [reveal, files]);

  /* Every file in the store lives in a folder, so a drop with no directory is
     refused here rather than sent and refused. */
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

  /* The server moves in one transaction, so the tree is re-read rather than
     patched in place. */
  const move = async (from, into, isDir) => {
    /* Dropped on the EDGE: a directory moves out to the top level and becomes a
       folder of its own; a file has nothing there to land on. */
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
    // Into itself or back where it is: no-ops, and the first would be a folder
    // swallowing its own subtree.
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

  /* What goes over the wire is a NAME, not a path. The server refuses a
     top-level folder while a run has it mounted; that arrives as an error. */
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

  /* Rows go, blobs stay: they are content-addressed and never collected, so the
     returned `batch` restores the same content — and the project links with it. */
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
        /* Read and clear the drag HERE: move() and add() can both return early,
           leaving an overlay that outlives the drop. */
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
              // dragend with no path means the drag was abandoned (Escape, or a
              // drop outside the panel) and no drop is coming.
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
        {/* Under the rows, where the deleted thing was; it stays until used or
            until the panel is left. */}
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

/* Flush commits what is on disk, so a running session still holding the old
   path is stale and is named rather than swallowed. */
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

/* `rename` is optional — `{path, text, onText, onCommit, onCancel, onStart}`,
   `path` being the row edited; passing none disables renaming. */
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

  /* Two clicks rather than a modal: the first arms the row, the second deletes. */
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

  /* Drawn in place of the name: Enter and blur commit, Escape cancels. */
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
                // A directory is the one thing that may be dragged OUT to the
                // top level, where it becomes a folder of its own.
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
            /* The bar sits under the LAST file of the target directory: dropping
               on a file means the folder holding it. */
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
