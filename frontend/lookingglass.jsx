/* Looking glass — the projects grid and live session detail. The only projects
   surface; chat, desk, approvals and computer are their own views in the rail. */

/* Checklist item shape mirrors `agent_module/events.py`: `todo_write` validates
   `pending | in_progress | done`. */
const TODO_DONE = "done";
const TODO_PENDING = "pending";

function isChecked(item) {
  const status = String((item && item.status) || "");
  return status === TODO_DONE || status === "completed";
}

function todoItem(text, status) {
  return { text: String(text), status: status || TODO_PENDING };
}

function pcStatus(status) {
  if (status === "running") return "work";
  if (status === "awaiting_approval") return "attn";
  if (status === "failed") return "err";
  return "ok";
}

function LookingGlassView({ onError, pulse, waiting: pending, onPulse, jump, onJumped, onOpenFile }) {
  const [projects, setProjects] = useState(null);
  const waiting = pending || [];
  const [openProject, setOpenProject] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [openSession, setOpenSession] = useState(null);
  const [making, setMaking] = useState(false);
  const [renaming, setRenaming] = useState(null);
  const [renameText, setRenameText] = useState("");

  const reload = useCallback(async () => {
    try {
      setProjects(await api.projects());
    } catch (e) {
      onError(e);
    }
  }, [onError]);

  const startRename = (project) => {
    setRenaming(project.id);
    setRenameText(project.title);
  };

  const commitRename = async () => {
    const id = renaming;
    const title = renameText.trim();
    setRenaming(null);
    if (!id || !title) return;
    const was = (projects || []).find((p) => p.id === id);
    if (was && was.title === title) return;
    setProjects((list) => (list || []).map((p) => (p.id === id ? { ...p, title } : p)));
    try {
      await api.renameProject(id, title);
      if (onPulse) onPulse();
    } catch (e) {
      onError(e);
      reload();
    }
  };

  useEffect(() => {
    let dead = false;
    (async () => {
      try {
        const list = await api.projects();
        if (dead) return;
        setProjects(list);
      } catch (e) {
        if (!dead) onError(e);
      }
    })();
    return () => {
      dead = true;
    };
  }, [pulse, onError]);

  useEffect(() => {
    if (!jump) return;
    setOpenSession(jump);
    onJumped();
  }, [jump, onJumped]);

  const open = async (project) => {
    setOpenProject(project);
    setSessions([]);
    try {
      const list = await api.projectSessions(project.id);
      setSessions(list);
      if (list.length === 1) setOpenSession(list[0].session_id);
    } catch (e) {
      onError(e);
    }
  };

  if (openSession) {
    return (
      /* Keyed by session: SessionDetail's local state is all about one session
         and must not carry across a switch. */
      <SessionDetail
        key={openSession}
        sessionId={openSession}
        project={openProject}
        onBack={() => setOpenSession(null)}
        onError={onError}
        onPulse={onPulse}
        onOpenFile={onOpenFile}
      />
    );
  }

  if (openProject) {
    return (
      <div className="lg-view">
        <div className="lg-ctxline">
          <button className="back-btn" onClick={() => setOpenProject(null)}>
            ← projects
          </button>
          <span className="path">
            <b>{openProject.title}</b>
          </span>
          <span className="grow" />
        </div>
        <div className="projects-grid">
          <div className="stack" style={{ maxWidth: 760 }}>
            {!sessions.length && <Empty glyph="○">nothing has run in this project yet</Empty>}
            {sessions.map((s) => (
              <div className="row" key={s.session_id} onClick={() => setOpenSession(s.session_id)}>
                <span className="label">
                  {s.status === "running" ? <Spinner /> : <Dot status={s.status} />}
                  <span className="text">
                    {s.title || "untitled session"}
                    <span className="src"> · {statusLabel(s.status, s.terminal_reason)}</span>
                  </span>
                </span>
                {s.open_questions > 0 && <span className="tag accent">needs you</span>}
                <span className="when">
                  {s.hops_used}/{s.hops_max} · {relTime(s.last_event_at)}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  const waitingFor = {};
  for (const item of waiting) waitingFor[item.project_id] = (waitingFor[item.project_id] || 0) + 1;

  return (
    <div className="lg-view">
      <div className="projects-grid">
        <div className="pg-head">
          <span className="kicker">projects</span>
          <button className="pg-new" title="new project" onClick={() => setMaking(true)}>
            +
          </button>
        </div>
        {projects === null && <Empty>reading…</Empty>}
        <div className="pg-grid">
          {(projects || []).map((p) => (
            <div className="proj-card" key={p.id} onClick={() => renaming !== p.id && open(p)}>
              <span className={"pc-status " + pcStatus(p.status_rollup)} title={statusLabel(p.status_rollup)} />
              {renaming === p.id ? (
                <input
                  className="pc-rename"
                  value={renameText}
                  autoFocus
                  spellCheck={false}
                  onClick={(e) => e.stopPropagation()}
                  onChange={(e) => setRenameText(e.target.value)}
                  onBlur={commitRename}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") commitRename();
                    if (e.key === "Escape") setRenaming(null);
                  }}
                />
              ) : (
                <div className="pc-name">
                  <span onDoubleClick={(e) => { e.stopPropagation(); startRename(p); }} title="double-click to rename">
                    {p.title}
                  </span>
                  <button
                    className="pc-rename-btn"
                    title="rename"
                    onClick={(e) => { e.stopPropagation(); startRename(p); }}
                  >
                    rename
                  </button>
                </div>
              )}
              <div className="pc-sess">
                {p.sessions === 1 ? "1 session" : p.sessions + " sessions"}
                {waitingFor[p.id] ? ` · ${waitingFor[p.id]} waiting on you` : ""}
              </div>
              <div className="pc-when">
                {statusLabel(p.status_rollup)} · {relTime(p.updated_at)}
              </div>
            </div>
          ))}
          {projects !== null && (
            <div className="proj-new" onClick={() => setMaking(true)}>
              new project
            </div>
          )}
        </div>
      </div>

      {making && (
        <NewProject
          onError={onError}
          onClose={() => setMaking(false)}
          onMade={async (project) => {
            setMaking(false);
            await reload();
            if (onPulse) onPulse();
            open({ id: project.id, title: project.title });
          }}
        />
      )}
    </div>
  );
}

/* The new-project modal. Folder choices come from `GET /folders` — the store's
   top-level segments — not from the projects, which would hide unlinked ones. */
function NewProject({ onClose, onMade, onError }) {
  const [name, setName] = useState("");
  const [mode, setMode] = useState("new");
  const [folders, setFolders] = useState(null);
  const [picked, setPicked] = useState(() => new Set());
  const [busy, setBusy] = useState(false);

  useEscape(true, onClose);

  useEffect(() => {
    if (mode !== "existing" || folders !== null) return;
    let dead = false;
    api
      .folders()
      .then((list) => !dead && setFolders(list))
      .catch((e) => !dead && onError(e));
    return () => {
      dead = true;
    };
  }, [mode, folders, onError]);

  const toggle = (folder) =>
    setPicked((current) => {
      const next = new Set(current);
      if (next.has(folder)) next.delete(folder);
      else next.add(folder);
      return next;
    });

  const slug = name.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "untitled";
  const linking = mode === "existing";
  const chosen = [...picked];
  const ready = !!name.trim() && !busy && (!linking || chosen.length > 0);

  const preview = linking
    ? chosen.length
      ? "links: " + chosen.map((f) => f + "/").join(", ")
      : "pick at least one folder"
    : "makes: " + slug + "/";

  const create = async () => {
    if (!ready) return;
    setBusy(true);
    try {
      onMade(await api.createProject(name.trim(), linking ? chosen : null));
    } catch (e) {
      setBusy(false);
      onError(e);
    }
  };

  return (
    <div className="np-over" onClick={onClose}>
      <div className="np" onClick={(e) => e.stopPropagation()}>
        <div className="np-head">
          <span className="kicker">new project</span>
          <button className="np-x" onClick={onClose}>✕</button>
        </div>
        <div className="np-body">
          <label className="np-field">
            <span className="np-label">name</span>
            <input
              value={name}
              autoFocus
              spellCheck={false}
              placeholder="what is this project for?"
              onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && create()}
            />
          </label>
          <div className="np-field">
            <span className="np-label">files</span>
            <div className={"np-pick" + (mode === "new" ? " on" : "")} onClick={() => setMode("new")}>
              <span className="np-radio" />
              <span className="np-copy">
                <span className="t">a new directory</span>
                <span className="s">a folder named after the project, starting empty</span>
              </span>
            </div>
            <div className={"np-pick" + (linking ? " on" : "")} onClick={() => setMode("existing")}>
              <span className="np-radio" />
              <span className="np-copy">
                <span className="t">an existing directory</span>
                <span className="s">link folders already in the store</span>
                {linking && (
                  <div className="np-list" onClick={(e) => e.stopPropagation()}>
                    {folders === null && <span className="np-none">reading…</span>}
                    {folders !== null && !folders.length && (
                      <span className="np-none">no folders yet — start a new one</span>
                    )}
                    {(folders || []).map((f) => (
                      <span
                        key={f.name}
                        className={"np-choice" + (picked.has(f.name) ? " on" : "")}
                        onClick={() => toggle(f.name)}
                      >
                        <span className="np-box">{picked.has(f.name) ? "✓" : ""}</span>
                        <span className="nm">{f.name}/</span>
                        <span className="n">{f.files} files</span>
                      </span>
                    ))}
                  </div>
                )}
              </span>
            </div>
          </div>
        </div>
        <div className="np-foot">
          <span className="np-preview">{preview}</span>
          <span className="np-acts">
            <button className="np-cancel" onClick={onClose}>cancel</button>
            <button className="np-go" disabled={!ready} onClick={create}>
              {busy ? "creating…" : "create"}
            </button>
          </span>
        </div>
      </div>
    </div>
  );
}

/* One session, live — reached from a project or from the desk's running list.
   A session need not belong to a project at all. */
function SessionDetail({ sessionId, project, onBack, onError, onPulse, onOpenFile }) {
  const stream = useStream(sessionId, onError, onPulse);
  const {
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
  } = stream;
  const [text, setText] = useState("");
  const [tab, setTab] = useState(() => localStorage.getItem("ark-canvas") || "files");
  const [headRename, setHeadRename] = useState(false);
  const [headText, setHeadText] = useState("");
  const tail = useRef(null);
  const composer = useRef(null);

  /* `auto` first so the box can shrink: `scrollHeight` never reports less than
     the height already set. The cap and internal scroll are the stylesheet's. */
  const grow = (el) => {
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  };

  /* Resetting the height belongs here: the value going empty does not re-run
     `grow`. */
  const submit = () => {
    const said = text.trim();
    if (!said) return;
    send(said);
    setText("");
    const el = composer.current;
    if (el) {
      el.style.height = "auto";
      el.focus();
    }
  };

  /* `drafting` is the one fact the server has no row for: a plan turn is in
     flight. The open plan itself is not held here — it is `questions`. */
  const [drafting, setDrafting] = useState(false);
  /* The plan approved in THIS window: the snapshot carries the same fact, but
     it was read before the approval, so the panel's steps would not seed. */
  const [approvedHere, setApprovedHere] = useState(null);

  /* Keyed off the SNAPSHOT's ids, not the grid's navigation state: a window
     opened from the desk has no `project` prop at all. */
  const commitHeadRename = async () => {
    const title = headText.trim();
    setHeadRename(false);
    const id = session && session.project_id;
    if (!id || !title || title === projectTitle) return;
    try {
      await api.renameProject(id, title);
      if (project) project.title = title;
      refreshSession();
      if (onPulse) onPulse();
    } catch (e) {
      onError(e);
    }
  };

  useEffect(() => {
    localStorage.setItem("ark-canvas", tab);
  }, [tab]);

  const openPlan = questions.find((q) => q.kind === "plan") || null;
  useEffect(() => {
    if (openPlan) {
      setDrafting(false);
      setApprovedHere(null);
    }
  }, [openPlan]);
  useEffect(() => {
    // A turn can end without proposing at all, so leaving `running` ends drafting.
    if (session && session.status !== "running") setDrafting(false);
  }, [session && session.status]);
  useEffect(() => {
    if (tail.current) tail.current.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [events.length, pending.length]);

  if (!session) return <div className="lg-view"><div className="projects-grid"><Empty>opening…</Empty></div></div>;

  const running = session.status === "running";
  const unattended = session.mode === "unattended";
  const projectTitle = session.project_title || (project && project.title) || null;

  const planCard = openPlan;
  const serverPlan = session.plan || null;
  const approvedPlan = approvedHere || (serverPlan && serverPlan.answer === "approve" ? serverPlan : null);

  /* idle + unattended is a stopped run and nothing else: an ordinary idle
     session is attended, and an unattended one is running or parked. */
  const held = session.status === "idle" && unattended;
  const cancelled = !!approvedPlan && session.status === "cancelled";
  const planSteps = (approvedPlan && approvedPlan.steps) || [];
  const seededSteps = (!todo || !todo.length) && planSteps.length > 0;
  const todoRows = seededSteps
    ? planSteps.map((step) => todoItem(step))
    : todo || [];
  /* A completed run never shows unchecked boxes: the harness sweeps the list at
     the terminal. Other terminals keep their partial list. */
  const doneRun = session.status === "completed";
  const shownRows = doneRun ? todoRows.map((r) => ({ ...r, status: TODO_DONE })) : todoRows;
  const showRun = !planCard && !drafting && !held && !running && !unattended;

  /* Cancel is the only press that spends the plan's approval; resume is a plain
     start, because the stop changed nothing to undo. */
  const cancelHeld = () =>
    api
      .cancel(sessionId)
      .then(() => {
        refreshSession();
        if (onPulse) onPulse();
      })
      .catch(onError);

  const resumeHeld = () =>
    api
      .resume(sessionId)
      .then(() => {
        refreshSession();
        if (onPulse) onPulse();
      })
      .catch(onError);

  return (
    <div className="lg-view">
      <div className="lg-ctxline">
        {onBack && (
          <button className="back-btn" onClick={onBack}>
            ← projects
          </button>
        )}
        {/* The project's name only, and from the SNAPSHOT — the header reads the
            same whether the window was opened from the grid or from the desk. */}
        <span className="path">
          {projectTitle &&
            (headRename ? (
              <input
                className="pc-rename"
                value={headText}
                autoFocus
                spellCheck={false}
                onChange={(e) => setHeadText(e.target.value)}
                onBlur={commitHeadRename}
                onKeyDown={(e) => {
                  if (e.key === "Enter") commitHeadRename();
                  if (e.key === "Escape") setHeadRename(false);
                }}
              />
            ) : (
              <b
                onDoubleClick={() => (setHeadText(projectTitle), setHeadRename(true))}
                title="double-click to rename"
                style={{ cursor: "text" }}
              >
                {projectTitle}
              </b>
            ))}
        </span>
        <span className={"status-pill" + (held ? " held" : "")}>
          {held ? (
            <span className="square" />
          ) : (
            <span className={"dot " + (running ? "live" : session.status === "awaiting_approval" ? "work" : "")} />
          )}
          {held ? "stopped" : statusLabel(session.status, session.terminal_reason)}
        </span>
        {!held && (
          <span className="lg-budget">
            hop {session.hops_used}/{session.hops_max}
          </span>
        )}
        {held && (
          <span className="lg-budget">holding at hop {session.hops_used}/{session.hops_max}</span>
        )}
        {(drafting || planCard) && (
          <span className="plan-chip">
            <Spinner />
            {planCard ? "plan awaiting approval" : "drafting a plan"}
          </span>
        )}
        <span className="grow" />
        <div className="lg-ctrls">
          {/* Three faces, never two at once: ▶ asks for a plan (a continuation
              on a cancelled run), ■ holds a running turn, ✕ spends the plan. */}
          {showRun && (
            <button
              className="run-btn"
              title={
                cancelled
                  ? "ark reads plan.md and the transcript, then proposes a continuation for your approval."
                  : "ark drafts a plan first. nothing runs until you approve it."
              }
              onClick={() => {
                setDrafting(true);
                api.approve(sessionId).catch((e) => {
                  setDrafting(false);
                  onError(e);
                });
              }}
            >
              <span className="glyph">▶</span>
              {cancelled ? "resume" : "autopilot"}
            </button>
          )}
          {running && (
            <button
              className="stop-btn"
              title={
                "holds the run at once. the plan's approval stands and the computer is kept, " +
                "so a message or resume picks it back up."
              }
              onClick={() =>
                api
                  .stop(sessionId)
                  .then((body) => {
                    refreshSession();
                    // `stopped: false` means the owning process is gone; the
                    // startup sweep is what ends the turn.
                    if (body && body.stopped === false) {
                      onError(
                        new ApiError(
                          "not_here",
                          "This run is not being driven by this server. It will be closed as interrupted; cancel to end it now."
                        )
                      );
                    }
                  })
                  .catch(onError)
              }
            >
              <span className="square" />
              stop
            </button>
          )}
          {/* A held run's two actions, and their only home: resume keeps the
              work, cancel spends it. */}
          {held && (
            <button
              className="run-btn"
              title="picks the run back up from where it stopped. the plan still stands and the computer is kept."
              onClick={() => resumeHeld()}
            >
              <span className="glyph">▶</span>
              resume
            </button>
          )}
          {held && (
            <button
              className="cancel-btn"
              title="ends the run for good and spends the plan's approval. resuming later means a new plan."
              onClick={() => cancelHeld()}
            >
              ✕ cancel
            </button>
          )}
        </div>
      </div>

      <div className="lg-body">
        <div className="stream-wrap">
          <div className="stream">
            {grouped(events).map((event) => (
              <StreamEvent key={event.seq} event={event} />
            ))}
            {/* Answered where they were asked — except a plan, which is not a
                question and gets the lane below. */}
            {questions
              .filter((q) => q.kind !== "plan")
              .map((q) => (
                <AskBlock
                  key={q.approval_id}
                  item={q}
                  hopLabel={session.hops_used}
                  onAnswered={refreshQuestions}
                  onError={onError}
                />
              ))}
            {pending.map((p) => (
              <div className="ev-block ev-user pending" key={p.id}>
                <span className="who">you</span>
                <div className="said">{p.text}</div>
              </div>
            ))}
            {/* The plan lane: drafting and the open card, never both at once. */}
            {drafting && !planCard && (
              <div className="plan-drafting">
                <Spinner />
                <span>ark is drafting {cancelled ? "a continuation" : "a plan"} for this run</span>
                <span className="grow" />
                <button className="link" onClick={() => api.cancel(sessionId).catch(onError)}>
                  cancel
                </button>
              </div>
            )}

            {planCard && (
              <PlanCard
                key={planCard.approval_id}
                item={planCard}
                onError={onError}
                onAnswered={(outcome) => {
                  if (outcome === "approved") {
                    setApprovedHere({
                      version: planCard.version || 1,
                      goal: (planCard.tool_args || {}).goal,
                      answer: "approve",
                      // Carried so the checklist seeds now, not on the next
                      // snapshot read.
                      steps: (planCard.tool_args || {}).steps || [],
                    });
                  }
                  // A reply wakes the session to propose again.
                  if (outcome === "replied") setDrafting(true);
                  refreshQuestions();
                  refreshSession();
                  if (onPulse) onPulse();
                }}
              />
            )}
            {/* Nothing plan-shaped below the feed: the header owns a held run's
                actions, the panel owns the steps, `plan.md` owns the rest. */}

            <div ref={tail} />
          </div>

          <form
            className="lg-composer"
            onSubmit={(e) => {
              e.preventDefault();
              submit();
            }}
          >
            <SessionTools sessionId={sessionId} onError={onError} />
            <span className="prompt">ark&gt;</span>
            {/* Enter sends, Shift+Enter is a newline; the height cap is the
                stylesheet's `max-height`. */}
            <textarea
              ref={composer}
              rows={1}
              value={text}
              onChange={(e) => {
                setText(e.target.value);
                grow(e.target);
              }}
              onKeyDown={(e) => {
                if (e.key !== "Enter" || e.shiftKey) return;
                // A newline mid-composition is the IME's, not a send.
                if (e.nativeEvent && e.nativeEvent.isComposing) return;
                e.preventDefault();
                submit();
              }}
              /* A stopped run resumes on what is typed here: kind `resume` is
                 exempt from the composer's 409. */
              placeholder={held ? "type to resume. your note is the next thing ark reads" : "suggest or steer this session…"}
              spellCheck={false}
              autoComplete="off"
            />
          </form>
        </div>

        <div className="ctx-panel">
          {/* The model's checklist, fed by `todo_write`; until the first one
              arrives it seeds from the approved plan's steps. */}
          <div className="todo-block">
            <span className="kicker">
              {seededSteps ? `steps · plan.md v${approvedPlan.version || 1}` : "todo"}
            </span>
            <div className="todo-list">
              {!shownRows.length ? (
                <span className="mute" style={{ fontSize: 11.5, fontStyle: "italic" }}>no steps yet</span>
              ) : (
                shownRows.map((item, i) => (
                  <label className={"todo-item" + (isChecked(item) ? " done" : "")} key={i}>
                    {/* The model owns the checklist; this is a readout, not a control. */}
                    <input type="checkbox" checked={isChecked(item)} readOnly />
                    <span>{item.text || item.title || String(item)}</span>
                  </label>
                ))
              )}
            </div>
          </div>

          <div className="ctx-tabs">
            <button className={tab === "files" ? "active" : ""} onClick={() => setTab("files")}>
              <span className="tab-label">working files</span>
            </button>
            <button className={tab === "browser" ? "active" : ""} onClick={() => setTab("browser")}>
              <span className="tab-label">
                browser{browserUrl ? " ●" : ""}
              </span>
            </button>
          </div>

          {tab === "files" ? (
            <FilesCanvas
              projectId={session.project_id}
              folders={session.folders || []}
              onError={onError}
              onOpenFile={onOpenFile}
            />
          ) : (
            <BrowserCanvas url={browserUrl} label={browserLabel} />
          )}
        </div>
      </div>
    </div>
  );
}

/* The tool budget. The meter reads `enabled / (llm.max_tools - ours)`: our own
   tools are always loaded and never spend the human's allowance. */

function SessionTools({ sessionId, onError }) {
  const [doc, setDoc] = useState(null);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(null);
  const [refused, setRefused] = useState(null);

  const load = useCallback(async () => {
    try {
      setDoc(await api.sessionTools(sessionId));
    } catch (e) {
      onError(e);
    }
  }, [sessionId, onError]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    if (open) load();
  }, [open, load]);

  useEscape(open, () => setOpen(false));

  const budget = doc ? doc.budget : 0;
  const used = doc ? doc.used : 0;
  const left = budget - used;
  const ratio = budget ? used / budget : 0;
  const meter = ratio >= 1 ? "stop" : ratio > 0.8 ? "work" : "";

  const toggle = async (row, blocked) => {
    if (blocked || busy) {
      // Refused here, with the numbers on the row; no request leaves the page.
      if (blocked && !row.enabled) setRefused(row.server);
      return;
    }
    setRefused(null);
    setBusy(row.server);
    try {
      setDoc(await api.setSessionTool(sessionId, row.server, !row.enabled));
    } catch (e) {
      onError(e);
      await load();
    } finally {
      setBusy(null);
    }
  };

  const clear = async () => {
    const on = (doc ? doc.servers : []).filter((s) => s.enabled);
    if (!on.length) return;
    setBusy("*");
    try {
      let latest = doc;
      for (const row of on) latest = await api.setSessionTool(sessionId, row.server, false);
      setDoc(latest);
    } catch (e) {
      onError(e);
      await load();
    } finally {
      setBusy(null);
    }
  };

  return (
    <React.Fragment>
      {open && <div className="tb-scrim" onClick={() => setOpen(false)} />}
      {open && (
        <div className="tb-panel" role="dialog" aria-label="tools in this session">
          <div className="tb-head">
            <div className="tb-head-row">
              <span className="kicker">tools in this session</span>
              <span className="tb-count">
                {used}
                <span className="mute">/{budget}</span>
              </span>
            </div>
            <div className="tb-meter">
              <div className={"tb-fill " + meter} style={{ width: Math.min(100, Math.round(ratio * 100)) + "%" }} />
            </div>
            <div className="tb-note">
              {!doc
                ? "reading…"
                : left <= 0
                  ? "cap reached — nothing else can be enabled until something is turned off"
                  : `${left} of ${budget} slots left · ${doc.ours} reserved for ark's own tools`}
            </div>
          </div>

          <div className="tb-list">
            {doc && !doc.servers.length && <div className="tb-empty">no servers are configured</div>}
            {(doc ? doc.servers : []).map((row) => {
              const connected = row.status === "connected";
              const fits = row.enabled || row.tool_count <= left;
              const blocked = !connected || !fits;
              return (
                <div
                  className={"tb-row" + (blocked ? " blocked" : "") + (row.enabled ? " on" : "")}
                  key={row.server}
                  onClick={() => toggle(row, blocked)}
                  title={row.name}
                >
                  <span className="tb-box">{row.enabled ? "✓" : ""}</span>
                  <span className="tb-name">
                    <span className="nm">{row.name}</span>
                    <span className="sub">
                      {!connected
                        ? "not connected — authorize in settings"
                        : !fits
                          ? "would exceed the cap"
                          : row.enabled
                            ? "on for this session"
                            : "off for this session"}
                    </span>
                  </span>
                  <span className="tb-n">{row.tool_count} tools</span>
                </div>
              );
            })}
          </div>

          <div className="tb-foot">
            <span className="tb-refused">
              {refused ? `${refused} needs more slots than are left` : ""}
            </span>
            <button type="button" className="tb-reset" onClick={clear} disabled={!!busy} title="back to ark's own tools only">
              reset
            </button>
          </div>
        </div>
      )}

      <button
        type="button"
        className={"tb-chip" + (open ? " open" : "")}
        title="mcp connectors in this session"
        onClick={() => setOpen((o) => !o)}
      >
        <span className={"tb-pip" + (ratio >= 1 ? " stop" : used ? " on" : "")} />
        <span className="k">tools</span>
        <span className="n">
          {doc ? used : "—"}
          <span className="mute">/{doc ? budget : "—"}</span>
        </span>
        <span className="caretglyph">{open ? "▾" : "▸"}</span>
      </button>
    </React.Fragment>
  );
}

/* The folders this project links, on the same `FileTree` and the same store as
   the Files tab. A linked folder reaches the agent only at the NEXT session,
   because claims are fixed for a session's life. */
function FilesCanvas({ projectId, folders, onError, onOpenFile }) {
  const [linked, setLinked] = useState(folders || []);
  const [linking, setLinking] = useState(false);
  const [choices, setChoices] = useState(null);
  const [busy, setBusy] = useState(false);
  // Bumped when a link lands: `load` closes over it, so the tree re-reads.
  const [pulse, setPulse] = useState(0);

  useEffect(() => {
    setLinked(folders || []);
  }, [folders]);

  const load = useCallback(() => api.files(projectId), [projectId, pulse]);

  useEffect(() => {
    if (!linking) return;
    let dead = false;
    api
      .folders()
      .then((list) => !dead && setChoices(list))
      .catch((e) => !dead && onError(e));
    return () => {
      dead = true;
    };
  }, [linking, onError]);

  const link = async (folder) => {
    if (linked.includes(folder)) return;
    setBusy(true);
    try {
      const body = await api.linkFolder(projectId, folder);
      setLinked(body.folders);
      setPulse((n) => n + 1);
    } catch (e) {
      onError(e);
    } finally {
      setBusy(false);
    }
  };

  if (!projectId) return <div className="ctx-content"><div className="dropzone">this session has no project</div></div>;

  const unlinked = (choices || []).filter((f) => !linked.includes(f.name));

  const header = () => (
    <React.Fragment>
      <div className="wf-head">
        <span className="kicker">linked folders</span>
        <button
          className={"wf-link" + (linking ? " on" : "")}
          title="link another folder from the store"
          onClick={() => setLinking((was) => !was)}
        >
          + link
        </button>
      </div>
      {linking && (
        <div className="wf-picker">
          {choices === null && <span className="wf-none">reading…</span>}
          {choices !== null && !unlinked.length && (
            <span className="wf-none">every folder in the store is already linked</span>
          )}
          {unlinked.map((f) => (
            <span key={f.name} className="wf-choice" onClick={() => !busy && link(f.name)}>
              <span className="np-box" />
              <span className="nm">{f.name}/</span>
              <span className="n">{f.files} files</span>
            </span>
          ))}
          <span className="wf-note">a folder linked now reaches the agent at the next session</span>
        </div>
      )}
    </React.Fragment>
  );

  return (
    <div className="ctx-content ctx-tree">
      <FileTree
        load={load}
        onOpen={(file) => onOpenFile && onOpenFile(file.path)}
        onError={onError}
        header={header}
        zoneIdle={{
          label: "drop files into a linked folder",
          empty: "this project links no folder yet",
        }}
      />
    </div>
  );
}

/* The browser's frames while it is browsing: live only, never events and never
   replayed. */
function BrowserCanvas({ url, label }) {
  const [frame, setFrame] = useState(null);
  const [big, setBig] = useState(false);

  useEffect(() => {
    if (!url) return undefined;
    setFrame(null);
    const source = new EventSource(url, { withCredentials: true });
    source.addEventListener("frame", (e) => {
      try {
        setFrame(JSON.parse(e.data).jpeg);
      } catch (err) {
        /* one dropped picture, not a broken pane */
      }
    });
    return () => source.close();
  }, [url]);

  useEscape(big, () => setBig(false));

  useEffect(() => {
    if (!url) setBig(false);
  }, [url]);

  if (!url) return <div className="ctx-content"><div className="waiting">no browser run in this session</div></div>;

  const picture = frame ? (
    <img className="frame" alt="what the browser is looking at" src={"data:image/jpeg;base64," + frame} />
  ) : (
    <div className="bw-hatch">
      <span className="pill-flat">waiting for the first frame</span>
    </div>
  );

  return (
    <div className="ctx-content">
      <div className="bw-card">
        <div className="bw-chrome">
          <span className="dot live" />
          <span className="bw-where">{label || "using the browser…"}</span>
          <button type="button" className="bw-expand" title="open larger" onClick={() => setBig(true)}>
            ⤢
          </button>
        </div>
        <div className="bw-shot" onClick={() => setBig(true)}>
          {picture}
        </div>
      </div>
      <div className="bw-foot">
        <span>streaming</span>
        <button type="button" className="bw-link" onClick={() => setBig(true)}>
          expand
        </button>
      </div>

      {big && (
        <div className="bw-over" onClick={() => setBig(false)}>
          <div className="bw-big" onClick={(e) => e.stopPropagation()}>
            <div className="bw-chrome">
              <span className="dot live" />
              <span className="bw-where">{label || "using the browser…"}</span>
              <span className="bw-fps">streaming</span>
              <button type="button" className="bw-close" onClick={() => setBig(false)}>
                ✕
              </button>
            </div>
            <div className="bw-shot big">{picture}</div>
          </div>
        </div>
      )}
    </div>
  );
}

