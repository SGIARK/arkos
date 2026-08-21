/* =========================================================
   looking glass — projects grid + live session detail

   The design's composition, on real data. The grid is the user's projects with
   their rollup dot; opening one lists its sessions; opening a session is the
   detail view — the transcript streaming live on the left, the plan and the
   canvas on the right, the composer underneath.

   This is the ONLY projects surface. Chat, desk, approvals and computer are
   their own views in the rail.
   ========================================================= */

/* The design's four statuses, from ours. `work` pings because something is
   actually happening; `attn` is the one that wants a person. */
function pcStatus(status) {
  if (status === "running") return "work";
  if (status === "awaiting_approval") return "attn";
  if (status === "failed") return "err";
  return "ok";
}

function LookingGlassView({ onError, pulse, onPulse, jump, onJumped }) {
  const [projects, setProjects] = useState(null);
  const [waiting, setWaiting] = useState([]);
  const [openProject, setOpenProject] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [openSession, setOpenSession] = useState(null);

  useEffect(() => {
    let dead = false;
    (async () => {
      try {
        const [list, attention] = await Promise.all([api.projects(), api.attention()]);
        if (dead) return;
        setProjects(list);
        setWaiting(attention);
      } catch (e) {
        if (!dead) onError(e);
      }
    })();
    return () => {
      dead = true;
    };
  }, [pulse, onError]);

  /* Opened from somewhere else — the desk's running list, say. The grid is
     still the surface; it just arrives already looking at one session. */
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
      // One session is the common case, and clicking twice to reach it is a
      // click too many.
      if (list.length === 1) setOpenSession(list[0].session_id);
    } catch (e) {
      onError(e);
    }
  };

  if (openSession) {
    return (
      <SessionDetail
        sessionId={openSession}
        project={openProject}
        onBack={() => setOpenSession(null)}
        onError={onError}
        onPulse={onPulse}
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
        {projects === null && <Empty>reading…</Empty>}
        {projects !== null && !projects.length && (
          <Empty glyph="◇">no projects yet — start something from the command bar</Empty>
        )}
        <div className="pg-grid">
          {(projects || []).map((p) => (
            <div className="proj-card" key={p.id} onClick={() => open(p)}>
              <span className={"pc-status " + pcStatus(p.status_rollup)} title={statusLabel(p.status_rollup)} />
              <div className="pc-name">{p.title}</div>
              <div className="pc-sess">
                {p.sessions === 1 ? "1 session" : p.sessions + " sessions"}
                {waitingFor[p.id] ? ` · ${waitingFor[p.id]} waiting on you` : ""}
              </div>
              <div className="pc-when">
                {statusLabel(p.status_rollup)} · {relTime(p.updated_at)}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* One session, live. The same component renders the home chat's window and any
   project session — they are the same thing (D5). */
function SessionDetail({ sessionId, project, onBack, onError, onPulse }) {
  const stream = useStream(sessionId, onError, onPulse);
  const { session, events, pending, questions, todo, browserUrl, send, refreshQuestions } = stream;
  const [text, setText] = useState("");
  const [tab, setTab] = useState(() => localStorage.getItem("ark-canvas") || "files");
  const tail = useRef(null);

  useEffect(() => {
    localStorage.setItem("ark-canvas", tab);
  }, [tab]);
  useEffect(() => {
    if (tail.current) tail.current.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [events.length, pending.length]);

  if (!session) return <div className="lg-view"><div className="projects-grid"><Empty>opening…</Empty></div></div>;

  const running = session.status === "running";

  return (
    <div className="lg-view">
      <div className="lg-ctxline">
        {onBack && (
          <button className="back-btn" onClick={onBack}>
            ← projects
          </button>
        )}
        <span className="path">
          <b>{(project && project.title) || session.title || "session"}</b>
          <span className="crumb">▸</span>
          {session.title || "untitled session"}
        </span>
        <span className="status-pill">
          <span className={"dot " + (running ? "live" : session.status === "awaiting_approval" ? "work" : "")} />
          {statusLabel(session.status, session.terminal_reason)}
        </span>
        <span className="lg-budget">
          hop {session.hops_used}/{session.hops_max}
        </span>
        <span className="grow" />
        <div className="lg-ctrls">
          {session.mode === "attended" && session.status === "idle" && (
            <button className="icon-round" title="let it run unattended" onClick={() => api.approve(sessionId).catch(onError)}>
              ▶
            </button>
          )}
          {running && (
            <button className="icon-round danger" title="cancel" onClick={() => api.cancel(sessionId).catch(onError)}>
              ✕
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
            {/* Answered where it was asked, not in a separate tray. */}
            {questions.map((q) => (
              <AskBlock key={q.approval_id} item={q} onAnswered={refreshQuestions} onError={onError} />
            ))}
            {pending.map((p) => (
              <div className="ev-block ev-user pending" key={p.id}>
                <span className="who">you</span>
                <div className="said">{p.text}</div>
              </div>
            ))}
            <div ref={tail} />
          </div>

          <form
            className="lg-composer"
            onSubmit={(e) => {
              e.preventDefault();
              if (text.trim()) {
                send(text.trim());
                setText("");
              }
            }}
          >
            <span className="prompt">ark&gt;</span>
            <input
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="suggest or steer this session…"
              spellCheck={false}
              autoComplete="off"
            />
          </form>
        </div>

        <div className="ctx-panel">
          <div className="todo-block">
            <span className="kicker">todo</span>
            <div className="todo-list">
              {!todo || !todo.length ? (
                <span className="mute" style={{ fontSize: 11.5, fontStyle: "italic" }}>no plan yet</span>
              ) : (
                todo.map((item, i) => (
                  <label className={"todo-item" + (item.status === "completed" ? " done" : "")} key={i}>
                    {/* The model owns the plan; this is a readout, not a control. */}
                    <input type="checkbox" checked={item.status === "completed"} readOnly />
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
            <FilesCanvas projectId={session.project_id} onError={onError} />
          ) : (
            <BrowserCanvas url={browserUrl} />
          )}
        </div>
      </div>
    </div>
  );
}

/* The project's durable files: uploaded here, mounted into the sandbox when a
   session takes its box, readable without waking anything. */
function FilesCanvas({ projectId, onError }) {
  const [files, setFiles] = useState(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    if (!projectId) return;
    try {
      setFiles(await api.files(projectId));
    } catch (e) {
      onError(e);
    }
  }, [projectId, onError]);

  useEffect(() => {
    load();
  }, [load]);

  const send = async (list) => {
    if (!projectId || !list || !list.length) return;
    setBusy(true);
    try {
      for (const file of list) await api.upload(projectId, file);
      await load();
    } catch (e) {
      onError(e);
    } finally {
      setBusy(false);
    }
  };

  if (!projectId) return <div className="ctx-content"><div className="dropzone">this session has no project</div></div>;

  return (
    <div
      className="ctx-content"
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        send(e.dataTransfer.files);
      }}
    >
      {(files || []).map((f) => (
        <div className="cv-entry" key={f.file_id} title={f.path}>
          <span className="nm">
            <span className="g">·</span>
            {f.path}
          </span>
          <span className="sz">{fileSize(f.size)}</span>
        </div>
      ))}
      <label className="dropzone">
        {busy ? "uploading…" : "drop files into this project"}
        <input type="file" multiple hidden onChange={(e) => send(e.target.files)} />
      </label>
    </div>
  );
}

/* The browser's frames while it is browsing. Never events, never replayed. */
function BrowserCanvas({ url }) {
  const [frame, setFrame] = useState(null);

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

  if (!url) return <div className="ctx-content"><div className="waiting">no browser run in this session</div></div>;
  return (
    <div className="ctx-content">
      {frame ? (
        <img className="frame" alt="what the browser is looking at" src={"data:image/jpeg;base64," + frame} />
      ) : (
        <div className="waiting">waiting for the first frame…</div>
      )}
    </div>
  );
}

function fileSize(n) {
  if (n === null || n === undefined) return "";
  if (n < 1024) return n + " b";
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " kb";
  return (n / 1024 / 1024).toFixed(1) + " mb";
}
