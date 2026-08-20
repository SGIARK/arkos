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
  const { session, events, pending, questions, todo, browserUrl, browserLabel, send, refreshQuestions } = stream;
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
            <SessionTools sessionId={sessionId} onError={onError} />
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
            <BrowserCanvas url={browserUrl} label={browserLabel} />
          )}
        </div>
      </div>
    </div>
  );
}

/* =========================================================
   the tool budget — the chip in the composer, and the panel behind it

   Choosing what this session can reach sits next to asking it for something,
   which is why the control is in the composer and not beside the claims. The
   meter reads `enabled / (llm.max_tools - ours)`: our own tools are always
   loaded and never spend the human's allowance, so the denominator moves on
   its own if we add one.

   A toggle that would overflow the cap is refused HERE, dim and with the
   numbers on the row, and fires no request. The API refuses it too — this is
   the half that makes the refusal legible instead of a 400 from somewhere else.

   What this panel does NOT claim is effect. Until Task 11.5 wires the toggles
   into the manifest, the prompt and the loop, a server turned on is recorded
   and displayed and nothing more, which is why no row here says a word about
   what the model was handed.
   ========================================================= */

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

  // A server connected in settings while this window was open is one the panel
  // should offer the moment it is asked for, not on the next reload.
  useEffect(() => {
    if (open) load();
  }, [open, load]);

  useEffect(() => {
    if (!open) return undefined;
    const key = (e) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [open]);

  const budget = doc ? doc.budget : 0;
  const used = doc ? doc.used : 0;
  const left = budget - used;
  const ratio = budget ? used / budget : 0;
  const meter = ratio >= 1 ? "stop" : ratio > 0.8 ? "work" : "";

  const toggle = async (row, blocked) => {
    if (blocked || busy) {
      // Refused in the panel, with the numbers, and no request leaves the page.
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
                          : !row.requires_auth
                            ? "shared connection"
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

/* The project's durable files: uploaded here, mounted into the sandbox when a
   session takes its box, readable without waking anything.

   The SAME tree the files view draws — `asTree` and `Branch`, directories
   closed until asked for. A project that has had a repository cloned into it is
   thousands of paths, and rendering them flat turned a 300px panel into a
   scroll through somebody else's `.git/objects`. The tree was written for
   exactly that and there is no reason for this panel to have its own idea of
   what a file list looks like. */
function FilesCanvas({ projectId, onError }) {
  const [files, setFiles] = useState(null);
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState(null);
  // Closed by default: what you came to look at is never the hundredth row.
  const [open, setOpen] = useState(() => new Set());

  const toggle = (path) =>
    setOpen((current) => {
      const next = new Set(current);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });

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
      {files === null && <div className="cv-entry loading">reading…</div>}
      {files !== null && files.length > 0 && (
        <Branch
          node={asTree(files)}
          path=""
          depth={0}
          open={open}
          onToggle={toggle}
          onRead={(file) => setSelected(file.path)}
          selected={selected}
        />
      )}
      <label className="dropzone">
        {busy ? "uploading…" : "drop files into this project"}
        <input type="file" multiple hidden onChange={(e) => send(e.target.files)} />
      </label>
    </div>
  );
}

/* The browser's frames while it is browsing. Never events, never replayed.

   A popout, not a fixed canvas: the 300px panel is where you notice the browser
   is working, and it is nowhere near enough to read a page in. Clicking the
   thumbnail opens the same single frame stream at a size you can actually see,
   over the conversation rather than instead of it, and escape puts it back.
   (The 11.4 design supersedes LG-2's right-panel wording here.) */
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

  useEffect(() => {
    if (!big) return undefined;
    const key = (e) => e.key === "Escape" && setBig(false);
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [big]);

  // The run ending takes the popout with it: an expanded still of a stream that
  // has stopped is a picture pretending to be a window.
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

function fileSize(n) {
  if (n === null || n === undefined) return "";
  if (n < 1024) return n + " b";
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " kb";
  return (n / 1024 / 1024).toFixed(1) + " mb";
}
