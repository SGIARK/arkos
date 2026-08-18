/* =========================================================
   The grid: every project as a bubble, and everything waiting on you.

   Parallelism is the grid — five things at once is five projects. A bubble
   carries one dot, rolled up by the API to the most urgent status any of its
   sessions is in, so the whole page is one request.
   ========================================================= */

function Dot({ status, title }) {
  return <span className={"dot dot-" + statusTone(status)} title={title || statusLabel(status)} />;
}

/* One project. Clicking it opens its sessions; the dot says whether anything in
   there wants a human. */
function ProjectBubble({ project, waiting, onOpen }) {
  return (
    <button className="bubble" onClick={() => onOpen(project)}>
      <div className="bubble-top">
        <Dot status={project.status_rollup} />
        <span className="bubble-title">{project.title}</span>
      </div>
      <div className="bubble-foot">
        <span>{project.sessions === 1 ? "1 session" : project.sessions + " sessions"}</span>
        <span>{relTime(project.updated_at)}</span>
      </div>
      {waiting > 0 && (
        <div className="bubble-waiting">{waiting === 1 ? "1 question" : waiting + " questions"}</div>
      )}
    </button>
  );
}

/* The sessions inside one project, once its bubble is open. */
function SessionRow({ session, onOpen }) {
  return (
    <button className="srow" onClick={() => onOpen(session.session_id)}>
      <Dot status={session.status} title={statusLabel(session.status, session.terminal_reason)} />
      <span className="srow-title">{session.title || "untitled session"}</span>
      <span className="srow-meta">
        {session.mode === "unattended" && <span className="tag">unattended</span>}
        {session.open_questions > 0 && <span className="tag tag-wait">needs you</span>}
        <span className="hops">
          {session.hops_used}/{session.hops_max}
        </span>
        <span>{relTime(session.last_event_at)}</span>
      </span>
    </button>
  );
}

/* What is waiting on the human, at whatever scope is open. The same query and
   the same row shape either way; answering here is answering in the window,
   because it is the same endpoint. */
function Attention({ items, onOpenSession }) {
  if (!items.length) return null;
  return (
    <div className="attention">
      <h2>Waiting on you</h2>
      {items.map((item) => (
        <button key={item.approval_id} className="arow" onClick={() => onOpenSession(item.session_id)}>
          <span className={"tag " + (item.kind === "ask" ? "tag-ask" : "tag-wait")}>{item.kind}</span>
          <span className="arow-prompt">{item.prompt}</span>
          <span className="arow-meta">
            {item.project_title || "no project"} · {relTime(item.created_at)}
          </span>
        </button>
      ))}
    </div>
  );
}

function Grid({ onOpenSession, onError }) {
  const [projects, setProjects] = useState([]);
  const [waiting, setWaiting] = useState([]);
  const [open, setOpen] = useState(null); // the project whose sessions are showing
  const [sessions, setSessions] = useState([]);
  const [goal, setGoal] = useState("");
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    try {
      const [projectList, attention] = await Promise.all([api.projects(), api.attention()]);
      setProjects(projectList);
      setWaiting(attention);
    } catch (e) {
      onError(e);
    } finally {
      setLoading(false);
    }
  }, [onError]);

  useEffect(() => {
    load();
  }, [load]);

  const openProject = async (project) => {
    setOpen(project);
    setSessions([]);
    try {
      setSessions(await api.projectSessions(project.id));
    } catch (e) {
      onError(e);
    }
  };

  /* Composing at grid level creates a project; composing inside one puts the
     session there. Placeful, and zero clicks for the common case. */
  const compose = async (e) => {
    e.preventDefault();
    const text = goal.trim();
    if (!text) return;
    setGoal("");
    try {
      const created = await api.start(text, open ? open.id : undefined);
      onOpenSession(created.session_id);
    } catch (err) {
      onError(err);
    }
  };

  const waitingByProject = {};
  for (const item of waiting) {
    waitingByProject[item.project_id] = (waitingByProject[item.project_id] || 0) + 1;
  }

  return (
    <div className="grid-view">
      <Attention items={open ? waiting.filter((w) => w.project_id === open.id) : waiting} onOpenSession={onOpenSession} />

      {open ? (
        <div className="project-open">
          <div className="project-head">
            <button className="link" onClick={() => setOpen(null)}>
              ← all projects
            </button>
            <h2>{open.title}</h2>
          </div>
          {sessions.map((session) => (
            <SessionRow key={session.session_id} session={session} onOpen={onOpenSession} />
          ))}
          {!sessions.length && <p className="empty">Nothing has run in this project yet.</p>}
        </div>
      ) : (
        <>
          <h2>Projects</h2>
          {loading && <p className="empty">Loading…</p>}
          {!loading && !projects.length && <p className="empty">No projects yet. Start something below.</p>}
          <div className="bubbles">
            {projects.map((project) => (
              <ProjectBubble
                key={project.id}
                project={project}
                waiting={waitingByProject[project.id] || 0}
                onOpen={openProject}
              />
            ))}
          </div>
        </>
      )}

      <form className="composer" onSubmit={compose}>
        <input
          value={goal}
          onChange={(e) => setGoal(e.target.value)}
          placeholder={open ? `Start something in ${open.title}…` : "Start something new…"}
          aria-label="What should ARK do?"
        />
        <button type="submit" disabled={!goal.trim()}>
          Start
        </button>
      </form>
    </div>
  );
}
