/* =========================================================
   The grid: every project as a bubble, and everything waiting on you.

   Parallelism is the grid — five things at once is five projects. A bubble
   carries one dot, rolled up by the API to the most urgent status any of its
   sessions is in, so the whole page is one request.
   ========================================================= */

/* One project. Clicking it opens its sessions; the dot says whether anything in
   there wants a human. */
function ProjectBubble({ project, waiting, onOpen }) {
  return (
    <div className="card bubble" onClick={() => onOpen(project)}>
      <div className="top">
        <span className="src">
          <Dot status={project.status_rollup} /> {statusLabel(project.status_rollup)}
        </span>
        <span className="when">{relTime(project.updated_at)}</span>
      </div>
      <div className="bubble-title">{project.title}</div>
      <div className="bubble-foot">
        <span>{project.sessions === 1 ? "1 session" : project.sessions + " sessions"}</span>
        {waiting > 0 && (
          <span className="bubble-waiting">{waiting === 1 ? "1 question" : waiting + " questions"}</span>
        )}
      </div>
    </div>
  );
}

/* The sessions inside one project, once its bubble is open. */
function SessionRow({ session, onOpen }) {
  return (
    <button className="row" onClick={() => onOpen(session.session_id)}>
      <span className="label">
        {session.status === "running" ? (
          <Spinner />
        ) : (
          <Dot status={session.status} title={statusLabel(session.status, session.terminal_reason)} />
        )}
        <span className="text">
          {session.title || "untitled session"}
          <span className="src"> · {statusLabel(session.status, session.terminal_reason)}</span>
        </span>
      </span>
      {session.mode === "unattended" && <span className="tag">unattended</span>}
      {session.open_questions > 0 && <span className="tag accent">needs you</span>}
      <span className="when">
        {session.hops_used}/{session.hops_max} · {relTime(session.last_event_at)}
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
      <div className="stack">
        {items.map((item) => (
          <button key={item.approval_id} className="row" onClick={() => onOpenSession(item.session_id)}>
            <span className="label">
              <Dot kind="work" />
              <span className="text">
                {item.prompt}
                <span className="src"> · {item.project_title || "no project"}</span>
              </span>
            </span>
            <span className="tag accent">{item.kind}</span>
            <span className="when">{relTime(item.created_at)}</span>
          </button>
        ))}
      </div>
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
          <div className="stack">
            {sessions.map((session) => (
              <SessionRow key={session.session_id} session={session} onOpen={onOpenSession} />
            ))}
          </div>
          {!sessions.length && <Empty glyph="◇">nothing has run in this project yet</Empty>}
        </div>
      ) : (
        <>
          <h2>Projects</h2>
          {loading && <Empty>reading…</Empty>}
          {!loading && !projects.length && <Empty glyph="◇">nothing here yet — start something below</Empty>}
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
        <button className="primary" type="submit" disabled={!goal.trim()}>
          start →
        </button>
      </form>
    </div>
  );
}
