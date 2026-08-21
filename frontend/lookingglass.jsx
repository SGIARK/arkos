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

function LookingGlassView({ onError, pulse, waiting: pending, onPulse, jump, onJumped, onOpenFile }) {
  const [projects, setProjects] = useState(null);
  // From App, which reads it once per pulse for every surface that shows it.
  const waiting = pending || [];
  const [openProject, setOpenProject] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [openSession, setOpenSession] = useState(null);
  const [making, setMaking] = useState(false);
  // The project whose name is being edited, and the text so far. One at a time.
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

  /* Enter and blur both commit, escape cancels — the export's contract, and the
     one people already have for renaming a file. An unchanged or empty name is
     a no-op rather than a request: renaming a thing to what it is called is not
     an edit. */
  const commitRename = async () => {
    const id = renaming;
    const title = renameText.trim();
    setRenaming(null);
    if (!id || !title) return;
    const was = (projects || []).find((p) => p.id === id);
    if (was && was.title === title) return;
    // Optimistic: the name is yours, and waiting a round trip to see your own
    // typing is the one latency a rename cannot have.
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
      /* Keyed by the session: every piece of local state in there — the plan
         lane's drafting flag, the resume note, the dismissed version — is about
         ONE session, and reusing the instance across two carried it over. */
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
            // Land in the new project, which is what the plus was for.
            open({ id: project.id, title: project.title });
          }}
        />
      )}
    </div>
  );
}

/* The new-project modal.

   Two ways to start, and they differ in what the project LINKS: folders that
   already exist in the store, or a fresh one named after the project. A project
   owns no folder either way — linking is a fact about which work reads and
   writes where, and the none-case folder is itself just linked (11.9).

   The choices come from `GET /folders`, which is the store's top-level segments
   grouped with their file counts. Not from the projects: a folder nothing links
   is still a folder, and deriving the list from projects would hide exactly the
   folders a new project is most likely to want. */
function NewProject({ onClose, onMade, onError }) {
  const [name, setName] = useState("");
  const [mode, setMode] = useState("new");
  const [folders, setFolders] = useState(null);
  const [picked, setPicked] = useState(() => new Set());
  const [busy, setBusy] = useState(false);

  useEscape(true, onClose);

  // Only when asked for: a person starting an empty project should not pay for
  // a listing they are not going to read.
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

  /* The footer previews the OUTCOME, because the two modes do different things
     and the difference is the whole choice: one points at files that exist, the
     other makes a folder that does not. */
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

/* One session, live. Every session is the same kind of thing (D5), so this is
   the only window there is — reached from a project, or from the desk's running
   list, and since 11.9 a session need not belong to a project at all. */
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
  // Renaming the project from its own session header: the same gesture as on
  // the card, because it is the same name.
  const [headRename, setHeadRename] = useState(false);
  const [headText, setHeadText] = useState("");
  const tail = useRef(null);

  /* The plan lane. Two pieces of local state, and each is a thing the server
     genuinely does not know:
       drafting  — the play button was pressed, or a reply was sent, and the
                   turn drafting the next plan is in flight. There is no plan
                   row to read yet.
       dismissed — a dismissed plan, kept visible as "nothing ran" until the
                   next thing happens. The row is closed; only this window
                   cares.
     The open plan itself is NOT held here. It is `questions`, and it is on
     screen exactly while the server says a plan is waiting: replying closes the
     row, the card goes with it, and the next plan arrives as a new card. */
  const [drafting, setDrafting] = useState(false);
  // The last note that resumed a held run, echoed beside the plan so the reason
  // the run picked back up is visible next to what it picked up.
  const [resumeNote, setResumeNote] = useState("");
  // The plan approved in THIS window. The snapshot's `plan` is the same fact
  // from the server and is what survives a reload, but it was read before the
  // approval, so the pinned card would not appear until then without this.
  const [approvedHere, setApprovedHere] = useState(null);
  /* Dismissing a finished plan clears it from THIS surface and nothing else:
     the row is a permanent fact and is never deleted. Kept per browser rather
     than in memory, because a reload resurrecting a plan you dismissed reads as
     the dismissal not having worked. */
  const dismissKey = `ark-plan-dismissed-${sessionId}`;
  const [dismissedVersion, setDismissedVersion] = useState(() => {
    const stored = Number(localStorage.getItem(dismissKey));
    return Number.isFinite(stored) && stored > 0 ? stored : 0;
  });
  const dismissPlan = (version) => {
    localStorage.setItem(dismissKey, String(version || 1));
    setDismissedVersion(version || 1);
    setApprovedHere(null);
    setResumeNote("");
  };

  /* Renaming the project from its own window. Keyed off the SNAPSHOT's ids
     rather than the grid's navigation state, so it works in a window opened
     from the desk — where there is no `project` prop at all — and the header
     re-reads the name from the server rather than from a mutated prop. */
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

  /* The lane's bookkeeping, all of it driven by rows arriving rather than by
     timers: a plan appearing ends the drafting, and a session that stops
     running ends it whatever else happened. */
  const openPlan = questions.find((q) => q.kind === "plan") || null;
  // A held run. It is a park like any other on the wire, and the surface it gets
  // is one transcript row plus the second face of the run button.
  const heldRun = questions.find((q) => q.kind === "resume") || null;
  useEffect(() => {
    if (openPlan) {
      setDrafting(false);
      setApprovedHere(null);
      setResumeNote("");
    }
  }, [openPlan]);
  useEffect(() => {
    // A turn that ended without a plan: the model answered in prose instead of
    // proposing. The spinner stops and the play button comes back rather than
    // waiting on something nothing is drafting.
    if (session && session.status !== "running") setDrafting(false);
  }, [session && session.status]);
  useEffect(() => {
    if (tail.current) tail.current.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [events.length, pending.length]);

  if (!session) return <div className="lg-view"><div className="projects-grid"><Empty>opening…</Empty></div></div>;

  const running = session.status === "running";
  const unattended = session.mode === "unattended";
  // Where an approved plan lands: the FIRST folder this session writes.
  const planFolder = (session.folders || [])[0] || null;
  /* The snapshot first, the navigation state second: a window opened from the
     desk has no `project` prop, and the header should not depend on how you
     got here. Null for a session with no project — the home chat is one. */
  const projectTitle = session.project_title || (project && project.title) || null;
  /* No directory chips in the header (11.9). A session may hold SEVERAL linked
     folders, so one chip was never going to be right, and where work lands is
     said by the plan card and the working-files pane — the two places that show
     it in full rather than in a slot that fits one. */

  // The open plan row, and the session's newest plan whatever became of it.
  const planCard = openPlan;
  /* Everything below reads the server's `plan` — its `answer` is what became of
     it — with one local override for the approval made in THIS window, which the
     snapshot was read before. Dismissing hides whatever version was dismissed. */
  const serverPlan = session.plan && session.plan.version > dismissedVersion ? session.plan : null;
  const approvedPlan = approvedHere || (serverPlan && serverPlan.answer === "approve" ? serverPlan : null);
  const abandonedPlan = serverPlan && serverPlan.answer === "decline" ? serverPlan : null;

  /* The faces of a plan that has been decided. `held` is a live park;
     `cancelled` is a spent one, and it is the only place the run button becomes
     "resume" — pressing it drafts a CONTINUATION rather than a fresh v1. */
  const held = !!heldRun;
  const cancelled = !!approvedPlan && session.status === "cancelled";
  // The pin outlives the run: an approved plan that finished still says where
  // it was saved. It goes only when the plan is dismissed.
  const showPin = !!approvedPlan;
  const stopStep = session.hops_used || 1;
  /* ▶ is offered when nothing is pending on the human and nothing is in flight:
     an idle session, a finished one, a dismissed plan, or a cancelled run
     (where it reads "resume"). A held run has its own two faces instead. */
  const showRun = !planCard && !drafting && !held && !running && !unattended;

  const cancelHeld = () =>
    api
      .answer(heldRun.approval_id, "decline")
      .then(() => {
        refreshQuestions();
        refreshSession();
        if (onPulse) onPulse();
      })
      .catch(onError);

  const resumeHeld = () =>
    api
      .answer(heldRun.approval_id, "approve")
      .then(() => {
        setResumeNote("");
        refreshQuestions();
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
        {/* THE PROJECT'S NAME, and nothing else. The session's own title was a
            second crumb here and it earned none of the room: it repeated the
            first line of the transcript underneath it, and where it did not —
            a session with no project — the fallback printed the SAME name
            twice, "Chat ▸ Chat", with a crumb between them promising a
            container that does not exist. A session with no project shows no
            name here rather than an invented one; the transcript says what it
            is. The title comes from the SNAPSHOT, so the header reads the same
            whether the window was opened from the grid or from the desk. */}
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
          {/* The run control has three faces and never two at once.

              ▶ asks for a PLAN — it does not hand the session over, which is
              what its tooltip promises and what the endpoint does. On a
              CANCELLED run it reads "resume" and drafts a continuation instead
              of a fresh v1: same request, different starting point, because the
              handoff tells the model to read plan.md and the transcript first.

              ■ stop holds a running turn. ✕ cancel is the second press and the
              only one that spends the plan. Before 11.8.6 the first press was
              the only press. */}
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
                "stops at the hop boundary. in-flight calls close as cancelled and count against " +
                "nothing; the plan's approval stands. if the run hangs, this becomes a hard cancel " +
                "after a grace period."
              }
              onClick={() => api.stop(sessionId).catch(() => api.cancel(sessionId).catch(onError))}
            >
              <span className="square" />
              stop
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
            {/* Answered where it was asked, not in a separate tray — except a
                plan, which is not a question and does not answer like one: it
                gets the lane below. */}
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
            {/* ---- the plan lane. Six faces, never two at once: drafting,
                    the open card, the running plan, a held run, a spent one, a
                    dismissed one. It lives INSIDE the transcript and scrolls
                    with it — a docked strip overlaid the conversation, and the
                    transcript is meant to be the only surface. ---- */}
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
                    });
                  }
                  // A reply wakes the session to propose again, so the lane goes
                  // straight back to drafting rather than blanking until the
                  // next card lands.
                  if (outcome === "replied") setDrafting(true);
                  refreshQuestions();
                  refreshSession();
                  if (onPulse) onPulse();
                }}
              />
            )}

            {/* The approved plan, collapsed to one line and the file it became.
                `plan.md` is not a phrase here — the run really does start from
                it. Held, the dot goes amber and stops pinging: a stopped run is
                alive, and it should not look like one that is working. */}
            {showPin && (
              <div className={"plan-pinned" + (held ? " held" : "")}>
                <span className="pin-dot" />
                <span className="goal">{approvedPlan.goal}</span>
                <span className="ver">v{approvedPlan.version}</span>
                {/* The real file, at its real path — the run starts from it, so
                    the chip opens it rather than naming it. It lands in the
                    session's FIRST linked folder (11.9), which is the first
                    one the session claimed. */}
                <span
                  className={"file" + (planFolder && onOpenFile ? " open" : "")}
                  title={planFolder ? `${planFolder}/plan.md` : "plan.md"}
                  onClick={() => planFolder && onOpenFile && onOpenFile(`${planFolder}/plan.md`)}
                >
                  plan.md
                </span>
                <span className="state">
                  {held ? "stopped" : planFolder ? `saved in ${planFolder}/` : "saved to the store"}
                </span>
              </div>
            )}

            {/* Why the run picked back up, next to what it picked up. */}
            {!held && resumeNote && showPin && (
              <div className="plan-resumed">
                <span className="lede">resumed with your note</span>
                <span className="said">"{resumeNote}"</span>
              </div>
            )}

            {held && (
              <div className="plan-stopped">
                <span className="square" />
                <span className="said">
                  stopped at step {stopStep}. in-flight calls closed, nothing counts against the plan.
                </span>
                <button className="go" onClick={resumeHeld}>
                  resume
                </button>
                <button className="link danger" onClick={cancelHeld}>
                  cancel
                </button>
              </div>
            )}

            {cancelled && (
              <div className="plan-spent">
                <span className="said">
                  run cancelled at step {stopStep}. the plan's approval is spent.
                </span>
                <span className="grow" />
                <button
                  className="link"
                  onClick={() => {
                    setDrafting(true);
                    api.approve(sessionId).catch((e) => {
                      setDrafting(false);
                      onError(e);
                    });
                  }}
                >
                  draft a continuation
                </button>
                <button className="link mute" onClick={() => dismissPlan(approvedPlan.version)}>
                  dismiss
                </button>
              </div>
            )}

            {abandonedPlan && !planCard && !drafting && (
              <div className="plan-spent">
                <span className="said">plan v{abandonedPlan.version} dismissed, nothing ran</span>
                <span className="grow" />
                <button
                  className="link"
                  onClick={() => {
                    setDrafting(true);
                    api.approve(sessionId).catch((e) => {
                      setDrafting(false);
                      onError(e);
                    });
                  }}
                >
                  draft again
                </button>
                <button className="link mute" onClick={() => dismissPlan(abandonedPlan.version)}>
                  dismiss
                </button>
              </div>
            )}

            <div ref={tail} />
          </div>

          <form
            className="lg-composer"
            onSubmit={(e) => {
              e.preventDefault();
              const said = text.trim();
              if (!said) return;
              // A held run resumes on what is typed here — kind `resume` is the
              // one park the composer may answer — and the note is the next
              // thing the model reads, so it is worth saying that it landed.
              if (held) setResumeNote(said);
              send(said);
              setText("");
            }}
          >
            <SessionTools sessionId={sessionId} onError={onError} />
            <span className="prompt">ark&gt;</span>
            <input
              value={text}
              onChange={(e) => setText(e.target.value)}
              /* A stopped run resumes on what is typed here — kind `resume` is
                 exempt from the composer's 409, because the plan it holds on is
                 already approved. Saying so is the difference between a held run
                 and a dead one. */
              placeholder={held ? "type to resume. your note is the next thing ark reads" : "suggest or steer this session…"}
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

  useEscape(open, () => setOpen(false));

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

/* WORKING FILES: the folders this project links, and nothing else.

   The SAME `FileTree` the Files tab draws, with the same powers — open, drag to
   move, drop to upload, double-click to rename. The two are scopes on one
   store, not two filesystems, so a file renamed here is renamed there, and a
   path clicked here is the path the tab lands on.

   What is local to this pane is the LINKING. `+ link` adds a folder the project
   does not yet link; the pane shows it at once and the AGENT sees it from the
   NEXT session, because claims are fixed for a session's life. That is stated
   on the control rather than hidden — a folder appearing under a running agent
   mid-hop would be a mount and a lease it was never told about. */
function FilesCanvas({ projectId, folders, onError, onOpenFile }) {
  const [linked, setLinked] = useState(folders || []);
  const [linking, setLinking] = useState(false);
  const [choices, setChoices] = useState(null);
  const [busy, setBusy] = useState(false);
  // Bumped when a link lands, so the tree re-reads with the new folder in it.
  const [pulse, setPulse] = useState(0);

  useEffect(() => {
    setLinked(folders || []);
  }, [folders]);

  const load = useCallback(() => api.files(projectId), [projectId, pulse]);

  // Only when the picker is opened: a listing nobody asked for is a request
  // spent on a control that is shut.
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

  useEscape(big, () => setBig(false));

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

