/* =========================================================
   views — desk · approvals · computer · chat, plus settings and sign-in

   The design's compositions on real data. `watching` is gone: nothing in the
   system watches sources on a schedule, and a view for a feature that does not
   exist is a promise the product cannot keep.
   ========================================================= */

/* ---------- DESK ---------- */

function DeskView({ onError, waiting: pending, onOpenSession }) {
  const waiting = pending || [];
  const [running, setRunning] = useState([]);
  const [projects, setProjects] = useState([]);

  useEffect(() => {
    let dead = false;
    (async () => {
      try {
        const [r, p] = await Promise.all([api.sessions("running"), api.projects()]);
        if (dead) return;
        setRunning(r);
        setProjects(p);
      } catch (e) {
        if (!dead) onError(e);
      }
    })();
    return () => {
      dead = true;
    };
    // `waiting` comes from App, so this re-reads when the pending set changes
    // — which is the same moment the running list is worth re-reading.
  }, [waiting, onError]);

  return (
    <div className="view">
      <PageHead
        title="the desk"
        lede="what is waiting on you, what is running, and where the work lives. nothing acts without your say-so."
      />
      <div className="zones">
        <section className="zone">
          <header>
            <span className="kicker">waiting on you</span>
            <span className="n">{waiting.length}</span>
          </header>
          <div className="stack">
            {waiting.length === 0 ? (
              <Empty glyph="✓">nothing waiting on you</Empty>
            ) : (
              waiting.map((a) => (
                <ApprovalCard key={a.approval_id} item={a} onResolve={onOpenSession ? () => {} : undefined} onError={onError} />
              ))
            )}
          </div>
        </section>

        <section className="zone">
          <header>
            <span className="kicker">running</span>
            <span className="n">{running.length}</span>
          </header>
          <div className="stack">
            {running.length === 0 ? (
              <Empty glyph="○">nothing running</Empty>
            ) : (
              running.map((s) => (
                <div className="row" key={s.session_id} onClick={() => onOpenSession(s.session_id)}>
                  <span className="label">
                    <Spinner />
                    <span className="text">
                      {s.title || "untitled session"}
                      <span className="src"> · {s.project_title || "no project"}</span>
                    </span>
                  </span>
                  <span className="when">
                    {s.hops_used}/{s.hops_max}
                  </span>
                </div>
              ))
            )}
          </div>
        </section>

        <section className="zone">
          <header>
            <span className="kicker">projects</span>
            <span className="n">{projects.length}</span>
          </header>
          <div className="stack">
            {projects.length === 0 ? (
              <Empty glyph="◇">no projects yet</Empty>
            ) : (
              projects.slice(0, 8).map((p) => (
                <div className="row" key={p.id}>
                  <span className="label">
                    <Dot status={p.status_rollup} />
                    <span className="text">{p.title}</span>
                  </span>
                  <span className="when">{relTime(p.updated_at)}</span>
                </div>
              ))
            )}
          </div>
        </section>
      </div>
    </div>
  );
}

/* ---------- APPROVALS ---------- */

function ApprovalsView({ onError, waiting, onResolved }) {
  // Null until App has read it once; the view says so rather than "all caught up".

  return (
    <div className="view">
      <PageHead
        title="approvals"
        lede="questions a run stopped to ask. answering here is answering in its window — one row, wherever you see it."
      />
      <div className="stack appr">
        {waiting === null && <Empty>reading…</Empty>}
        {waiting !== null && waiting.length === 0 && <Empty glyph="✓">all caught up — nothing waiting on you</Empty>}
        {(waiting || []).map((a) => (
          <ApprovalCard
            key={a.approval_id}
            item={a}
            onResolve={onResolved}
            onError={onError}
          />
        ))}
      </div>
    </div>
  );
}

/* ---------- COMPUTER ---------- */

/* The filesystem, read from the store rather than from a booted box. The files
   a session works on live here whether or not anything is awake (D27), so this
   view wakes nothing — and what it shows is exactly what the next materialize
   will put on the disk. */
function ComputerView({ onError }) {
  const [projects, setProjects] = useState([]);
  const [projectId, setProjectId] = useState("");
  const [files, setFiles] = useState(null);
  const [open, setOpen] = useState(null);
  // Directories start closed: a clone is hundreds of files and none of them is
  // what you came to look at.
  const [expanded, setExpanded] = useState(() => new Set());
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  // `target` is where the next drop or pick lands, "" being the project root;
  // `moving` is the path being dragged, which is what makes a drop a rearrange
  // rather than an upload.
  const [moving, setMoving] = useState(null);
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState("");
  const [target, setTarget] = useState("");
  const cancelled = useRef(false);
  // The editor, as the design draws it. `draft` is null when reading; a string
  // means the pane is in edit mode, dirty or not.
  const [draft, setDraft] = useState(null);
  const [saving, setSaving] = useState(false);

  // What the panel counts. The sentinels are structure, not content.
  const shown = files ? files.filter((f) => !isSentinel(f.path)) : null;

  const toggle = (path) =>
    setExpanded((current) => {
      const next = new Set(current);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });

  useEffect(() => {
    api
      .projects()
      .then((list) => {
        setProjects(list);
        if (list.length && !projectId) setProjectId(list[0].id);
      })
      .catch(onError);
  }, [onError]);

  useEffect(() => {
    if (!projectId) return;
    setFiles(null);
    setOpen(null);
    setExpanded(new Set());
    setCreating(false);
    setNewName("");
    setTarget("");
    api.files(projectId).then(setFiles).catch(onError);
  }, [projectId, onError]);

  const read = async (file) => {
    setDraft(null);
    setOpen({ path: file.path, loading: true });
    try {
      setOpen(await api.file(projectId, file.file_id));
    } catch (e) {
      setOpen(null);
      onError(e);
    }
  };

  /* The store is the durable copy, so saving here is not a scratch edit: a
     session already holding this project is written through and reads it on its
     next turn, and every other session gets it at its next materialize. */
  const save = async () => {
    if (draft === null || !open) return;
    setSaving(true);
    try {
      await api.saveFile(projectId, open.path, draft);
      setOpen({ ...open, text: draft, size: new Blob([draft]).size });
      setDraft(null);
      setFiles(await api.files(projectId));
    } catch (e) {
      onError(e);
    } finally {
      setSaving(false);
    }
  };

  // Only text is editable. A binary file has no draft to hold and no editor
  // that would not corrupt it.
  const canEdit = !!open && !open.loading && !open.binary;
  const dirty = draft !== null && draft !== (open && open.text);

  /* Dropped or picked, they go to the store — the same endpoint the session
     window's panel uses, because it is the same shelf. A session already
     holding this project is written through and reads them on its next turn. */
  const add = async (list, dir) => {
    if (!projectId || !list || !list.length) return;
    const into = dir === undefined ? target : dir;
    setBusy(true);
    try {
      for (const file of list) await api.upload(projectId, file, into);
      setFiles(await api.files(projectId));
    } catch (e) {
      onError(e);
    } finally {
      setBusy(false);
      setDragging(false);
    }
  };

  /* Enter or blur commits, Escape cancels. Leading and trailing slashes come
     off so the name cannot read as absolute; `a/b` nests, and every prefix is
     opened so what you just made is on screen instead of inside a shut parent.
     The new folder becomes the target, because a folder you cannot put a file
     in is furniture.

     It goes to the server before it is drawn: the tree it appears in is the
     store's, so a folder held here until a file arrived would be a second
     version of the truth that the next reload would contradict. */
  const commitFolder = async () => {
    /* Escape unmounts the input, and removing a focused element fires blur —
       which would commit the name the user just cancelled. The flag is read and
       cleared here so that blur does nothing. */
    if (cancelled.current) {
      cancelled.current = false;
      return;
    }
    const name = newName.trim().replace(/^\/+|\/+$/g, "");
    setCreating(false);
    setNewName("");
    if (!name) return;
    setBusy(true);
    try {
      await api.newFolder(projectId, name);
      setFiles(await api.files(projectId));
      setExpanded((current) => {
        const next = new Set(current);
        const parts = name.split("/");
        for (let i = 0; i < parts.length; i++) next.add(parts.slice(0, i + 1).join("/"));
        return next;
      });
      setTarget(name);
    } catch (e) {
      onError(e);
    } finally {
      setBusy(false);
    }
  };

  /* Rearranging. The store moves in one transaction and every live box is
     corrected in the same request, so the only copy that can be behind is this
     one — which is why the tree is re-read rather than patched in place. */
  const move = async (from, into) => {
    const dest = into || "";
    const to = dest ? `${dest}/${from.split("/").pop()}` : from.split("/").pop();
    // Into itself, or back where it already is: both are no-ops, and the first
    // would be a folder swallowing its own subtree.
    if (to === from || dest === from || dest.startsWith(from + "/")) return;
    setBusy(true);
    try {
      const result = await api.moveFile(projectId, from, to);
      setFiles(await api.files(projectId));
      if (dest) setExpanded((current) => new Set(current).add(dest));
      // The row id survives a move, so what is open is still open — under a new name.
      if (open && (open.path === from || open.path.startsWith(from + "/"))) {
        setOpen({ ...open, path: to + open.path.slice(from.length) });
      }
      if (result.stale_sessions && result.stale_sessions.length) {
        onError(
          new ApiError(
            "move_stale",
            `Moved in the store, but ${result.stale_sessions.length} running session(s) still hold the old path on disk.`
          )
        );
      }
    } catch (e) {
      onError(e);
    } finally {
      setBusy(false);
      setMoving(null);
      setTarget("");
    }
  };

  return (
    <div className="view" style={{ padding: 0, height: "100%" }}>
      <div
        className="computer"
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
          const into = target;
          setDragging(false);
          setMoving(null);
          setTarget("");
          // Lifted from the tree: a rearrange. From the desktop: an upload.
          if (lifted) move(lifted, into);
          else add(e.dataTransfer.files, into);
        }}
      >
        <div className="cv-files">
          <div className="cv-project">
            <select value={projectId} onChange={(e) => setProjectId(e.target.value)}>
              {projects.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.title}
                </option>
              ))}
            </select>
          </div>
          <div className="cv-head">
            <span className="path">
              {busy ? "working…" : shown ? `${shown.length} file${shown.length === 1 ? "" : "s"}` : "…"}
            </span>
            <span className="cv-acts">
              <label className="cv-add" title="attach files to this project">
                + file
                <input
                  type="file"
                  multiple
                  hidden
                  onChange={(e) => {
                    // Into the folder you just made, else the one you are reading.
                    add(e.target.files, target || dirOf(open && open.path));
                    e.target.value = "";
                  }}
                />
              </label>
              <span
                className="cv-add"
                title="new folder"
                onClick={() => {
                  // If a blur never came to clear it, the last Escape must not
                  // swallow this one.
                  cancelled.current = false;
                  setCreating(true);
                  setNewName("");
                }}
              >
                + folder
              </span>
            </span>
          </div>
          <div className="cv-entries">
            {creating && (
              <div className="cv-newdir">
                <span className="g">▸</span>
                <input
                  value={newName}
                  autoFocus
                  spellCheck={false}
                  placeholder="folder name"
                  onChange={(e) => setNewName(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") commitFolder();
                    if (e.key === "Escape") {
                      cancelled.current = true;
                      setCreating(false);
                      setNewName("");
                    }
                  }}
                  onBlur={commitFolder}
                />
              </div>
            )}
            {files === null && <div className="cv-entry loading">reading…</div>}
            {files !== null && !files.length && !creating && (
              <div className="cv-entry loading">no files in this project</div>
            )}
            {files !== null && files.length > 0 && (
              <Branch
                node={asTree(files)}
                path=""
                depth={0}
                open={expanded}
                onToggle={toggle}
                onRead={read}
                selected={open ? open.path : null}
                dropTarget={dragging ? target : null}
                onTarget={setTarget}
                onLift={(path) => {
                  setMoving(path);
                  // dragend with nothing lifted means the drag was abandoned —
                  // Escape, or a drop outside the panel — and no drop is coming.
                  if (!path) {
                    setDragging(false);
                    setTarget("");
                  }
                }}
                lifted={moving}
              />
            )}
          </div>
        </div>

        <div className="cv-read">
          {/* Over the reader, never over the tree: the tree is what you are
              aiming at, and the row underline is the answer to "where". */}
          {dragging && (
            <div className="cv-drop">
              <span>
                {moving ? "move" : "drop to add"} to{" "}
                {target || (projects.find((p) => p.id === projectId) || {}).title || "this project"}
              </span>
            </div>
          )}
          <div className="cv-read-head">
            <span className="path">
              {open ? open.path : "select a file to read"}
              {dirty && <span className="dirty"> ●</span>}
            </span>
            <span className="acts">
              {open && !open.loading && <span>{fileSize(open.size)}</span>}
              {canEdit && (
                <button
                  className="cv-edit"
                  onClick={() => setDraft(draft === null ? open.text || "" : null)}
                >
                  {draft === null ? "edit" : "reading"}
                </button>
              )}
              {dirty && (
                <button className="cv-revert" onClick={() => setDraft(open.text || "")}>
                  revert
                </button>
              )}
            </span>
          </div>
          {draft !== null ? (
            <React.Fragment>
              <textarea
                className="cv-editor"
                value={draft}
                spellCheck={false}
                onChange={(e) => setDraft(e.target.value)}
              />
              <div className="cv-editbar">
                <span>
                  {dirty
                    ? "unsaved changes in this project's copy"
                    : "editing the project's copy — sessions mount it on their next box"}
                </span>
                <button className="cv-save" onClick={save} disabled={saving || !dirty}>
                  {saving ? "saving…" : "save"}
                </button>
              </div>
            </React.Fragment>
          ) : open && open.loading ? (
            <div className="cv-read-body" style={{ fontStyle: "italic", color: "var(--ink-mute)" }}>
              reading…
            </div>
          ) : open && open.binary ? (
            <div className="cv-binary">this file is not text — {fileSize(open.size)} of it</div>
          ) : open ? (
            <pre className="cv-read-body">
              {(open.text || "").split("\n").map((line, i) => (
                <div key={i}>
                  <span className="ln">{String(i + 1).padStart(3, " ")}</span>
                  {line || " "}
                </div>
              ))}
            </pre>
          ) : (
            <div className="cv-read-body" style={{ color: "var(--ink-mute)", fontStyle: "italic" }}>
              the project's files live in the store, not on a computer. click one to read it — nothing has to
              be awake.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}


/* ---------- SETTINGS ---------- */

function SettingsModal({ user, onClose, onSignOut, onError }) {
  const [rows, setRows] = useState(null);
  const [problem, setProblem] = useState(null);
  const [busy, setBusy] = useState({});
  const [links, setLinks] = useState({});
  const poll = useRef(null);

  const refresh = useCallback(async () => {
    try {
      setRows(await api.connections());
      setProblem(null);
    } catch (e) {
      setProblem(e.message || "could not read connections");
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  /* Re-read when this window becomes the one being looked at again.

     The OAuth popup is on another origin, so nothing tells this page when it
     finishes — which is why this used to poll `api.connections()` every two
     seconds. Contracts forbids polling anywhere, and it was also the wrong
     shape: it burned requests while the user was still typing a password, and
     stopped the moment they tabbed away. Coming back to the tab, or the popup
     closing, IS the event. `postMessage` from the callback page arrives too
     (`/oauth/callback/{server}` posts one before it closes), and whichever
     lands first wins — `refresh` is idempotent. */
  useEffect(() => {
    const again = () => {
      if (document.visibilityState === "visible") refresh();
    };
    const posted = (e) => {
      if (e.data && e.data.type === "arkos:connection") refresh();
    };
    window.addEventListener("focus", again);
    document.addEventListener("visibilitychange", again);
    window.addEventListener("message", posted);
    return () => {
      window.removeEventListener("focus", again);
      document.removeEventListener("visibilitychange", again);
      window.removeEventListener("message", posted);
      if (poll.current) clearInterval(poll.current);
    };
  }, [refresh]);

  /* The popup closing is the other end of the same signal, and it is the only
     one a user who never leaves the tab will produce. One watcher, cleared as
     soon as the window is gone — this checks a boolean, it does not fetch. */
  function watch(server, popup) {
    if (!popup) return;
    if (poll.current) clearInterval(poll.current);
    poll.current = setInterval(() => {
      if (!popup.closed) return;
      clearInterval(poll.current);
      poll.current = null;
      refresh();
    }, 500);
  }

  function connect(server) {
    // Opened SYNCHRONOUSLY inside the click handler: after an await the browser
    // has lost the user gesture and blocks it silently.
    const popup = window.open("about:blank", "ark_oauth", "width=560,height=720");
    setBusy((b) => ({ ...b, [server]: true }));
    setLinks((m) => ({ ...m, [server]: null }));

    api
      .connect(server)
      .then((result) => {
        if (result.status === "connected") {
          if (popup) popup.close();
          refresh();
          return;
        }
        if (result.setup_url) {
          if (popup && !popup.closed) popup.location.href = result.setup_url;
          else setLinks((m) => ({ ...m, [server]: result.setup_url }));
          watch(server, popup);
        } else {
          if (popup) popup.close();
          setProblem("could not start authorization for " + server);
          refresh();
        }
      })
      .catch((e) => {
        if (popup) popup.close();
        setProblem(e.message || String(e));
      })
      .finally(() => setBusy((b) => ({ ...b, [server]: false })));
  }

  async function disconnect(server) {
    setBusy((b) => ({ ...b, [server]: true }));
    try {
      await api.disconnect(server);
      await refresh();
    } catch (e) {
      setProblem(e.message || String(e));
    } finally {
      setBusy((b) => ({ ...b, [server]: false }));
    }
  }

  return (
    <div className="overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h2>settings</h2>
        <p className="sub">connections and account.</p>

        <section>
          <span className="kicker">tools &amp; connections</span>
          {rows === null && <div className="soft" style={{ fontSize: 12 }}>loading…</div>}
          {problem && <div className="soft" style={{ fontSize: 12, color: "var(--stop)" }}>{problem}</div>}
          {rows !== null && !rows.length && (
            <div className="soft" style={{ fontSize: 12 }}>
              no servers configured. add entries to <code>mcp_servers</code> in config.yaml.
            </div>
          )}
          {(rows || []).map((row) => {
            const connected = row.status === "connected";
            return (
              <div className="conn" key={row.server}>
                <span className="meta">
                  <Dot kind={connected ? "live" : ""} />
                  <span className="nm">{row.name || row.server}</span>
                </span>
                <span style={{ display: "flex", alignItems: "center", gap: 12 }}>
                  <span className={"st" + (connected ? " on" : "")}>
                    {connected ? "connected" : row.requires_auth ? row.status : "shared"}
                  </span>
                  {!row.requires_auth ? (
                    <span className="soft" style={{ fontSize: 10 }}>always on</span>
                  ) : connected ? (
                    <button className="btn" disabled={busy[row.server]} onClick={() => disconnect(row.server)}>
                      disconnect
                    </button>
                  ) : links[row.server] ? (
                    <a className="btn primary" href={links[row.server]} target="ark_oauth" rel="noopener" onClick={() => watch(row.server, null)}>
                      authorize →
                    </a>
                  ) : (
                    <button className="btn primary" disabled={busy[row.server]} onClick={() => connect(row.server)}>
                      {busy[row.server] ? "…" : "connect"}
                    </button>
                  )}
                </span>
              </div>
            );
          })}
        </section>

        <section>
          <span className="kicker">account</span>
          <div className="soft" style={{ fontSize: 12, lineHeight: 1.9 }}>
            signed in as <b style={{ color: "var(--ink)" }}>{(user && (user.email || user.user_id)) || "—"}</b>
          </div>
        </section>

        <div className="foot">
          <span className="mute" style={{ fontSize: 10.5 }}>changes save immediately</span>
          <div style={{ display: "flex", gap: 8 }}>
            <button className="btn danger" onClick={onSignOut}>sign out</button>
            <button className="btn" onClick={onClose}>close</button>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ---------- SIGN IN ---------- */

/* The design's login card, with the flow that actually exists: Supabase takes
   the email and password, we take the token once and turn it into the cookie.
   No signup link — accounts are made in the dashboard until there are real
   users to self-serve. */
function Login({ gone, onSignedIn }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState(null);
  const first = useRef(null);

  useEffect(() => {
    if (!gone && first.current) first.current.focus();
  }, [gone]);

  async function submit() {
    if (!email.trim() || !password) return;
    setBusy(true);
    setProblem(null);
    try {
      onSignedIn(await api.signIn(email.trim(), password));
    } catch (e) {
      setProblem(e.message || "sign-in failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className={"login" + (gone ? " gone" : "")}>
      <div className="login-card">
        <div className="mark-lg">
          ark<span className="pip" />
        </div>
        <p>
          your digital life, handled in the background. sign in and pick up wherever the last conversation
          left off.
        </p>
        <div className="field">
          <label>email</label>
          <input
            ref={first}
            type="email"
            value={email}
            placeholder="you@example.com"
            spellCheck={false}
            autoComplete="username"
            onChange={(e) => setEmail(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && submit()}
          />
        </div>
        <div className="field">
          <label>password</label>
          <input
            type="password"
            value={password}
            placeholder="••••••••"
            autoComplete="current-password"
            onChange={(e) => setPassword(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && submit()}
          />
        </div>
        <div className="go">
          <span className="hint">press enter to continue</span>
          <button className="btn primary lg" onClick={submit} disabled={busy || !email.trim() || !password}>
            {busy ? "…" : "enter →"}
          </button>
        </div>
        {problem && <p className="problem">{problem}</p>}
      </div>
    </div>
  );
}
