/* =========================================================
   views — desk · approvals · files, plus settings and sign-in

   The design's compositions on real data. `watching` is gone: nothing in the
   system watches sources on a schedule, and a view for a feature that does not
   exist is a promise the product cannot keep. `chat` went the same way in 11.8,
   and `computer` is `files` — named for what it shows.
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

/* ---------- FILES ---------- */

/* THE STORE ITSELF. One flat namespace per user: the header dropdowns are its
   top-level segments — `triage/`, `notes/` — and a folder appears the moment a
   file lands under a new first segment. There is no project picker, because a
   project does not own a directory: it LINKS folders, several of them, and
   renaming one cannot touch this view (11.9).

   The tree is `FileTree`, the SAME component the session window's working-files
   pane uses, with the same powers in both: open, drag to move, drop to upload,
   double-click to rename. What differs is the scope it loads and what a click
   does — everything else is behaviour, and behaviour has one implementation.

   Read from the store rather than from a booted box. The files a session works
   on live here whether or not anything is awake (D27), so this view wakes
   nothing — and what it shows is exactly what the next materialize will put on
   the disk. */
function ComputerView({ onError, jumpTo, onJumped }) {
  const [open, setOpen] = useState(null);
  const [count, setCount] = useState(null);
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState("");
  const [drag, setDrag] = useState({ dragging: false, target: "", moving: null });
  const cancelled = useRef(false);
  // The editor, as the design draws it. `draft` is null when reading; a string
  // means the pane is in edit mode, dirty or not.
  const [draft, setDraft] = useState(null);
  const [saving, setSaving] = useState(false);
  // Bumped to make the tree re-read after a write that happened outside it.
  const [pulse, setPulse] = useState(0);

  const load = useCallback(() => api.storeFiles(), [pulse]);

  const read = useCallback(
    async (file) => {
      setDraft(null);
      setOpen({ path: file.path, loading: true });
      try {
        setOpen(await api.file(file.file_id));
      } catch (e) {
        setOpen(null);
        onError(e);
      }
    },
    [onError]
  );

  /* The store is the durable copy, so saving here is not a scratch edit: a
     session already holding this folder is written through and reads it on its
     next turn, and every other session gets it at its next materialize. */
  const save = async () => {
    if (draft === null || !open) return;
    setSaving(true);
    try {
      await api.saveFile(open.path, draft);
      setOpen({ ...open, text: draft, size: new Blob([draft]).size });
      setDraft(null);
      setPulse((n) => n + 1);
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

  /* Enter or blur commits, Escape cancels. Leading and trailing slashes come
     off so the name cannot read as absolute; `a/b` nests.

     It goes to the server before it is drawn: this IS the store's tree, so a
     folder held here until a file arrived would be a second version of the
     truth that the next reload would contradict. */
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
    try {
      await api.newFolder(name);
      setPulse((n) => n + 1);
    } catch (e) {
      onError(e);
    }
  };

  // What the panel counts. The sentinels are structure, not content.
  const shown = count === null ? null : count.filter((f) => !isSentinel(f.path));

  /* `add` is the TREE's uploader, handed back so the header's `+ file` and a
     drop are the same code — a second copy here is how the two panes drifted
     apart in the first place. */
  const header = ({ busy, add }) => (
    <React.Fragment>
      <div className="cv-head">
        <span className="path">
          {busy ? "working…" : shown ? `${shown.length} file${shown.length === 1 ? "" : "s"}` : "…"}
        </span>
        <span className="cv-acts">
          <label className="cv-add" title="add files to the store">
            + file
            <input
              type="file"
              multiple
              hidden
              onChange={(e) => {
                // Into the folder you are aiming at, else the one you are reading.
                add(e.target.files, drag.target || dirOf(open && open.path));
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
    </React.Fragment>
  );

  return (
    <div className="view" style={{ padding: 0, height: "100%" }}>
      <div className="computer">
        <div className="cv-files">
          <FileTree
            load={load}
            onOpen={read}
            onError={onError}
            onFiles={setCount}
            reveal={jumpTo}
            onRevealed={onJumped}
            header={header}
            onDragState={setDrag}
            zoneIdle={{ label: "drop files into a folder", empty: "nothing in the store yet" }}
          />
        </div>

        <div className="cv-read">
          {/* Over the reader, never over the tree: the tree is what you are
              aiming at, and the row underline is the answer to "where". It names
              the TARGET, and says so plainly when there is not one — the top
              level is not a destination. */}
          {drag.dragging && (
            <div className={"cv-drop" + (drag.target || drag.movingDir ? "" : " nowhere")}>
              <span>
                {drag.target
                  ? `${drag.moving ? "move" : "drop"} into ${drag.target}/`
                  : drag.movingDir
                    ? `move ${drag.moving.split("/").pop()}/ out to the top level`
                    : "aim at a folder — the top level holds folders, not files"}
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
                    ? "unsaved changes in the store's copy"
                    : "editing the store's copy — sessions mount it on their next box"}
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
                  {line || " "}
                </div>
              ))}
            </pre>
          ) : (
            <div className="cv-read-body" style={{ color: "var(--ink-mute)", fontStyle: "italic" }}>
              your files live in the store, not on a computer. click one to read it — nothing has to be
              awake. a folder is a top-level directory here, and a project links the ones it works in.
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
  /* Which disconnect has been asked for once and is waiting to be meant. */
  const [armed, setArmed] = useState(null);
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

     The consent popup is on Arcade's origin and then the provider's, so nothing
     tells this page when it finishes — which is why this used to poll
     `api.connections()` every two seconds. Contracts forbids polling anywhere,
     and it was also the wrong shape: it burned requests while the user was
     still typing a password, and stopped the moment they tabbed away. Coming
     back to the tab, or the popup closing, IS the event. */
  useEffect(() => {
    const again = () => {
      if (document.visibilityState === "visible") refresh();
    };
    window.addEventListener("focus", again);
    document.addEventListener("visibilitychange", again);
    return () => {
      window.removeEventListener("focus", again);
      document.removeEventListener("visibilitychange", again);
      if (poll.current) clearInterval(poll.current);
    };
  }, [refresh]);

  /* The popup closing is the other end of the same signal, and it is the only
     one a user who never leaves the tab will produce. One watcher, cleared as
     soon as the window is gone — this checks a boolean, it does not fetch. */
  function watch(popup) {
    if (!popup) return;
    if (poll.current) clearInterval(poll.current);
    poll.current = setInterval(() => {
      if (!popup.closed) return;
      clearInterval(poll.current);
      poll.current = null;
      refresh();
    }, 500);
  }

  /* The link is already in hand: `GET /connections` asks Arcade for consent
     state and gets the url back in the same answer, so the popup opens INSIDE
     the click. After an await the browser has lost the user gesture and blocks
     it silently, which is the whole reason the url travels with the row. */
  function connect(row) {
    const href = links[row.server] || row.setup_url;
    if (href) {
      watch(window.open(href, "ark_oauth", "width=560,height=720"));
      return;
    }
    /* No link on the row — it expired, or this app was just added. Mint one and
       render it as an anchor: the user's click on THAT is a fresh gesture. */
    setBusy((b) => ({ ...b, [row.server]: true }));
    api
      .connect(row.server)
      .then((result) => {
        if (result.status === "connected") return refresh();
        if (result.setup_url) setLinks((m) => ({ ...m, [row.server]: result.setup_url }));
        else setProblem("could not start authorization for " + (row.name || row.server));
      })
      .catch((e) => setProblem(e.message || String(e)))
      .finally(() => setBusy((b) => ({ ...b, [row.server]: false })));
  }

  /* Disconnecting is shared whenever services sign in through one account, so a
     service with siblings is asked twice: the first click says what will go,
     the second means it. */
  async function disconnect(row) {
    const shared = row.shares_with || [];
    if (shared.length && armed !== row.server) {
      setArmed(row.server);
      return;
    }
    setArmed(null);
    setBusy((b) => ({ ...b, [row.server]: true }));
    try {
      await api.disconnect(row.server);
      setLinks({});
      await refresh();
    } catch (e) {
      setProblem(e.message || String(e));
    } finally {
      setBusy((b) => ({ ...b, [row.server]: false }));
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
            const asking = armed === row.server;
            const shared = row.shares_with || [];
            return (
              <div className="conn" key={row.server}>
                <span className="meta">
                  <Dot kind={connected ? "live" : ""} />
                  <span className="nm">{row.name || row.server}</span>
                  {/* What the click is about to do, before it does it. */}
                  {!connected && !!(row.scopes || []).length && (
                    <span className="soft" style={{ fontSize: 10, marginLeft: 8 }}>
                      grants {scopeNames(row.scopes).join(", ")}
                    </span>
                  )}
                  {connected && !!shared.length && (
                    <span className="soft" style={{ fontSize: 10, marginLeft: 8 }}>
                      shares a sign-in with {shared.join(", ")}
                    </span>
                  )}
                </span>
                <span style={{ display: "flex", alignItems: "center", gap: 12 }}>
                  <span className={"st" + (connected ? " on" : "")}>
                    {connected ? `connected · ${row.tool_count} tools` : row.status}
                  </span>
                  {connected ? (
                    <button
                      className={asking ? "btn danger" : "btn"}
                      disabled={busy[row.server]}
                      onClick={() => disconnect(row)}
                    >
                      {asking ? `also disconnects ${shared.join(" and ")} — confirm` : "disconnect"}
                    </button>
                  ) : links[row.server] ? (
                    <a
                      className="btn primary"
                      href={links[row.server]}
                      target="ark_oauth"
                      rel="noopener"
                      onClick={() => watch(null)}
                    >
                      authorize →
                    </a>
                  ) : (
                    <button className="btn primary" disabled={busy[row.server]} onClick={() => connect(row)}>
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

/* An OAuth scope url is unreadable and its last segment is not:
   `https://www.googleapis.com/auth/gmail.readonly` is "gmail.readonly". Shown
   so the human can see what a connect grants without reading a url. */
function scopeNames(scopes) {
  return (scopes || []).map((scope) => {
    const tail = String(scope).split("/").filter(Boolean).pop();
    return tail || String(scope);
  });
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
