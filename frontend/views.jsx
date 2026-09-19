/* =========================================================
   views — desk · approvals · files, plus settings and sign-in
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
    // `waiting` is a dep on purpose: App changing the pending set is also the
    // moment the running list is worth re-reading.
  }, [waiting, onError]);

  return (
    <div className="view viewin">
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
  // `waiting` is null until App has read it once — distinct from an empty list.
  return (
    <div className="view viewin">
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

/* One flat namespace per user; folders are the top-level path segments, and a
   project links folders rather than owning them. Reads the store directly, so
   it wakes no box (D27); `FileTree` is shared with the session working-files
   pane and only the scope it loads differs. */
function ComputerView({ onError, jumpTo, onJumped }) {
  const [open, setOpen] = useState(null);
  const [count, setCount] = useState(null);
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState("");
  const [drag, setDrag] = useState({ dragging: false, target: "", moving: null });
  const cancelled = useRef(false);
  // `draft` null means reading; any string means edit mode, dirty or not.
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

  /* Writes the durable copy, not a scratch edit: a session already holding the
     folder is written through, others pick it up at their next materialize. */
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

  const canEdit = !!open && !open.loading && !open.binary;
  const dirty = draft !== null && draft !== (open && open.text);

  /* Enter or blur commits, Escape cancels; outer slashes are stripped so the
     name cannot read as absolute, and `a/b` nests. The server is written before
     the tree redraws — no client-only folder that a reload would contradict. */
  const commitFolder = async () => {
    /* Escape unmounts the input and removing a focused element fires blur, which
       would commit the cancelled name; read and clear the flag so blur is a no-op. */
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

  // Sentinels are structure, not content, so the count excludes them.
  const shown = count === null ? null : count.filter((f) => !isSentinel(f.path));

  /* `add` is the tree's own uploader, handed back so `+ file` and a drop run
     the same code. */
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
                // Into the folder being aimed at, else the one being read.
                add(e.target.files, drag.target || dirOf(open && open.path));
                e.target.value = "";
              }}
            />
          </label>
          <span
            className="cv-add"
            title="new folder"
            onClick={() => {
              // A previous Escape may have left the flag set with no blur to clear it.
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
    <div className="view viewin" style={{ padding: 0, height: "100%" }}>
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
          {/* Sits over the reader, never the tree, and names the drop target —
              the top level takes folders, not files. */}
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
  /* Server whose disconnect has been clicked once and awaits confirmation. */
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

  /* The consent popup is cross-origin (Composio, then the provider), so nothing
     signals this page when it finishes; regaining focus is the event, and
     contracts forbids polling `api.connections()`. */
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

  /* The only signal a user who never leaves the tab produces; one watcher at a
     time, and it reads `popup.closed` rather than fetching. */
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

  /* `GET /connections` returns the setup url with the row so the popup can open
     synchronously inside the click — after an await the browser has lost the
     user gesture and blocks it silently. */
  function connect(row) {
    const href = links[row.server] || row.setup_url;
    if (href) {
      watch(window.open(href, "ark_oauth", "width=560,height=720"));
      return;
    }
    /* No link on the row: mint one and render it as an anchor, since the click
       on that anchor is a fresh gesture the browser will honour. */
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

  /* One sign-in can back several services, so a row with siblings takes two
     clicks: the first names what else goes, the second confirms. */
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
                  {/* Scopes are shown only before connecting: what the click grants. */}
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

/* Last path segment of an OAuth scope url: `.../auth/gmail.readonly` reads as
   "gmail.readonly". */
function scopeNames(scopes) {
  return (scopes || []).map((scope) => {
    const tail = String(scope).split("/").filter(Boolean).pop();
    return tail || String(scope);
  });
}

/* ---------- SIGN IN ---------- */

/* Sign-up, sign-in and Google are three ways to get a Supabase token and one
   way to be signed in: `api` trades any of them for our cookie via
   `POST /auth/session`, the only endpoint that reads a bearer. */
function Login({ gone, onSignedIn, problem: arrived, notice, startMode }) {
  // in | up | forgot
  const [mode, setMode] = useState(startMode || "in");
  // An error carried in from the boot belongs to the screen as it opened, not
  // to whatever the person does next.
  const [stale, setStale] = useState(false);
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState(null);
  /* "signup" | "reset" — a flow that ends in the mail rather than in the app.
     Set on EVERY outcome that sends nothing back, because an address that is
     already registered, rate-limited, or inside the 60s window is
     indistinguishable from success and must not read as a dead button. */
  const [sent, setSent] = useState(null);
  const first = useRef(null);

  const up = mode === "up";
  const forgot = mode === "forgot";
  const shown = problem || (stale ? null : arrived);

  useEffect(() => {
    if (!gone && first.current) first.current.focus();
  }, [gone, mode]);

  const ready = forgot
    ? email.trim()
    : up
      ? name.trim() && email.trim() && password.length >= MIN_PASSWORD
      : email.trim() && password;

  async function submit() {
    if (!ready || busy) return;
    setBusy(true);
    setProblem(null);
    try {
      if (forgot) {
        await api.sendReset(email.trim());
        setSent("reset");
      } else if (up) {
        const out = await api.signUp(name.trim(), email.trim(), password);
        if (out.confirm) setSent("signup");
        else onSignedIn(out.me);
      } else {
        onSignedIn(await api.signIn(email.trim(), password));
      }
    } catch (e) {
      setProblem(
        forgot
          ? "could not send the reset email"
          : up
            ? e.message || "could not create the account"
            : "that email and password do not match, or the address has not been confirmed yet"
      );
    } finally {
      setBusy(false);
    }
  }

  async function google() {
    if (busy) return;
    setBusy(true);
    setProblem(null);
    try {
      // Navigates the tab away; nothing after this runs on success.
      await api.signInWithGoogle();
    } catch (e) {
      setProblem(e.message || "could not start google sign-in");
      setBusy(false);
    }
  }

  /* Email confirmation is ON: the account is unusable until the link is clicked.
     Supabase gives the same answer for an address that already exists, so this
     screen must not distinguish the two either. */
  if (sent) {
    return (
      <div className={"auth" + (gone ? " gone" : "")}>
        <AuthAside />
        <div className="auth-right">
          <MailSent
            kind={sent}
            email={email.trim()}
            onBack={() => {
              setSent(null);
              setMode("in");
              setPassword("");
            }}
          />
        </div>
      </div>
    );
  }

  return (
    <div className={"auth" + (gone ? " gone" : "")}>
      <AuthAside />
      <div className="auth-right">
        <div className="auth-card">
          <div className="auth-head">
            <span className="kicker">{forgot ? "reset password" : up ? "new account" : "sign in"}</span>
            <span className="title">
              {forgot ? "we will email you a link" : up ? "make an account" : "welcome back"}
            </span>
          </div>

          <div className="auth-fields">
            {up && (
              <label className="auth-field">
                <span className="lab">name</span>
                <input
                  ref={up ? first : null}
                  value={name}
                  spellCheck={false}
                  autoComplete="name"
                  placeholder="what should buddy call you?"
                  onChange={(e) => setName(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && submit()}
                />
              </label>
            )}
            <label className="auth-field">
              <span className="lab">username</span>
              <input
                ref={up ? null : first}
                type="email"
                value={email}
                spellCheck={false}
                autoComplete="username"
                placeholder="nathaniel@buddy.computer"
                onChange={(e) => setEmail(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && submit()}
              />
            </label>
            {!forgot && (
            <label className="auth-field">
              <span className="lab-row">
                <span className="lab">password</span>
                {!up && (
                  <span
                    className="auth-forgot"
                    onClick={() => {
                      setMode("forgot");
                      setProblem(null);
                      setStale(true);
                    }}
                  >
                    forgot
                  </span>
                )}
              </span>
              <input
                type="password"
                value={password}
                placeholder={up ? `at least ${MIN_PASSWORD} characters` : "••••••••"}
                autoComplete={up ? "new-password" : "current-password"}
                onChange={(e) => setPassword(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && submit()}
              />
            </label>
            )}
          </div>

          <div className="auth-actions">
            <button className="auth-cta" onClick={submit} disabled={busy || !ready}>
              {busy ? "…" : forgot ? "email me a link" : up ? "create account" : "sign in"}
            </button>
            {/* Google is a way IN, not a way to reset a password it does not
                hold — hidden here rather than offered and refused. */}
            {!forgot && (
              <React.Fragment>
                <div className="auth-or">
                  <span className="rule" />
                  or
                  <span className="rule" />
                </div>
                <button className="auth-google" onClick={google} disabled={busy}>
                  <span className="g">G</span>
                  continue with google
                </button>
              </React.Fragment>
            )}
          </div>

          {notice && !problem && <p className="auth-note quiet">{notice}</p>}
          {shown && <p className="auth-problem">{shown}</p>}

          <div className="auth-switch">
            {forgot ? (
              <span
                className="auth-link"
                onClick={() => {
                  setMode("in");
                  setProblem(null);
                  setStale(true);
                }}
              >
                back to sign in
              </span>
            ) : (
              <React.Fragment>
                {up ? "already have an account?" : "no account yet?"}{" "}
                <span
                  className="auth-link"
                  onClick={() => {
                    setMode(up ? "in" : "up");
                    setProblem(null);
                    setStale(true);
                  }}
                >
                  {up ? "sign in" : "make one"}
                </span>
              </React.Fragment>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/* THE FLOW ALWAYS SAYS SOMETHING. A signup or reset that sends no mail — the
   address is already registered, the sender is rate-limited, or we are inside
   Supabase's 60s per-user window — is indistinguishable from one that did, and
   silence reads as a dead button. Each flow has ONE sentence it says whether or
   not the address exists; the two flows say different things, because
   anti-enumeration is about telling exists from not-exists WITHIN a flow, not
   about signup and reset sounding alike. */
// Supabase's own per-user interval, so the button can say why it is dim
// instead of failing at the server.
const RESEND_WAIT = 60;
// One password rule, stated wherever a password is typed.
const MIN_PASSWORD = 8;

function MailSent({ kind, email, onBack }) {
  const reset = kind === "reset";
  /* Starts COOLING. This mounts at second 0 of Supabase's per-user window —
     the mail was just sent — so a button enabled here is a button that cannot
     work, and its refusal would name the window and thereby the account. */
  const [cooling, setCooling] = useState(RESEND_WAIT);
  const [note, setNote] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    const id = setInterval(() => setCooling((n) => (n > 0 ? n - 1 : 0)), 1000);
    return () => clearInterval(id);
  }, []);

  async function again() {
    if (cooling || busy) return;
    setBusy(true);
    setNote(null);
    try {
      if (reset) await api.sendReset(email);
      else await api.resendSignup(email);
      setNote("sent again — check your inbox and your spam folder.");
      setCooling(RESEND_WAIT);
    } catch (e) {
      /* FIXED COPY, never the provider's. Its refusals name the per-user
         window, which answers "does this address exist?" — the question this
         whole screen exists not to answer. */
      setNote("if there is anything to send, it is on its way. try again in a minute.");
      setCooling(RESEND_WAIT);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="auth-card">
      <div className="auth-head">
        <span className="kicker">check your email</span>
        <span className="title">{reset ? "reset on its way" : "almost there"}</span>
      </div>
      <p className="auth-note">
        {reset ? (
          "if buddy knows this address, a reset link is on its way"
        ) : (
          <React.Fragment>
            check your email — if <b>{email}</b> is new to buddy, a confirmation is on its way. click the
            link and you are in.
          </React.Fragment>
        )}
      </p>
      <div className="auth-actions">
        <button className="auth-google" onClick={again} disabled={!!cooling || busy}>
          {cooling ? `send again in ${cooling}s` : busy ? "…" : "send it again"}
        </button>
      </div>
      {note && <p className="auth-note quiet">{note}</p>}
      <div className="auth-switch">
        <span className="auth-link" onClick={onBack}>
          back to sign in
        </span>
      </div>
    </div>
  );
}

/* The recovery landing. The link carries a token good for exactly one thing —
   changing the password — so this screen is the whole of what it can do, and
   it is reachable only with that token in hand. */
function ResetPassword({ onDone, onGiveUp }) {
  const [password, setPassword] = useState("");
  const [again, setAgain] = useState("");
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState(null);
  const first = useRef(null);

  useEffect(() => {
    if (first.current) first.current.focus();
  }, []);

  const mismatch = again.length > 0 && password !== again;
  const ready = password.length >= MIN_PASSWORD && password === again;

  async function submit() {
    if (!ready || busy) return;
    setBusy(true);
    setProblem(null);
    try {
      await api.completeReset(password);
      onDone();
    } catch (e) {
      setProblem(e.message || "could not set the new password");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="auth">
      <AuthAside />
      <div className="auth-right">
        <div className="auth-card">
          <div className="auth-head">
            <span className="kicker">reset password</span>
            <span className="title">choose a new one</span>
          </div>
          <div className="auth-fields">
            <label className="auth-field">
              <span className="lab">new password</span>
              <input
                ref={first}
                type="password"
                value={password}
                placeholder={`at least ${MIN_PASSWORD} characters`}
                autoComplete="new-password"
                onChange={(e) => setPassword(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && submit()}
              />
            </label>
            <label className="auth-field">
              <span className="lab">again</span>
              <input
                type="password"
                value={again}
                placeholder="••••••••"
                autoComplete="new-password"
                onChange={(e) => setAgain(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && submit()}
              />
            </label>
          </div>
          <div className="auth-actions">
            <button className="auth-cta" onClick={submit} disabled={busy || !ready}>
              {busy ? "…" : "set new password"}
            </button>
          </div>
          {mismatch && <p className="auth-note quiet">those two do not match yet.</p>}
          {problem && <p className="auth-problem">{problem}</p>}
          <div className="auth-switch">
            <span className="auth-link" onClick={onGiveUp}>
              back to sign in
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

function AuthAside() {
  return (
    <div className="auth-aside">
      <div className="auth-mark">
        <span className="glyph">b</span>
        <span className="pip" />
        <span className="word">buddy</span>
      </div>
      <div className="auth-pitch">
        <h1>
          a digital intern
          <br />
          still on the job
          <br />
          next week.
          <span className="caret" />
        </h1>
        <p>
          buddy drives a real browser and the services you connect, with durable storage that survives
          between runs. workflows can span days, and it asks before anything leaves your account.
        </p>
      </div>
      <div className="auth-tags">
        <span>approvals first</span>
        <span>a real browser</span>
        <span>durable storage</span>
      </div>
    </div>
  );
}
