/* =========================================================
   views — desk · approvals · computer · chat, plus settings and sign-in

   The design's compositions on real data. `watching` is gone: nothing in the
   system watches sources on a schedule, and a view for a feature that does not
   exist is a promise the product cannot keep.
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

/* ---------- DESK ---------- */

function DeskView({ onError, pulse, onOpenSession }) {
  const [waiting, setWaiting] = useState([]);
  const [running, setRunning] = useState([]);
  const [projects, setProjects] = useState([]);

  useEffect(() => {
    let dead = false;
    (async () => {
      try {
        const [a, r, p] = await Promise.all([api.attention(), api.sessions("running"), api.projects()]);
        if (dead) return;
        setWaiting(a);
        setRunning(r);
        setProjects(p);
      } catch (e) {
        if (!dead) onError(e);
      }
    })();
    return () => {
      dead = true;
    };
  }, [pulse, onError]);

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
                <ApprovalCard key={a.approval_id} item={a} onResolve={() => setWaiting((w) => w.filter((x) => x.approval_id !== a.approval_id))} onError={onError} />
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

function ApprovalsView({ onError, pulse }) {
  const [waiting, setWaiting] = useState(null);

  useEffect(() => {
    let dead = false;
    api
      .attention()
      .then((a) => !dead && setWaiting(a))
      .catch((e) => !dead && onError(e));
    return () => {
      dead = true;
    };
  }, [pulse, onError]);

  return (
    <div className="view">
      <PageHead
        title="approvals"
        lede="questions a run stopped to ask. answering here is answering in its window — one row, wherever you see it."
      />
      <div className="stack" style={{ maxWidth: 620 }}>
        {waiting === null && <Empty>reading…</Empty>}
        {waiting !== null && waiting.length === 0 && <Empty glyph="✓">all caught up — nothing waiting on you</Empty>}
        {(waiting || []).map((a) => (
          <ApprovalCard
            key={a.approval_id}
            item={a}
            onResolve={() => setWaiting((w) => w.filter((x) => x.approval_id !== a.approval_id))}
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
    api.files(projectId).then(setFiles).catch(onError);
  }, [projectId, onError]);

  const read = async (file) => {
    setOpen({ path: file.path, loading: true });
    try {
      setOpen(await api.file(projectId, file.file_id));
    } catch (e) {
      setOpen(null);
      onError(e);
    }
  };

  return (
    <div className="view view-wide" style={{ padding: 0, height: "100%" }}>
      <div className="computer">
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
            <span className="path">{files ? `${files.length} file${files.length === 1 ? "" : "s"}` : "…"}</span>
          </div>
          <div className="cv-entries">
            {files === null && <div className="cv-entry loading">reading…</div>}
            {files !== null && !files.length && <div className="cv-entry loading">no files in this project</div>}
            {(files || []).map((f) => (
              <div
                className={"cv-entry" + (open && open.path === f.path ? " sel" : "")}
                key={f.file_id}
                onClick={() => read(f)}
              >
                <span className="nm">
                  <span className="g">·</span>
                  {f.path}
                </span>
                <span className="sz">{fileSize(f.size)}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="cv-read">
          <div className="cv-read-head">
            <span>{open ? open.path : "select a file to read"}</span>
            {open && !open.loading && <span>{fileSize(open.size)}</span>}
          </div>
          {open && open.loading ? (
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

/* ---------- CHAT ---------- */

/* The home session: the standing conversation, in the design's chat shape.
   Same session and same events as any window — this one just reads as a
   conversation because that is what it is. */
function ChatView({ sessionId, onError, onPulse, composer }) {
  const { session, events, pending, questions, refreshQuestions } = useStream(sessionId, onError, onPulse);
  const tail = useRef(null);

  useEffect(() => {
    if (tail.current) tail.current.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [events.length, pending.length]);

  if (!sessionId) {
    return (
      <div className="view">
        <PageHead title="chat" />
        <Empty glyph="◇">no home session yet — sign out and back in, and the server will make one</Empty>
      </div>
    );
  }

  const said = grouped(events).filter((e) => e.kind === "user" || e.kind === "content");

  return (
    <div className="view">
      <PageHead
        title="chat"
        lede="think out loud. this conversation stands between sessions, and what is worth keeping goes to memory."
      />
      <div className="chat-wrap">
        {!said.length && !pending.length && <Empty glyph="○">nothing said yet — the bar below is the way in</Empty>}
        {said.map((m) => (
          <div className={"msg " + (m.kind === "user" ? "user" : "buddy")} key={m.seq}>
            <span className="who">{m.kind === "user" ? "you" : "ark"}</span>
            <span className="bubble">{m.text}</span>
          </div>
        ))}
        {pending.map((p) => (
          <div className="msg user pending" key={p.id}>
            <span className="who">you</span>
            <span className="bubble">{p.text}</span>
          </div>
        ))}
        {questions.map((q) => (
          <AskBlock key={q.approval_id} item={q} onAnswered={refreshQuestions} onError={onError} />
        ))}
        {session && session.status === "running" && (
          <div className="ev-status">
            <span className="spin" />
            working…
          </div>
        )}
        <div ref={tail} />
      </div>
      {composer}
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
    return () => {
      if (poll.current) clearInterval(poll.current);
    };
  }, [refresh]);

  /* Watch until the server says connected or the popup closes: the browser
     gets no callback when a popup on another origin finishes. */
  function watch(server, popup) {
    if (poll.current) clearInterval(poll.current);
    poll.current = setInterval(async () => {
      let current = [];
      try {
        current = await api.connections();
      } catch (e) {
        return;
      }
      setRows(current);
      const row = current.find((r) => r.server === server);
      if ((row && row.status === "connected") || (popup && popup.closed)) {
        clearInterval(poll.current);
        poll.current = null;
        setTimeout(refresh, 400);
      }
    }, 2000);
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
