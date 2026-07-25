/* =========================================================
   views
   ========================================================= */

function PageHead({ title, accent, lede }) {
  return (
    <div className="head">
      <h1>{title}{accent && <span className="accent">{accent}</span>}<span className="caret" /></h1>
      {lede && <div className="lede">{lede}</div>}
    </div>
  );
}

/* ---------- DESK ---------- */
function DeskView({ data, onResolve }) {
  return (
    <div className="view">
      <PageHead
        title="buddy's desk"
        lede="your digital life, handled in the background. buddy watches, triages, and workshops plans for your approval."
      />
      <div className="zones">
        <section className="zone">
          <header>
            <span className="kicker">pending approvals</span>
            <span className="n">{data.approvals.length}</span>
          </header>
          <div className="stack">
            {data.approvals.length === 0
              ? <Empty glyph="✓">nothing waiting on you</Empty>
              : data.approvals.map((a) => <ApprovalCard key={a.id} item={a} onResolve={onResolve} />)}
          </div>
        </section>

        <section className="zone">
          <header>
            <span className="kicker">active tasks</span>
            <span className="n">{data.tasks.filter((t) => t.state === "run").length}</span>
          </header>
          <div className="stack">
            {data.tasks.length === 0
              ? <Empty glyph="○">buddy is idle</Empty>
              : data.tasks.map((t) => <TaskRow key={t.id} item={t} />)}
          </div>
        </section>

        <section className="zone">
          <header>
            <span className="kicker">watching</span>
            <span className="n">{data.watching.length}</span>
          </header>
          <div className="stack">
            {data.watching.map((w) => <WatchRow key={w.id} item={w} />)}
          </div>
        </section>
      </div>
    </div>
  );
}

/* ---------- WATCHING ---------- */
function WatchingView({ data }) {
  return (
    <div className="view">
      <PageHead title="watching" lede="the sources buddy keeps an eye on, and how often it checks." />
      <div className="stack" style={{ maxWidth: 820 }}>
        {data.watching.map((w) => <WatchRow key={w.id} item={w} />)}
      </div>
    </div>
  );
}

/* ---------- APPROVALS ---------- */
function ApprovalsView({ data, onResolve }) {
  return (
    <div className="view">
      <PageHead title="approvals" lede="plans buddy has workshopped and is holding for your ok. nothing acts without it." />
      <div className="stack" style={{ maxWidth: 620 }}>
        {data.approvals.length === 0
          ? <Empty glyph="✓">all caught up — nothing waiting on you</Empty>
          : data.approvals.map((a) => <ApprovalCard key={a.id} item={a} onResolve={onResolve} />)}
      </div>
    </div>
  );
}

/* ---------- CHAT ---------- */
function ChatView({ data }) {
  return (
    <div className="view">
      <PageHead title="chat" lede="think out loud with buddy. it remembers, and turns the useful bits into standing rules." />
      <div className="chat-wrap">
        {data.chat.map((m, i) => (
          <div className={"msg " + (m.who === "you" ? "user" : "buddy")} key={i}>
            <span className="who">{m.who === "you" ? data.user : "buddy"}</span>
            <span className="bubble">{m.text}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ---------- COMPUTER ---------- */
function ComputerView({ data }) {
  const [path, setPath] = useState("/home/" + data.user);
  const [file, setFile] = useState(null);
  const entries = data.files[path] || [];

  function open(e) {
    if (e.up) { goUp(); return; }
    if (e.dir) { setPath(path === "/home/" + data.user ? path + "/" + e.name : path + "/" + e.name); setFile(null); return; }
    setFile(path + "/" + e.name);
  }
  function goUp() {
    const parts = path.split("/").filter(Boolean);
    parts.pop();
    setPath("/" + parts.join("/"));
    setFile(null);
  }

  const body = file ? data.fileBodies[file] : null;

  return (
    <div className="view view-wide" style={{ padding: 0, height: "100%" }}>
      <div className="computer">
        <div className="cv-files">
          <div className="cv-head">
            <span className="path">{path}</span>
            <button className="icon-btn" onClick={goUp} disabled={path === "/"}>up ↑</button>
          </div>
          <div className="cv-entries">
            {entries.map((e, i) => (
              <div className={"cv-entry" + (e.dir ? " dir" : "") + (file === path + "/" + e.name ? " sel" : "")} key={i} onClick={() => open(e)}>
                <span className="nm"><span className="g">{e.up ? "↑" : e.dir ? "▸" : "·"}</span>{e.name}</span>
                {e.size && <span className="sz">{e.size}</span>}
              </div>
            ))}
          </div>
        </div>
        <div className="cv-read">
          <div className="cv-read-head">
            <span>{file ? file : "select a file to read"}</span>
            {data.computerTasks.some((t) => t.state === "run") && <span style={{ display: "flex", alignItems: "center", gap: 8 }}><span className="spin" /> 1 running</span>}
          </div>
          {body ? (
            <pre className="cv-read-body">{body.split("\n").map((ln, i) => (
              <div key={i}><span className="ln">{String(i + 1).padStart(2, " ")}</span>{ln || "\u00a0"}</div>
            ))}</pre>
          ) : (
            <div className="cv-read-body" style={{ color: "var(--ink-mute)", fontStyle: "italic" }}>
              buddy keeps a small workspace here. click a file on the left to read it.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ---------- SETTINGS MODAL ---------- */
function SettingsModal({ data, onClose, onSignOut }) {
  const [conns, setConns] = useState([
    { id: "gmail", nm: "gmail", st: "on" },
    { id: "gcal", nm: "google calendar", st: "on" },
    { id: "linear", nm: "linear", st: "on" },
    { id: "github", nm: "github", st: "off" },
    { id: "slack", nm: "slack", st: "off" },
  ]);
  function toggle(id) {
    setConns((cs) => cs.map((c) => c.id === id ? { ...c, st: c.st === "on" ? "off" : "on" } : c));
  }
  return (
    <div className="overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h2>settings</h2>
        <p className="sub">connections, account, and backend.</p>

        <section>
          <span className="kicker">tools &amp; connections</span>
          {conns.map((c) => (
            <div className="conn" key={c.id}>
              <span className="meta"><Dot kind={c.st === "on" ? "live" : ""} /> <span className="nm">{c.nm}</span></span>
              <span style={{ display: "flex", alignItems: "center", gap: 12 }}>
                <span className={"st" + (c.st === "on" ? " on" : "")}>{c.st === "on" ? "connected" : "off"}</span>
                <button className={"btn" + (c.st === "on" ? "" : " primary")} onClick={() => toggle(c.id)}>
                  {c.st === "on" ? "disconnect" : "connect"}
                </button>
              </span>
            </div>
          ))}
        </section>

        <section>
          <span className="kicker">account</span>
          <div className="soft" style={{ fontSize: 12, lineHeight: 1.9 }}>
            signed in as <b style={{ color: "var(--ink)" }}>{data.user}</b><br />
            backend · <b style={{ color: "var(--ink)" }}>{data.backend}</b>
          </div>
        </section>

        <div className="foot">
          <span className="mute" style={{ fontSize: 10.5 }}>changes save automatically</span>
          <div style={{ display: "flex", gap: 8 }}>
            <button className="btn danger" onClick={onSignOut}>sign out</button>
            <button className="btn" onClick={onClose}>close</button>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ---------- LOGIN ---------- */
function Login({ gone, onEnter }) {
  const [name, setName] = useState("");
  const inputRef = useRef(null);
  useEffect(() => { if (!gone && inputRef.current) inputRef.current.focus(); }, [gone]);
  function submit() { if (name.trim()) onEnter(name.trim()); }
  return (
    <div className={"login" + (gone ? " gone" : "")}>
      <div className="login-card">
        <div className="mark-lg">ark<span className="pip" /></div>
        <p>buddy handles your digital life in the background. pick a name to begin — no passwords, no email. just you and your buddy.</p>
        <div className="field">
          <label>username</label>
          <input ref={inputRef} value={name} placeholder="e.g. nate" spellCheck={false}
            onChange={(e) => setName(e.target.value)} onKeyDown={(e) => e.key === "Enter" && submit()} />
        </div>
        <div className="field opt">
          <label>backend url — optional override</label>
          <input placeholder="ark.mit.edu" spellCheck={false} />
        </div>
        <div className="go">
          <span className="hint">press enter to continue</span>
          <button className="btn primary lg" onClick={submit} disabled={!name.trim()}>enter →</button>
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { PageHead, DeskView, WatchingView, ApprovalsView, ChatView, ComputerView, SettingsModal, Login });
