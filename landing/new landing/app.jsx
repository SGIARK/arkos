/* =========================================================
   app — root: state, routing, theme, rail, command bar
   ========================================================= */

const NAV = ["desk", "watching", "approvals", "computer", "looking glass", "chat"];

function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem("ark-theme") || "light");
  const [authed, setAuthed] = useState(() => !!localStorage.getItem("ark-user"));
  const [loginGone, setLoginGone] = useState(() => !!localStorage.getItem("ark-user"));
  const [view, setView] = useState(() => {
    const h = location.hash.replace("#", "");
    return NAV.includes(h) ? h : "desk";
  });
  const [settings, setSettings] = useState(false);
  const [data, setData] = useState(() => {
    const u = localStorage.getItem("ark-user");
    return { ...SEED, user: u || SEED.user };
  });
  const [floaters, setFloaters] = useState([]);
  const inputRef = useRef(null);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("ark-theme", theme);
  }, [theme]);

  useEffect(() => {
    function key(e) {
      if (e.key === "/" && document.activeElement !== inputRef.current) {
        e.preventDefault(); inputRef.current && inputRef.current.focus();
      }
      if (e.key === "Escape") {
        if (settings) setSettings(false);
        else if (inputRef.current) { inputRef.current.value = ""; inputRef.current.blur(); }
      }
    }
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [settings]);

  function enter(name) {
    localStorage.setItem("ark-user", name);
    setData((d) => ({ ...d, user: name }));
    setAuthed(true);
    setTimeout(() => setLoginGone(true), 30);
  }
  function signOut() {
    localStorage.removeItem("ark-user");
    setSettings(false);
    setLoginGone(false);
    setTimeout(() => setAuthed(false), 450);
  }

  function resolveApproval(id, verb) {
    setData((d) => {
      const item = d.approvals.find((a) => a.id === id);
      const approvals = d.approvals.filter((a) => a.id !== id);
      let tasks = d.tasks;
      if (verb === "approved" && item) {
        tasks = [
          { id: "tn" + Date.now(), state: "run", when: "running 0s",
            text: item.title.toLowerCase(), src: "from " + item.src,
            events: [{ k: "start", t: "approved — executing plan" }] },
          ...d.tasks,
        ];
      }
      return { ...d, approvals, tasks };
    });
  }

  function send(text) {
    const id = "f" + Date.now();
    setFloaters((f) => [...f, { id, who: data.user, text }]);
    setData((d) => ({ ...d, chat: [...d.chat, { who: "you", text }] }));
    // buddy replies, then both fold away
    setTimeout(() => {
      const reply = REPLIES[Math.floor(Math.random() * REPLIES.length)];
      setFloaters((f) => [...f, { id: id + "r", who: "buddy", text: reply }]);
      setData((d) => ({ ...d, chat: [...d.chat, { who: "buddy", text: reply }] }));
      setTimeout(() => {
        setFloaters((f) => f.map((x) => (x.id === id || x.id === id + "r") ? { ...x, fold: true } : x));
        setTimeout(() => setFloaters((f) => f.filter((x) => x.id !== id && x.id !== id + "r")), 460);
      }, 2600);
    }, 700);
  }

  function onKey(e) {
    if (e.key === "Enter" && e.target.value.trim()) {
      send(e.target.value.trim());
      e.target.value = "";
    }
  }

  const views = {
    desk: <DeskView data={data} onResolve={resolveApproval} />,
    watching: <WatchingView data={data} />,
    approvals: <ApprovalsView data={data} onResolve={resolveApproval} />,
    computer: <ComputerView data={data} />,
    "looking glass": <LookingGlassView />,
    chat: <ChatView data={data} />,
  };

  const pending = data.approvals.length;

  return (
    <React.Fragment>
      <div className={"app" + (view === "looking glass" ? " no-ambient" : "")}>
        {/* rail */}
        <div className="rail">
          <div className="mark"><span className="glyph">a</span><span className="pip" /></div>
          <nav>
            {NAV.map((v) => (
              <a key={v} className={(view === v ? "active" : "") + (v === "approvals" && pending > 0 ? " alert" : "")} onClick={() => setView(v)}>
                {v}
              </a>
            ))}
          </nav>
          <div className="foot">
            <button className="theme-btn" onClick={() => setTheme((t) => t === "light" ? "dark" : "light")}>
              {theme === "light" ? "dark" : "light"}
            </button>
          </div>
        </div>

        {/* topbar */}
        <div className="topbar">
          <div className="crumbs">
            <span>buddy <b>v1</b></span>
            <span className="sep">/</span>
            <span>user <b>{data.user}</b></span>
            <span className="sep">/</span>
            <span>{data.backend}</span>
          </div>
          <div className="right">
            <span className={"pill" + (pending > 0 ? " attn" : "")} onClick={() => setView("approvals")}>
              {pending > 0 && <Dot kind="work" />}{pending} pending
            </span>
            <button className="icon-btn" onClick={() => setSettings(true)}>settings</button>
          </div>
        </div>

        {/* main */}
        <main key={view}>{views[view]}</main>

        {/* ambient command bar */}
        {view !== "looking glass" && <div className="ambient">
          <span className="prompt">ark&gt;</span>
          <input ref={inputRef} spellCheck={false} autoComplete="off"
            placeholder="tell buddy what to do. or just think out loud." onKeyDown={onKey} />
          <div className="hints">
            <span className="hint"><kbd>/</kbd> focus</span>
            <span className="hint"><kbd>enter</kbd> send</span>
            <span className="hint"><kbd>esc</kbd> clear</span>
          </div>
          <div className="floaters">
            {floaters.map((f) => (
              <div className={"floater" + (f.fold ? " fold" : "")} key={f.id}>
                <span className="who">{f.who}</span>
                {f.text}
              </div>
            ))}
          </div>
        </div>}
      </div>

      {settings && <SettingsModal data={data} onClose={() => setSettings(false)} onSignOut={signOut} />}
      {!loginGone && <Login gone={authed} onEnter={enter} />}
    </React.Fragment>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
