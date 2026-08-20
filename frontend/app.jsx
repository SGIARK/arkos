/* =========================================================
   app — root: state, routing, theme, rail, command bar

   The design's frame. `watching` is not in the nav: nothing in the system
   watches a source on a schedule, and a rail entry for a feature that does not
   exist is a promise the product cannot keep. Everything else is here — desk,
   approvals, files, projects, chat.

   `projects` is the projects surface and only that; it was called "looking
   glass" until the 11.4 design renamed it to what it shows, and `computer`
   became `files` for the same reason. The ambient bar at the bottom talks to
   the home session, which is the standing conversation; projects hides it,
   because a session's own composer is in its window.
   ========================================================= */

const NAV = ["desk", "approvals", "files", "projects", "chat"];

/* The hash a browser may still be holding from before the rename. Landing on
   the desk because a bookmark says "computer" would be a small betrayal of a
   link somebody saved. */
const NAV_ALIAS = { computer: "files", "looking glass": "projects" };

function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem("ark-theme") || "light");
  const [user, setUser] = useState(null);
  const [booting, setBooting] = useState(true);
  const [gone, setGone] = useState(false);
  const [view, setView] = useState(() => {
    const hash = decodeURIComponent(location.hash.replace("#", ""));
    const named = NAV_ALIAS[hash] || hash;
    return NAV.includes(named) ? named : "chat";
  });
  const [settings, setSettings] = useState(false);
  const [error, setError] = useState(null);
  const [floaters, setFloaters] = useState([]);
  // Bumped when something happened that the counts are made of.
  const [pulse, setPulse] = useState(0);
  const [waiting, setWaiting] = useState([]);
  // A session opened from somewhere other than the grid — the desk, say.
  const [jump, setJump] = useState(null);
  const inputRef = useRef(null);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("ark-theme", theme);
  }, [theme]);

  useEffect(() => {
    location.hash = encodeURIComponent(view);
  }, [view]);

  /* The cookie may already be good, so the page asks before it offers a form. */
  useEffect(() => {
    api
      .me()
      .then((who) => {
        setUser(who);
        setGone(true);
      })
      .catch(() => setUser(null))
      .finally(() => setBooting(false));
  }, []);

  const onError = useCallback((e) => {
    // A missing cookie is a sign-in, not an error to shout about.
    if (e && (e.code === "unauthenticated" || e.code === "http_401")) {
      setUser(null);
      setGone(false);
      return;
    }
    setError(e);
  }, []);

  const bump = useCallback(() => setPulse((n) => n + 1), []);

  /* The pending count in the topbar, and the alert dot on the rail. */
  useEffect(() => {
    if (!user) return;
    api.attention().then(setWaiting).catch(() => {});
  }, [user, pulse]);

  useEffect(() => {
    function key(e) {
      if (e.key === "/" && document.activeElement !== inputRef.current) {
        e.preventDefault();
        if (inputRef.current) inputRef.current.focus();
      }
      if (e.key === "Escape") {
        if (settings) setSettings(false);
        else if (inputRef.current) {
          inputRef.current.value = "";
          inputRef.current.blur();
        }
      }
    }
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [settings]);

  async function signIn(who) {
    setUser(who);
    setTimeout(() => setGone(true), 30);
  }

  async function signOut() {
    try {
      await api.signOut();
    } catch (e) {
      /* whatever the server said, this browser is done with the session */
    }
    setSettings(false);
    setGone(false);
    setTimeout(() => setUser(null), 450);
  }

  /* The ambient bar speaks to the home session: one standing conversation,
     reachable from every view. What is said floats up and folds away, and the
     chat view has the durable copy. */
  const home = user && user.home_session_id;

  async function say(text) {
    if (!home) return;
    const id = "f" + Date.now();
    setFloaters((f) => [...f, { id, who: "you", text }]);
    setTimeout(() => {
      setFloaters((f) => f.map((x) => (x.id === id ? { ...x, fold: true } : x)));
      setTimeout(() => setFloaters((f) => f.filter((x) => x.id !== id)), 460);
    }, 2400);
    try {
      await api.send(home, text);
      bump();
    } catch (e) {
      onError(e);
    }
  }

  function onKey(e) {
    if (e.key === "Enter" && e.target.value.trim()) {
      say(e.target.value.trim());
      e.target.value = "";
    }
  }

  if (booting) return <div className="login" />;
  if (!user) return <Login gone={gone} onSignedIn={signIn} />;

  const ambient = (
    <div className="ambient">
      <span className="prompt">ark&gt;</span>
      <input
        ref={inputRef}
        spellCheck={false}
        autoComplete="off"
        placeholder="tell ark what to do. or just think out loud."
        onKeyDown={onKey}
      />
      <div className="hints">
        <span className="hint">
          <kbd>/</kbd> focus
        </span>
        <span className="hint">
          <kbd>enter</kbd> send
        </span>
        <span className="hint">
          <kbd>esc</kbd> clear
        </span>
      </div>
      <div className="floaters">
        {floaters.map((f) => (
          <div className={"floater" + (f.fold ? " fold" : "")} key={f.id}>
            <span className="who">{f.who}</span>
            {f.text}
          </div>
        ))}
      </div>
    </div>
  );

  const views = {
    desk: <DeskView onError={onError} pulse={pulse} onOpenSession={(id) => { setJump(id); setView("projects"); }} />,
    approvals: <ApprovalsView onError={onError} pulse={pulse} />,
    files: <ComputerView onError={onError} />,
    projects: <LookingGlassView onError={onError} pulse={pulse} onPulse={bump} jump={jump} onJumped={() => setJump(null)} />,
    chat: <ChatView sessionId={home} onError={onError} onPulse={bump} />,
  };

  const pending = waiting.length;
  const bare = view === "projects";

  return (
    <React.Fragment>
      <div className={"app" + (bare ? " no-ambient" : "")}>
        <div className="rail">
          <div className="mark">
            <span className="glyph">a</span>
            <span className="pip" />
          </div>
          <nav>
            {NAV.map((v) => (
              <a
                key={v}
                className={(view === v ? "active" : "") + (v === "approvals" && pending > 0 ? " alert" : "")}
                onClick={() => setView(v)}
              >
                {v}
              </a>
            ))}
          </nav>
          <div className="foot">
            <button className="theme-btn" onClick={() => setTheme((t) => (t === "light" ? "dark" : "light"))}>
              {theme === "light" ? "dark" : "light"}
            </button>
          </div>
        </div>

        <div className="topbar">
          <div className="crumbs">
            <span>
              ark <b>v1</b>
            </span>
            <span className="sep">/</span>
            <span>
              user <b>{user.email || user.user_id}</b>
            </span>
          </div>
          <div className="right">
            <span className={"pill" + (pending > 0 ? " attn" : "")} onClick={() => setView("approvals")}>
              {pending > 0 && <Dot kind="work" />}
              {pending} pending
            </span>
            <button className="icon-btn" onClick={() => setSettings(true)}>
              settings
            </button>
          </div>
        </div>

        <main key={view}>
          {error && (
            <div className="banner" role="alert">
              <span>{error.message || error.code || "something failed"}</span>
              <button onClick={() => setError(null)}>dismiss</button>
            </div>
          )}
          {views[view]}
        </main>

        {!bare && ambient}
      </div>

      {settings && (
        <SettingsModal user={user} onClose={() => setSettings(false)} onSignOut={signOut} onError={onError} />
      )}
    </React.Fragment>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
