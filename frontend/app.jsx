/* =========================================================
   app — root: state, routing, theme, rail, command bar

   The design's frame. `watching` is not in the nav: nothing in the system
   watches a source on a schedule, and a rail entry for a feature that does not
   exist is a promise the product cannot keep. The nav is the whole surface —
   desk, approvals, files, projects.

   `projects` is the projects surface and only that; it was called "looking
   glass" until the 11.4 design renamed it to what it shows, and `computer`
   became `files` for the same reason.

   Chat and the ambient bar are GONE (11.8): the design export removed them, and
   a session's own composer is where you talk to a session. The home session
   (LG-1.7) still exists backend-side and has no surface at all — the buddy's
   return or retirement is a future card. 11.9 finished taking the furniture out
   from under it: it holds no project, claims no folder, and the window's header
   no longer invents a name for the container it does not have.
   ========================================================= */

const NAV = ["desk", "approvals", "files", "projects"];

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
    return NAV.includes(named) ? named : "projects";
  });
  const [settings, setSettings] = useState(false);
  const [error, setError] = useState(null);
  // Bumped when something happened that the counts are made of.
  const [pulse, setPulse] = useState(0);
  // Fetched ONCE per pulse and passed down. Every surface that shows what is
  // waiting shows the same unscoped list, and four components each fetching it
  // was four identical requests on every change. Null means not yet read.
  const [waiting, setWaiting] = useState(null);
  // A session opened from somewhere other than the grid — the desk, say.
  const [jump, setJump] = useState(null);
  // A file the working-files pane asked the Files tab to land on. Same store,
  // same path: the pane and the tab are two views of one namespace (11.9).
  const [openFile, setOpenFile] = useState(null);

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

  // `/` focused the ambient bar, which is gone; escape still closes settings.
  useEscape(settings, () => setSettings(false));

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

  if (booting) return <div className="login" />;
  if (!user) return <Login gone={gone} onSignedIn={signIn} />;

  const views = {
    desk: <DeskView onError={onError} waiting={waiting} onOpenSession={(id) => { setJump(id); setView("projects"); }} />,
    approvals: <ApprovalsView onError={onError} waiting={waiting} onResolved={bump} />,
    files: <ComputerView onError={onError} jumpTo={openFile} onJumped={() => setOpenFile(null)} />,
    projects: (
      <LookingGlassView
        onError={onError}
        pulse={pulse}
        waiting={waiting}
        onPulse={bump}
        jump={jump}
        onJumped={() => setJump(null)}
        onOpenFile={(path) => {
          setOpenFile(path);
          setView("files");
        }}
      />
    ),
  };

  const pending = (waiting || []).length;

  return (
    <React.Fragment>
      <div className="app no-ambient">
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

      </div>

      {settings && (
        <SettingsModal user={user} onClose={() => setSettings(false)} onSignOut={signOut} onError={onError} />
      )}
    </React.Fragment>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
