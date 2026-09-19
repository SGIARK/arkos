const NAV = ["desk", "approvals", "files", "projects"];

/* Hashes bookmarked under the old view names still have to resolve. */
const NAV_ALIAS = { computer: "files", "looking glass": "projects" };

function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem("arkos-theme") || "light");
  const [user, setUser] = useState(null);
  const [booting, setBooting] = useState(true);
  const [gone, setGone] = useState(false);
  // A refusal carried back on the OAuth fragment; only the sign-in screen shows it.
  const [authProblem, setAuthProblem] = useState(null);
  // Abandoning the reset screen must leave somewhere to go.
  const [recoveryDone, setRecoveryDone] = useState(false);
  const [authNotice, setAuthNotice] = useState(null);
  const [view, setView] = useState(() => {
    const hash = decodeURIComponent(location.hash.replace("#", ""));
    const named = NAV_ALIAS[hash] || hash;
    return NAV.includes(named) ? named : "projects";
  });
  const [settings, setSettings] = useState(false);
  const [error, setError] = useState(null);
  const [pulse, setPulse] = useState(0);
  // Fetched once per pulse and passed down to every surface; null means not yet read.
  const [waiting, setWaiting] = useState(null);
  const [jump, setJump] = useState(null);
  // Path the Files tab should land on: pane and tab are two views of one namespace.
  const [openFile, setOpenFile] = useState(null);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("arkos-theme", theme);
  }, [theme]);

  useEffect(() => {
    location.hash = encodeURIComponent(view);
  }, [view]);

  /* The OAuth return must be asked before `me()`: that load carries a token on
     the fragment and no cookie yet, so `me()` would 401 and flash the form. */
  useEffect(() => {
    let dead = false;
    (async () => {
      try {
        const returned = await api.returnFromOAuth();
        const who = returned || (await api.me());
        if (dead) return;
        setUser(who);
        setGone(true);
      } catch (e) {
        if (dead) return;
        setUser(null);
        if (e && e.code === "sign_in_failed") {
          setAuthProblem(api.linkExpired() ? "that reset link has expired — ask for another" : e.message);
        }
      } finally {
        if (!dead) setBooting(false);
      }
    })();
    return () => {
      dead = true;
    };
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

  /* Keyed to `user`, NOT to `pulse`: the account-wide list must not depend on
     any window being mounted. */
  const readWaiting = useCallback(() => {
    api.attention().then(setWaiting).catch(() => {});
  }, []);

  useEffect(() => {
    if (!user) return undefined;
    readWaiting();
    /* One subscription for the account; the first frame arrives on connect. */
    return api.watchAttention(readWaiting);
  }, [user, readWaiting]);

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
  /* A reset link outranks EVERYTHING, including an existing session. Guarding
     this on `!user` meant that clicking it while still signed in dropped the
     token and opened the app as normal — the one person who asked to change
     their password, silently refused. */
  if (api.recoveryPending() && !recoveryDone) {
    return (
      <ResetPassword
        onDone={() => {
          setRecoveryDone(true);
          setAuthNotice("password changed — sign in with it");
        }}
        onGiveUp={() => setRecoveryDone(true)}
      />
    );
  }
  if (!user)
    return (
      <Login
        gone={gone}
        onSignedIn={signIn}
        problem={authProblem}
        notice={authNotice}
        startMode={api.linkExpired() ? "forgot" : "in"}
      />
    );

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
            <span className="glyph">b</span>
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
              arkos <b>v1</b>
            </span>
            <span className="sep">/</span>
            <span>
              {/* Name if one was given at sign-up; an older account has none. */}
              user <b>{user.display_name || user.email || user.user_id}</b>
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
