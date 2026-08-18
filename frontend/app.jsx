/* =========================================================
   Root: sign in, then the chat you already have.

   The landing is the chat, not the grid. The product is a companion with
   memory, so opening the app is walking up to your desk where ARK already is —
   the home session, an ordinary attended session the server made on first
   login. Looking Glass is the Projects tab, one click away, where work lives in
   its project bubbles.

   Two tabs and a rail. The old nav (desk / tasks / watching / approvals /
   computer / chat) described an architecture that no longer exists: a session
   is the conversation and the run, so watching one and talking to it are the
   same window.
   ========================================================= */

function SignIn({ onSignedIn }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState(null);

  const submit = async (e) => {
    e.preventDefault();
    setBusy(true);
    setProblem(null);
    try {
      onSignedIn(await api.signIn(email, password));
    } catch (err) {
      setProblem(err.message || "Sign-in failed.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="signin">
      <form onSubmit={submit}>
        <h1>ARK</h1>
        <input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="you@example.com"
          autoComplete="username"
          required
        />
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Password"
          autoComplete="current-password"
          required
        />
        <button className="primary" type="submit" disabled={busy || !email || !password}>
          {busy ? "signing in…" : "sign in →"}
        </button>
        {problem && <p className="problem">{problem}</p>}
        {/* No reset link: accounts are made in the Supabase dashboard until
            there are real users to self-serve. */}
      </form>
    </div>
  );
}

/* Failures surface here rather than in a console nobody has open. */
function Problem({ error, onDismiss }) {
  if (!error) return null;
  return (
    <div className="banner" role="alert">
      <span>{error.message || error.code || "Something failed."}</span>
      <button className="link" onClick={onDismiss}>
        dismiss
      </button>
    </div>
  );
}

function App() {
  const [user, setUser] = useState(null);
  const [booting, setBooting] = useState(true);
  const [tab, setTab] = useState("chat");
  // Only meaningful in the Projects tab; the Chat tab always shows home.
  const [openSession, setOpenSession] = useState(null);
  const [error, setError] = useState(null);
  const [settings, setSettings] = useState(false);
  const [theme, setTheme] = useState(() => localStorage.getItem("ark-theme") || "light");
  // Bumped when something happened that the rail's two sections are made of.
  const [pulse, setPulse] = useState(0);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("ark-theme", theme);
  }, [theme]);

  /* The cookie may already be good, so the page asks before it offers a form. */
  useEffect(() => {
    api
      .me()
      .then(setUser)
      .catch(() => setUser(null))
      .finally(() => setBooting(false));
  }, []);

  const onError = useCallback((e) => {
    // An expired or missing cookie is not an error to shout about; it is a
    // sign-in.
    if (e && (e.code === "unauthenticated" || e.code === "http_401")) {
      setUser(null);
      return;
    }
    setError(e);
  }, []);

  const bump = useCallback(() => setPulse((n) => n + 1), []);

  const home = user && user.home_session_id;

  /* One place decides where a session id lands: home opens the Chat tab,
     anything else opens it in Projects. */
  const openSessionAnywhere = useCallback(
    (sessionId) => {
      if (home && sessionId === home) {
        setTab("chat");
        setOpenSession(null);
      } else {
        setTab("projects");
        setOpenSession(sessionId);
      }
      bump();
    },
    [home, bump],
  );

  const signOut = async () => {
    try {
      await api.signOut();
    } catch (e) {
      // Whatever the server said, this browser is done with the session.
    }
    setUser(null);
    setOpenSession(null);
    setSettings(false);
    setTab("chat");
  };

  if (booting) return <div className="empty boot">…</div>;
  if (!user) return <SignIn onSignedIn={setUser} />;

  const showChat = tab === "chat";

  return (
    <div className="app">
      <header className="topbar">
        <span className="brand">ARK</span>
        <nav className="tabs">
          <button
            className={"tab" + (showChat ? " on" : "")}
            onClick={() => {
              setTab("chat");
              bump();
            }}
          >
            Chat
          </button>
          <button
            className={"tab" + (!showChat ? " on" : "")}
            onClick={() => {
              setTab("projects");
              setOpenSession(null);
              bump();
            }}
          >
            Projects
          </button>
        </nav>
        <span className="who">{user.email || user.user_id}</span>
      </header>

      <Problem error={error} onDismiss={() => setError(null)} />

      {settings && (
        <SettingsModal user={user} onClose={() => setSettings(false)} onSignOut={signOut} />
      )}

      <div className="body">
        <Rail
          refreshKey={pulse}
          openSession={showChat ? home : openSession}
          onOpenSession={openSessionAnywhere}
          onSettings={() => setSettings(true)}
          theme={theme}
          onTheme={setTheme}
          onError={onError}
        />

        <main className="surface">
          {showChat ? (
            home ? (
              /* No back link: the tabs are how you leave the chat. */
              <SessionWindow sessionId={home} onError={onError} onActivity={bump} />
            ) : (
              <Empty glyph="◇">
                no home session yet — sign out and back in, and the server will make one
              </Empty>
            )
          ) : openSession ? (
            <SessionWindow
              sessionId={openSession}
              onBack={() => setOpenSession(null)}
              onError={onError}
              onActivity={bump}
            />
          ) : (
            <Grid onOpenSession={openSessionAnywhere} onError={onError} />
          )}
        </main>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
