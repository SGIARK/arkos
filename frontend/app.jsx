/* =========================================================
   Root: sign in, then the grid or one window.

   Two surfaces and nothing else. The old nav (desk / tasks / watching /
   approvals / computer / chat) described an architecture that no longer
   exists — a session is the conversation and the run, so watching one and
   talking to it are the same window.
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
        <button type="submit" disabled={busy || !email || !password}>
          {busy ? "Signing in…" : "Sign in"}
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
  const [openSession, setOpenSession] = useState(null);
  const [error, setError] = useState(null);
  const [theme, setTheme] = useState(() => localStorage.getItem("ark-theme") || "light");

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

  const signOut = async () => {
    try {
      await api.signOut();
    } catch (e) {
      // Whatever the server said, this browser is done with the session.
    }
    setUser(null);
    setOpenSession(null);
  };

  if (booting) return <div className="empty boot">…</div>;
  if (!user) return <SignIn onSignedIn={setUser} />;

  return (
    <div className="app">
      <header className="topbar">
        <span className="brand" onClick={() => setOpenSession(null)}>
          ARK
        </span>
        <span className="topbar-right">
          <button className="link" onClick={() => setTheme(theme === "light" ? "dark" : "light")}>
            {theme === "light" ? "dark" : "light"}
          </button>
          <span className="who">{user.email || user.user_id}</span>
          <button className="link" onClick={signOut}>
            sign out
          </button>
        </span>
      </header>

      <Problem error={error} onDismiss={() => setError(null)} />

      {openSession ? (
        <SessionWindow sessionId={openSession} onBack={() => setOpenSession(null)} onError={onError} />
      ) : (
        <Grid onOpenSession={setOpenSession} onError={onError} />
      )}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
