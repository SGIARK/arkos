/* =========================================================
   The rail: peripheral vision, across both tabs.

   Two sections and no timer. "Needs you" is every unanswered question at
   account scope; "Running" is every live session with the project it belongs
   to. Both reload on navigation and when an open session's stream says
   something that could have changed them — a lifecycle move or a terminal.

   Deliberately NOT a live feed of everything happening everywhere: that needs
   the hold-back window (G27) and is deferred. Peripheral vision that is a few
   seconds stale is fine; a polling loop to avoid that is not.
   ========================================================= */

function Rail({ refreshKey, openSession, onOpenSession, onSettings, theme, onTheme, onError }) {
  const [waiting, setWaiting] = useState([]);
  const [running, setRunning] = useState([]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [attention, live] = await Promise.all([api.attention(), api.sessions("running")]);
        if (cancelled) return;
        setWaiting(attention);
        setRunning(live);
      } catch (e) {
        if (!cancelled) onError(e);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [refreshKey, onError]);

  return (
    <aside className="rail">
      <div className="rail-section">
        <span className="kicker">Needs you</span>
        {!waiting.length && <p className="empty rail-empty">Nothing waiting.</p>}
        {waiting.map((item) => (
          <button
            key={item.approval_id}
            className={"rail-item" + (item.session_id === openSession ? " here" : "")}
            onClick={() => onOpenSession(item.session_id)}
            title={item.prompt}
          >
            <span className="dot dot-wait" />
            <span className="rail-text">{item.prompt}</span>
            <span className="rail-sub">{item.project_title || item.session_title || "session"}</span>
          </button>
        ))}
      </div>

      <div className="rail-section">
        <span className="kicker">Running</span>
        {!running.length && <p className="empty rail-empty">Nothing running.</p>}
        {running.map((session) => (
          <button
            key={session.session_id}
            className={"rail-item" + (session.session_id === openSession ? " here" : "")}
            onClick={() => onOpenSession(session.session_id)}
            title={session.title || "untitled session"}
          >
            <span className="dot dot-run" />
            <span className="rail-text">{session.title || "untitled session"}</span>
            <span className="rail-sub">
              {session.project_title || "no project"} · {session.hops_used}/{session.hops_max}
            </span>
          </button>
        ))}
      </div>

      <div className="rail-foot">
        <button className="link" onClick={() => onTheme(theme === "light" ? "dark" : "light")}>
          {theme === "light" ? "dark" : "light"}
        </button>
        <button className="link" onClick={onSettings}>
          settings
        </button>
      </div>
    </aside>
  );
}
