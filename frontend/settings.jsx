/* =========================================================
   Settings: connections, account, sign out.

   Ported from the panel deleted with views.jsx rather than rewritten, because
   its popup choreography was already correct and hard-won — see `connect`. Only
   the endpoint calls changed: /connections, /connections/{server}/connect and
   DELETE /connections/{server} replace the old services API.

   The model's own auth_required message tells people to authorize a server
   "from the connections panel in Settings", so this panel existing is part of
   that message being true.
   ========================================================= */

function SettingsModal({ user, onClose, onSignOut }) {
  const [rows, setRows] = useState(null);
  const [problem, setProblem] = useState(null);
  const [busy, setBusy] = useState({});
  // A popup the browser blocked: the setup url is surfaced as a link instead.
  const [links, setLinks] = useState({});
  const poll = useRef(null);

  const refresh = useCallback(async () => {
    try {
      setRows(await api.connections());
      setProblem(null);
    } catch (e) {
      setProblem(e.message || "Could not read connections.");
    }
  }, []);

  useEffect(() => {
    refresh();
    return () => {
      if (poll.current) clearInterval(poll.current);
    };
  }, [refresh]);

  /* Watch until the server reports connected or the popup closes. This is the
     only timer in the app and it exists for the length of one OAuth flow: the
     browser gets no callback when a popup on another origin finishes. */
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
    // The popup is opened SYNCHRONOUSLY, inside the click handler. Opening it
    // after `await api.connect(...)` loses the user gesture and the browser
    // blocks it silently — the "nothing happens" bug. So a blank window is
    // opened now and navigated once Smithery hands back the setup url.
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
          // The return url is the backend's business now: it passes its own
          // callback to Smithery when it mints the connection.
          if (popup && !popup.closed) popup.location.href = result.setup_url;
          else setLinks((m) => ({ ...m, [server]: result.setup_url }));
          watch(server, popup);
        } else {
          if (popup) popup.close();
          setProblem("Could not start authorization for " + server + ".");
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
        <h2>Settings</h2>

        <section>
          <span className="kicker">Tools &amp; connections</span>
          {rows === null && <Empty>reading…</Empty>}
          {problem && <p className="problem">{problem}</p>}
          {rows !== null && !rows.length && (
            <Empty glyph="◇">
              no servers configured — add entries to <code>mcp_servers</code> in config.yaml
            </Empty>
          )}
          {(rows || []).map((row) => {
            const connected = row.status === "connected";
            return (
              <div className="conn" key={row.server}>
                <span className="conn-name">
                  <Dot kind={connected ? "live" : ""} />
                  {row.name || row.server}
                  {connected && row.tool_count > 0 && (
                    <span className="conn-count">{row.tool_count} tools</span>
                  )}
                </span>
                <span className="conn-action">
                  <span className="conn-status">
                    {connected ? "connected" : row.requires_auth ? row.status : "shared"}
                  </span>
                  {!row.requires_auth ? (
                    <span className="empty">always on</span>
                  ) : connected ? (
                    <button className="ghost" disabled={busy[row.server]} onClick={() => disconnect(row.server)}>
                      disconnect
                    </button>
                  ) : links[row.server] ? (
                    /* The popup was blocked, so the flow becomes a link the
                       human clicks themselves. */
                    <a
                      className="button"
                      href={links[row.server]}
                      target="ark_oauth"
                      rel="noopener"
                      onClick={() => watch(row.server, null)}
                    >
                      authorize →
                    </a>
                  ) : (
                    <button className="primary" disabled={busy[row.server]} onClick={() => connect(row.server)}>
                      {busy[row.server] ? "…" : "connect"}
                    </button>
                  )}
                </span>
              </div>
            );
          })}
        </section>

        <section>
          <span className="kicker">Account</span>
          <div className="empty bare">
            signed in as <b>{(user && (user.email || user.user_id)) || "—"}</b>
          </div>
        </section>

        <div className="modal-foot">
          <button className="ghost danger" onClick={onSignOut}>
            sign out
          </button>
          <button onClick={onClose}>close</button>
        </div>
      </div>
    </div>
  );
}
