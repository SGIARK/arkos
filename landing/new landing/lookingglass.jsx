/* =========================================================
   looking glass — projects grid + live session detail
   ========================================================= */

const LG_PROJECTS = [
  { id: "p1", name: "q3 competitor teardown", session: "scrape competitor pricing", status: "work", when: "active · hop 7/15" },
  { id: "p2", name: "weekly linear triage", session: "close stale issues", status: "attn", when: "needs input · fri" },
  { id: "p3", name: "travel — sf offsite", session: "find flights + hotel", status: "err", when: "failed · 3d ago" },
  { id: "p4", name: "inbox cleanup", session: "unsubscribe sweep", status: "ok", when: "done · 2h ago" },
];

function LookingGlassView() {
  const [openId, setOpenId] = useState(null);
  const [running, setRunning] = useState(true);
  const [tab, setTab] = useState("files");
  const [openArgs, setOpenArgs] = useState({});
  const [openResult, setOpenResult] = useState({});

  const proj = LG_PROJECTS.find((p) => p.id === openId);

  if (!proj) {
    return (
      <div className="lg-view">
        <div className="projects-grid">
          <div className="pg-grid">
            {LG_PROJECTS.map((p) => (
              <div className="proj-card" key={p.id} onClick={() => setOpenId(p.id)}>
                <span className={"pc-status " + p.status} title={p.status === "work" ? "working" : p.status === "attn" ? "needs attention" : p.status === "err" ? "failed" : "done"} />
                <div className="pc-name">{p.name}</div>
                <div className="pc-sess">{p.session}</div>
                <div className="pc-when">{p.when}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="lg-view">
      <div className="lg-ctxline">
        <button className="back-btn" onClick={() => setOpenId(null)}>← projects</button>
        <span className="path"><b>{proj.name}</b><span className="crumb">▸</span>{proj.session}</span>
        <span className="status-pill"><span className={"dot " + (running ? "live" : "work")} />{running ? "running" : "paused"}</span>
        <span className="lg-budget">hop 7/15 · 4m12s</span>
        <span className="grow" />
        <div className="lg-ctrls">
          <button className="icon-round" title={running ? "pause" : "resume"} onClick={() => setRunning((r) => !r)}>{running ? "❙❙" : "▶"}</button>
          <button className="icon-round danger" title="cancel">✕</button>
        </div>
      </div>

      <div className="lg-body">
        <div className="stream-wrap">
          <div className="stream">
            <div className="ev-block ev-assist"><span className="who">buddy</span><p>starting with the two competitors you flagged — vantage and northline. pulling current list pricing before i touch anything else.</p></div>

            <div className="ev-block ev-tool">
              <div className="row1"><span className="arrow">→</span><span className="name">browser_task</span><span className="kicker">tool</span></div>
              <div className={"args" + (openArgs.a1 ? "" : " collapsed")} onClick={() => setOpenArgs((s) => ({ ...s, a1: !s.a1 }))}>{'{ url: "vantage.io/pricing", goal: "capture plan names, prices, and billing period" }'}</div>
            </div>

            <div className="ev-block">
              <div className="ev-result">
                <div className="rhead"><span className="src"><span className="dot live" />vantage.io/pricing</span><button className="expand" onClick={() => setOpenResult((s) => ({ ...s, r1: !s.r1 }))}>{openResult.r1 ? "collapse" : "expand"}</button></div>
                <pre className={openResult.r1 ? "open" : ""}>{`starter — $29/mo, billed monthly. 3 seats, 10gb storage.
growth — $89/mo or $890/yr, unlimited seats up to 25, 100gb.
enterprise — "contact sales", custom seats + storage, sso, sla.
note: prices shown in usd, a small "eu pricing" toggle exists top-right.`}</pre>
                <div className="note">4,208 chars · showing first 2,000</div>
              </div>
            </div>

            <div className="ev-block ev-tool">
              <div className="row1"><span className="arrow">→</span><span className="name">browser_task</span><span className="kicker">tool</span></div>
              <div className={"args" + (openArgs.a2 ? "" : " collapsed")} onClick={() => setOpenArgs((s) => ({ ...s, a2: !s.a2 }))}>{'{ url: "northline.com/pricing", goal: "capture plan names, prices, and billing period" }'}</div>
            </div>

            <div className="ev-block">
              <div className="ev-result error">
                <div className="rhead"><span className="src"><span className="dot stop" />northline.com/pricing</span><button className="expand" onClick={() => setOpenResult((s) => ({ ...s, r2: !s.r2 }))}>{openResult.r2 ? "collapse" : "expand"}</button></div>
                <pre className={openResult.r2 ? "open" : ""}>timeout after 30s — page sat behind a cloudflare interstitial and never resolved.</pre>
                <div className="note">error · retried once, same result</div>
              </div>
            </div>

            <div className="ev-block ev-assist"><span className="who">buddy</span><p>northline's pricing page is gated behind a bot check, so the direct fetch won't work. i pulled their last cached pricing from the wayback machine instead — dated 6 weeks ago, close enough to flag as "approximate" in the teardown.</p></div>

            <div className="ev-block ev-status"><span className="spin" />using the browser…</div>

            <div className="ev-block ev-ask">
              <span className="who">buddy — needs input</span>
              two pricing pages found for northline: a us page and an eu page, about 12% apart after fx. which region should the teardown use as the baseline?
              <div className="opts"><span className="opt">us pricing</span><span className="opt">eu pricing</span><span className="opt">show both</span></div>
            </div>

            <div className="ev-block ev-assist"><span className="who">buddy</span><p>while i wait on that — drafting the comparison table now with everything confirmed so far. i'll drop the northline row in once you pick a region.<span className="caret" style={{ marginLeft: 2 }} /></p></div>
          </div>

          <div className="lg-composer">
            <span className="prompt">ark&gt;</span>
            <input placeholder="suggest or steer this session…" spellCheck={false} autoComplete="off" />
            <span className="attach">+ attach</span>
          </div>
        </div>

        <div className="ctx-panel">
          <div className="todo-block">
            <span className="kicker">todo</span>
            <div className="todo-list">
              <label className="todo-item done"><input type="checkbox" defaultChecked /><span>pull vantage pricing</span></label>
              <label className="todo-item done"><input type="checkbox" defaultChecked /><span>find northline fallback source</span></label>
              <label className="todo-item"><input type="checkbox" /><span>confirm region baseline w/ nate</span></label>
              <label className="todo-item"><input type="checkbox" /><span>draft comparison table</span></label>
              <label className="todo-item"><input type="checkbox" /><span>write teardown summary</span></label>
            </div>
          </div>
          <div className="ctx-tabs">
            <button className={tab === "files" ? "active" : ""} onClick={() => setTab("files")}><span className="tab-label">working files</span></button>
            <button className={tab === "memory" ? "active" : ""} onClick={() => setTab("memory")}><span className="tab-label">project memory</span></button>
          </div>
          {tab === "files" ? (
            <div className="ctx-content">
              <div className="cv-entry"><span className="nm"><span className="g">·</span>pricing.csv</span><span className="sz">4.2 kb</span></div>
              <div className="cv-entry"><span className="nm"><span className="g">·</span>teardown.md</span><span className="sz">1.1 kb</span></div>
              <div className="cv-entry"><span className="nm"><span className="g">·</span>vantage-screenshot.png</span><span className="sz">220 kb</span></div>
              <div className="cv-entry"><span className="nm"><span className="g">·</span>northline-archived.pdf</span><span className="sz">640 kb</span></div>
              <div className="dropzone">drop files into this project</div>
            </div>
          ) : (
            <div className="ctx-content">
              <div className="mem-note"><span className="k">standing rule</span>always cite source + date for competitor pricing — it goes stale fast.</div>
              <div className="mem-note"><span className="k">standing rule</span>prefer us pricing as the default baseline unless told otherwise.</div>
              <div className="mem-note"><span className="k">learned</span>northline blocks automated fetches; wayback machine is a reliable fallback.</div>
              <div className="mem-note" style={{ borderBottom: "none" }}><span className="k">learned</span>nate reads teardowns as tables first, prose second — lead with the comparison.</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { LookingGlassView });
