/* =========================================================
   components — shared atoms + cards
   ========================================================= */
const { useState, useRef, useEffect } = React;

/* ---- atoms ---- */
function Dot({ kind }) { return <span className={"dot" + (kind ? " " + kind : "")} />; }

function Empty({ glyph, children }) {
  return (
    <div className="empty">
      <span className="glyph">{glyph}</span>
      {children}
    </div>
  );
}

/* ---- approval card (workshopped plan awaiting your ok) ---- */
function ApprovalCard({ item, onResolve }) {
  const [noteOpen, setNoteOpen] = useState(false);
  const [gone, setGone] = useState(false);
  const ref = useRef(null);

  function resolve(verb) {
    if (ref.current) {
      ref.current.style.transition = "opacity .35s, transform .35s, margin .35s, max-height .35s, padding .35s, border-color .35s";
      ref.current.style.maxHeight = ref.current.offsetHeight + "px";
      requestAnimationFrame(() => {
        ref.current.style.maxHeight = "0px";
        ref.current.style.opacity = "0";
        ref.current.style.paddingTop = "0px";
        ref.current.style.paddingBottom = "0px";
        ref.current.style.marginBottom = "-12px";
        ref.current.style.borderColor = "transparent";
        ref.current.style.transform = "translateY(-4px)";
      });
    }
    setGone(true);
    setTimeout(() => onResolve(item.id, verb), 360);
  }

  return (
    <div className="card approval" ref={ref} style={{ overflow: "hidden" }}>
      <div className="top">
        <span className="src"><Dot kind="work" /> {item.src}</span>
        <span className="tag accent">{item.tag}</span>
      </div>
      <div className="title">{item.title}</div>
      <div className="body">{item.body}</div>
      <div className="plan">
        <ol>{item.plan.map((p, i) => <li key={i}>{p}</li>)}</ol>
        <div className="toolrow">
          {item.tools.map((t) => <span className="chip" key={t}>{t}</span>)}
        </div>
      </div>
      <textarea
        className={"note" + (noteOpen ? " open" : "")}
        placeholder="add a note or tweak before approving…"
      />
      <div className="actions">
        <button className="btn ghost grow" style={{ textAlign: "left", flex: "0 0 auto" }} onClick={() => setNoteOpen((o) => !o)}>
          {noteOpen ? "− note" : "+ note"}
        </button>
        <span className="grow" />
        <button className="btn" disabled={gone} onClick={() => resolve("declined")}>decline</button>
        <button className="btn primary" disabled={gone} onClick={() => resolve("approved")}>approve →</button>
      </div>
    </div>
  );
}

/* ---- task row (expandable event log) ---- */
function TaskRow({ item }) {
  const [open, setOpen] = useState(false);
  const running = item.state === "run";
  return (
    <div className={"row col" + (open ? " expanded" : "")}>
      <div className="row-top" style={{ cursor: item.events ? "pointer" : "default" }} onClick={() => item.events && setOpen((o) => !o)}>
        <span className="label">
          {running ? <span className="spin" /> : <Dot kind={item.state === "done" ? "live" : "stop"} />}
          <span className="text">{item.text} <span className="src">· {item.src}</span></span>
        </span>
        <span className="when">{item.when}</span>
      </div>
      {item.events && (
        <div className={"events" + (open ? " open" : "")}>
          {item.events.map((e, i) => (
            <div className="ev" key={i}>
              <span className="k">{e.k}</span>
              <span className={e.ok ? "ok" : ""}>{e.t}{e.ok ? " ✓" : ""}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ---- watch row ---- */
function WatchRow({ item }) {
  return (
    <div className="row">
      <span className="label">
        <Dot kind={item.live ? "live" : ""} />
        <span className="text">{item.src}{item.note ? <span className="src"> · {item.note}</span> : null}</span>
      </span>
      <span className="when">{item.cadence}</span>
    </div>
  );
}

Object.assign(window, { Dot, Empty, ApprovalCard, TaskRow, WatchRow });
