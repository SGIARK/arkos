/* =========================================================
   Shared atoms and cards, ported from the pre-rewrite frontend.

   These are the originals' markup and behaviour, re-pointed at the endpoints
   that exist now. The class names, the structure and the choreography are not
   new work — only what they talk to is. What was genuinely bound to the deleted
   architecture (the task rows, the plan card's fenced-block approval flow, the
   demo login) is gone with it; everything here outlived its data source.
   ========================================================= */

/* The status dot. `live` pings, because a running thing should look running. */
function Dot({ kind, status, title }) {
  const tone = kind || toneFor(status);
  return <span className={"dot" + (tone ? " " + tone : "")} title={title || statusLabel(status)} />;
}

/* The old vocabulary was live / work / stop, and it is a better one than five
   flat colours: one thing is alive, one wants you, one has stopped. */
function toneFor(status) {
  if (status === "running") return "live";
  if (status === "awaiting_approval") return "work";
  if (status === "failed") return "stop";
  return "";
}

function Empty({ glyph, children }) {
  return (
    <div className="empty">
      {glyph && <span className="glyph">{glyph}</span>}
      {children}
    </div>
  );
}

function Spinner() {
  return <span className="spin" />;
}

/* The approval card, ported. It collapses itself on resolve rather than
   vanishing, and it carries the note field: answering a question and adding a
   sentence about why are the same gesture, so they are the same control.

   The old one took a workshopped plan; this one takes an `approval` row, which
   is the same thing the backend now calls a question. */
function ApprovalCard({ item, onResolve, onError }) {
  const [noteOpen, setNoteOpen] = useState(false);
  const [gone, setGone] = useState(false);
  const card = useRef(null);
  const note = useRef(null);

  const isAsk = item.kind === "ask";

  async function resolve(answer) {
    const extra = note.current ? note.current.value.trim() : "";
    const body = extra ? `${answer}\n\n${extra}` : answer;
    setGone(true);

    // Collapse in place: the card shrinking is what says "answered", and the
    // list closing over it is quieter than a row disappearing.
    if (card.current) {
      const height = card.current.offsetHeight;
      card.current.style.transition =
        "opacity .35s var(--ease), transform .35s var(--ease), margin .35s var(--ease), " +
        "max-height .35s var(--ease), padding .35s var(--ease), border-color .35s var(--ease)";
      card.current.style.maxHeight = height + "px";
      requestAnimationFrame(() => {
        if (!card.current) return;
        card.current.style.maxHeight = "0px";
        card.current.style.opacity = "0";
        card.current.style.paddingTop = "0px";
        card.current.style.paddingBottom = "0px";
        card.current.style.marginBottom = "-12px";
        card.current.style.borderColor = "transparent";
        card.current.style.transform = "translateY(-4px)";
      });
    }

    try {
      await api.answer(item.approval_id, body);
      setTimeout(onResolve, 360);
    } catch (e) {
      setGone(false);
      if (card.current) card.current.removeAttribute("style");
      onError(e);
    }
  }

  return (
    <div className="card approval" ref={card} style={{ overflow: "hidden" }}>
      <div className="top">
        <span className="src">
          <Dot kind="work" /> {item.session_title || "session"}
        </span>
        <span className="tag accent">{isAsk ? "answer" : "approve / decline"}</span>
      </div>
      <div className="title">{item.prompt}</div>
      {item.project_title && <div className="body">{item.project_title}</div>}

      {isAsk ? (
        <AskAnswer disabled={gone} onSend={resolve} />
      ) : (
        <>
          <textarea
            ref={note}
            className={"note" + (noteOpen ? " open" : "")}
            placeholder="add a note or a condition before approving…"
          />
          <div className="actions">
            <button className="btn ghost" onClick={() => setNoteOpen((o) => !o)}>
              {noteOpen ? "− note" : "+ note"}
            </button>
            <span className="grow" />
            <button className="btn" disabled={gone} onClick={() => resolve("no")}>
              decline
            </button>
            <button className="btn primary" disabled={gone} onClick={() => resolve("yes")}>
              approve →
            </button>
          </div>
        </>
      )}
    </div>
  );
}

/* An ask wants prose, not a verdict, so it gets the field rather than the pair
   of buttons. */
function AskAnswer({ disabled, onSend }) {
  const [text, setText] = useState("");
  return (
    <form
      className="actions"
      onSubmit={(e) => {
        e.preventDefault();
        if (text.trim()) onSend(text.trim());
      }}
    >
      <input
        className="answer"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="your answer…"
        disabled={disabled}
      />
      <button className="btn primary" type="submit" disabled={disabled || !text.trim()}>
        send →
      </button>
    </form>
  );
}
