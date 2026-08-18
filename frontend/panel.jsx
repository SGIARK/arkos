/* =========================================================
   The right panel: a pinned TODO above one canvas.

   Three things wanted this space and they are not peers. The TODO is pinned,
   because hiding a five-line checklist behind a tab is worse than showing it.
   The canvas is one of two — the project's working files, or the live browser —
   and you pick, because choosing to watch is the difference between watching
   and being interrupted. Project memory is not here: it would be a tab for a
   feature that is not in the product yet.

   The whole panel collapses, and what it was showing is remembered per browser.
   ========================================================= */

/* Pinned, always visible during a run. Five lines is the point. */
function TodoPanel({ items }) {
  if (!items || !items.length) return null;
  return (
    <div className="panel-block">
      <span className="kicker">Plan</span>
      {items.map((item, i) => (
        <div key={i} className={"todo-item todo-" + (item.status || "pending")}>
          <span className="todo-mark">
            {item.status === "completed" ? "✓" : item.status === "in_progress" ? "▸" : "○"}
          </span>
          {item.text || item.title || String(item)}
        </div>
      ))}
    </div>
  );
}

/* The project's durable files: uploaded here, mounted into the sandbox when a
   session takes its box, and readable without waking anything. */
function FilesCanvas({ projectId, onError }) {
  const [files, setFiles] = useState(null);
  const [busy, setBusy] = useState(false);
  const [over, setOver] = useState(false);

  const load = useCallback(async () => {
    if (!projectId) return;
    try {
      setFiles(await api.files(projectId));
    } catch (e) {
      onError(e);
    }
  }, [projectId, onError]);

  useEffect(() => {
    load();
  }, [load]);

  const send = async (list) => {
    if (!projectId || !list || !list.length) return;
    setBusy(true);
    try {
      for (const file of list) await api.upload(projectId, file);
      await load();
    } catch (e) {
      onError(e);
    } finally {
      setBusy(false);
    }
  };

  if (!projectId) return <p className="empty panel-empty">This session has no project.</p>;

  return (
    <div
      className={"files" + (over ? " over" : "")}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        send(e.dataTransfer.files);
      }}
    >
      {files === null && <p className="empty panel-empty">Loading…</p>}
      {files !== null && !files.length && <p className="empty panel-empty">No files yet.</p>}
      {(files || []).map((file) => (
        <div className="file" key={file.file_id} title={file.path}>
          <span className="file-name">{file.path}</span>
          <span className="file-size">{fileSize(file.size)}</span>
        </div>
      ))}
      <label className="drop">
        {busy ? "uploading…" : "drop files here, or choose"}
        <input type="file" multiple hidden onChange={(e) => send(e.target.files)} />
      </label>
    </div>
  );
}

/* The browser's frames, while it is browsing. Mounted from the `status` event
   that announced the url; frames are a side-channel and never events, so there
   is nothing to replay and nothing to scroll back to. */
function BrowserCanvas({ url }) {
  const [frame, setFrame] = useState(null);

  useEffect(() => {
    if (!url) return undefined;
    setFrame(null);
    const source = new EventSource(url, { withCredentials: true });
    source.addEventListener("frame", (e) => {
      try {
        setFrame(JSON.parse(e.data).jpeg);
      } catch (err) {
        /* one dropped picture, not a broken pane */
      }
    });
    return () => source.close();
  }, [url]);

  if (!url) return <p className="empty panel-empty">No browser run yet.</p>;
  return frame ? (
    <img className="frame" alt="what the browser is looking at" src={"data:image/jpeg;base64," + frame} />
  ) : (
    <p className="empty panel-empty">waiting for the first frame…</p>
  );
}

function RightPanel({ projectId, todo, browserUrl, onError }) {
  const [open, setOpen] = useState(() => localStorage.getItem("ark-panel") !== "closed");
  const [canvas, setCanvas] = useState(() => localStorage.getItem("ark-canvas") || "files");

  useEffect(() => {
    localStorage.setItem("ark-panel", open ? "open" : "closed");
  }, [open]);
  useEffect(() => {
    localStorage.setItem("ark-canvas", canvas);
  }, [canvas]);

  if (!open) {
    return (
      <button className="panel-tab" onClick={() => setOpen(true)} title="show the panel">
        ‹
      </button>
    );
  }

  return (
    <aside className="panel">
      <TodoPanel items={todo} />

      <div className="panel-block panel-canvas">
        <div className="canvas-tabs">
          <button className={"tab" + (canvas === "files" ? " on" : "")} onClick={() => setCanvas("files")}>
            Files
          </button>
          <button className={"tab" + (canvas === "browser" ? " on" : "")} onClick={() => setCanvas("browser")}>
            Browser
            {/* Available, not demanding: a live run gets a dot, never focus. */}
            {browserUrl && <span className="dot dot-run" />}
          </button>
          <button className="link panel-hide" onClick={() => setOpen(false)}>
            hide
          </button>
        </div>

        {canvas === "files" ? (
          <FilesCanvas projectId={projectId} onError={onError} />
        ) : (
          <BrowserCanvas url={browserUrl} />
        )}
      </div>
    </aside>
  );
}

function fileSize(n) {
  if (n === null || n === undefined) return "";
  if (n < 1024) return n + " b";
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " kb";
  return (n / 1024 / 1024).toFixed(1) + " mb";
}
