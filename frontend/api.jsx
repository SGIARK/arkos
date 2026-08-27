/* =========================================================
   The data layer: sign-in, the endpoint table, and the stream.

   Everything here talks to the API on our own origin. That is not a
   convenience: the session cookie is httpOnly and SameSite=Lax, so a
   cross-origin backend would leave every EventSource unauthenticated. The page
   never holds a token — supabase-js hands one over once at sign-in, the backend
   turns it into a cookie, and the browser does the rest.
   ========================================================= */

/* Every file here is a <script type="text/babel">, so they share one global
   lexical scope: the hooks are destructured once, in the first script the page
   loads, and a second `const` of these names anywhere would throw. */
const { useState, useEffect, useCallback, useRef } = React;

/* The API is wherever this page came from. /app is served by the same app. */
const API = location.origin;

/* Supabase's client, built from GET /auth/config on first use. Only sign-in
   touches it; every other call in this file is to our own API. */
let _supabase = null;

async function supabaseClient() {
  if (_supabase) return _supabase;
  const cfg = await request("GET", "/auth/config");
  if (!cfg.supabase_url || !cfg.anon_key) {
    throw new ApiError(
      "not_configured",
      "Sign-in is not configured: the server has no Supabase URL or publishable key.",
    );
  }
  _supabase = window.supabase.createClient(cfg.supabase_url, cfg.anon_key, {
    auth: {
      // The cookie is the session. Persisting a second one in localStorage would
      // be a copy of a credential we deliberately do not keep.
      persistSession: false,
      autoRefreshToken: false,
      /* IMPLICIT, not PKCE (12.1). PKCE is supabase-js's default and is the
         better flow when the client keeps a session — but it stores a code
         verifier between the redirect out and the redirect back, and with
         `persistSession: false` there is nowhere to keep one, so the exchange
         would fail every time. Implicit hands the token straight back on the
         URL fragment, which is exactly what this app wants: a token to trade
         for a cookie, once, and never store. The fragment never reaches the
         server, and `returnFromOAuth` clears it before it can be shared. */
      flowType: "implicit",
      /* We read the fragment ourselves in `returnFromOAuth`, because the token
         is not the session here — the cookie is, and the token is only good for
         the one exchange that mints it. */
      detectSessionInUrl: false,
    },
  });
  return _supabase;
}

/* Trade a Supabase access token for our cookie. THE one session-establishment
   path (12.1): password sign-in, sign-up and Google all end here, because
   `POST /auth/session` is the only endpoint that reads a bearer token and the
   only thing that mints a cookie. A second path would be a second place for
   identity to be decided. */
async function exchange(token) {
  const response = await fetch(API + "/auth/session", {
    method: "POST",
    credentials: "same-origin",
    headers: { Authorization: "Bearer " + token },
  });
  if (!response.ok) {
    const shape = await response.json().catch(() => ({}));
    throw new ApiError(shape.code || "sign_in_failed", shape.message || "The server rejected the token.");
  }
  return api.me();
}

class ApiError extends Error {
  constructor(code, message, retryable) {
    super(message || code);
    this.code = code;
    this.retryable = !!retryable;
  }
}

/* One shape for every failure, because the backend has one: {code, message,
   retryable}. A response that is not JSON is still an error, not a crash. */
async function request(method, path, body) {
  const opts = { method, credentials: "same-origin", headers: {} };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  let response;
  try {
    response = await fetch(API + path, opts);
  } catch (e) {
    throw new ApiError("offline", "Could not reach the server.", true);
  }
  if (response.status === 204) return null;

  let payload = null;
  try {
    payload = await response.json();
  } catch (e) {
    payload = null;
  }
  if (!response.ok) {
    const shape = payload || {};
    throw new ApiError(
      shape.code || "http_" + response.status,
      shape.message || response.statusText,
      shape.retryable,
    );
  }
  return payload;
}

/* The wire carries `{seq, ts, kind, version, payload:{...}}`; every renderer
   wants one flat event. Flattened here, once, at the boundary — so "store-shape
   is wire-shape" is true of what the components actually receive, and no
   renderer has to know the envelope exists. */
function asEvent(raw) {
  const { payload, ...rest } = raw || {};
  return { ...rest, ...(payload || {}) };
}

/* Multipart goes around `request` rather than through it: the browser sets its
   own boundary, and a Content-Type we invented would break it.

   The path is REQUIRED and carries the folder: the store is one flat namespace
   per user and every file in it lives in a folder, so there is no root to drop
   something into. */
async function _postFile(blob, path, whenItFails) {
  const form = new FormData();
  form.append("file", blob, path.split("/").pop());
  form.append("path", path);
  const response = await fetch(`${API}/files`, {
    method: "POST",
    credentials: "same-origin",
    body: form,
  });
  if (!response.ok) {
    const shape = await response.json().catch(() => ({}));
    throw new ApiError(shape.code || "upload_failed", shape.message || whenItFails);
  }
  return response.json();
}

const api = {
  ApiError,

  /* --- who is here ------------------------------------------------------ */

  me: () => request("GET", "/auth/me"),

  /* Sign in with Supabase, then trade the token for our cookie. The token is
     used once, here, and never stored: from the next request on, identity is
     the httpOnly cookie, which script cannot read. */
  async signIn(email, password) {
    const client = await supabaseClient();
    const { data, error } = await client.auth.signInWithPassword({ email, password });
    if (error) throw new ApiError("sign_in_failed", error.message);
    const token = data && data.session && data.session.access_token;
    if (!token) throw new ApiError("sign_in_failed", "Supabase returned no session.");
    return exchange(token);
  },

  /* Make an account (12.1). The NAME rides as user_metadata, so it arrives on
     the signed token and `POST /auth/session` can trust it; nothing about the
     name is sent in a body anywhere.

     Two endings, and the caller has to tell them apart. With email confirmation
     ON — which it is — Supabase returns a user and NO session, because the
     account is not usable until the link is clicked. That is a success, not a
     failure, and it is reported as `{confirm: true}` rather than by throwing.
     With confirmation off, a session comes straight back and this signs in.

     A repeat sign-up on an existing address does NOT error: Supabase answers
     with a user carrying an empty `identities` array, deliberately, so the form
     cannot be used to enumerate who has an account. We honour that — the caller
     says "check your email" either way — because doing otherwise would rebuild
     the leak by hand. */
  async signUp(name, email, password) {
    const client = await supabaseClient();
    const { data, error } = await client.auth.signUp({
      email,
      password,
      options: {
        data: { name },
        // Stated rather than inherited: without it the confirmation link lands
        // on whatever the dashboard's Site URL happens to be, which is a
        // different setting in a different place that nothing here can see.
        // Same trailing slash, and the same allowlist requirement.
        emailRedirectTo: API + "/app/",
      },
    });
    if (error) throw new ApiError("sign_up_failed", error.message);
    const token = data && data.session && data.session.access_token;
    if (!token) return { confirm: true };
    return { confirm: false, me: await exchange(token) };
  },

  /* Google is only another way to GET a Supabase token (12.1) — the cookie it
     ends at is the same cookie, minted by the same endpoint. `redirectTo` is
     this exact origin, which is what keeps contracts' same-origin `/app` rule
     true through the round trip: the cookie is SameSite=Lax and a cross-site
     landing would arrive without it. This navigates the tab away; nothing after
     it runs. */
  async signInWithGoogle() {
    const client = await supabaseClient();
    const { error } = await client.auth.signInWithOAuth({
      provider: "google",
      // TRAILING SLASH, deliberately. `/app` is a StaticFiles mount, so the
      // server answers a slashless request with a 307 to `/app/`. Landing on
      // the final URL means the return leg never depends on what survives a
      // redirect. NOTE: Supabase matches this against its Redirect Allowlist,
      // so the allowlist must name `<origin>/app/` — an entry for `/app` alone
      // will refuse this and the error comes back on the fragment.
      options: { redirectTo: API + "/app/" },
    });
    if (error) throw new ApiError("sign_in_failed", error.message);
  },

  /* The return leg. Implicit flow puts the token on the fragment, so this reads
     it, trades it for the cookie, and STRIPS the fragment before anything can
     copy the URL out of the address bar with a live access token in it.

     Returns null when there is nothing to return from, which is the ordinary
     case on every load, so the caller can always call it. An `error` in the
     fragment is Google or Supabase refusing — surfaced, and the fragment
     cleared, or a reload would replay the same failure forever. */
  async returnFromOAuth() {
    const hash = location.hash || "";
    /* SAY SO EITHER WAY. This ran silently and returned null on a load that
       should have carried a token, and a silent no-op is indistinguishable from
       code that never ran at all — which is exactly how the first Google round
       trip failed: the console showed one 401 from `me()` and nothing else, and
       the absence proved nothing. `[auth]` so it is greppable in a busy log. */
    if (!hash.includes("access_token=") && !hash.includes("error=")) {
      console.log("[auth] no fragment on", location.pathname, "— nothing to exchange");
      return null;
    }
    const params = new URLSearchParams(hash.slice(1));
    // Cleared BEFORE the exchange, so a token cannot be copied out of the
    // address bar while the request is in flight.
    history.replaceState(null, "", location.pathname + location.search);
    const failed = params.get("error_description") || params.get("error");
    if (failed) {
      console.log("[auth] provider refused:", failed);
      throw new ApiError("sign_in_failed", failed);
    }
    const token = params.get("access_token");
    if (!token) {
      // A fragment that carried neither a token nor an error is a shape nobody
      // predicted; naming its keys is what makes it diagnosable.
      console.log("[auth] fragment with no access_token, keys:", [...params.keys()].join(","));
      return null;
    }
    console.log("[auth] fragment found on", location.pathname, "— exchanging for the cookie");
    return exchange(token);
  },

  async signOut() {
    await request("DELETE", "/auth/session");
    if (_supabase) await _supabase.auth.signOut().catch(() => {});
  },

  /* --- what is here ----------------------------------------------------- */

  projects: () => request("GET", "/projects"),
  /* Deliberately, rather than as a side effect of starting a session.

     `folders` is what it LINKS — store folders that already exist, by name. A
     project owns none of them, so linking costs nothing and unlinking would
     take nothing away. Picking none makes a folder named after the project,
     which then appears in the Files tab like any other. */
  createProject: (title, folders) =>
    request("POST", "/projects", folders && folders.length ? { title, folders } : { title }),
  /* Link one more folder to a project. The pane shows it at once; the AGENT
     sees it from the next session, because claims are fixed per session. */
  linkFolder: (projectId, folder) => request("POST", `/projects/${projectId}/folders`, { folder }),
  renameProject: (projectId, title) => request("PATCH", `/projects/${projectId}`, { title }),
  sessions: (status) => request("GET", status ? `/sessions?status=${encodeURIComponent(status)}` : "/sessions"),
  projectSessions: (projectId) => request("GET", `/projects/${projectId}/sessions`),
  async session(sessionId) {
    const body = await request("GET", `/sessions/${sessionId}`);
    return { ...body, recent_events: (body.recent_events || []).map(asEvent) };
  },
  /* The whole store: one flat namespace per user, folder-first paths. What the
     Files tab draws, and where its headers come from. */
  storeFiles: () => request("GET", "/files"),
  /* The store's folders with their file counts — the modal's checklist and the
     `+ link` picker. Derived from the paths, never a table. */
  folders: () => request("GET", "/folders"),
  /* The same rows, narrowed to what one project LINKS: the working-files pane.
     Same paths, so clicking one lands on it in the Files tab. */
  files: (projectId) => request("GET", `/projects/${projectId}/files`),
  file: (fileId) => request("GET", `/files/${fileId}`),
  /* One query at three scopes: nothing is the Command Center, a project is its
     list, a session is one window. */
  attention: (scope) => {
    const query = !scope
      ? ""
      : scope.session_id
        ? `?session_id=${encodeURIComponent(scope.session_id)}`
        : `?project_id=${encodeURIComponent(scope.project_id)}`;
    return request("GET", "/attention" + query);
  },

  /* Uploading a file and saving an edit are the SAME call: `put_file` upserts
     on (project_id, path), so writing a path that already exists replaces it,
     and write-through puts it in any box already holding the project.

     A folder is a path prefix, not a row: the store keeps flat paths and the
     tree is derived from them, so uploading INTO a directory is uploading a
     file whose path carries it. Absent, the name alone lands it at the root. */
  upload: (file, dir) =>
    _postFile(file, dir ? `${dir}/${file.name}` : file.name, `Could not upload ${file.name}.`),
  saveFile: (path, text) =>
    _postFile(new Blob([text], { type: "text/plain" }), path, `Could not save ${path}.`),

  /* A folder is durable the moment it is named: the server writes a zero-byte
     sentinel inside it, because a folder is a path segment and not a row.
     Nothing about it lives only in this tab. */
  newFolder: (path) => request("POST", "/folders", { path }),

  /* Rename or reparent a file or a directory inside the store. Blobs never move
     — they are content-addressed — so this is a row edit the server also pushes
     into any live sandbox, which is what stops a running turn from undoing it.
     Moving a top-level FOLDER is refused: that is its own card. */
  moveFile: (from, to) => request("POST", "/files/move", { from, to }),

  /* Rename anything in the store — a file, a directory, or a top-level folder.
     `name` is a NAME: a `/` in it is refused rather than quietly making this a
     move. Renaming a top-level folder carries the projects that link it and the
     claims that mount it with the paths, and is refused while a running session
     has it mounted (`409 folder_busy`). */
  renameFile: (path, name) => request("POST", "/files/rename", { path, name }),

  /* Delete a file or a whole subtree. The rows go and the BLOBS do not — they
     are content-addressed and never collected — so the `batch` this returns
     takes it back exactly, the same content under the same id. A delete that
     empties a folder takes the folder and the project links that named it, and
     `folders` says which stopped existing. */
  deleteFile: (path) => request("DELETE", "/files", { path }),
  undoDelete: (batch) => request("POST", "/files/undo", { batch }),

  /* --- connections ------------------------------------------------------ */

  /* One row per connector, each carrying what the next click does: `scopes`,
     what a connect is about to grant, and `shares_with`, the sibling services a
     disconnect takes with it. A Composio connected account is per toolkit, so
     this is normally empty.
     `setup_url` is a live consent link, so the panel can open the popup inside
     the click rather than after an await, which the browser would block. */
  connections: () => request("GET", "/connections"),

  /* Mints a fresh consent link when the row's has gone stale. Calls no tool and
     connects nothing: the popup is what connects. */
  connect: (server) => request("POST", `/connections/${encodeURIComponent(server)}/connect`),

  /* Revokes at Composio and answers with what actually went — which may be more than
     one service whenever they share a sign-in. */
  disconnect: (server) => request("DELETE", `/connections/${encodeURIComponent(server)}`),

  /* --- what one session may reach --------------------------------------- */

  /* The meter and the per-server rows behind the composer's tools chip. The
     write returns the whole document, so the chip re-renders from what the
     server now holds rather than from what the click assumed. */
  sessionTools: (sessionId) => request("GET", `/sessions/${sessionId}/tools`),
  setSessionTool: (sessionId, server, enabled) =>
    request("PUT", `/sessions/${sessionId}/tools/${encodeURIComponent(server)}`, { enabled }),

  /* --- the session's live disk ------------------------------------------ */

  /* The sandbox filesystem while the session is awake. Nothing here boots a
     box: a parked or finished session 404s, which is the honest answer. */
  sandboxDir: (sessionId, path) =>
    request("GET", `/sessions/${sessionId}/fs${path ? `?path=${encodeURIComponent(path)}` : ""}`),
  sandboxFile: (sessionId, path) =>
    request("GET", `/sessions/${sessionId}/fs/file?path=${encodeURIComponent(path)}`),

  /* --- what a human may do ---------------------------------------------- */

  /* Every one of these is a suggestion or a decision. None of them executes a
     tool: the human steers the session, and the session acts. */
  start: (goal, projectId) =>
    request("POST", "/sessions", projectId ? { goal, project_id: projectId } : { goal }),
  send: (sessionId, text) => request("POST", `/sessions/${sessionId}/messages`, { text }),
  answer: (approvalId, answer) => request("POST", `/approvals/${approvalId}/respond`, { answer }),
  approve: (sessionId) => request("POST", `/sessions/${sessionId}/approve`),
  /* Stop holds a run; cancel ends it. The same teardown on the server, landing
     differently: stop leaves the session idle with its mode kept, so the plan
     still stands, and cancel is terminal and spends it. */
  stop: (sessionId) => request("POST", `/sessions/${sessionId}/stop`),
  /* Pick a stopped run back up with nothing added. The mode was kept, so an
     idle unattended session resumes UNATTENDED from its plan. Saying something
     instead is `send` — the words land in the fold and the run reads them. */
  resume: (sessionId) => request("POST", `/sessions/${sessionId}/resume`),
  cancel: (sessionId) => request("POST", `/sessions/${sessionId}/cancel`),

  /* --- the stream ------------------------------------------------------- */

  /* One EventSource for the ACCOUNT, opened once at sign-in. A frame carries no
     row: it means "read /attention again". Returns its own unsubscribe, which
     is what a `useEffect` cleanup wants. */
  watchAttention(onChange) {
    const source = new EventSource(`${API}/attention/stream`, { withCredentials: true });
    source.addEventListener("attention", () => onChange());
    /* No error handler beyond the default: there is no state to repair and no
       cursor to resume from, so a drop costs one stale list until the browser
       reconnects on its own and the server's opening frame refetches. */
    return () => source.close();
  },


  /* One EventSource per open session. No polling anywhere: the snapshot gives
     the tail of the log and its last seq, and the stream carries everything
     after it. On a drop the browser reconnects on its own and sends
     Last-Event-ID; `last_event_id` here is only for the first connect, which
     has no header to send. */
  stream(sessionId, afterSeq, onEvent, onError) {
    const url = `${API}/sessions/${sessionId}/events?last_event_id=${afterSeq || 0}`;
    const source = new EventSource(url, { withCredentials: true });

    // Every event kind arrives as its own SSE `event:` name, so one generic
    // listener is not enough — but the payload shape is identical, so one
    // handler is.
    const handle = (e) => {
      let payload = null;
      try {
        payload = JSON.parse(e.data);
      } catch (err) {
        return;
      }
      onEvent(asEvent(payload));
    };
    for (const kind of EVENT_KINDS) source.addEventListener(kind, handle);
    source.addEventListener("error", (e) => {
      // A stream that failed on the server sends a final `error` frame with a
      // body; a dropped connection sends an event with none, and the browser
      // is already reconnecting.
      if (e.data && onError) {
        try {
          onError(JSON.parse(e.data));
        } catch (err) {
          onError({ code: "stream_failed", message: "The stream failed." });
        }
      }
    });
    return source;
  },
};

/* The event vocabulary, from contracts.md. A kind absent from this list is not
   rendered, so a new one is added here and in the renderer together. */
const EVENT_KINDS = [
  "user",
  "content",
  "reasoning",
  "tool_call",
  "tool_result",
  "status",
  "todo",
  "budget",
  "lifecycle",
  "view_transform",
  "done",
];

/* --- small shared helpers ------------------------------------------------ */

function relTime(iso) {
  if (!iso) return "";
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return "";
  const s = Math.max(1, Math.floor((Date.now() - t) / 1000));
  if (s < 60) return s + "s ago";
  if (s < 3600) return Math.floor(s / 60) + "m ago";
  if (s < 86400) return Math.floor(s / 3600) + "h ago";
  return Math.floor(s / 86400) + "d ago";
}

/* Why a run stopped, in words. The vocabulary is `done.reason` and the pill is
   read by people, so each reason says what actually happened: 11.8.5 split
   `model_error` into three, and "failed: model_error" for a Postgres blip was
   exactly the confusion that split fixed. */
const REASON_LABEL = {
  stalled_progress: "stalled — no progress",
  model_error: "the model errored",
  internal_error: "we errored",
  max_hops: "out of hops",
  wall_clock: "out of time",
  context_overflow: "context full",
  interrupted: "interrupted",
};

function statusLabel(status, terminalReason) {
  if (status === "awaiting_approval") return "waiting on you";
  if (status === "failed" && terminalReason) {
    return "failed: " + (REASON_LABEL[terminalReason] || terminalReason.replace(/_/g, " "));
  }
  return String(status || "").replace("_", " ");
}
