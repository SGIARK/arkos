/* =========================================================
   The data layer: sign-in, the endpoint table, and the stream.

   Everything here talks to the API on our own origin: the session cookie is
   httpOnly and SameSite=Lax, so a cross-origin backend would leave every
   EventSource unauthenticated. The page never holds a token.
   ========================================================= */

/* Every file here is a <script type="text/babel">, so they share one global
   lexical scope: the hooks are destructured once, in the first script the page
   loads, and a second `const` of these names anywhere would throw. */
const { useState, useEffect, useCallback, useRef } = React;

const API = location.origin;

/* Must run at module scope in the first script `index.html` loads, before the
   router writes the hash: the OAuth token and the route share location.hash,
   and whoever writes second wins. Stripping also keeps a live access token out
   of the address bar. */
const _landing = (function captureAuthLanding() {
  const hash = location.hash || "";
  const query = new URLSearchParams(location.search || "");
  /* Supabase sends links in two shapes depending on the template: the default
     `{{ .ConfirmationURL }}` redirects and lands with a fragment, while a
     `{{ .TokenHash }}` template lands with `?token_hash=&type=` and needs
     `verifyOtp`. Both are handled — the templates are the owner's and this code
     cannot see which shape they use. */
  if (query.get("token_hash")) {
    const type = query.get("type") || "";
    history.replaceState(null, "", location.pathname);
    return { recovery: type === "recovery", otp: query.get("token_hash"), otpType: type };
  }
  if (!hash.includes("access_token=") && !hash.includes("error=")) return {};
  history.replaceState(null, "", location.pathname + location.search);
  const p = new URLSearchParams(hash.slice(1));
  return {
    // A recovery token opens the new-password screen; it never buys a cookie.
    recovery: p.get("type") === "recovery",
    token: p.get("access_token") || "",
    refresh: p.get("refresh_token") || "",
    failed: p.get("error_description") || p.get("error") || "",
    // The one error worth its own copy: a reset link that has been used or has
    // aged out, which is otherwise painted as a sign-in failure.
    expired: p.get("error_code") === "otp_expired",
  };
})();

/* The `amr` claim and nothing else. A JWT payload is base64url and readable by
   anyone holding the token, so this discloses nothing the browser did not
   already have — but it is deliberately narrow: the rest of the payload is not
   ours to print into a console log. */
function _amrOf(token) {
  try {
    const body = String(token || "").split(".")[1];
    if (!body) return "(no payload)";
    const json = JSON.parse(atob(body.replace(/-/g, "+").replace(/_/g, "/")));
    return "amr" in json ? JSON.stringify(json.amr) : "(no amr claim)";
  } catch (e) {
    return "(unreadable)";
  }
}

/* Errors that mean NO MAIL WENT OUT and why must not be said aloud: the
   address is already registered, or we are inside the per-user window. Both
   answer the question "does this address have an account?", so they end on the
   same screen as a successful send instead of under the form. Anything else —
   a malformed address, a weak password — is about what was typed and is shown. */
function _isUndisclosableSendOutcome(error) {
  if (!error) return false;
  if (error.status === 429) return true;
  const code = String(error.code || "");
  if (["over_email_send_rate_limit", "email_exists", "user_already_exists", "over_request_rate_limit"].includes(code)) {
    return true;
  }
  const said = String(error.message || "").toLowerCase();
  return /after \d+ seconds|rate limit|already registered|already exists/.test(said);
}

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
      // The cookie is the session; a second copy in localStorage would be a
      // credential we deliberately do not keep.
      persistSession: false,
      autoRefreshToken: false,
      /* Implicit, not supabase-js's default PKCE: PKCE stores a code verifier
         across the redirect, and `persistSession: false` has nowhere to keep
         one, so the exchange would fail every time. */
      flowType: "implicit",
      /* We read the fragment ourselves in `returnFromOAuth`: the cookie is the
         session, and the token is only good for the one exchange that mints it. */
      detectSessionInUrl: false,
    },
  });
  return _supabase;
}

/* Trade a Supabase access token for our cookie. `POST /auth/session` is the only
   endpoint that reads a bearer token and the only thing that mints a cookie, so
   password sign-in, sign-up and Google all end here. */
async function exchange(token) {
  const response = await fetch(API + "/auth/session", {
    method: "POST",
    credentials: "same-origin",
    headers: { Authorization: "Bearer " + token },
  });
  if (!response.ok) {
    const shape = await response.json().catch(() => ({}));
    /* ALWAYS `sign_in_failed`, never the server's own code. Every rejection
       here is `unauthenticated`, which is also what a signed-out `me()` says —
       and the boot must show one and not the other, or a token the server
       refuses leaves a blank form and no explanation. */
    throw new ApiError("sign_in_failed", shape.message || "The server rejected the sign-in.");
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

/* The wire carries `{seq, ts, kind, version, payload:{...}}`; flattened once
   here, at the boundary, so no renderer has to know the envelope exists. */
function asEvent(raw) {
  const { payload, ...rest } = raw || {};
  return { ...rest, ...(payload || {}) };
}

/* Multipart goes around `request`: the browser sets its own boundary, and a
   Content-Type we invented would break it. `path` is REQUIRED and carries the
   folder — the store is one flat namespace per user with no root. */
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

  /* The token is used once, here, and never stored: from the next request on,
     identity is the httpOnly cookie, which script cannot read. */
  async signIn(email, password) {
    const client = await supabaseClient();
    const { data, error } = await client.auth.signInWithPassword({ email, password });
    if (error) throw new ApiError("sign_in_failed", error.message);
    const token = data && data.session && data.session.access_token;
    if (!token) throw new ApiError("sign_in_failed", "Supabase returned no session.");
    return exchange(token);
  },

  /* The name rides as user_metadata so it arrives on the signed token. Supabase
     returns no session when email confirmation is on (`{confirm: true}`, a
     success), and no error for an existing address (empty `identities`). */
  async signUp(name, email, password) {
    const client = await supabaseClient();
    const { data, error } = await client.auth.signUp({
      email,
      password,
      options: {
        data: { name },
        // Stated rather than inherited: without it the confirmation link lands on
        // the dashboard's Site URL. Same trailing slash and allowlist requirement.
        emailRedirectTo: API + "/app/",
      },
    });
    // Nothing was sent, and saying why would answer "does this address exist?".
    if (error && _isUndisclosableSendOutcome(error)) return { confirm: true };
    if (error) throw new ApiError("sign_up_failed", error.message);
    const token = data && data.session && data.session.access_token;
    /* A session here means the dashboard's "Confirm email" toggle is OFF, so
       nothing was sent and the address was never proved. Signing in is still
       what the flow asks for, but it is a misconfiguration and says so. */
    if (token) {
      console.log("[auth] signUp returned a session — email confirmation is OFF in Supabase");
      return { confirm: false, me: await exchange(token) };
    }
    return { confirm: true };
  },

  /* `redirectTo` must be this exact origin: the cookie is SameSite=Lax and a
     cross-site landing would arrive without it. This navigates the tab away, so
     nothing after it runs. */
  async signInWithGoogle() {
    const client = await supabaseClient();
    const { error } = await client.auth.signInWithOAuth({
      provider: "google",
      // TRAILING SLASH: `/app` is a StaticFiles mount that 307s to `/app/`, and
      // Supabase matches this against its Redirect Allowlist, so the allowlist
      // must name `<origin>/app/` — an entry for `/app` alone refuses this.
      options: { redirectTo: API + "/app/" },
    });
    if (error) throw new ApiError("sign_in_failed", error.message);
  },

  /* Returns null when there is nothing to return from, the ordinary case on
     every load, so the caller can always call it. */
  async returnFromOAuth() {
    /* Reads the stash, never `location.hash`: by the time this runs the router
       has already written over the real fragment. */
    if (_landing.failed) {
      console.log("[auth] provider refused:", _landing.failed);
      throw new ApiError("sign_in_failed", _landing.failed);
    }
    if (_landing.recovery) {
      console.log("[auth] recovery link — opening the new-password screen");
      // ONE claim, printed once, so a live reset run confirms the vocabulary the
      // server's denylist is written against. `amr` names how the token was
      // obtained; nothing else from the payload is read or logged.
      console.log("[auth] recovery token amr:", _amrOf(_landing.token));
      return null;
    }
    if (_landing.otp) {
      console.log("[auth] token_hash link (", _landing.otpType, ") — verifying");
      const client = await supabaseClient();
      const { data, error } = await client.auth.verifyOtp({
        token_hash: _landing.otp,
        type: _landing.otpType || "email",
      });
      if (error) throw new ApiError("sign_in_failed", "That link has expired. Ask for another.");
      const token = data && data.session && data.session.access_token;
      if (!token) return null;
      return exchange(token);
    }
    if (!_landing.token) {
      console.log("[auth] no fragment on", location.pathname, "— nothing to exchange");
      return null;
    }
    console.log("[auth] fragment found on", location.pathname, "— exchanging for the cookie");
    return exchange(_landing.token);
  },

  /* This load came from a reset link, so the app owes a new-password screen
     rather than a session. */
  recoveryPending: () => !!_landing.recovery && !!(_landing.token || _landing.otp),

  /* A reset link that was already used or has aged out. Its own copy, because
     painting it as a sign-in failure tells the person nothing about the link
     they just clicked. */
  linkExpired: () => !!_landing.expired,

  /* Send the reset mail. Resolves the same way whether or not the address has
     an account: Supabase does not say, and neither may we. */
  async sendReset(email) {
    const client = await supabaseClient();
    const { error } = await client.auth.resetPasswordForEmail(email, {
      redirectTo: API + "/app/",
    });
    if (error && !_isUndisclosableSendOutcome(error)) throw new ApiError("reset_failed", error.message);
  },

  /* Finish the reset. The recovery token is only good for this: it is put in
     as an in-memory session, spent on the password change, and the session that
     comes back is what buys the cookie — so the new password is live and the
     person is signed in in one step. */
  /* Change the password, then STOP. The recovery token is never traded for a
     cookie: the server cannot tell one from a password sign-in, so a link seen
     by a mail scanner, a shared mailbox or a forward would buy a seven-day
     session without changing anything the owner would notice. Signing in with
     the new password is the proof that it took. */
  async completeReset(password) {
    const client = await supabaseClient();
    if (_landing.otp) {
      const { error } = await client.auth.verifyOtp({ token_hash: _landing.otp, type: "recovery" });
      if (error) throw new ApiError("link_expired", "That reset link has expired. Ask for another.");
    } else {
      const { error } = await client.auth.setSession({
        access_token: _landing.token,
        refresh_token: _landing.refresh,
      });
      if (error) throw new ApiError("link_expired", "That reset link has expired. Ask for another.");
    }
    const { data: live } = await client.auth.getSession();
    const proof = (live && live.session && live.session.access_token) || _landing.token;
    const { error } = await client.auth.updateUser({ password });
    if (error) throw new ApiError("reset_failed", error.message);
    /* SIGN OUT EVERYWHERE (12.2.5). The reason someone resets a password is
       often that somebody else has their session; leaving those alive is the
       one outcome the reset was meant to prevent. Best effort — the password
       has already changed and a failure here must not read as one. */
    await fetch(API + "/auth/sessions/revoke", {
      method: "POST",
      credentials: "same-origin",
      headers: { Authorization: "Bearer " + proof },
    }).catch(() => {});
    // Drop the in-memory session so nothing is left holding a recovery token.
    await client.auth.signOut().catch(() => {});
    _landing.recovery = false;
    _landing.token = "";
    _landing.refresh = "";
    _landing.otp = "";
    console.log("[auth] password changed — sign in with it");
  },

  /* Ask for the confirmation mail again. Supabase rate-limits this per user;
     the caller owns the cooldown so the button can say why it is dim. */
  async resendSignup(email) {
    const client = await supabaseClient();
    const { error } = await client.auth.resend({
      type: "signup",
      email,
      options: { emailRedirectTo: API + "/app/" },
    });
    if (error && !_isUndisclosableSendOutcome(error)) throw new ApiError("resend_failed", error.message);
  },

  async signOut() {
    await request("DELETE", "/auth/session");
    if (_supabase) await _supabase.auth.signOut().catch(() => {});
  },

  /* --- what is here ----------------------------------------------------- */

  projects: () => request("GET", "/projects"),
  /* `folders` LINKS store folders that already exist, by name; a project owns
     none of them. Picking none makes a folder named after the project. */
  createProject: (title, folders) =>
    request("POST", "/projects", folders && folders.length ? { title, folders } : { title }),
  /* The pane shows it at once; the AGENT sees it from the next session, because
     claims are fixed per session. */
  linkFolder: (projectId, folder) => request("POST", `/projects/${projectId}/folders`, { folder }),
  renameProject: (projectId, title) => request("PATCH", `/projects/${projectId}`, { title }),
  sessions: (status) => request("GET", status ? `/sessions?status=${encodeURIComponent(status)}` : "/sessions"),
  projectSessions: (projectId) => request("GET", `/projects/${projectId}/sessions`),
  async session(sessionId) {
    const body = await request("GET", `/sessions/${sessionId}`);
    return { ...body, recent_events: (body.recent_events || []).map(asEvent) };
  },
  /* The whole store: one flat namespace per user, folder-first paths. */
  storeFiles: () => request("GET", "/files"),
  /* The store's folders with their file counts, derived from the paths rather
     than from a table. */
  folders: () => request("GET", "/folders"),
  /* The same rows, narrowed to what one project LINKS; same paths as the Files
     tab, so clicking one lands on it there. */
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

  /* `put_file` upserts on (project_id, path), so writing an existing path
     replaces it. A folder is a path prefix, not a row: uploading into a
     directory means putting it in the path; absent, the name lands at the root. */
  upload: (file, dir) =>
    _postFile(file, dir ? `${dir}/${file.name}` : file.name, `Could not upload ${file.name}.`),
  saveFile: (path, text) =>
    _postFile(new Blob([text], { type: "text/plain" }), path, `Could not save ${path}.`),

  /* The server writes a zero-byte sentinel inside the folder, because a folder
     is a path segment and not a row. */
  newFolder: (path) => request("POST", "/folders", { path }),

  /* A row edit; blobs are content-addressed and never move. Moving a top-level
     FOLDER is refused. */
  moveFile: (from, to) => request("POST", "/files/move", { from, to }),

  /* `name` is a NAME: a `/` in it is refused rather than making this a move.
     Renaming a top-level folder carries its project links and claims, and is
     refused while a running session holds its write lease (`409 folder_busy`). */
  renameFile: (path, name) => request("POST", "/files/rename", { path, name }),

  /* The rows go and the BLOBS do not, so the returned `batch` takes it back
     exactly. A delete that empties a folder takes the folder and the project
     links that named it. */
  deleteFile: (path) => request("DELETE", "/files", { path }),
  undoDelete: (batch) => request("POST", "/files/undo", { batch }),

  /* --- connections ------------------------------------------------------ */

  /* Each row carries `scopes` (what a connect grants) and `shares_with` (what a
     disconnect takes with it — normally empty, Composio grants are per toolkit).
     `setup_url` is a live consent link, so the popup can open inside the click
     rather than after an await, which the browser would block. */
  connections: () => request("GET", "/connections"),

  /* Mints a fresh consent link when the row's has gone stale. Calls no tool and
     connects nothing: the popup is what connects. */
  connect: (server) => request("POST", `/connections/${encodeURIComponent(server)}/connect`),

  /* Revokes at Composio and answers with what actually went — which may be more
     than one service whenever they share a sign-in. */
  disconnect: (server) => request("DELETE", `/connections/${encodeURIComponent(server)}`),

  /* --- what one session may reach --------------------------------------- */

  /* The write returns the whole document, so the chip re-renders from what the
     server now holds rather than from what the click assumed. */
  sessionTools: (sessionId) => request("GET", `/sessions/${sessionId}/tools`),
  setSessionTool: (sessionId, server, enabled) =>
    request("PUT", `/sessions/${sessionId}/tools/${encodeURIComponent(server)}`, { enabled }),

  /* --- what a human may do ---------------------------------------------- */

  /* None of these executes a tool: the human steers the session, and the
     session acts. */
  start: (goal, projectId) =>
    request("POST", "/sessions", projectId ? { goal, project_id: projectId } : { goal }),
  send: (sessionId, text) => request("POST", `/sessions/${sessionId}/messages`, { text }),
  answer: (approvalId, answer) => request("POST", `/approvals/${approvalId}/respond`, { answer }),
  approve: (sessionId) => request("POST", `/sessions/${sessionId}/approve`),
  /* Same teardown, different landing: stop leaves the session idle with its mode
     kept, so the plan still stands; cancel is terminal and spends it. */
  stop: (sessionId) => request("POST", `/sessions/${sessionId}/stop`),
  /* The mode was kept, so an idle unattended session resumes UNATTENDED from its
     plan. Saying something instead is `send`. */
  resume: (sessionId) => request("POST", `/sessions/${sessionId}/resume`),
  cancel: (sessionId) => request("POST", `/sessions/${sessionId}/cancel`),

  /* --- the stream ------------------------------------------------------- */

  /* One EventSource for the ACCOUNT, opened once at sign-in. A frame carries no
     row: it means "read /attention again". Returns its own unsubscribe. */
  watchAttention(onChange) {
    const source = new EventSource(`${API}/attention/stream`, { withCredentials: true });
    source.addEventListener("attention", () => onChange());
    /* No error handler beyond the default: there is no state to repair and no
       cursor to resume from, so a drop costs one stale list. */
    return () => source.close();
  },

  /* One EventSource per open session, no polling: the snapshot gives the tail of
     the log and its last seq, and the stream carries everything after it.
     `last_event_id` is only for the first connect; a reconnect sends the
     Last-Event-ID header itself. */
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
      // body; a dropped connection sends an event with none.
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

/* Keyed by `done.reason`, and read by people, so each reason says what actually
   happened. */
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
