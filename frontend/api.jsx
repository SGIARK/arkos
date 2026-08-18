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
    // The cookie is the session. Persisting a second one in localStorage would
    // be a copy of a credential we deliberately do not keep.
    auth: { persistSession: false, autoRefreshToken: false },
  });
  return _supabase;
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
  },

  async signOut() {
    await request("DELETE", "/auth/session");
    if (_supabase) await _supabase.auth.signOut().catch(() => {});
  },

  /* --- what is here ----------------------------------------------------- */

  projects: () => request("GET", "/projects"),
  projectSessions: (projectId) => request("GET", `/projects/${projectId}/sessions`),
  session: (sessionId) => request("GET", `/sessions/${sessionId}`),
  files: (projectId) => request("GET", `/projects/${projectId}/files`),
  attention: (projectId) =>
    request("GET", projectId ? `/attention?project_id=${encodeURIComponent(projectId)}` : "/attention"),

  /* --- what a human may do ---------------------------------------------- */

  /* Every one of these is a suggestion or a decision. None of them executes a
     tool: the human steers the session, and the session acts. */
  start: (goal, projectId) =>
    request("POST", "/sessions", projectId ? { goal, project_id: projectId } : { goal }),
  send: (sessionId, text) => request("POST", `/sessions/${sessionId}/messages`, { text }),
  answer: (approvalId, answer) => request("POST", `/approvals/${approvalId}/respond`, { answer }),
  approve: (sessionId) => request("POST", `/sessions/${sessionId}/approve`),
  cancel: (sessionId) => request("POST", `/sessions/${sessionId}/cancel`),

  /* --- the stream ------------------------------------------------------- */

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
      onEvent(payload);
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

/* The dot. One vocabulary for a project's rollup and a session's own status,
   because they are the same statuses and the grid shows both. */
function statusTone(status) {
  if (status === "running") return "run";
  if (status === "awaiting_approval") return "wait";
  if (status === "failed") return "stop";
  if (status === "completed" || status === "cancelled") return "done";
  return "idle";
}

function statusLabel(status, terminalReason) {
  if (status === "awaiting_approval") return "waiting on you";
  if (status === "failed" && terminalReason) return "failed: " + terminalReason;
  return String(status || "").replace("_", " ");
}
