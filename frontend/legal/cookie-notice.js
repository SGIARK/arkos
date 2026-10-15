/* The one-time cookie notice for the public pages (landing, legal, help).
   arkos stores only what it needs, so this informs; it asks nothing.
   Plain script, no framework; brings its own styles so it looks the same everywhere. */
(function () {
  "use strict";

  var KEY = "arkos-cookie-ack";

  function seen() {
    try {
      return window.localStorage.getItem(KEY) === "1";
    } catch (e) {
      return false; // storage blocked: show it every time, never throw
    }
  }

  function remember() {
    try {
      window.localStorage.setItem(KEY, "1");
    } catch (e) {
      /* storage blocked: nothing to keep */
    }
  }

  function show() {
    if (seen() || document.getElementById("cookie-notice-bar")) return;

    var style = document.createElement("style");
    style.textContent =
      "#cookie-notice-bar{position:fixed;left:16px;right:16px;bottom:16px;z-index:50;" +
      "max-width:640px;margin:0 auto;display:flex;flex-wrap:wrap;align-items:center;gap:8px 16px;" +
      "padding:12px 14px;border:1px solid var(--line,#e6e3da);border-radius:7px;" +
      "background:var(--surface,#fdfcf9);color:var(--ink-soft,#54524b);" +
      "font:12px/1.5 var(--mono,ui-monospace,monospace);box-shadow:0 12px 32px -20px rgba(0,0,0,.35)}" +
      "#cookie-notice-bar p{margin:0;flex:1 1 260px}" +
      "#cookie-notice-bar .acts{display:flex;align-items:center;gap:14px;margin-left:auto}" +
      "#cookie-notice-bar a{color:var(--accent,#3f7a57)}" +
      "#cookie-notice-bar button{font:inherit;cursor:pointer;padding:5px 14px;border-radius:6px;" +
      "border:1px solid var(--ink,#1c1b18);background:var(--ink,#1c1b18);color:var(--paper,#f7f6f1)}";
    document.head.appendChild(style);

    var bar = document.createElement("div");
    bar.id = "cookie-notice-bar";
    bar.setAttribute("role", "region");
    bar.setAttribute("aria-label", "Cookie notice");

    var text = document.createElement("p");
    text.textContent =
      "arkos uses one cookie to keep you signed in and remembers a few settings in your " +
      "browser, like your theme. No ads, no analytics.";

    var acts = document.createElement("div");
    acts.className = "acts";

    var link = document.createElement("a");
    link.href = "/app/legal/cookies.html";
    link.textContent = "Cookie policy";

    var ok = document.createElement("button");
    ok.type = "button";
    ok.textContent = "OK";
    ok.addEventListener("click", function () {
      remember();
      bar.remove();
    });

    acts.appendChild(link);
    acts.appendChild(ok);
    bar.appendChild(text);
    bar.appendChild(acts);
    document.body.appendChild(bar);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", show);
  } else {
    show();
  }
})();
