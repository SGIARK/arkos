/* The public pages use the app's theme: light unless the visitor chose dark in arkos
   (saved as `arkos-theme`). Loaded in <head>, before the stylesheet applies, so dark
   pages never flash light. Without storage or JavaScript the page stays light, the app's default. */
(function () {
  "use strict";
  try {
    if (window.localStorage.getItem("arkos-theme") === "dark") {
      document.documentElement.setAttribute("data-theme", "dark");
    }
  } catch (e) {
    /* storage blocked: light */
  }
})();
