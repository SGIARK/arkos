# Concepts

A short inventory of arkos behaviors, grown one planning cycle at a time. Each entry gives a purpose, a scenario and the source paths.

## Browser task

**Purpose:** let the agent do something on the web in a real browser that it cannot do with search alone.

**Scenario:** the model calls `browser_task` with a goal. The tool takes one of the user's browser slots and connects to the Browserless container over CDP. Then it hands the goal to a browser-use agent. Each call gets a fresh, signed-out browser, which is closed when the run ends, so nothing carries over between calls. The run is asked to stop at `browser.wall_clock_s` and cut off at `browser.hard_timeout_s`. The slot is freed on every exit, so another session of the same user can browse at once.

**Sources:** `tool_module/browser/tool.py`, `tool_module/browser/stream.py`, `config_module/config.yaml` (`browser:`).

## Resource lease

**Purpose:** stop two sessions from using one shared resource at the same time, without locking it forever when a process dies.

**Scenario:** a session asks for a resource through `ToolContext.lease`. The runner inserts or renews one `resource_leases` row with an expiry, and waits with a "waiting for…" status while another session holds it. It gives up after `leases.wait_timeout_s`. A write claim holds its folder for the session's whole run. The browser is a pool: the session takes the first free key out of `browser:<user>:0` to `:N-1`, where N is `browser.max_concurrent_per_user`. It gives the key back through `ToolContext.unlease` when the call ends. Every lease a session still holds is released when the session ends or parks.

**Sources:** `harness_module/leases.py`, `harness_module/runner.py` (`_lease`, `_unlease`, `_claim_browser`, `_release_leases`), `tool_module/envelope.py` (`ToolContext`).
