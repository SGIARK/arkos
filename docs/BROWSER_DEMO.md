# Persistent browser: acceptance test for the November 30 demo

Each user has one real browser that stays open between agent turns. The agent drives it with the `browser_*` tools, and a person can use it from the app's browser pane while the agent is paused.

## Steps

1. In one session, ask arkos to open a page with a text field and type something into it.
2. When that turn ends, ask arkos what the field says. It should still say what it typed, because the page stayed open.
3. In the browser pane, click the field, replace the text, and press type.
4. Ask arkos what the field says now. Its next snapshot should show your text, not its own.

While arkos is working, the pane says "arkos is using the browser" and won't take input. Each user has one browser, and every tool acts on it directly, with no separate browser agent in between.

## Result

On 2026-11-20 the same steps passed as an automated test (`tests/test_browser_integration.py`) against a local Browserless container. The page kept the agent's text between turns, the agent's next snapshot showed the person's change, and the browser was the same one throughout.

On 2026-11-20 the same plumbing passed on ark itself, driving the tools directly without the model.

On 2026-11-21 the first two steps ran in the app on ark, with Qwen3-8B. On the first turn arkos called `browser_open` and `browser_type`, and its snapshot showed the field holding "hello". On the second turn it answered "The field contains hello" without taking a new snapshot, so that answer came from its memory of the first turn rather than from looking at the page. Steps 3 and 4, where a person changes the page, have not been run in the app yet. For the demo, ask it to take a browser_snapshot before answering, so the second turn shows the page really stayed open.
