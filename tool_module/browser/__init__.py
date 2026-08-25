"""The browser, on a leash.

A rented specialist behind the tool boundary: `browser_task` hands a goal to
`browser_use`'s own agent and reports back through the same envelope every other
tool uses. Its progress is `status` events in the session log, its video is an
ephemeral side-channel that is never an event, and its budget is enforced by
asking it to stop rather than by killing it mid-step.
"""
