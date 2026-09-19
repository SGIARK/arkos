-- The home session: the chat you land in, standing.
--
-- One column, set once on first login, and nothing else in the system reads it
-- as a special case. The row it points at is an ordinary attended session with
-- an ordinary project — it may sit idle forever, wake when someone types, and
-- move through the lifecycle exactly like any other. What makes it "home" is
-- that the app opens it by default, which is a routing fact, not a state one.
--
-- ON DELETE SET NULL rather than CASCADE: deleting a session must never delete
-- the person whose chat it was. A null here just means the next login makes a
-- new one.

ALTER TABLE users
    ADD COLUMN IF NOT EXISTS home_session_id UUID REFERENCES sessions(id) ON DELETE SET NULL;
