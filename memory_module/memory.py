# memory.py
import os
import uuid
import sys
from psycopg2 import pool
from typing import Dict
from mem0 import Memory as Mem0Memory
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model_module.ArkModelNew import (
    Message,
    UserMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)


from typing import Type

ROLE_TO_CLASS: Dict[str, Type[Message]] = {
    "system": SystemMessage,
    "user": UserMessage,
    "assistant": AIMessage,
    "tool": ToolMessage,
}


CLASS_TO_ROLE: Dict[Type[Message], str] = {
    SystemMessage: "system",
    UserMessage: "user",
    AIMessage: "assistant",
    ToolMessage: "tool",
}


# Global Mem0 config ---------------------
# Load .env file
load_dotenv()
os.environ["OPENAI_API_KEY"] = "sk"

config = {
    "vector_store": {
        "provider": "supabase",
        "config": {
            "connection_string": os.environ["DB_URL"],
            "collection_name": "memories",
            "index_method": "hnsw",
            "index_measure": "cosine_distance",
        },
    },
    "llm": {
        "provider": "vllm",
        "config": {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "vllm_base_url": "http://localhost:30000/v1",
        },
    },
    "embedder": {
        "provider": "huggingface",
        "config": {"huggingface_base_url": "http://localhost:4444/v1"},
    },
}

# Global connection pool (initialized lazily)
_connection_pool = None
_pool_lock = threading.Lock()

# Background executor for non-blocking mem0 operations
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mem0_bg")


def _get_pool(db_url: str):
    """Get or create the global connection pool."""
    global _connection_pool
    if _connection_pool is None:
        with _pool_lock:
            if _connection_pool is None:
                _connection_pool = pool.ThreadedConnectionPool(
                    minconn=1, maxconn=10, dsn=db_url
                )
    return _connection_pool


class Memory:
    """
    Connects agent to supabase backend for long
    and short term memories

    """

    def __init__(
        self, user_id: str, session_id: str, db_url: str, use_long_term: bool = True
    ):
        self.user_id = user_id
        self.db_url = db_url
        self.use_long_term = use_long_term  # Toggle for long-term memory

        # Initialize connection pool
        self._pool = _get_pool(db_url)

        # initialize mem0 (lazy - only if needed)
        self._mem0 = None
        if self.use_long_term:
            self._mem0 = Mem0Memory.from_config(config)

        # session handling
        self.session_id = session_id if session_id is not None else str(uuid.uuid4())

    def start_new_session(self):
        """Start a new chat session."""
        self.session_id = str(uuid.uuid4())
        return self.session_id

    def serialize(self, message: Message) -> str:
        """
        Convert a Message subclass into the string stored in Postgres.
        Store role separately in the role column.
        """
        return message.model_dump_json()

    def deserialize(self, message: str, role: str) -> Message:
        """
        Convert the stored Postgres string back into the correct Message subclass.
        Requires the role column value.
        """
        cls = ROLE_TO_CLASS.get(role)
        if cls is None:
            raise ValueError(f"Unknown role: {role}")
        return cls.model_validate_json(message)

    def _add_to_mem0_background(self, content: str, metadata: dict):
        """Background task to add to mem0 (non-blocking)."""
        try:
            if self._mem0:
                self._mem0.add(
                    messages=content, metadata=metadata, user_id=self.user_id
                )
        except Exception as e:
            print(f"[mem0 background] Error: {e}")

    def add_memory(self, message) -> bool:
        """Add a single turn to Postgres (fast) + Mem0 in background."""
        try:
            role = CLASS_TO_ROLE[type(message)]

            # Store in Postgres immediately (fast, using pool)
            conn = self._pool.getconn()
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO conversation_context (user_id, session_id, role, message)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (self.user_id, self.session_id, role, self.serialize(message)),
                )
                conn.commit()
                cur.close()
            finally:
                self._pool.putconn(conn)

            # Store in mem0 in background — only user/tool messages, not AI responses
            # AI responses may be hallucinated; storing them poisons future routing
            if self.use_long_term and self._mem0 and not isinstance(message, AIMessage):
                metadata = {
                    "user_id": self.user_id,
                    "session_id": self.session_id,
                    "role": role,
                }
                _executor.submit(
                    self._add_to_mem0_background, message.content, metadata
                )

            return True

        except Exception:
            import traceback

            traceback.print_exc()
            return False

    def retrieve_long_memory(
        self, context: list = [], mem0_limit: int = 10
    ) -> SystemMessage:
        """Retrieve relevant long term memories for the current user."""
        # Skip if long-term memory is disabled
        if not self.use_long_term or not self._mem0:
            return SystemMessage(content="")

        try:
            # Build query from recent context only (faster)
            query = " ".join(m.content for m in context[-2:] if hasattr(m, "content"))

            if not query.strip():
                return SystemMessage(content="")

            results = self._mem0.search(
                query=query,
                user_id=self.user_id,
                limit=mem0_limit,  # Reduced from 50 to 10
            )

            memory_entries = [
                f"{r.get('role', 'user')}: {r['memory']}"
                for r in results.get("results", [])
            ]

            if not memory_entries:
                return SystemMessage(content="")

            memory_string = "retrieved memories:\n" + "\n".join(memory_entries)
            return SystemMessage(content=memory_string)

        except Exception as e:
            print(f"[retrieve_long_memory] Error: {e}")
            return SystemMessage(content="")

    def retrieve_short_memory(self, turns):
        """Retrieve relevant short term memories for the current user"""
        try:
            conn = self._pool.getconn()
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT role, message
                    FROM (
                        SELECT id, role, message
                        FROM conversation_context
                        WHERE user_id = %s
                        ORDER BY id DESC
                        LIMIT %s
                    ) sub
                    ORDER BY id ASC
                    """,
                    (self.user_id, turns),
                )
                rows = cur.fetchall()
                cur.close()
            finally:
                self._pool.putconn(conn)

            return [self.deserialize(message=msg, role=role) for role, msg in rows]

        except Exception as e:
            print(f"[retrieve_short_memory] Error: {e}")
            return []


if __name__ == "__main__":
    test_instance = Memory(
        user_id="alice_test", session_id="session_test", db_url=os.environ["DB_URL"]
    )

    print(
        test_instance.add_memory(
            SystemMessage(content="My favorite color is blue and I live in New York")
        )
    )

    context = test_instance.retrieve_short_memory(turns=2)
    print(context)

    print(test_instance.retrieve_long_memory(context))
