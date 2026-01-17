# memory.py
import os
import sys
import uuid
from typing import Any

import psycopg2
from dotenv import load_dotenv
from mem0 import Memory as Mem0Memory

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from model_module.ArkModelNew import (
    AIMessage,
    Message,
    SystemMessage,
    ToolMessage,
    UserMessage,
)

ROLE_TO_CLASS: dict[str, type[Message]] = {
    "system": SystemMessage,
    "user": UserMessage,
    "assistant": AIMessage,
    "tool": ToolMessage,
}


CLASS_TO_ROLE: dict[type[Message], str] = {
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


class Memory:
    """
    Connects agent to supabase backend for long
    and short term memories

    """

    def __init__(self, user_id: str, session_id: str | None, db_url: str):
        self.user_id = user_id
        self.db_url = db_url

        # initialize mem0
        self.mem0 = Mem0Memory.from_config(config)

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

    def add_memory(self, message) -> bool:
        """Add a single turn to Mem0 + Postgres."""
        try:
            role = CLASS_TO_ROLE[type(message)]

            metadata = {
                "user_id": self.user_id,
                "session_id": self.session_id,
                "role": role,
            }

            # store in mem0
            self.mem0.add(messages=message.content, metadata=metadata, user_id=self.user_id)

            conn = psycopg2.connect(self.db_url)
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
            conn.close()

            return True

        except Exception as e:
            import traceback

            traceback.print_exc()
            raise
            print(e)
            return False

    def retrieve_long_memory(
        self, context: list | None = None, mem0_limit: int = 50
    ) -> dict[str, Any]:
        """Retrieve relevant long term memories for the current user."""
        if context is None:
            context = []
        try:
            # Mem0 vector retrieval

            query = ""

            for message in context:
                query += f" \n {message.content}"

            results = self.mem0.search(
                query=query,
                user_id=self.user_id,
                limit=mem0_limit,
            )

            memory_entries = [
                f"{r.get('role', 'user')}: {r['memory']}" for r in results.get("results", [])
            ]

            memory_string = "retrieved memories:\n" + "\n".join(memory_entries)

            return SystemMessage(content=memory_string)

        except Exception:
            import traceback

            traceback.print_exc()
            raise

            return "retrieval_failed"

    def retrieve_short_memory(self, turns):
        """Retrieve relevant short term memories for the current user"""
        try:
            conn = psycopg2.connect(self.db_url)
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
            conn.close()

            return [self.deserialize(message=msg, role=role) for role, msg in rows]

        except Exception as e:
            print(e)
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
