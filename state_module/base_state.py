from typing import Any, Literal, Optional

from pydantic import BaseModel


class StateOutput(BaseModel):
    # content is Optional: edge case where the model returns nothing should not crash the loop
    content: Optional[str] = None
    completion_signal: Literal["complete", "incomplete", "error", "needs_input"]
    # dict[str, Any]: tool and reasoning results are heterogeneous JSON objects
    structured_data: Optional[dict[str, Any]] = None
    error_detail: Optional[str] = None
