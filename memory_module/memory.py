# memory.py
import csv
import os
import json
from typing import Dict, Any

class Memory:
    def __init__(self, agent_id: str, filename: str = "memory.csv"):
        self.agent_id = agent_id
        self.filename = filename
        self.state_stack = []  # [{state, intent, tool, scratchpad}]

        if not os.path.exists(self.filename):
            with open(self.filename, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["agent_id", "state", "intent", "tool", "scratchpad"])
                writer.writeheader()

    def push_state(self, state_obj: Dict[str, Any]):
        if not isinstance(state_obj.get("state"), str):
            raise ValueError("State must be a string. Use State.<name>.name to get the state name.")

        # If scratchpad exists and is not a string, serialize it to JSON
        scratchpad = state_obj.get("scratchpad", {})
        if isinstance(scratchpad, dict):
            scratchpad_str = json.dumps(scratchpad)
        else:
            scratchpad_str = scratchpad

        self.state_stack.append({
            "state": state_obj.get("state"),
            "intent": state_obj.get("intent"),
            "tool": state_obj.get("tool"),
            "scratchpad": scratchpad  # keep as dict in memory
        })
        self._write_to_csv({
            "state": state_obj.get("state"),
            "intent": state_obj.get("intent"),
            "tool": state_obj.get("tool"),
            "scratchpad": scratchpad_str
        })

    def pop_state(self) -> Dict[str, Any]:
        if self.state_stack:
            return self.state_stack.pop()
        return {}

    def peek_state(self) -> Dict[str, Any]:
        return self.state_stack[-1] if self.state_stack else {}

    def update_scratchpad(self, updates: Dict[str, Any]):
        if not self.state_stack:
            return
        current = self.state_stack[-1]
        if "scratchpad" not in current or not isinstance(current["scratchpad"], dict):
            current["scratchpad"] = {}
        current["scratchpad"].update(updates)

    def _write_to_csv(self, state_obj: Dict[str, Any]):
        with open(self.filename, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["agent_id", "state", "intent", "tool", "scratchpad"])
            writer.writerow({
                "agent_id": self.agent_id,
                "state": state_obj.get("state"),
                "intent": state_obj.get("intent"),
                "tool": state_obj.get("tool"),
                "scratchpad": state_obj.get("scratchpad")
            })

