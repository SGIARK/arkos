"""Tests for state_module/base_state.py: StateOutput model."""

import pytest
from pydantic import ValidationError

from state_module.base_state import StateOutput


class TestStateOutput:
    def test_acceptance_test(self):
        """Reproduces the acceptance test from the issue verbatim."""
        result = StateOutput(content="ok", completion_signal="complete")
        assert result.content == "ok"
        assert result.completion_signal == "complete"

    def test_content_is_optional(self):
        """content=None is valid: model returning nothing must not crash the loop."""
        result = StateOutput(completion_signal="error")
        assert result.content is None

    def test_defaults(self):
        """structured_data and error_detail default to None."""
        result = StateOutput(completion_signal="incomplete")
        assert result.structured_data is None
        assert result.error_detail is None

    def test_all_completion_signals_accepted(self):
        """All four valid signals are accepted; documents the allowed enum."""
        for signal in ("complete", "incomplete", "error", "needs_input"):
            result = StateOutput(completion_signal=signal)
            assert result.completion_signal == signal

    def test_invalid_completion_signal_rejected(self):
        """Literal constraint is enforced — unknown signals are rejected."""
        with pytest.raises(ValidationError):
            StateOutput(completion_signal="unknown")

    def test_completion_signal_required(self):
        """completion_signal has no default; callers must always declare intent."""
        with pytest.raises(ValidationError):
            StateOutput(content="hello")
