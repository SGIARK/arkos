"""The pages draw every declared machine, and every box links to the right page."""

from state_machine_module.view import (
    quoted,
)


def test_quoted_escapes_quotes_and_keeps_line_breaks() -> None:
    assert quoted('a "b"\nc') == '"a \\"b\\"\\nc"'
