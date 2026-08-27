"""Typed errors for the model client layer."""


class ModelError(Exception):
    """The only exception `model_module.client.generate` raises.

    Args:
        kind: one of timeout, connect, rate_limit, server_error, bad_request,
            auth, stream, internal, unknown.
        retry_after: seconds the PROVIDER asked us to wait, when it said so.
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        kind: str = "unknown",
        cause: Exception | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.kind = kind
        self.retry_after = retry_after
        # Attempts spent before giving up; set by the retry loop, not here.
        self.attempts = 1
        self.cause = cause


class OutputValidationError(Exception):
    """Raised when the model responded but its output does not match the required schema.

    `detail` is safe to feed back into context; `raw` is logged only, never shown.
    """

    def __init__(self, detail: str, *, raw: str | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.raw = raw
