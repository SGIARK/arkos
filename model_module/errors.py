"""Typed errors for the model client layer."""


class ModelError(Exception):
    """
    The only exception `model_module.client.generate` raises.

    Args:
        message: human-readable description of the failure.
        retryable: True for transport failures worth another attempt.
        kind: one of timeout, connect, rate_limit, server_error, bad_request,
            auth, stream, internal, unknown.
        cause: the original exception, preserved for logging.
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        kind: str = "unknown",
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.kind = kind
        self.cause = cause


class OutputValidationError(Exception):
    """
    Raised when the model responded but its output does not match the required schema.

    `detail` is model-actionable and safe to feed back into context; `raw` is
    logged only, never shown.
    """

    def __init__(self, detail: str, *, raw: str | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.raw = raw
