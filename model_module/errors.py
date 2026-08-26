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
        retry_after: seconds the PROVIDER asked us to wait, when it said so.
            A 429 usually carries one, and honouring it beats guessing: our
            backoff curve is a guess about a number the other side knows.
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
        # How many attempts were spent before giving up. Set by the retry loop
        # so a terminal can say "5 attempts" rather than only naming the last
        # failure, which reads as if nothing was tried.
        self.attempts = 1
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
