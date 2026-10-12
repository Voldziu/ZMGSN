"""Domain exceptions. Messages must be safe to show to the LLM (no absolute paths, no secrets)."""


class ToolEvalError(Exception):
    """Base class for errors that are reported back to the agent instead of halting the run."""


class WorkspaceError(ToolEvalError):
    """A file operation was rejected or failed inside the sandboxed workspace."""


class ExternalServiceError(ToolEvalError):
    """A remote service (arXiv, Wikipedia, ...) failed or returned unusable data."""


class InvalidInputError(ToolEvalError):
    """The caller supplied input that cannot be processed."""


class PromptNotFoundError(ToolEvalError):
    """The requested prompt name or version does not exist."""
