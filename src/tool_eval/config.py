"""Application settings loaded from environment variables and the `.env` file."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Served model names whose chat template rejects several tool calls in one assistant message
# (the Llama 3.1 template raises "This model only supports single tool-calls at once!").
SINGLE_TOOL_CALL_PRESETS = frozenset({"llama"})


class Settings(BaseSettings):
    """Validated configuration for the LLM, the agent and the research tools."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    llm_base_url: str = "http://localhost:8000/v1"
    llm_api_key: SecretStr = SecretStr("EMPTY")
    llm_model: str = "qwen"
    llm_max_tokens: int = Field(default=4096, ge=1)
    llm_enable_thinking: bool = False
    # Unset: preset default (see SINGLE_TOOL_CALL_PRESETS). False asks for one tool call per reply.
    llm_parallel_tool_calls: bool | None = None

    # One-tool-per-turn models need about 25 calls for a five-paper review; 30 leaves headroom.
    agent_max_steps: int = Field(default=30, ge=1)
    agent_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    prompt_version: str = "v1"

    # Inclusive submission-date window for arXiv searches; keeps runs reproducible.
    arxiv_date_from: date | None = None
    arxiv_date_to: date | None = None

    workspace_dir: Path = Path("workspace")
    # Shared download cache so repeated runs do not hit arXiv again; disabled when unset.
    pdf_cache_dir: Path | None = None
    http_timeout_seconds: float = Field(default=30.0, gt=0)
    max_file_bytes: int = Field(default=20_000_000, gt=0)

    text_max_chars: int = Field(default=6000, gt=0)
    chunk_size: int = Field(default=3000, gt=400)
    pipeline_max_chunks: int = Field(default=6, ge=1)
    review_paper_count: int = Field(default=5, ge=1, le=10)

    @property
    def effective_parallel_tool_calls(self) -> bool | None:
        """Parallel-tool-call policy sent to the server.

        Returns:
            `False` to ask for one tool call per reply, `True` to allow several, `None` to leave
            it to the server's default.
        """
        if self.llm_parallel_tool_calls is not None:
            return self.llm_parallel_tool_calls
        return False if self.llm_model in SINGLE_TOOL_CALL_PRESETS else None

    @model_validator(mode="after")
    def _check_date_window(self) -> Self:
        if (
            self.arxiv_date_from is not None
            and self.arxiv_date_to is not None
            and self.arxiv_date_from > self.arxiv_date_to
        ):
            msg = "ARXIV_DATE_FROM must not be later than ARXIV_DATE_TO"
            raise ValueError(msg)
        return self


_override: ContextVar[Settings | None] = ContextVar("settings_override", default=None)


@lru_cache
def _load_settings() -> Settings:
    return Settings()


def get_settings() -> Settings:
    """Return the active settings: a `use_settings` override, else the cached `.env` settings.

    Returns:
        The validated `Settings` for the current context.
    """
    return _override.get() or _load_settings()


@contextmanager
def use_settings(settings: Settings) -> Iterator[None]:
    """Make `get_settings()` return `settings` inside the `with` block (including tool threads).

    Args:
        settings: Settings to activate for the current context.

    Yields:
        Nothing; the override is removed on exit.
    """
    token = _override.set(settings)
    try:
        yield
    finally:
        _override.reset(token)
