"""Data models for scenarios and per-run results (one `RunRecord` per JSONL line)."""

from datetime import date, datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

# `failed`: agent broke the run (e.g. context overflow), scored; `error`: infrastructure, retried.
RunStatus = Literal["completed", "step_limit", "loop", "failed", "error"]


class Scenario(BaseModel):
    """One task: a research topic and the arXiv submission-date window that counts."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    topic: str = Field(min_length=3)
    date_from: date
    date_to: date

    @model_validator(mode="after")
    def _check_window(self) -> Self:
        if self.date_from > self.date_to:
            msg = f"Scenario {self.id}: date_from is later than date_to"
            raise ValueError(msg)
        return self


class CriterionResult(BaseModel):
    """Outcome of one deterministic check."""

    model_config = ConfigDict(frozen=True)

    name: str
    passed: bool
    detail: str = ""


class OutcomeScore(BaseModel):
    """All criteria for a run; `success` means every criterion passed.

    Whether the run itself succeeded also depends on its status; see `RunRecord.success`.
    """

    model_config = ConfigDict(frozen=True)

    criteria: list[CriterionResult]
    score: float
    success: bool

    @classmethod
    def from_criteria(cls, criteria: list[CriterionResult]) -> Self:
        """Aggregate criteria into a fractional score and an all-or-nothing success flag.

        Args:
            criteria: Individual check results.

        Returns:
            The aggregated score.
        """
        passed = sum(item.passed for item in criteria)
        return cls(
            criteria=criteria,
            score=passed / len(criteria) if criteria else 0.0,
            success=bool(criteria) and passed == len(criteria),
        )


class CostMetrics(BaseModel):
    """Resource use of one run, split into agent-level and tool-internal LLM calls.

    The fields that default to `None` were added later. `None` means "not measured" (a record
    written before the field existed), never zero, so averages can leave such records out.
    """

    model_config = ConfigDict(frozen=True)

    agent_llm_calls: int = 0
    tool_llm_calls: int = 0
    agent_input_tokens: int = 0
    agent_output_tokens: int = 0
    tool_input_tokens: int = 0
    tool_output_tokens: int = 0
    llm_calls_without_usage: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    # Building the agent plus the agent loop; records written before scoring was left out
    # (those without `tool_seconds`) also include scoring the review.
    wall_seconds: float = 0.0
    # Summed durations of calls; calls that overlap in time (parallel tool calls) all count.
    agent_llm_seconds: float | None = None
    tool_llm_seconds: float | None = None
    # Tool calls, including the LLM calls made inside them.
    tool_seconds: float | None = None
    # Input tokens of the first agent LLM call (system prompt, task and tool schemas): the
    # prompt overhead of the level. `None` also when that call reported no usage.
    first_call_input_tokens: int | None = None
    # Agent tool calls that never started a tool run: arguments that could not be parsed, and
    # tool names the level does not offer. Neither counts in `tool_calls` or `tool_errors`.
    invalid_tool_calls: int | None = None
    unknown_tool_calls: int | None = None

    @property
    def total_tokens(self) -> int:
        """Return input plus output tokens over agent and tool-internal calls."""
        return (
            self.agent_input_tokens
            + self.agent_output_tokens
            + self.tool_input_tokens
            + self.tool_output_tokens
        )

    @property
    def llm_calls(self) -> int:
        """Return the number of LLM calls over agent and tool-internal calls."""
        return self.agent_llm_calls + self.tool_llm_calls

    @property
    def tool_other_seconds(self) -> float | None:
        """Return the time tools spent outside LLM calls (network, PDF parsing, files)."""
        if self.tool_seconds is None or self.tool_llm_seconds is None:
            return None
        return max(0.0, round(self.tool_seconds - self.tool_llm_seconds, 3))

    @property
    def malformed_tool_calls(self) -> int | None:
        """Return the agent's tool calls that never started a tool run (`None` if unknown)."""
        if self.invalid_tool_calls is None or self.unknown_tool_calls is None:
            return None
        return self.invalid_tool_calls + self.unknown_tool_calls


class RunRecord(BaseModel):
    """Everything needed to analyse and reproduce a single agent run."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    level: str
    scenario_id: str
    repeat: int
    status: RunStatus
    error: str | None = None
    outcome: OutcomeScore | None = None
    cost: CostMetrics
    model: str
    temperature: float
    max_steps: int
    # Loop-guard limits (AGENT_MAX_REPEATED_CALLS, AGENT_MAX_TOOL_ERROR_STREAK); unknown in
    # records written before the loop guard existed.
    max_repeated_calls: int | None = None
    max_tool_error_streak: int | None = None
    prompt_version: str
    system_prompt_sha256: str
    task_prompt_sha256: str
    review_paper_count: int | None = None
    chunk_size: int | None = None
    text_max_chars: int | None = None
    pipeline_max_chunks: int | None = None
    # Hash of the settings above that change results; differing hashes are not comparable.
    config_sha256: str = ""
    # Sampling seed of the run; deliberately not part of config_sha256.
    seed: int | None = None
    # Seed of the level order within each (repeat, scenario) block (`runner.block_order`) and
    # the run's 1-based position in its block's planned order over all levels of the tool set;
    # not part of config_sha256. `started_at` shows the order in which runs actually ran.
    order_seed: int | None = None
    block_position: int | None = None
    # MLflow run and trace of this run when tracing was on (MLFLOW_TRACKING_URI).
    mlflow_run_id: str | None = None
    trace_id: str | None = None
    date_from: date
    date_to: date
    started_at: datetime

    @computed_field  # type: ignore[prop-decorator]
    @property
    def success(self) -> bool:
        """Whether the run succeeded: it completed and passed every check.

        `step_limit`, `loop` and `failed` runs never succeed, even when every check passed; their
        `outcome.score` still gives partial credit. The value is written to the results file
        but recomputed on every read, so records written before this rule follow it too.
        """
        return self.status == "completed" and self.outcome is not None and self.outcome.success
