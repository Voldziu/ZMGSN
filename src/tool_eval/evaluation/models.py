"""Data models for scenarios and per-run results (one `RunRecord` per JSONL line)."""

from datetime import date, datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

RunStatus = Literal["completed", "step_limit", "error"]


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
    """All criteria for a run; `success` means every criterion passed."""

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
    """Resource use of one run, split into agent-level and tool-internal LLM calls."""

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
    wall_seconds: float = 0.0

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
    prompt_version: str
    system_prompt_sha256: str
    task_prompt_sha256: str
    date_from: date
    date_to: date
    started_at: datetime
