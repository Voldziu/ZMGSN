"""Data models for scenarios and per-run results (one `RunRecord` per JSONL line)."""

from datetime import date, datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tool_eval.evaluation.server_identity import ServerIdentity

# `failed`: agent broke the run (e.g. context overflow), scored; `error`: infrastructure, retried.
RunStatus = Literal["completed", "step_limit", "loop", "failed", "error"]


# Requirement a variant task adds to its base scenario (see `scenarios.load_scenarios`).
Requirement = Literal["table", "polish", "no_surveys"]
REQUIREMENTS: tuple[Requirement, ...] = ("table", "polish", "no_surveys")
TaskType = Literal["base", "variant"]


class Scenario(BaseModel):
    """One task: a research topic and the arXiv submission-date window that counts.

    A variant names its base scenario (`base`) and the one requirement it adds; a base scenario
    has neither.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    topic: str = Field(min_length=3)
    date_from: date
    date_to: date
    requirement: Requirement | None = None
    base: str | None = None

    @model_validator(mode="after")
    def _check_window(self) -> Self:
        if self.date_from > self.date_to:
            msg = f"Scenario {self.id}: date_from is later than date_to"
            raise ValueError(msg)
        if (self.requirement is None) != (self.base is None):
            msg = f"Scenario {self.id}: a variant needs both `base` and `requirement`"
            raise ValueError(msg)
        return self

    @property
    def task_type(self) -> TaskType:
        """`variant` for a scenario with an added requirement, else `base`."""
        return "base" if self.requirement is None else "variant"

    @property
    def base_id(self) -> str:
        """Id of the base scenario; a base scenario's own id."""
        return self.base or self.id


class CriterionResult(BaseModel):
    """Outcome of one deterministic check."""

    model_config = ConfigDict(frozen=True)

    name: str
    passed: bool
    detail: str = ""


class OutcomeScore(BaseModel):
    """All checks of a run.

    `criteria` are the format checks (structure, APA references, titles, bibliography); `score`
    is the fraction of them that passed and `success` whether all did. `facts` are checked on
    every arXiv id in the file, whatever its formatting. `requirement` is the added requirement
    of a variant when it can be checked without the judge (`table`, `polish`); it is `None` for
    base tasks and for `no_surveys`, which the judge decides. Whether the run succeeded also
    depends on its status; see `RunRecord`.
    """

    model_config = ConfigDict(frozen=True)

    criteria: list[CriterionResult]
    score: float
    success: bool
    facts: list[CriterionResult] = []
    requirement: CriterionResult | None = None

    @classmethod
    def from_criteria(
        cls,
        criteria: list[CriterionResult],
        *,
        facts: list[CriterionResult] | None = None,
        requirement: CriterionResult | None = None,
    ) -> Self:
        """Aggregate format checks into a fractional score and an all-or-nothing flag.

        Args:
            criteria: Format check results.
            facts: Fact check results.
            requirement: Result of a variant's deterministic requirement check.

        Returns:
            The aggregated outcome.
        """
        passed = sum(item.passed for item in criteria)
        return cls(
            criteria=criteria,
            score=passed / len(criteria) if criteria else 0.0,
            success=bool(criteria) and passed == len(criteria),
            facts=facts or [],
            requirement=requirement,
        )

    @property
    def facts_pass(self) -> bool | None:
        """Whether every fact check passed; `None` for records scored before fact checks."""
        return all(f.passed for f in self.facts) if self.facts else None


# Lowest rating (on 1-5) that counts as passing a judge scale.
JUDGE_PASS = 4


class Judgment(BaseModel):
    """The LLM judge's ratings of one run's review (see `evaluation.judge`).

    Ratings are integers 1-5, or `None` when the judge's answer could not be parsed.
    `faithfulness` has one rating per sampled summary. `surveys` maps each cited paper of a
    `no_surveys` variant to whether the judge considers it a survey; `None` for other tasks.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    config_sha256: str
    judge_sha256: str
    valid_review: int | None = None
    relevance: int | None = None
    faithfulness: list[int | None] = []
    surveys: dict[str, bool | None] | None = None

    @property
    def passes(self) -> bool | None:
        """Every rating at least `JUDGE_PASS`; `None` when a rating is missing."""
        ratings = [self.valid_review, self.relevance, *self.faithfulness]
        if not self.faithfulness or any(r is None for r in ratings):
            return None
        return all(r is not None and r >= JUDGE_PASS for r in ratings)


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
    # Distinct characters of each paper's extracted text (`texts/<stem>.txt`) that reached the
    # agent or a model in the run, keyed by file stem.
    chars_read: dict[str, int] | None = None
    # arXiv API requests and PDF downloads of the run that missed the shared caches.
    arxiv_requests: int | None = None
    pdf_downloads: int | None = None
    # Agent replies whose text holds a tool call the server's parser missed.
    text_tool_calls: int | None = None
    # Tool errors caused by external services (network, arXiv HTTP errors, missing PDFs) and
    # by the agent (every other failed call); they add up to `tool_errors`.
    tool_errors_environment: int | None = None
    tool_errors_agent: int | None = None

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
    # Sampling, token limit, thinking mode and served model; unknown in records written before
    # they entered the configuration hash.
    top_p: float | None = None
    top_k: int | None = None
    llm_max_tokens: int | None = None
    enable_thinking: bool | None = None
    server: ServerIdentity | None = None
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
    # Base task or variant with an added requirement; older records are base tasks.
    task_type: TaskType = "base"
    requirement: Requirement | None = None
    base_scenario_id: str | None = None
    started_at: datetime

    @computed_field  # type: ignore[prop-decorator]
    @property
    def success(self) -> bool:
        """Format success: the run completed and passed every format check.

        `step_limit`, `loop` and `failed` runs never succeed, even when every check passed; their
        `outcome.score` still gives partial credit. The value is written to the results file
        but recomputed on every read, so records written before this rule follow it too.
        """
        return self.status == "completed" and self.outcome is not None and self.outcome.success

    @property
    def format_success(self) -> bool:
        """The run completed and passed every format check (the original success rule)."""
        return self.success

    @property
    def reliable(self) -> bool:
        """The agent ended the run itself (status `completed`)."""
        return self.status == "completed"

    def artifact_valid(self, judgment: Judgment | None) -> bool | None:
        """Whether the review is valid whatever the run's status.

        Valid means every fact check passed, every judge rating is at least `JUDGE_PASS` and a
        variant meets its requirement. `None` when it cannot be decided yet: the judgment or a
        rating is missing, or the record predates the fact checks.

        Args:
            judgment: The judge's ratings of this run, if any.

        Returns:
            True, False or None.
        """
        if self.outcome is None:
            return None
        facts = self.outcome.facts_pass
        if facts is False:
            return False
        requirement = self._requirement_met(judgment)
        if requirement is False:
            return False
        if facts is None or judgment is None or judgment.passes is None or requirement is None:
            return None
        return judgment.passes

    def success_with(self, judgment: Judgment | None) -> bool | None:
        """Artifact valid and reliable; `None` while artifact validity is undecided."""
        valid = self.artifact_valid(judgment)
        if valid is None:
            return None if self.reliable else False
        return valid and self.reliable

    def _requirement_met(self, judgment: Judgment | None) -> bool | None:
        if self.requirement is None:
            return True
        if self.requirement == "no_surveys":
            if judgment is None or judgment.surveys is None:
                return None
            if any(judgment.surveys.values()):
                return False
            return None if any(v is None for v in judgment.surveys.values()) else True
        result = self.outcome.requirement if self.outcome else None
        return None if result is None else result.passed
