"""LLM judge: rates each scored run's review after the evaluation, blind to model and level.

For every run the judge makes:
- one review call (topic, the variant's requirement statement, the review text) rating whether
  the file is a valid review and whether its papers fit the topic;
- one faithfulness call for each of up to two paper sections, chosen by a seed derived from the
  run id, comparing the section's summary with the beginning of the paper's extracted text;
- for `no_surveys` variants, one call per cited paper deciding from its arXiv title and abstract
  whether it is a survey.

Every answer is constrained to a Pydantic schema of integers 1-5 or booleans; a reply that does
not fit is retried once and then stored as missing. Judgments go to `<results>.judge.jsonl` next
to the results file, keyed by run id and configuration hash; run records are never changed.
"""

import hashlib
import json
import random
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tool_eval.evaluation.checks import Lookup, cited_ids, paper_sections
from tool_eval.evaluation.models import Judgment, RunRecord, Scenario
from tool_eval.evaluation.server_identity import ServerIdentity
from tool_eval.prompts import load_prompt
from tool_eval.services.papers import paper_file_stem
from tool_eval.utils.logger import get_logger
from tool_eval.utils.text import truncate

logger = get_logger(__name__)

JUDGE_PROMPT_VERSION = "v1"
REVIEW_MAX_CHARS = 20_000
SOURCE_MAX_CHARS = 12_000
SAMPLED_SECTIONS = 2
JUDGE_SUFFIX = ".judge.jsonl"


class ReviewRating(BaseModel):
    """Whether the file is a valid review, and whether its papers fit the topic."""

    model_config = ConfigDict(extra="forbid")

    valid_review: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)


class SummaryRating(BaseModel):
    """Faithfulness of one summary to its paper."""

    model_config = ConfigDict(extra="forbid")

    faithfulness: int = Field(ge=1, le=5)


class SurveyDecision(BaseModel):
    """Whether a paper is a survey or review."""

    model_config = ConfigDict(extra="forbid")

    is_survey: bool


# Asks the judge model: (schema, system prompt, user text) -> parsed answer; raises on bad output.
Ask = Callable[[type[BaseModel], str, str], BaseModel]


def chat_ask(*, base_url: str, model: str, api_key: str = "EMPTY") -> Ask:
    """An `Ask` that calls a served chat model with structured output, deterministically.

    Args:
        base_url: OpenAI-compatible base URL.
        model: Served model name.
        api_key: API key (vLLM ignores it).

    Returns:
        The ask function.
    """
    llm = ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0.0,
        top_p=1.0,
        seed=0,
        max_completion_tokens=64,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    def ask(schema: type[BaseModel], system: str, user: str) -> BaseModel:
        structured = llm.with_structured_output(schema, method="json_schema", strict=True)
        result = structured.invoke([SystemMessage(system), HumanMessage(user)])
        if not isinstance(result, schema):
            msg = f"judge returned {type(result).__name__}, expected {schema.__name__}"
            raise TypeError(msg)
        return result

    return ask


def judge_hash(identity: ServerIdentity, served_name: str) -> str:
    """Hash of the judge's prompts, settings and served model, stored with every judgment.

    Args:
        identity: Identity of the server answering as judge.
        served_name: Served model name of the judge.

    Returns:
        Hex SHA-256 digest.
    """
    payload = {
        "prompts": {
            name: load_prompt(name, JUDGE_PROMPT_VERSION).sha256
            for name in ("judge_review", "judge_summary", "judge_survey")
        },
        "settings": [REVIEW_MAX_CHARS, SOURCE_MAX_CHARS, SAMPLED_SECTIONS, 0.0, 0],
        "model": served_name,
        "server": identity.model_dump(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def judge_run(
    record: RunRecord,
    *,
    workspace: Path,
    scenario: Scenario,
    requirement_statement: str | None,
    ask: Ask,
    lookup: Lookup,
    judge_sha256: str,
) -> Judgment:
    """Rate one run's review.

    Args:
        record: The run's record.
        workspace: The run's workspace (review and extracted texts).
        scenario: The run's scenario (topic, requirement).
        requirement_statement: The variant's requirement as stated to the agent, if any.
        ask: Asks the judge model.
        lookup: arXiv metadata for survey decisions.
        judge_sha256: Hash of the judge configuration.

    Returns:
        The judgment; ratings the judge could not give are `None`.
    """
    review = workspace / "reviews" / "review.md"
    base = Judgment(
        run_id=record.run_id, config_sha256=record.config_sha256, judge_sha256=judge_sha256
    )
    if not review.is_file():
        return base
    text = review.read_text(encoding="utf-8", errors="replace")
    task = f"Topic: {scenario.topic}"
    if requirement_statement:
        task += f"\n{requirement_statement}"
    rating = _ask_or_none(
        ask,
        ReviewRating,
        load_prompt("judge_review", JUDGE_PROMPT_VERSION).text,
        f"{task}\n\nReview:\n{truncate(text, max_chars=REVIEW_MAX_CHARS)}",
    )
    faithfulness = [
        _faithfulness(ask, summary, workspace / "texts" / f"{paper_file_stem(arxiv_id)}.txt")
        for arxiv_id, summary in _sampled_sections(text, record.run_id)
    ]
    surveys = None
    if scenario.requirement == "no_surveys":
        surveys = _surveys(ask, cited_ids(text), lookup)
    return base.model_copy(
        update={
            "valid_review": rating.valid_review if rating else None,
            "relevance": rating.relevance if rating else None,
            "faithfulness": faithfulness,
            "surveys": surveys,
        }
    )


def judge_results(
    output: Path,
    *,
    runs_dir: Path,
    scenarios: Mapping[str, Scenario],
    requirement_statement: Callable[[Scenario], str | None],
    ask: Ask,
    lookup: Lookup,
    judge_sha256: str,
) -> tuple[int, int]:
    """Judge every scored run of a results file that has no judgment with this judge hash yet.

    Args:
        output: Results file (JSONL of run records).
        runs_dir: Directory with one workspace per run.
        scenarios: Scenarios by id.
        requirement_statement: The requirement sentence a variant's agent saw.
        ask: Asks the judge model.
        lookup: arXiv metadata for survey decisions.
        judge_sha256: Hash of the judge configuration.

    Returns:
        Runs judged in this call, and runs skipped because they were judged already.
    """
    path = judgment_file(output)
    done = {
        (j.run_id, j.config_sha256) for j in load_judgments(path) if j.judge_sha256 == judge_sha256
    }
    judged = skipped = 0
    for line in output.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = RunRecord.model_validate_json(line)
        if record.status == "error" or record.outcome is None:
            continue
        if (record.run_id, record.config_sha256) in done:
            skipped += 1
            continue
        scenario = scenarios.get(record.scenario_id)
        if scenario is None:
            logger.warning(
                "judge: scenario %s unknown; skipped %s", record.scenario_id, record.run_id
            )
            continue
        judgment = judge_run(
            record,
            workspace=runs_dir / record.run_id,
            scenario=scenario,
            requirement_statement=requirement_statement(scenario),
            ask=ask,
            lookup=lookup,
            judge_sha256=judge_sha256,
        )
        with path.open("a", encoding="utf-8") as handle:
            handle.write(judgment.model_dump_json() + "\n")
        judged += 1
        logger.info(
            "judged %s: valid=%s relevance=%s faithfulness=%s",
            record.run_id,
            judgment.valid_review,
            judgment.relevance,
            judgment.faithfulness,
        )
    return judged, skipped


def judgment_file(output: Path) -> Path:
    """File holding the judgments of a results file."""
    return output.with_name(output.stem + JUDGE_SUFFIX)


def load_judgments(path: Path) -> list[Judgment]:
    """Judgments stored in `path`; empty when it does not exist. Later lines win."""
    if not path.is_file():
        return []
    return [
        Judgment.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _ask_or_none[A: BaseModel](ask: Ask, schema: type[A], system: str, user: str) -> A | None:
    for attempt in (1, 2):
        try:
            answer = ask(schema, system, user)
            return schema.model_validate(answer.model_dump())
        except (ValidationError, ValueError, TypeError) as exc:
            logger.warning(
                "judge answer did not fit %s (attempt %d): %s", schema.__name__, attempt, exc
            )
    return None


def _faithfulness(ask: Ask, summary: str, source: Path) -> int | None:
    if not source.is_file():
        return 1  # nothing was extracted, so the summary cannot reflect the paper's text
    paper = truncate(
        source.read_text(encoding="utf-8", errors="replace"), max_chars=SOURCE_MAX_CHARS
    )
    rating = _ask_or_none(
        ask,
        SummaryRating,
        load_prompt("judge_summary", JUDGE_PROMPT_VERSION).text,
        f"Summary:\n{summary}\n\nPaper text:\n{paper}",
    )
    return rating.faithfulness if rating else None


def _surveys(ask: Ask, ids: Sequence[str], lookup: Lookup) -> dict[str, bool | None]:
    papers = lookup(list(ids)) if ids else {}
    decisions: dict[str, bool | None] = {}
    for arxiv_id in ids:
        paper = papers.get(arxiv_id)
        if paper is None:
            decisions[arxiv_id] = None
            continue
        answer = _ask_or_none(
            ask,
            SurveyDecision,
            load_prompt("judge_survey", JUDGE_PROMPT_VERSION).text,
            f"Title: {paper.title}\n\nAbstract: {paper.abstract}",
        )
        decisions[arxiv_id] = answer.is_survey if answer else None
    return decisions


def _sampled_sections(text: str, run_id: str) -> list[tuple[str, str]]:
    """Up to two (arxiv_id, summary) pairs, the same ones for the same run every time."""
    sections = paper_sections(text)
    seed = int(hashlib.sha256(run_id.encode()).hexdigest()[:16], 16)
    chosen = random.Random(seed).sample(sections, min(SAMPLED_SECTIONS, len(sections)))
    return chosen
