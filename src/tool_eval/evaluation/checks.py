"""Deterministic outcome checks for the review file produced by an agent run."""

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from langdetect import (  # type: ignore[import-untyped]
    DetectorFactory,
    LangDetectException,
    detect,
)

from tool_eval.evaluation.models import CriterionResult, OutcomeScore, Scenario
from tool_eval.exceptions import InvalidInputError
from tool_eval.services.arxiv import ArxivPaper, fetch_arxiv_papers
from tool_eval.services.papers import paper_file_stem

Lookup = Callable[[Sequence[str]], Mapping[str, ArxivPaper | None]]

_ID = r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})"
_ARXIV_URL = re.compile(rf"https://arxiv\.org/abs/({_ID})(?:v\d+)?")
_APA_LINE = re.compile(
    r"^[^\W\d_][\w'’.-]*(?: [\w'’.-]+)*, [A-Z]\..*\(\d{4}\)\. .+\. arXiv\. https://arxiv\.org/abs/\S+$"
)
_SECTION_HEADING = re.compile(r"^##\s+(?:\d+\.\s*)?(.+?)\s*$")
_MIN_SUMMARY_CHARS = 40
CRITERIA = (
    "file_exists",
    "section_count",
    "summaries_present",
    "citations_apa",
    "ids_exist",
    "dates_in_window",
    "titles_match",
    "bibliography_matches",
    "no_duplicates",
    "papers_read",
)

FACTS = ("papers_cited", "cited_ids_exist", "cited_in_window", "text_extracted")

# Bump whenever a criterion is added or changes meaning; it is part of the run config hash.
# 2: papers_read (every cited paper's full text was extracted in the run).
# 3: fact checks on every arXiv id in the file, and the table/polish requirement checks.
CHECKS_VERSION = 3

# Every arXiv id in the file: abs/pdf URLs (any version) and `arXiv:<id>` mentions.
_ANY_ID = re.compile(rf"(?:arxiv\.org/(?:abs|pdf)/|arXiv:\s?)({_ID})(?:v\d+)?", re.IGNORECASE)
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_URL = re.compile(r"https?://\S+")
# Text of a paper block shorter than this is too little to tell its language.
_MIN_LANGUAGE_CHARS = 40


class ArxivLookup:
    """Looks up arXiv metadata by id and remembers answers across runs."""

    def __init__(self, *, timeout: float, cache_dir: Path | None = None) -> None:
        """Create an empty cache.

        Args:
            timeout: HTTP timeout in seconds.
            cache_dir: On-disk arXiv response cache shared with the tools, or `None`.
        """
        self._timeout = timeout
        self._cache_dir = cache_dir
        self._cache: dict[str, ArxivPaper | None] = {}

    def __call__(self, ids: Sequence[str]) -> Mapping[str, ArxivPaper | None]:
        """Return metadata (or `None` for unknown ids) for every requested id.

        Args:
            ids: arXiv identifiers without version suffix.

        Returns:
            Mapping from id to its paper, or `None` if arXiv does not know it.

        Raises:
            ExternalServiceError: If arXiv cannot be reached.
        """
        unknown = [item for item in dict.fromkeys(ids) if item not in self._cache]
        if unknown:
            try:
                for paper in fetch_arxiv_papers(
                    unknown, timeout=self._timeout, cache_dir=self._cache_dir
                ):
                    self._cache[paper.arxiv_id] = paper
            except InvalidInputError:
                # At least one id is missing; ask one by one to find out which.
                for item in unknown:
                    self._cache[item] = self._fetch_single(item)
            for item in unknown:
                self._cache.setdefault(item, None)
        return {item: self._cache[item] for item in ids}

    def _fetch_single(self, arxiv_id: str) -> ArxivPaper | None:
        try:
            return fetch_arxiv_papers([arxiv_id], timeout=self._timeout, cache_dir=self._cache_dir)[
                0
            ]
        except InvalidInputError:
            return None


@dataclass(frozen=True)
class _Section:
    title: str
    body: str
    citation: str
    summary: str
    arxiv_id: str | None


def evaluate_review(
    review_file: Path,
    *,
    workspace: Path,
    scenario: Scenario,
    expected_papers: int,
    lookup: Lookup,
) -> OutcomeScore:
    """Check a review file against the scenario; every criterion is always reported.

    Args:
        review_file: Path of the Markdown file the agent should have written.
        workspace: The run's own workspace; `papers_read` looks for extracted texts there.
        scenario: Scenario providing the allowed submission-date window.
        expected_papers: Required number of papers.
        lookup: Resolves arXiv ids to metadata.

    Returns:
        Per-criterion results with an aggregated score.

    Raises:
        ExternalServiceError: If arXiv cannot be reached while checking ids.
    """
    if not review_file.is_file():
        missing = [
            CriterionResult(name=name, passed=False, detail="review file missing")
            for name in (*CRITERIA, *FACTS)
        ]
        requirement = None
        if scenario.requirement in ("table", "polish"):
            requirement = _check("requirement_met", False, "review file missing")
        return OutcomeScore.from_criteria(
            missing[: len(CRITERIA)], facts=missing[len(CRITERIA) :], requirement=requirement
        )
    text = review_file.read_text(encoding="utf-8", errors="replace")
    sections, bibliography = _parse(text)
    ids = [section.arxiv_id for section in sections if section.arxiv_id]
    cited = cited_ids(text)
    papers = lookup(list(dict.fromkeys([*ids, *cited]))) if ids or cited else {}

    results = [
        CriterionResult(name="file_exists", passed=True),
        _check(
            "section_count",
            len(sections) == expected_papers,
            f"found {len(sections)}, expected {expected_papers}",
        ),
        _check(
            "summaries_present",
            bool(sections) and all(len(s.summary) >= _MIN_SUMMARY_CHARS for s in sections),
            "every section needs a summary",
        ),
        _check(
            "citations_apa",
            bool(sections) and all(_APA_LINE.match(s.citation) for s in sections),
            "every section needs one APA reference with an arXiv URL",
        ),
        _check(
            "ids_exist",
            bool(sections) and len(ids) == len(sections) and all(papers.get(i) for i in ids),
            "every section must cite an existing arXiv id",
        ),
        _check(
            "dates_in_window",
            bool(ids) and all(_in_window(papers.get(i), scenario) for i in ids),
            f"published must be within {scenario.date_from}..{scenario.date_to}",
        ),
        _check(
            "titles_match",
            bool(sections) and all(_title_matches(s, papers) for s in sections),
            "section heading and citation must contain the real arXiv title",
        ),
        _check(
            "bibliography_matches",
            bibliography is not None and _bibliography_ids(bibliography) == set(ids) and bool(ids),
            "Bibliography must list exactly the cited ids",
        ),
        _check("no_duplicates", bool(ids) and len(ids) == len(set(ids)), "papers must be distinct"),
        _papers_read(ids, workspace),
    ]
    return OutcomeScore.from_criteria(
        results,
        facts=_facts(cited, papers, workspace, scenario, expected_papers),
        requirement=_requirement(text, cited, papers, scenario),
    )


def cited_ids(text: str) -> list[str]:
    """Every distinct arXiv id in `text`, in order of first mention, without version suffix.

    Args:
        text: Review text.

    Returns:
        The ids.
    """
    return list(dict.fromkeys(match.group(1) for match in _ANY_ID.finditer(text)))


def _facts(
    cited: Sequence[str],
    papers: Mapping[str, ArxivPaper | None],
    workspace: Path,
    scenario: Scenario,
    expected_papers: int,
) -> list[CriterionResult]:
    unknown = [i for i in cited if not papers.get(i)]
    outside = [i for i in cited if papers.get(i) and not _in_window(papers.get(i), scenario)]
    unread = [i for i in cited if not (workspace / "texts" / f"{paper_file_stem(i)}.txt").is_file()]
    return [
        _check(
            "papers_cited",
            len(cited) >= expected_papers,
            f"cites {len(cited)} distinct papers, expected at least {expected_papers}",
        ),
        _check(
            "cited_ids_exist", bool(cited) and not unknown, f"unknown ids: {', '.join(unknown)}"
        ),
        _check(
            "cited_in_window",
            bool(cited) and not unknown and not outside,
            f"outside {scenario.date_from}..{scenario.date_to}: {', '.join(outside or unknown)}",
        ),
        _check(
            "text_extracted",
            bool(cited) and not unread,
            f"text not extracted in this run: {', '.join(unread)}" if cited else "no ids cited",
        ),
    ]


def _requirement(
    text: str, cited: Sequence[str], papers: Mapping[str, ArxivPaper | None], scenario: Scenario
) -> CriterionResult | None:
    if scenario.requirement == "table":
        return _table_requirement(text, cited, papers)
    if scenario.requirement == "polish":
        return _polish_requirement(text, cited)
    return None  # base task, or no_surveys (decided by the judge)


def _paper_blocks(text: str, cited: Sequence[str]) -> tuple[int | None, list[list[str]]]:
    """Index of the first paper heading and the line blocks of paper sections.

    A paper section is a heading (any level below the first-level title) followed, before the
    next heading, by a line naming a cited id.
    """
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if _HEADING.match(line)]
    blocks: list[list[str]] = []
    first: int | None = None
    wanted = set(cited)
    for n, start in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        block = lines[start:end]
        heading = _HEADING.match(block[0])
        if heading and len(heading.group(1)) == 1:
            continue
        if any(set(cited_ids(line)) & wanted for line in block[1:]):
            blocks.append(block)
            first = start if first is None else first
    return first, blocks


def paper_sections(text: str) -> list[tuple[str, str]]:
    """The paper sections of a review as (arXiv id, summary), whatever the heading level.

    A paper section is a heading below the first-level title followed by a line naming an arXiv
    id; its summary is its text without the heading, URLs and lines naming an id.

    Args:
        text: Review text.

    Returns:
        One pair per section, in order.
    """
    _, blocks = _paper_blocks(text, cited_ids(text))
    sections = []
    for block in blocks:
        ids = [i for line in block[1:] for i in cited_ids(line)]
        summary = " ".join(
            line.strip()
            for line in block[1:]
            if line.strip() and not _URL.search(line) and not cited_ids(line)
        )
        sections.append((ids[0], summary))
    return sections


def _table_requirement(
    text: str, cited: Sequence[str], papers: Mapping[str, ArxivPaper | None]
) -> CriterionResult:
    lines = text.splitlines()
    first_section, _ = _paper_blocks(text, cited)
    title = next((i for i, line in enumerate(lines) if line.startswith("# ")), 0)
    end = first_section if first_section is not None else len(lines)
    region = lines[title:end]
    separator = next((i for i, line in enumerate(region) if _TABLE_SEPARATOR.match(line)), None)
    if separator is None or separator == 0 or not region[separator - 1].strip().startswith("|"):
        return _check("requirement_met", False, "no Markdown table before the first paper section")
    rows: list[str] = []
    for line in region[separator + 1 :]:
        if not line.strip().startswith("|"):
            break
        rows.append(line)
    normalized_rows = [_normalize(row) for row in rows]
    missing = []
    for arxiv_id in cited:
        paper = papers.get(arxiv_id)
        title_key = _normalize(paper.title) if paper else None
        found = any(
            arxiv_id in row or (title_key is not None and title_key in normalized)
            for row, normalized in zip(rows, normalized_rows, strict=True)
        )
        if not found:
            missing.append(arxiv_id)
    return _check(
        "requirement_met",
        bool(cited) and not missing,
        f"papers missing from the table: {', '.join(missing)}" if cited else "no ids cited",
    )


def _polish_requirement(text: str, cited: Sequence[str]) -> CriterionResult:
    _, blocks = _paper_blocks(text, cited)
    languages = []
    for block in blocks:
        body = [
            line
            for line in block[1:]
            if line.strip() and not _URL.search(line) and not cited_ids(line)
        ]
        prose = _URL.sub("", " ".join(body)).strip()
        if len(prose) >= _MIN_LANGUAGE_CHARS:
            languages.append(_language(prose))
    not_polish = [lang for lang in languages if lang != "pl"]
    return _check(
        "requirement_met",
        bool(languages) and not not_polish,
        f"summaries detected as {', '.join(sorted(set(not_polish)))}"
        if languages
        else "no paper summaries found",
    )


def _language(text: str) -> str:
    DetectorFactory.seed = 0  # langdetect is random otherwise
    try:
        return str(detect(text))
    except LangDetectException:
        return "unknown"


def _papers_read(ids: Sequence[str], workspace: Path) -> CriterionResult:
    # The tools always extract a paper's full text to texts/<stem>.txt in the run workspace
    # before summarizing it (all levels), so its presence proves the paper was read here.
    unread = [
        arxiv_id
        for arxiv_id in dict.fromkeys(ids)
        if not (workspace / "texts" / f"{paper_file_stem(arxiv_id)}.txt").is_file()
    ]
    if not ids:
        return _check("papers_read", False, "the review cites no arXiv papers")
    return _check("papers_read", not unread, f"not read in this run: {', '.join(unread)}")


def _check(name: str, passed: bool, detail: str) -> CriterionResult:
    return CriterionResult(name=name, passed=passed, detail="" if passed else detail)


def _parse(text: str) -> tuple[list[_Section], str | None]:
    blocks: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        match = _SECTION_HEADING.match(line) if line.startswith("## ") else None
        if match:
            blocks.append((match.group(1), []))
        elif blocks:
            blocks[-1][1].append(line)
    sections: list[_Section] = []
    bibliography: str | None = None
    for title, lines in blocks:
        if title.lower().startswith("bibliography"):
            bibliography = "\n".join(lines)
            continue
        sections.append(_build_section(title, lines))
    return sections, bibliography


def _build_section(title: str, lines: list[str]) -> _Section:
    citation_lines = [line.strip() for line in lines if "arxiv.org/abs/" in line]
    citation = citation_lines[0] if citation_lines else ""
    summary = " ".join(
        line.strip() for line in lines if line.strip() and "arxiv.org/abs/" not in line
    )
    match = _ARXIV_URL.search(citation)
    return _Section(
        title=title,
        body="\n".join(lines),
        citation=citation,
        summary=summary,
        arxiv_id=match.group(1) if match else None,
    )


def _bibliography_ids(bibliography: str) -> set[str]:
    return {match.group(1) for match in _ARXIV_URL.finditer(bibliography)}


def _in_window(paper: ArxivPaper | None, scenario: Scenario) -> bool:
    return paper is not None and scenario.date_from <= paper.published <= scenario.date_to


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def _title_matches(section: _Section, papers: Mapping[str, ArxivPaper | None]) -> bool:
    paper = papers.get(section.arxiv_id) if section.arxiv_id else None
    if paper is None:
        return False
    title = _normalize(paper.title)
    return title in _normalize(section.title) and title in _normalize(section.citation)
