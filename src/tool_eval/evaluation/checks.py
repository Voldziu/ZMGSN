"""Deterministic outcome checks for the review file produced by an agent run."""

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from tool_eval.evaluation.models import CriterionResult, OutcomeScore, Scenario
from tool_eval.exceptions import InvalidInputError
from tool_eval.services.arxiv import ArxivPaper, fetch_arxiv_papers

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
)


class ArxivLookup:
    """Looks up arXiv metadata by id and remembers answers across runs."""

    def __init__(self, *, timeout: float) -> None:
        """Create an empty cache.

        Args:
            timeout: HTTP timeout in seconds.
        """
        self._timeout = timeout
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
                for paper in fetch_arxiv_papers(unknown, timeout=self._timeout):
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
            return fetch_arxiv_papers([arxiv_id], timeout=self._timeout)[0]
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
    review_file: Path, *, scenario: Scenario, expected_papers: int, lookup: Lookup
) -> OutcomeScore:
    """Check a review file against the scenario; every criterion is always reported.

    Args:
        review_file: Path of the Markdown file the agent should have written.
        scenario: Scenario providing the allowed submission-date window.
        expected_papers: Required number of papers.
        lookup: Resolves arXiv ids to metadata.

    Returns:
        Per-criterion results with an aggregated score.

    Raises:
        ExternalServiceError: If arXiv cannot be reached while checking ids.
    """
    if not review_file.is_file():
        return OutcomeScore.from_criteria(
            [
                CriterionResult(name=name, passed=False, detail="review file missing")
                for name in CRITERIA
            ]
        )
    text = review_file.read_text(encoding="utf-8", errors="replace")
    sections, bibliography = _parse(text)
    ids = [section.arxiv_id for section in sections if section.arxiv_id]
    papers = lookup(list(dict.fromkeys(ids))) if ids else {}

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
    ]
    return OutcomeScore.from_criteria(results)


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
