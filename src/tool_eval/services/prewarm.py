"""Fill the shared arXiv and PDF caches before an evaluation.

For every scenario the level-3 search (topic, twice the review size, scenario window) runs once,
each candidate's metadata is looked up on its own, and each candidate's PDF is downloaded into
the PDF cache. Missing PDFs (404) end up cached as missing. Runs then find the same environment
whatever order they run in.
"""

import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from tool_eval.config import Settings
from tool_eval.evaluation.models import Scenario
from tool_eval.exceptions import PaperNotFoundError, ToolEvalError
from tool_eval.services import papers
from tool_eval.services.arxiv import fetch_arxiv_papers, search_arxiv
from tool_eval.services.network import network_meter
from tool_eval.services.reports import CANDIDATE_FACTOR
from tool_eval.utils.logger import get_logger

logger = get_logger(__name__)


class PrewarmReport(BaseModel):
    """What `prewarm` added to the caches."""

    scenarios: int = 0
    candidates: int = 0
    arxiv_requests: int = 0
    pdf_downloads: int = 0
    missing_pdfs: list[str] = []
    failed: list[str] = []


def prewarm(scenarios: Sequence[Scenario], *, settings: Settings) -> PrewarmReport:
    """Fill the arXiv response cache and the PDF cache for `scenarios`.

    Args:
        scenarios: Scenarios whose candidates are cached; variants share their base's topic.
        settings: Settings with `arxiv_cache_dir` and `pdf_cache_dir` set.

    Returns:
        Counts of what reached the network, and the papers whose PDF is missing or failed.
    """
    report = PrewarmReport()
    seen: set[str] = set()
    with tempfile.TemporaryDirectory() as scratch, network_meter() as network:
        workspace = Path(scratch)
        for scenario in scenarios:
            key = f"{scenario.topic}|{scenario.date_from}|{scenario.date_to}"
            if key in seen:
                continue
            seen.add(key)
            report.scenarios += 1
            candidates = search_arxiv(
                scenario.topic,
                max_results=CANDIDATE_FACTOR * settings.review_paper_count,
                date_from=scenario.date_from,
                date_to=scenario.date_to,
                timeout=settings.http_timeout_seconds,
            )
            for paper in candidates:
                report.candidates += 1
                try:
                    fetch_arxiv_papers([paper.arxiv_id], timeout=settings.http_timeout_seconds)
                    papers.download_paper_pdf(
                        paper.arxiv_id,
                        settings=settings.model_copy(update={"workspace_dir": workspace}),
                    )
                    logger.info("prewarm: %s cached", paper.arxiv_id)
                except PaperNotFoundError:
                    report.missing_pdfs.append(paper.arxiv_id)
                except ToolEvalError as exc:
                    logger.warning("prewarm: %s failed: %s", paper.arxiv_id, exc)
                    report.failed.append(paper.arxiv_id)
                shutil.rmtree(workspace / "papers", ignore_errors=True)
        report.arxiv_requests = network["arxiv_requests"]
        report.pdf_downloads = network["pdf_downloads"]
    return report
