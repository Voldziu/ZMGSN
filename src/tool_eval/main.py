"""Command-line entry point.

`run` executes the evaluation, `prewarm` fills the arXiv and PDF caches beforehand, `judge`
rates the reviews with an LLM judge afterwards, and `report` summarizes results.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from tool_eval.config import get_settings
from tool_eval.evaluation.checks import ArxivLookup
from tool_eval.evaluation.judge import chat_ask, judge_hash, judge_results
from tool_eval.evaluation.models import Scenario
from tool_eval.evaluation.report import (
    load_records,
    load_results,
    render_markdown,
    summarize,
    write_csv,
)
from tool_eval.evaluation.rescore import rescore
from tool_eval.evaluation.runner import arxiv_cache_dir, requirement_prompt, run_evaluation
from tool_eval.evaluation.scenarios import load_scenarios
from tool_eval.evaluation.server_identity import read_server_identity
from tool_eval.exceptions import InvalidInputError, TrackingServerError
from tool_eval.services.prewarm import prewarm
from tool_eval.tools.registry import TOOLSETS
from tool_eval.utils.logger import close_logging, configure_logging, get_logger

# Not `__name__`: run as `python -m tool_eval.main`, this module is `__main__`, whose logger would
# sit outside `tool_eval` and drop its INFO lines.
logger = get_logger("tool_eval.main")

DEFAULT_RESULTS = Path("results/results.jsonl")
LOG_SUFFIX = ".log"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface.

    Args:
        argv: Arguments without the program name; defaults to `sys.argv[1:]`.

    Returns:
        Process exit code.
    """
    # LangSmith tracing is configured through LANGSMITH_* variables, which only exist in os.environ.
    load_dotenv()
    args = _parser().parse_args(argv)
    configure_logging(_log_file(args))
    try:
        if args.command == "run":
            return _run(args)
        if args.command == "prewarm":
            return _prewarm(args)
        if args.command == "judge":
            return _judge(args)
        if args.command == "rescore":
            return _rescore(args)
        return _report(args)
    except (InvalidInputError, TrackingServerError) as exc:
        logger.error("%s", exc)  # noqa: TRY400 - user-facing message, no traceback needed
        return 2
    finally:
        close_logging()


def _log_file(args: argparse.Namespace) -> Path | None:
    """`run` logs next to its results file; `report` and a rejected output log to the console."""
    if args.command != "run" or args.output.suffix == LOG_SUFFIX:
        return None
    return Path(args.output).with_suffix(LOG_SUFFIX)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tool_eval")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run scenarios against tool levels")
    run.add_argument("--scenarios", type=Path, default=Path("scenarios/scenarios.json"))
    run.add_argument("--only", nargs="+", metavar="ID", help="restrict to these scenario ids")
    run.add_argument("--levels", nargs="+", default=list(TOOLSETS), choices=list(TOOLSETS))
    run.add_argument("--repeats", type=int, default=10)
    run.add_argument("--temperature", type=float, default=0.7)
    run.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    run.add_argument("--runs-dir", type=Path, default=Path("workspace/runs"))
    run.add_argument(
        "--order-seed",
        type=int,
        help="seed of the level order within each block "
        "(default: the seed recorded in --output, else random)",
    )

    warm = commands.add_parser("prewarm", help="fill the arXiv and PDF caches for scenarios")
    warm.add_argument("--scenarios", type=Path, default=Path("scenarios/scenarios.json"))
    warm.add_argument("--only", nargs="+", metavar="ID", help="restrict to these scenario ids")
    warm.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("workspace/runs"),
        help="runs directory of the evaluation; locates the default caches",
    )

    judge = commands.add_parser("judge", help="rate the reviews of results files with an LLM judge")
    judge.add_argument("inputs", type=Path, nargs="+", metavar="input", help="results files")
    judge.add_argument("--scenarios", type=Path, default=Path("scenarios/scenarios.json"))
    judge.add_argument(
        "--runs-root",
        type=Path,
        default=Path("workspace/preset-runs"),
        help="run workspaces are <runs-root>/<model>/<run_id> (as written by eval-presets)",
    )
    judge.add_argument("--judge-model", default="qwen", help="served name of the judge model")

    again = commands.add_parser("rescore", help="recompute the checks of saved runs")
    again.add_argument("inputs", type=Path, nargs="+", metavar="input", help="results files")
    again.add_argument("--scenarios", type=Path, default=Path("scenarios/scenarios.json"))
    again.add_argument(
        "--runs-root",
        type=Path,
        default=Path("workspace/preset-runs"),
        help="run workspaces are <runs-root>/<model>/<run_id>",
    )

    report = commands.add_parser("report", help="summarize one or more results files")
    report.add_argument(
        "inputs",
        type=Path,
        nargs="*",
        default=[DEFAULT_RESULTS],
        metavar="input",
        help=f"results files, for example one per model (default: {DEFAULT_RESULTS})",
    )
    report.add_argument(
        "--csv", type=Path, help="also write one row per model and level to this CSV file"
    )
    return parser


def _run(args: argparse.Namespace) -> int:
    if args.output.suffix == LOG_SUFFIX:
        msg = f"--output must not end in {LOG_SUFFIX}; that name is the evaluation's log file"
        raise InvalidInputError(msg)
    scenarios = _select(args)
    settings = get_settings().model_copy(update={"agent_temperature": args.temperature})
    executed = run_evaluation(
        scenarios=scenarios,
        levels=args.levels,
        repeats=args.repeats,
        output=args.output,
        runs_dir=args.runs_dir,
        settings=settings,
        lookup=ArxivLookup(
            timeout=settings.http_timeout_seconds,
            cache_dir=arxiv_cache_dir(settings, args.runs_dir),
        ),
        order_seed=args.order_seed,
    )
    logger.info(
        "Executed %d runs; results in %s, log in %s", executed, args.output, _log_file(args)
    )
    return 0


def _select(args: argparse.Namespace) -> list[Scenario]:
    scenarios = load_scenarios(args.scenarios)
    if args.only:
        unknown = set(args.only) - {s.id for s in scenarios}
        if unknown:
            msg = f"Unknown scenario ids: {', '.join(sorted(unknown))}"
            raise InvalidInputError(msg)
        scenarios = [s for s in scenarios if s.id in args.only]
    return scenarios


def _prewarm(args: argparse.Namespace) -> int:
    base = get_settings()
    settings = base.model_copy(
        update={
            "pdf_cache_dir": base.pdf_cache_dir or args.runs_dir.parent / "pdf_cache",
            "arxiv_cache_dir": arxiv_cache_dir(base, args.runs_dir),
        }
    )
    result = prewarm(_select(args), settings=settings)
    logger.info(
        "prewarm: %d topics, %d candidates, %d arXiv requests and %d PDF downloads sent; "
        "missing PDFs: %s; failed: %s",
        result.scenarios,
        result.candidates,
        result.arxiv_requests,
        result.pdf_downloads,
        " ".join(result.missing_pdfs) or "none",
        " ".join(result.failed) or "none",
    )
    return 0


def _judge(args: argparse.Namespace) -> int:
    settings = get_settings()
    scenarios = {s.id: s for s in load_scenarios(args.scenarios)}
    identity = read_server_identity(settings.llm_base_url, served_name=args.judge_model)
    judge_sha = judge_hash(identity, args.judge_model)
    ask = chat_ask(base_url=settings.llm_base_url, model=args.judge_model)
    lookup = ArxivLookup(
        timeout=settings.http_timeout_seconds,
        cache_dir=settings.arxiv_cache_dir or Path("workspace/arxiv_cache"),
    )
    for path in args.inputs:
        records = load_records(path)
        if not records:
            continue
        judged, skipped = judge_results(
            path,
            runs_dir=args.runs_root / records[0].model,
            scenarios=scenarios,
            requirement_statement=lambda s: (
                requirement_prompt(s.requirement, settings) if s.requirement else None
            ),
            ask=ask,
            lookup=lookup,
            judge_sha256=judge_sha,
        )
        logger.info("%s: judged %d runs, %d already judged", path, judged, skipped)
    return 0


def _rescore(args: argparse.Namespace) -> int:
    settings = get_settings()
    scenarios = {s.id: s for s in load_scenarios(args.scenarios)}
    lookup = ArxivLookup(
        timeout=settings.http_timeout_seconds,
        cache_dir=settings.arxiv_cache_dir or Path("workspace/arxiv_cache"),
    )
    for path in args.inputs:
        records = load_records(path)
        if not records:
            continue
        written, missing = rescore(
            path, runs_dir=args.runs_root / records[0].model, scenarios=scenarios, lookup=lookup
        )
        logger.info(
            "%s: rescored into %s; kept old outcome (no workspace): %s",
            path,
            written,
            " ".join(missing) or "none",
        )
    return 0


def _report(args: argparse.Namespace) -> int:
    missing = [str(path) for path in args.inputs if not path.is_file()]
    if missing:
        msg = f"Results file not found: {', '.join(missing)}"
        raise InvalidInputError(msg)
    records = load_results(args.inputs)
    # The report is the program's output, so it goes to stdout rather than the log.
    print(render_markdown(records))  # noqa: T201
    if args.csv:
        write_csv(summarize(records), args.csv)
        logger.info("Wrote %s", args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
