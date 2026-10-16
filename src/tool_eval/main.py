"""Command-line entry point: `run` executes the evaluation, `report` summarizes its results."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from tool_eval.config import get_settings
from tool_eval.evaluation.checks import ArxivLookup
from tool_eval.evaluation.report import load_records, render_markdown
from tool_eval.evaluation.runner import run_evaluation
from tool_eval.evaluation.scenarios import load_scenarios
from tool_eval.exceptions import InvalidInputError
from tool_eval.tools.registry import TOOLSETS
from tool_eval.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_RESULTS = Path("results/results.jsonl")


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
    try:
        if args.command == "run":
            return _run(args)
        return _report(args)
    except InvalidInputError as exc:
        logger.error("%s", exc)  # noqa: TRY400 - user-facing message, no traceback needed
        return 2


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

    report = commands.add_parser("report", help="summarize a results file")
    report.add_argument("input", type=Path, nargs="?", default=DEFAULT_RESULTS)
    return parser


def _run(args: argparse.Namespace) -> int:
    scenarios = load_scenarios(args.scenarios)
    if args.only:
        unknown = set(args.only) - {s.id for s in scenarios}
        if unknown:
            msg = f"Unknown scenario ids: {', '.join(sorted(unknown))}"
            raise InvalidInputError(msg)
        scenarios = [s for s in scenarios if s.id in args.only]
    settings = get_settings().model_copy(update={"agent_temperature": args.temperature})
    executed = run_evaluation(
        scenarios=scenarios,
        levels=args.levels,
        repeats=args.repeats,
        output=args.output,
        runs_dir=args.runs_dir,
        settings=settings,
        lookup=ArxivLookup(timeout=settings.http_timeout_seconds),
    )
    logger.info("Executed %d runs; results in %s", executed, args.output)
    return 0


def _report(args: argparse.Namespace) -> int:
    if not args.input.is_file():
        msg = f"Results file not found: {args.input}"
        raise InvalidInputError(msg)
    # The report is the program's output, so it goes to stdout rather than the log.
    print(render_markdown(load_records(args.input)))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
