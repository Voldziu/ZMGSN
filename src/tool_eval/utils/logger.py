"""Logging set-up: modules get their logger from `get_logger`; only the command line adds handlers.

Importing the package installs no handlers, so tests and other callers never write a log file;
their lines go wherever the caller's logging sends them. `configure_logging`, called by
`tool_eval.main`, sends lines to the console and, for an evaluation, to the log file next to its
results. Inside `run_context`, every line carries the run's ID, also when a tool's service writes
it from one of the agent's worker threads.
"""

import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from tqdm import tqdm

ROOT_LOGGER_NAME = "tool_eval"
LOG_LEVEL_VARIABLE = "TOOL_EVAL_LOG_LEVEL"

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s:%(lineno)d %(run_tag)s%(message)s"
_installed: list[logging.Handler] = []
_current_run: ContextVar[str | None] = ContextVar("current_run", default=None)


class _RunTag(logging.Filter):
    """Adds `run_tag` to every record: `[<run ID>] ` inside `run_context`, empty outside."""

    def filter(self, record: logging.LogRecord) -> bool:
        run_id = _current_run.get()
        record.run_tag = f"[{run_id}] " if run_id else ""
        return True


@contextmanager
def run_context(run_id: str) -> Iterator[None]:
    """Put `run_id` on every log line written inside the block.

    The ID lives in a context variable, which LangChain copies into the threads that run tools,
    the same way the run's settings reach them.

    Args:
        run_id: ID of the evaluation run.

    Yields:
        Nothing; the ID is removed on exit.
    """
    token = _current_run.set(run_id)
    try:
        yield
    finally:
        _current_run.reset(token)


def get_logger(name: str) -> logging.Logger:
    """Return the logger of a module.

    Args:
        name: Logger name, normally `__name__` of the calling module.

    Returns:
        A logger below the `tool_eval` logger. It writes nowhere until `configure_logging` runs.
    """
    return logging.getLogger(name)


def configure_logging(log_file: Path | None = None) -> None:
    """Send log lines to the console and, when a file is given, to that file.

    The console shows lines from `TOOL_EVAL_LOG_LEVEL` (default INFO) up. The file keeps every
    `tool_eval` line from DEBUG up; it is appended to and never rotated. Other libraries' lines
    reach both from WARNING up. Handlers from an earlier call are closed and replaced.

    Args:
        log_file: File the log is appended to; `None` logs to the console only.
    """
    close_logging()
    console = logging.StreamHandler(_ConsoleStream())
    console.setLevel(os.environ.get(LOG_LEVEL_VARIABLE, "INFO").upper())
    handlers: list[logging.Handler] = [console]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        handlers.append(file_handler)
    root = logging.getLogger()
    formatter = logging.Formatter(_FORMAT)
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.addFilter(_RunTag())
        root.addHandler(handler)
        _installed.append(handler)
    root.setLevel(logging.WARNING)
    logging.getLogger(ROOT_LOGGER_NAME).setLevel(logging.DEBUG)


def close_logging() -> None:
    """Remove and close the handlers that `configure_logging` installed."""
    root = logging.getLogger()
    for handler in _installed:
        root.removeHandler(handler)
        handler.close()
    _installed.clear()
    logging.getLogger(ROOT_LOGGER_NAME).setLevel(logging.NOTSET)


class _ConsoleStream:
    """Stderr for the console handler, written through `tqdm.write`.

    With progress bars open, `tqdm.write` prints the line above them and redraws them; with no
    bars it writes the text unchanged. Stderr is looked up on every write, so a redirected or
    captured stderr is honoured.
    """

    def write(self, text: str) -> None:
        tqdm.write(text, file=sys.stderr, end="")

    def flush(self) -> None:
        sys.stderr.flush()
