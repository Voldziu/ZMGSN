"""Central logging setup: every module gets its logger from `get_logger`."""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT_LOGGER_NAME = "tool_eval"
LOG_FILE = Path(os.environ.get("TOOL_EVAL_LOG_FILE", "logs/tool_eval.log"))
LOG_LEVEL = os.environ.get("TOOL_EVAL_LOG_LEVEL", "INFO").upper()

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s:%(lineno)d %(message)s"


def _configure() -> None:
    root = logging.getLogger(ROOT_LOGGER_NAME)
    if root.handlers:
        return
    formatter = logging.Formatter(_FORMAT)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    # The file keeps everything (DEBUG) so a run can be inspected afterwards.
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
        delay=True,
    )
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(LOG_LEVEL)
    for handler in (file_handler, console_handler):
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes to the console and the log file.

    Args:
        name: Logger name, normally `__name__` of the calling module.

    Returns:
        A logger below the `tool_eval` logger; handlers are installed on first use.
    """
    _configure()
    return logging.getLogger(name)
