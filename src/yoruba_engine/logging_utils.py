"""Logging setup.

The package never configures the root logger on import; ``configure_logging`` is
called explicitly by the CLI. Library consumers (a FastAPI app) keep control of
their own logging configuration.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(
    level: int | str = logging.INFO,
    log_file: Path | None = None,
) -> None:
    """Attach a stderr handler, and optionally a file handler, to the package logger.

    Args:
        level: Standard logging level or its name.
        log_file: If given, also write logs here. Parent dirs are created.
    """
    logger = logging.getLogger("yoruba_engine")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(_FORMAT)

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``yoruba_engine`` namespace."""
    return logging.getLogger(f"yoruba_engine.{name}")
