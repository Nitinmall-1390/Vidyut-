"""
VIDYUT Logging Utility
===========================================================================
Structured logging with rotating file handler + coloured console output.
All modules import get_logger() from here for consistency.
===========================================================================
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ANSI colour codes for console
_COLOURS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
    "RESET":    "\033[0m",
}


class _ColouredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        colour = _COLOURS.get(record.levelname, "")
        reset = _COLOURS["RESET"]
        record.levelname = f"{colour}{record.levelname}{reset}"
        return super().format(record)


def get_logger(
    name: str,
    level: Optional[str] = None,
    log_dir: Optional[Path] = None,
) -> logging.Logger:
    """
    Return a named logger with console + optional rotating file handler.

    Parameters
    ----------
    name : str
        Module / component name (e.g. "vidyut.ingestion").
    level : str, optional
        Override log level. Falls back to LOG_LEVEL env var or INFO.
    log_dir : Path, optional
        If supplied, writes to <log_dir>/<name>.log with 10 MB rotation.
    """
    import os

    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured — return as-is
        return logger

    resolved_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logger.setLevel(resolved_level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        _ColouredFormatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)
    )
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / f"{name.replace('.', '_')}.log",
            maxBytes=10 * 1024 * 1024,   # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(
            logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)
        )
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
