"""Central logging configuration for the vocalis daemon and CLI.

Every module in vocalis logs through the single "vocalis" logger configured
here, so log level and destinations are controlled in one place. Never log
secrets, credentials, or full user text content — only status and metadata.
"""
import logging
from pathlib import Path

__all__ = ["configure_logging"]

_LOGGER_NAME = "vocalis"
_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    """Configure and return the shared "vocalis" logger.

    Safe to call multiple times (e.g. once at daemon startup, once in
    tests) — it will not attach duplicate handlers for the same log file.

    Args:
        level: Standard logging level name, e.g. "INFO", "DEBUG".
        log_file: Optional path to also write logs to a file. Parent
            directories are created if missing.

    Returns:
        The configured logger named "vocalis".
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper()))
    formatter = logging.Formatter(_FORMAT)

    has_console_handler = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not has_console_handler:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file is not None:
        already_has_this_file = any(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file
            for h in logger.handlers
        )
        if not already_has_this_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
