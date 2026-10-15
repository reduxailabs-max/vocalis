"""Tests for vocalis.logging_setup.configure_logging."""
import logging
from pathlib import Path

import pytest

from vocalis.logging_setup import configure_logging, _LOGGER_NAME


@pytest.fixture(autouse=True)
def _clear_logger_handlers() -> None:
    """Clear all handlers from the vocalis logger before each test."""
    logger = logging.getLogger(_LOGGER_NAME)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)


def test_returns_logger_named_vocalis() -> None:
    logger = configure_logging()
    assert logger.name == "vocalis"


def test_sets_requested_level() -> None:
    logger = configure_logging(level="DEBUG")
    assert logger.level == logging.DEBUG


def test_writes_to_log_file(tmp_path: Path) -> None:
    log_file = tmp_path / "vocalis.log"
    logger = configure_logging(level="INFO", log_file=log_file)
    logger.info("hello from test")
    for handler in logger.handlers:
        handler.flush()
    assert log_file.exists()
    assert "hello from test" in log_file.read_text()


def test_idempotent_does_not_duplicate_handlers(tmp_path: Path) -> None:
    log_file = tmp_path / "vocalis.log"
    configure_logging(level="INFO", log_file=log_file)
    logger = configure_logging(level="INFO", log_file=log_file)
    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1
