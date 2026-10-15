"""Tests for vocalis.entry — the lightweight `vocalis` console-script entrypoint."""
from unittest.mock import patch

import pytest


def test_entrypoint_without_flags_runs_main() -> None:
    """With no flags, entrypoint() imports and calls vocalis.cli.main()."""
    with patch("vocalis.cli.main") as fake_main:
        from vocalis.entry import entrypoint
        entrypoint([])
    fake_main.assert_called_once_with()


def test_entrypoint_settings_flag_opens_settings_and_exits_with_its_code() -> None:
    """With --settings, entrypoint() calls open_settings() and exits with its code, never touching main()."""
    from vocalis.entry import entrypoint
    with patch("vocalis.entry.open_settings", return_value=0) as fake_open, patch("vocalis.cli.main") as fake_main:
        with pytest.raises(SystemExit) as exited:
            entrypoint(["--settings"])
    assert exited.value.code == 0
    fake_open.assert_called_once()
    fake_main.assert_not_called()
