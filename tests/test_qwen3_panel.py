"""Tests for the Settings window's Qwen3-TTS add/remove panel and its progress dialog."""
import threading
from pathlib import Path
from typing import Any, Callable

import pytest
from PySide6.QtWidgets import QPlainTextEdit, QProgressBar, QLabel

from vocalis.config import VocalisConfig
from vocalis.installer.progress_display import ProgressDisplay
from vocalis.settings import LiveSettings
from vocalis.ui import qwen3_panel as panel_module
from vocalis.ui.qwen3_panel import (
    ADD_PHASE,
    REMOVE_PHASE,
    STATUS_INSTALLED,
    STATUS_MISSING,
    STATUS_NO_INSTALLER,
    STATUS_PARTIAL,
    STOPPED_TEXT,
    PhaseDialog,
    Qwen3Panel,
)
from vocalis.ui.settings_window import SettingsWindow


class FakeRunner:
    """A scripted stand-in for PhaseRunner."""

    def __init__(self, on_line: Callable[[str], None], lines: list[str], code: int, block: bool) -> None:
        self.on_line, self.lines, self.code, self.block = on_line, lines, code, block
        self.gate = threading.Event()
        self.terminated = False

    def run(self) -> int:
        for line in self.lines:
            self.on_line(line)
        if self.block:
            self.gate.wait(5)
            return 143 if self.terminated else self.code
        return self.code

    def terminate(self) -> None:
        self.terminated = True
        self.gate.set()


class Factory:
    """Records what it was asked to run; one scripted outcome per call."""

    def __init__(self, lines: list[str] | None = None, code: int = 0, block: bool = False) -> None:
        self.lines, self.code, self.block = lines or [], code, block
        self.calls: list[tuple[Path, str, tuple[str, ...]]] = []
        self.runners: list[FakeRunner] = []

    def __call__(self, installer_dir: Path, phase: str, on_line: Callable[[str], None], extra_args: Any = ()) -> Any:
        self.calls.append((installer_dir, phase, tuple(extra_args)))
        runner = FakeRunner(on_line, self.lines, self.code, self.block)
        self.runners.append(runner)
        return runner


@pytest.fixture
def installer_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "installer"
    folder.mkdir()
    (folder / "install.sh").write_text("# fake")
    return folder


def make_panel(
    qtbot: Any, installer_dir: Path, tmp_path: Path, factory: Factory, packages: bool = False, model: bool = False
) -> tuple[Qwen3Panel, dict[str, bool]]:
    state = {"packages": packages, "model": model}
    config = VocalisConfig()
    config.engines.qwen3_tts_model_dir = str(tmp_path / "qwen")
    panel = Qwen3Panel(
        lambda: config,
        installer_dir=lambda: installer_dir,
        runner_factory=factory,
        packages_installed=lambda: state["packages"],
        model_present=lambda path: state["model"],
    )
    qtbot.addWidget(panel)
    return panel, state


def test_not_installed_offers_download_but_not_remove(qtbot: Any, installer_dir: Path, tmp_path: Path) -> None:
    panel, _ = make_panel(qtbot, installer_dir, tmp_path, Factory())
    assert panel.status.text() == STATUS_MISSING
    assert panel.add_button.isEnabled() and not panel.remove_button.isEnabled()
    assert panel.is_installed() is False


def test_installed_offers_remove_but_not_download(qtbot: Any, installer_dir: Path, tmp_path: Path) -> None:
    panel, _ = make_panel(qtbot, installer_dir, tmp_path, Factory(), packages=True, model=True)
    assert panel.status.text() == STATUS_INSTALLED
    assert panel.remove_button.isEnabled() and not panel.add_button.isEnabled()
    assert panel.is_installed() is True


@pytest.mark.parametrize(("packages", "model"), [(True, False), (False, True)])
def test_partly_installed_offers_both_buttons(
    qtbot: Any, installer_dir: Path, tmp_path: Path, packages: bool, model: bool
) -> None:
    panel, _ = make_panel(qtbot, installer_dir, tmp_path, Factory(), packages=packages, model=model)
    assert panel.status.text() == STATUS_PARTIAL
    assert panel.add_button.isEnabled() and panel.remove_button.isEnabled()
    assert panel.is_installed() is False


def test_without_the_kept_installer_nothing_can_be_changed(qtbot: Any, tmp_path: Path) -> None:
    panel, _ = make_panel(qtbot, tmp_path / "missing", tmp_path, Factory(), packages=True, model=True)
    assert panel.status.text() == STATUS_NO_INSTALLER
    assert not panel.add_button.isEnabled() and not panel.remove_button.isEnabled()


def test_add_runs_the_add_phase_and_shows_the_download(qtbot: Any, installer_dir: Path, tmp_path: Path) -> None:
    factory = Factory(["STEP 2/2 Downloading the voice", "PROGRESS 500 1000 100 5"])
    panel, state = make_panel(qtbot, installer_dir, tmp_path, factory)
    changed: list[int] = []
    panel.changed.connect(lambda: changed.append(1))
    state.update(packages=True, model=True)
    panel.add_button.click()
    qtbot.waitUntil(lambda: bool(changed), timeout=5000)
    assert factory.calls == [(installer_dir, ADD_PHASE, ())]
    dialog = panel._dialog  # noqa: SLF001
    assert dialog is not None and dialog.button.text() == "Close"
    assert dialog.status.text() == panel_module.ADDED_TEXT
    assert panel.status.text() == STATUS_INSTALLED
    assert dialog.bar.value() == 100


def test_remove_asks_first_and_does_nothing_when_declined(qtbot: Any, installer_dir: Path, tmp_path: Path) -> None:
    factory = Factory()
    panel, _ = make_panel(qtbot, installer_dir, tmp_path, factory, packages=True, model=True)
    panel._confirm_remove = lambda: False  # type: ignore[method-assign]  # noqa: SLF001
    panel.remove_button.click()
    assert factory.calls == []


def test_remove_runs_the_remove_phase_when_confirmed(qtbot: Any, installer_dir: Path, tmp_path: Path) -> None:
    factory = Factory(["STEP 1/3 Removing"])
    panel, state = make_panel(qtbot, installer_dir, tmp_path, factory, packages=True, model=True)
    panel._confirm_remove = lambda: True  # type: ignore[method-assign]  # noqa: SLF001
    changed: list[int] = []
    panel.changed.connect(lambda: changed.append(1))
    state.update(packages=False, model=False)
    panel.remove_button.click()
    qtbot.waitUntil(lambda: bool(changed), timeout=5000)
    assert factory.calls == [(installer_dir, REMOVE_PHASE, ())]
    assert panel.status.text() == STATUS_MISSING
    assert panel._dialog.status.text() == panel_module.REMOVED_TEXT  # type: ignore[union-attr]  # noqa: SLF001


def test_buttons_stay_off_while_a_phase_is_running(qtbot: Any, installer_dir: Path, tmp_path: Path) -> None:
    factory = Factory(["STEP 1/2 Working"], block=True)
    panel, _ = make_panel(qtbot, installer_dir, tmp_path, factory)
    panel.add_button.click()
    qtbot.waitUntil(lambda: bool(factory.runners), timeout=5000)
    assert not panel.add_button.isEnabled() and not panel.remove_button.isEnabled()
    factory.runners[0].terminate()
    qtbot.waitUntil(lambda: panel._dialog is not None and not panel._dialog.is_running, timeout=5000)  # noqa: SLF001


def test_a_failed_phase_tells_the_person_and_reports_no_success(
    qtbot: Any, installer_dir: Path, tmp_path: Path
) -> None:
    factory = Factory(["boom"], code=1)
    panel, _ = make_panel(qtbot, installer_dir, tmp_path, factory)
    panel.add_button.click()
    qtbot.waitUntil(lambda: panel._dialog is not None and not panel._dialog.is_running, timeout=5000)  # noqa: SLF001
    dialog = panel._dialog  # noqa: SLF001
    assert dialog is not None
    assert "Something went wrong" in dialog.status.text()
    assert dialog.bar.value() == 0
    assert panel.add_button.isEnabled()


def test_stop_ends_the_phase_and_keeps_partial_downloads(qtbot: Any, installer_dir: Path) -> None:
    factory = Factory(["STEP 1/2 Working"], block=True)
    dialog = PhaseDialog("Adding", ADD_PHASE, installer_dir, factory)
    qtbot.addWidget(dialog)
    outcomes: list[bool] = []
    dialog.outcome.connect(outcomes.append)
    dialog.start()
    qtbot.waitUntil(lambda: bool(factory.runners), timeout=5000)
    dialog.button.click()
    qtbot.waitUntil(lambda: bool(outcomes), timeout=5000)
    assert factory.runners[0].terminated
    assert outcomes == [False]
    assert dialog.status.text() == STOPPED_TEXT
    assert dialog.button.text() == "Close"


def test_closing_the_dialog_while_it_runs_means_stop(qtbot: Any, installer_dir: Path) -> None:
    factory = Factory([], block=True)
    dialog = PhaseDialog("Adding", ADD_PHASE, installer_dir, factory)
    qtbot.addWidget(dialog)
    outcomes: list[bool] = []
    dialog.outcome.connect(outcomes.append)
    dialog.start()
    qtbot.waitUntil(lambda: bool(factory.runners), timeout=5000)
    dialog.reject()
    qtbot.waitUntil(lambda: bool(outcomes), timeout=5000)
    assert factory.runners[0].terminated
    dialog.reject()  # now idle: closes normally
    assert not dialog.isVisible()


def test_close_button_closes_a_finished_dialog(qtbot: Any, installer_dir: Path) -> None:
    dialog = PhaseDialog("Adding", ADD_PHASE, installer_dir, Factory())
    qtbot.addWidget(dialog)
    outcomes: list[bool] = []
    dialog.outcome.connect(outcomes.append)
    dialog.show()
    dialog.start()
    qtbot.waitUntil(lambda: bool(outcomes), timeout=5000)
    assert outcomes == [True]
    dialog.button.click()
    assert not dialog.isVisible()


def test_the_default_runner_builds_a_phase_runner(tmp_path: Path) -> None:
    runner = panel_module._default_runner(tmp_path, ADD_PHASE, lambda line: None, ("--x",))  # noqa: SLF001
    assert runner._phase == ADD_PHASE and runner._extra_args == ("--x",)  # noqa: SLF001


def test_progress_display_handles_steps_downloads_and_plain_lines(qtbot: Any) -> None:
    bar, status, log = QProgressBar(), QLabel(), QPlainTextEdit()
    for widget in (bar, status, log):
        qtbot.addWidget(widget)
    display = ProgressDisplay(bar, status, log)
    display.start()
    assert bar.maximum() == 0 and status.text() == "Starting…"
    display.handle("STEP 2/4 Getting voices")
    assert status.text() == "Getting voices" and bar.value() == 25
    event = display.handle("PROGRESS 500 1000 100 5")
    assert event is not None and event.kind == "progress"
    assert bar.maximum() == 1000 and bar.value() == 500
    assert "500 B of 1.0 KB" in status.text() and "Getting voices" in status.text()
    assert log.toPlainText() == ""
    assert display.handle("plain words") is None
    assert "plain words" in log.toPlainText()
    display.finish(True, "All set")
    assert bar.value() == 100 and status.text() == "All set"
    display.finish(False)
    assert bar.value() == 0


class FakePanel(Qwen3Panel):
    """A panel whose installed state the test controls."""

    def __init__(self, installed: bool) -> None:
        self.installed = installed
        super().__init__(
            lambda: VocalisConfig(),
            installer_dir=lambda: Path("/nowhere"),
            packages_installed=lambda: self.installed,
            model_present=lambda path: self.installed,
        )


def _window(qtbot: Any, tmp_path: Path, installed: bool) -> tuple[SettingsWindow, FakePanel]:
    panel = FakePanel(installed)
    settings = LiveSettings(VocalisConfig(), tmp_path / "config.toml")
    window = SettingsWindow(settings, qwen3_panel=panel)
    qtbot.addWidget(window)
    return window, panel


def _qwen_item(window: SettingsWindow) -> tuple[Any, int]:
    combo = window._quick_engine_combo  # noqa: SLF001
    index = combo.findData("qwen3_tts")
    return combo, index


def test_settings_greys_out_qwen3_choices_when_it_is_not_installed(qtbot: Any, tmp_path: Path) -> None:
    window, _ = _window(qtbot, tmp_path, installed=False)
    combo, index = _qwen_item(window)
    assert "not installed" in combo.itemText(index)
    assert not combo.model().item(index).isEnabled()
    assert not window._voice_combo.isEnabled()  # noqa: SLF001
    assert combo.model().item(combo.findData("kokoro")).isEnabled()


def test_settings_enables_qwen3_choices_once_it_is_installed(qtbot: Any, tmp_path: Path) -> None:
    window, panel = _window(qtbot, tmp_path, installed=False)
    panel.installed = True
    panel.changed.emit()
    combo, index = _qwen_item(window)
    assert "not installed" not in combo.itemText(index)
    assert combo.model().item(index).isEnabled()
    assert window._voice_combo.isEnabled()  # noqa: SLF001


def test_the_panel_sits_on_the_voice_tab(qtbot: Any, tmp_path: Path) -> None:
    window, panel = _window(qtbot, tmp_path, installed=True)
    voice_page = window._tabs.widget(1)  # noqa: SLF001
    assert voice_page is not None and voice_page.isAncestorOf(panel)
