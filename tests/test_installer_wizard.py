"""Tests for the setup wizard window and the vocalis-setup program."""
import signal
import threading
from pathlib import Path
from typing import Any, Callable

import pytest
from PySide6.QtWidgets import QLabel, QPushButton, QWizard

import vocalis.installer.app as app_module
import vocalis.installer.wizard as wizard_module
import vocalis.installer.wizard_pages as pages_module
from vocalis.installer.mic_test import MicResult
from vocalis.installer.plan import InstalledGpu, make_plan
from vocalis.installer.wizard import SetupWizard

NEXT = QWizard.WizardButton.NextButton
FINISH = QWizard.WizardButton.FinishButton


class FakeRunner:
    """A scripted stand-in for PhaseRunner."""

    def __init__(self, on_line: Callable[[str], None], lines: list[str], code: int,
                 block: bool = False) -> None:
        self.on_line, self.lines, self.code = on_line, lines, code
        self.gate = threading.Event()
        self.block = block
        self.terminated = False
        self.stubborn = False

    def run(self) -> int:
        for line in self.lines:
            self.on_line(line)
        if self.block:
            self.gate.wait(5)
            return 143 if self.terminated else self.code
        return self.code

    def terminate(self) -> None:
        self.terminated = True
        if not self.stubborn:
            self.gate.set()


class Factory:
    """Records every runner it builds; outcomes are scripted per phase."""

    def __init__(self, script: dict[str, list[tuple[list[str], int, bool]]]) -> None:
        self.script = script
        self.built: list[tuple[str, FakeRunner]] = []
        self.args: list[tuple[str, tuple[str, ...]]] = []

    def __call__(self, installer_dir: Path, phase: str, on_line: Callable[[str], None],
                 extra_args: Any = ()) -> Any:
        self.args.append((phase, tuple(extra_args)))
        lines, code, block = self.script[phase].pop(0)
        runner = FakeRunner(on_line, lines, code, block)
        self.built.append((phase, runner))
        return runner


def ok(lines: list[str] | None = None) -> tuple[list[str], int, bool]:
    return (lines or [], 0, False)


@pytest.fixture
def log_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "share" / "install.log"
    monkeypatch.setattr(pages_module, "INSTALL_LOG_PATH", path)
    return path


@pytest.fixture
def messages(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    counts = {"stopped": 0, "hint": 0}
    monkeypatch.setattr(SetupWizard, "_show_stopped_message",
                        lambda self: counts.__setitem__("stopped", counts["stopped"] + 1))
    monkeypatch.setattr(SetupWizard, "_show_settings_hint",
                        lambda self: counts.__setitem__("hint", counts["hint"] + 1))
    monkeypatch.setattr(wizard_module, "_pause", lambda seconds: None)
    return counts


REAL_INSTALL_SIGNALS = app_module.install_signal_handlers


@pytest.fixture(autouse=True)
def no_real_signal_handlers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "install_signal_handlers", lambda app, wizard: None)


def build(qtbot: Any, factory: Factory, mic: MicResult | None = None,
          opener: Callable[[], int] = lambda: 0) -> SetupWizard:
    wizard = SetupWizard(
        Path("/unused"),
        plan_reader=lambda: make_plan(InstalledGpu("RTX 4090", 24000)),
        runner_factory=factory,
        mic_tester=lambda: mic or MicResult("heard", 0.5, "Great"),
        settings_opener=opener,
    )
    qtbot.addWidget(wizard)
    wizard.show()
    return wizard


def to_installing(wizard: SetupWizard) -> None:
    wizard.next()
    wizard.next()


def wait_next(qtbot: Any, wizard: SetupWizard) -> None:
    qtbot.waitUntil(lambda: wizard.button(NEXT).isEnabled(), timeout=5000)


def test_pages_come_in_the_expected_order(qtbot, log_path, messages) -> None:  # type: ignore[no-untyped-def]
    wizard = build(qtbot, Factory({"models": [ok()], "finish": [ok()]}))
    names = [wizard.page(i).objectName() for i in wizard.pageIds()]
    assert names == ["welcome", "found", "installing", "microphone", "done"]
    assert wizard.button(NEXT).text() == "Start"


def test_found_page_shows_the_plan_summary(qtbot, log_path, messages) -> None:  # type: ignore[no-untyped-def]
    wizard = build(qtbot, Factory({"models": [ok()], "finish": [ok()]}))
    wizard.next()
    text = wizard.page(1).findChild(QLabel, "found_text").text()  # type: ignore[union-attr]
    assert "RTX 4090" in text
    assert "Qwen3-TTS" in text


def test_found_page_offers_qwen3_unticked_and_explains_it(qtbot, log_path, messages) -> None:  # type: ignore[no-untyped-def]
    wizard = build(qtbot, Factory({"models": [ok()], "finish": [ok()]}))
    wizard.next()
    box = wizard.found_page.qwen3_checkbox
    assert not box.isChecked()
    assert box.isEnabled()
    assert "optional" in box.text()
    assert "Kokoro already sounds great" in wizard.found_page.qwen3_hint.text()


def test_found_page_warns_when_the_graphics_card_is_too_small(qtbot, log_path, messages) -> None:  # type: ignore[no-untyped-def]
    wizard = SetupWizard(
        Path("/unused"),
        plan_reader=lambda: make_plan(InstalledGpu("Tiny", 2048)),
        runner_factory=Factory({"models": [ok()]}),
        mic_tester=lambda: MicResult("heard", 0.5, "Great"),
    )
    qtbot.addWidget(wizard)
    wizard.show()
    wizard.next()
    assert wizard.found_page.qwen3_checkbox.isEnabled()
    assert "Not recommended" in wizard.found_page.qwen3_hint.text()


def test_models_phase_gets_no_flag_by_default(qtbot, log_path, messages) -> None:  # type: ignore[no-untyped-def]
    factory = Factory({"models": [ok()]})
    wizard = build(qtbot, factory)
    to_installing(wizard)
    qtbot.waitUntil(lambda: bool(factory.args), timeout=5000)
    assert factory.args == [("models", ())]


def test_ticking_qwen3_passes_the_flag_to_the_models_phase(qtbot, log_path, messages) -> None:  # type: ignore[no-untyped-def]
    factory = Factory({"models": [ok()]})
    wizard = build(qtbot, factory)
    wizard.next()
    wizard.found_page.qwen3_checkbox.setChecked(True)
    wizard.next()
    qtbot.waitUntil(lambda: bool(factory.args), timeout=5000)
    assert factory.args == [("models", ("--with-qwen3",))]


def test_installing_page_shows_step_text_and_progress(qtbot, log_path, messages) -> None:  # type: ignore[no-untyped-def]
    factory = Factory({"models": [(["STEP 1/2 Getting voices", "NOTE be patient", "plain"], 0, True)]})
    wizard = build(qtbot, factory)
    to_installing(wizard)
    page = wizard.installing_page
    qtbot.waitUntil(lambda: "be patient" in page.log_pane.toPlainText(), timeout=5000)
    assert page.status.text() == "Getting voices"
    assert page.bar.value() == 0
    assert "plain" in page.log_pane.toPlainText()
    factory.built[0][1].gate.set()
    qtbot.waitUntil(lambda: page.bar.value() == 100, timeout=5000)


def test_progress_bar_follows_the_step_index(qtbot, log_path, messages) -> None:  # type: ignore[no-untyped-def]
    factory = Factory({"models": [(["STEP 1/2 One", "STEP 2/2 Two"], 0, True)]})
    wizard = build(qtbot, factory)
    to_installing(wizard)
    qtbot.waitUntil(lambda: wizard.installing_page.status.text() == "Two", timeout=5000)
    assert wizard.installing_page.bar.value() == 50
    factory.built[0][1].gate.set()
    qtbot.waitUntil(lambda: wizard.installing_page.bar.value() == 100, timeout=5000)


def test_installing_success_enables_next(qtbot, log_path, messages) -> None:  # type: ignore[no-untyped-def]
    wizard = build(qtbot, Factory({"models": [ok()]}))
    to_installing(wizard)
    wait_next(qtbot, wizard)
    assert wizard.installing_page.bar.value() == 100


def test_installing_failure_shows_try_again_and_reruns_the_same_phase(  # type: ignore[no-untyped-def]
        qtbot, log_path, messages) -> None:
    factory = Factory({"models": [(["boom"], 1, False), ok()]})
    wizard = build(qtbot, factory)
    to_installing(wizard)
    page = wizard.installing_page
    qtbot.waitUntil(lambda: not page.retry.isHidden(), timeout=5000)
    assert "Something went wrong" in page.error.text()
    assert not wizard.button(NEXT).isEnabled()
    page.retry.click()
    wait_next(qtbot, wizard)
    assert [phase for phase, _ in factory.built] == ["models", "models"]
    assert factory.built[0][1] is not factory.built[1][1]
    assert wizard.exit_code == 1


def test_lines_are_appended_to_the_log_file(qtbot, log_path, messages) -> None:  # type: ignore[no-untyped-def]
    wizard = build(qtbot, Factory({"models": [ok(["STEP 1/1 Go", "hello"])]}))
    to_installing(wizard)
    wait_next(qtbot, wizard)
    assert log_path.read_text(encoding="utf-8").splitlines()[1:] == ["STEP 1/1 Go", "hello"]


def test_unwritable_log_never_crashes(qtbot, tmp_path, monkeypatch, messages) -> None:  # type: ignore[no-untyped-def]
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setattr(pages_module, "INSTALL_LOG_PATH", blocker / "sub" / "install.log")
    wizard = build(qtbot, Factory({"models": [ok(["hello"])]}))
    to_installing(wizard)
    wait_next(qtbot, wizard)


def test_closing_mid_install_terminates_the_runner_and_shows_the_stopped_message(  # type: ignore[no-untyped-def]
        qtbot, log_path, messages) -> None:
    factory = Factory({"models": [(["STEP 1/2 Working"], 0, True)]})
    wizard = build(qtbot, factory)
    to_installing(wizard)
    qtbot.waitUntil(lambda: bool(factory.built), timeout=5000)
    wizard.close()
    assert factory.built[0][1].terminated
    assert messages["stopped"] == 1
    assert wizard.exit_code == 1
    assert "stopped" in wizard.final_message


@pytest.mark.parametrize("verdict", ["quiet", "no_device"])
def test_microphone_warning_does_not_block_next_or_finish(  # type: ignore[no-untyped-def]
        qtbot, log_path, messages, verdict: str) -> None:
    mic = MicResult(verdict, 0.0, f"warning {verdict}")  # type: ignore[arg-type]
    wizard = build(qtbot, Factory({"models": [ok()], "finish": [ok()]}), mic=mic)
    to_installing(wizard)
    wait_next(qtbot, wizard)
    wizard.next()
    wizard.findChild(QPushButton, "mic_test_button").click()  # type: ignore[union-attr]
    qtbot.waitUntil(lambda: wizard.microphone_page.result_label.text() == f"warning {verdict}",
                    timeout=5000)
    assert wizard.button(NEXT).isEnabled()
    wizard.next()
    qtbot.waitUntil(lambda: wizard.button(FINISH).isEnabled(), timeout=5000)


def test_done_page_runs_the_finish_phase_after_the_microphone_page(  # type: ignore[no-untyped-def]
        qtbot, log_path, messages) -> None:
    factory = Factory({"models": [ok()], "finish": [ok()]})
    wizard = build(qtbot, factory)
    to_installing(wizard)
    wait_next(qtbot, wizard)
    assert [p for p, _ in factory.built] == ["models"]
    wizard.next()
    assert [p for p, _ in factory.built] == ["models"]
    wizard.next()
    qtbot.waitUntil(lambda: wizard.button(FINISH).isEnabled(), timeout=5000)
    assert [p for p, _ in factory.built] == ["models", "finish"]
    assert wizard.done_page.status.text() == "Vocalis is running"
    assert wizard.exit_code == 0


def _finish(qtbot: Any, factory: Factory, opener: Callable[[], int]) -> SetupWizard:
    wizard = build(qtbot, factory, opener=opener)
    to_installing(wizard)
    wait_next(qtbot, wizard)
    wizard.next()
    wizard.next()
    qtbot.waitUntil(lambda: wizard.button(FINISH).isEnabled(), timeout=5000)
    with qtbot.waitSignal(wizard.accepted, timeout=5000):
        wizard.button(FINISH).click()
    return wizard


def test_finish_opens_settings_and_retries_until_it_succeeds(qtbot, log_path, messages) -> None:  # type: ignore[no-untyped-def]
    results = iter([1, 1, 0])
    calls: list[int] = []

    def opener() -> int:
        calls.append(1)
        return next(results)

    _finish(qtbot, Factory({"models": [ok()], "finish": [ok()]}), opener)
    assert len(calls) == 3
    assert messages["hint"] == 0


def test_finish_gives_up_after_three_tries_without_crashing(qtbot, log_path, messages) -> None:  # type: ignore[no-untyped-def]
    calls: list[int] = []

    def opener() -> int:
        calls.append(1)
        return 1

    wizard = _finish(qtbot, Factory({"models": [ok()], "finish": [ok()]}), opener)
    assert len(calls) == 3
    assert messages["hint"] == 1
    assert wizard.exit_code == 0


def test_finish_survives_an_opener_that_raises(qtbot, log_path, messages) -> None:  # type: ignore[no-untyped-def]
    def opener() -> int:
        raise RuntimeError("no service")

    _finish(qtbot, Factory({"models": [ok()], "finish": [ok()]}), opener)
    assert messages["hint"] == 1


def test_wizard_uses_the_theme_in_app_main(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    shown: list[Any] = []
    themed: list[Any] = []

    class FakeApp:
        def exec(self) -> int:
            return 0

    class FakeWizard:
        exit_code = 0
        final_message = ""

        def __init__(self, installer_dir: Path) -> None:
            self.installer_dir = installer_dir

        def show(self) -> None:
            shown.append(self.installer_dir)

    monkeypatch.setattr(app_module, "_make_app", lambda argv: FakeApp())
    monkeypatch.setattr(app_module, "apply_theme", themed.append)
    monkeypatch.setattr(app_module, "SetupWizard", FakeWizard)
    assert app_module.main(["--installer-dir", str(tmp_path)]) == 0
    assert shown == [tmp_path]
    assert len(themed) == 1


def test_app_main_returns_one_and_prints_a_message_when_stopped(monkeypatch, capsys, tmp_path) -> None:  # type: ignore[no-untyped-def]
    class FakeApp:
        def exec(self) -> int:
            return 0

    class FakeWizard:
        exit_code = 1
        final_message = "Setup was stopped."

        def __init__(self, installer_dir: Path) -> None:
            pass

        def show(self) -> None:
            pass

    monkeypatch.setattr(app_module, "_make_app", lambda argv: FakeApp())
    monkeypatch.setattr(app_module, "apply_theme", lambda app: None)
    monkeypatch.setattr(app_module, "SetupWizard", FakeWizard)
    assert app_module.main(["--installer-dir", str(tmp_path)]) == 1
    assert "Setup was stopped." in capsys.readouterr().err


def test_app_main_returns_three_when_no_window_can_open(monkeypatch, capsys, tmp_path) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(app_module, "_has_display", lambda: False)
    assert app_module.main(["--installer-dir", str(tmp_path)]) == 3
    assert "could not open a window" in capsys.readouterr().err


def test_app_main_returns_three_when_the_application_cannot_start(  # type: ignore[no-untyped-def]
        monkeypatch, capsys, tmp_path) -> None:
    def broken(argv: list[str]) -> Any:
        raise RuntimeError("no platform")

    monkeypatch.setattr(app_module, "_make_app", broken)
    assert app_module.main(["--installer-dir", str(tmp_path)]) == 3
    assert "no platform" in capsys.readouterr().err


def test_has_display_reads_the_environment(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    for name in ("QT_QPA_PLATFORM", "DISPLAY", "WAYLAND_DISPLAY"):
        monkeypatch.delenv(name, raising=False)
    assert not app_module._has_display()
    monkeypatch.setenv("DISPLAY", ":0")
    assert app_module._has_display()


def test_app_main_requires_installer_dir() -> None:
    with pytest.raises(SystemExit) as info:
        app_module.main([])
    assert info.value.code == 2


def test_failure_text_and_stopped_text_show_the_log_path(qtbot, log_path, messages) -> None:  # type: ignore[no-untyped-def]
    factory = Factory({"models": [(["boom"], 1, False)]})
    wizard = build(qtbot, factory)
    to_installing(wizard)
    page = wizard.installing_page
    qtbot.waitUntil(lambda: not page.retry.isHidden(), timeout=5000)
    assert "Details are saved in" in page.error.text()
    assert str(log_path) in page.error.text() or "~" in page.error.text()
    assert "Setup did not finish. Run the installer again to finish." in wizard.final_message
    assert "press Try again" not in wizard.final_message
    assert "Details are saved in" in wizard.final_message


def test_stopped_final_message_has_the_log_path(qtbot, log_path, messages) -> None:  # type: ignore[no-untyped-def]
    wizard = build(qtbot, Factory({"models": [ok()]}))
    assert "Setup was stopped." in wizard.final_message
    assert "Details are saved in" in wizard.final_message


def test_log_hint_uses_a_tilde_for_the_home_folder(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(pages_module, "INSTALL_LOG_PATH", Path.home() / "x" / "install.log")
    assert pages_module.log_hint() == "Details are saved in ~/x/install.log"


def test_a_successful_retry_clears_the_failure_message(qtbot, log_path, messages) -> None:  # type: ignore[no-untyped-def]
    factory = Factory({"models": [(["boom"], 1, False), ok()]})
    wizard = build(qtbot, factory)
    to_installing(wizard)
    page = wizard.installing_page
    qtbot.waitUntil(lambda: not page.retry.isHidden(), timeout=5000)
    page.retry.click()
    wait_next(qtbot, wizard)
    assert "did not finish" not in wizard.final_message


def test_failure_and_try_again_on_the_finish_page(qtbot, log_path, messages) -> None:  # type: ignore[no-untyped-def]
    factory = Factory({"models": [ok()], "finish": [(["bad"], 2, False), ok()]})
    wizard = build(qtbot, factory)
    to_installing(wizard)
    wait_next(qtbot, wizard)
    wizard.next()
    wizard.next()
    page = wizard.done_page
    qtbot.waitUntil(lambda: not page.retry.isHidden(), timeout=5000)
    assert not wizard.button(FINISH).isEnabled()
    page.retry.click()
    qtbot.waitUntil(lambda: wizard.button(FINISH).isEnabled(), timeout=5000)
    assert [p for p, _ in factory.built] == ["models", "finish", "finish"]
    assert wizard.exit_code == 0


def test_each_attempt_writes_a_header_line_to_the_log(qtbot, log_path, messages) -> None:  # type: ignore[no-untyped-def]
    wizard = build(qtbot, Factory({"models": [ok(["hi"])]}))
    to_installing(wizard)
    wait_next(qtbot, wizard)
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("--- models phase ") and lines[0].endswith(" ---")
    assert lines[1] == "hi"


def test_the_bar_is_busy_until_the_first_step(qtbot, log_path, messages) -> None:  # type: ignore[no-untyped-def]
    factory = Factory({"models": [(["plain"], 0, True)]})
    wizard = build(qtbot, factory)
    to_installing(wizard)
    page = wizard.installing_page
    qtbot.waitUntil(lambda: "plain" in page.log_pane.toPlainText(), timeout=5000)
    assert page.bar.maximum() == 0
    factory.built[0][1].gate.set()
    wait_next(qtbot, wizard)
    assert page.bar.maximum() == 100


def test_close_with_a_stubborn_runner_still_exits_one_without_aborting(  # type: ignore[no-untyped-def]
        qtbot, log_path, messages, monkeypatch) -> None:
    monkeypatch.setattr(pages_module, "_THREAD_WAIT_MS", 50)
    factory = Factory({"models": [(["STEP 1/2 Working"], 0, True)]})
    wizard = build(qtbot, factory)
    to_installing(wizard)
    qtbot.waitUntil(lambda: bool(factory.built), timeout=5000)
    runner = factory.built[0][1]
    runner.stubborn = True
    wizard.close()
    assert runner.terminated
    assert wizard.exit_code == 1
    assert pages_module.running_threads()
    exits: list[int] = []
    monkeypatch.setattr("os._exit", exits.append)
    assert app_module._leave(1) == 1 or exits == [1]
    assert exits == [1]
    runner.gate.set()
    qtbot.waitUntil(lambda: not pages_module.running_threads(), timeout=5000)
    assert app_module._leave(1) == 1


def test_closing_after_a_clean_stop_leaves_no_running_thread(qtbot, log_path, messages) -> None:  # type: ignore[no-untyped-def]
    factory = Factory({"models": [(["STEP 1/2 Working"], 0, True)]})
    wizard = build(qtbot, factory)
    to_installing(wizard)
    qtbot.waitUntil(lambda: bool(factory.built), timeout=5000)
    wizard.close()
    assert not pages_module.running_threads()


def test_finish_shows_opening_text_and_can_be_closed_while_opening(  # type: ignore[no-untyped-def]
        qtbot, log_path, messages) -> None:
    gate = threading.Event()

    def opener() -> int:
        gate.wait(5)
        return 0

    wizard = build(qtbot, Factory({"models": [ok()], "finish": [ok()]}), opener=opener)
    to_installing(wizard)
    wait_next(qtbot, wizard)
    wizard.next()
    wizard.next()
    qtbot.waitUntil(lambda: wizard.button(FINISH).isEnabled(), timeout=5000)
    wizard.button(FINISH).click()
    assert "Opening Settings" in wizard.done_page.status.text()
    with qtbot.waitSignal(wizard.rejected, timeout=2000):
        wizard.reject()
    assert messages["stopped"] == 0
    assert wizard.exit_code == 0
    gate.set()
    qtbot.waitUntil(lambda: not pages_module.running_threads(), timeout=5000)


def test_a_kill_signal_stops_the_runner_and_exits_one_without_a_dialog(  # type: ignore[no-untyped-def]
        qtbot, log_path, messages) -> None:
    from PySide6.QtWidgets import QApplication

    factory = Factory({"models": [(["STEP 1/2 Working"], 0, True)]})
    wizard = build(qtbot, factory)
    to_installing(wizard)
    qtbot.waitUntil(lambda: bool(factory.built), timeout=5000)
    saved = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGHUP, signal.SIGTERM)}
    try:
        app = QApplication.instance()
        assert isinstance(app, QApplication)
        handler = REAL_INSTALL_SIGNALS(app, wizard)
        assert signal.getsignal(signal.SIGTERM) is handler
        handler(signal.SIGTERM, None)
    finally:
        for number, old in saved.items():
            signal.signal(number, old)
    assert factory.built[0][1].terminated
    assert wizard.exit_code == 1
    assert messages["stopped"] == 0
    assert "Setup was stopped." in wizard.final_message
    assert not pages_module.running_threads()


def test_download_progress_fills_the_bar_and_shows_speed(qtbot, log_path, messages) -> None:  # type: ignore[no-untyped-def]
    lines = ["STEP 1/2 Getting voices", "PROGRESS 1200000000 3400000000 12500000 176", "PROGRESS 1300000000 3400000000 0 -1"]
    factory = Factory({"models": [(lines, 0, True)]})
    wizard = build(qtbot, factory)
    to_installing(wizard)
    page = wizard.installing_page
    qtbot.waitUntil(lambda: page.bar.value() == 1300000000 * 1000 // 3400000000, timeout=5000)
    assert page.bar.maximum() == 1000
    assert page.status.text() == "Getting voices — 1.3 GB of 3.4 GB"
    factory.built[0][1].gate.set()
    qtbot.waitUntil(lambda: page.bar.value() == 100, timeout=5000)


def test_download_status_has_speed_and_time_left(qtbot, log_path, messages) -> None:  # type: ignore[no-untyped-def]
    wizard = build(qtbot, Factory({"models": [ok()], "finish": [ok()]}))
    to_installing(wizard)
    page = wizard.installing_page
    page._on_line("STEP 1/2 Getting voices")
    page._on_line("PROGRESS 1200000000 3400000000 12500000 176")
    assert page.status.text() == "Getting voices — 1.2 GB of 3.4 GB · 12.5 MB/s · about 3 min left"


def test_progress_lines_stay_out_of_the_log(qtbot, log_path, messages) -> None:  # type: ignore[no-untyped-def]
    wizard = build(qtbot, Factory({"models": [ok()], "finish": [ok()]}))
    to_installing(wizard)
    page = wizard.installing_page
    page._on_line("PROGRESS 5 10 1 1")
    assert "PROGRESS" not in page.log_pane.toPlainText()
    assert not log_path.exists() or "PROGRESS" not in log_path.read_text()
