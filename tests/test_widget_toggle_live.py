"""The floating widget follows the Settings window live (no restart)."""
from vocalis.config import VocalisConfig
from vocalis.settings import LiveSettings
from vocalis.ui.floating_widget import FloatingWidget
from vocalis.ui.settings_window import SettingsWindow
from vocalis.ui.status_hub import ReadingControls, StatusHub


def _build(qtbot, tmp_path, config):  # type: ignore[no-untyped-def]
    settings = LiveSettings(config, tmp_path / "config.toml")
    hub = StatusHub()
    widget = FloatingWidget(hub, settings.floating_widget_enabled)
    window = SettingsWindow(settings, on_applied=widget.apply_enabled)
    qtbot.addWidget(window)
    qtbot.addWidget(widget)
    return settings, hub, widget, window


def test_ticking_only_the_widget_box_shows_the_widget_on_the_next_reading(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = VocalisConfig(floating_widget_enabled=False)
    _settings, hub, widget, window = _build(qtbot, tmp_path, config)  # type: ignore[no-untyped-call]
    window._notifications_box.setChecked(False)  # noqa: SLF001
    window._overlay_box.setChecked(False)  # noqa: SLF001
    window._widget_box.setChecked(True)  # noqa: SLF001
    window._save_button.click()  # noqa: SLF001
    hub.begin_reading("Quick read", ReadingControls(stop=lambda: None))
    qtbot.waitUntil(widget.isVisible, timeout=2000)
