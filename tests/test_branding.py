"""The shipped logo files load and reach the application."""

from __future__ import annotations

from importlib import resources

import pytest
from PySide6.QtWidgets import QApplication

from vocalis.ui import branding


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])  # type: ignore[return-value]


def test_every_icon_size_is_shipped_inside_the_package() -> None:
    folder = resources.files("vocalis.assets.icons")
    for size in branding.ICON_SIZES:
        assert folder.joinpath(f"vocalis-{size}.png").is_file()
    assert folder.joinpath("vocalis-mark.png").is_file()


def test_app_icon_contains_every_size(qapp: QApplication) -> None:
    icon = branding.app_icon()
    sizes = {size.width() for size in icon.availableSizes()}
    assert sizes == set(branding.ICON_SIZES)


def test_logo_pixmap_is_a_real_image(qapp: QApplication) -> None:
    pixmap = branding.logo_pixmap()
    assert not pixmap.isNull()
    assert pixmap.width() == pixmap.height()


def test_apply_branding_sets_window_icon_and_menu_name(qapp: QApplication) -> None:
    branding.apply_branding(qapp)
    assert not qapp.windowIcon().isNull()
    assert qapp.desktopFileName() == "vocalis"


def test_desktop_entry_uses_the_vocalis_icon() -> None:
    from pathlib import Path

    entry = Path(__file__).resolve().parent.parent / "packaging" / "desktop" / "vocalis.desktop"
    assert "Icon=vocalis" in entry.read_text()
