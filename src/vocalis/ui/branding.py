"""The Vocalis logo: loading the shipped icon files for windows and the menu."""

from __future__ import annotations

from importlib import resources

from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QApplication

APP_ID = "vocalis"
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256, 512)
_MARK_NAME = "vocalis-mark.png"


def _icon_bytes(name: str) -> bytes:
    """Read one shipped icon file from the package."""
    return resources.files("vocalis.assets.icons").joinpath(name).read_bytes()


def app_icon() -> QIcon:
    """The Vocalis app icon with every shipped size (the desktop picks the best)."""
    icon = QIcon()
    for size in ICON_SIZES:
        pixmap = QPixmap()
        if pixmap.loadFromData(_icon_bytes(f"vocalis-{size}.png")):
            icon.addPixmap(pixmap)
    return icon


def logo_pixmap() -> QPixmap:
    """The bare logo mark (no tile), for banners."""
    pixmap = QPixmap()
    pixmap.loadFromData(_icon_bytes(_MARK_NAME))
    return pixmap


def apply_branding(app: QApplication) -> None:
    """Give the whole application the Vocalis icon and menu identity."""
    app.setWindowIcon(app_icon())
    app.setDesktopFileName(APP_ID)
