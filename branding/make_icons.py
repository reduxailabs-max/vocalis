"""Turn the rendered Remotion logo into the icon PNGs the app ships.

Usage (from the project root, after ``npm run render`` in branding/logo):

    QT_QPA_PLATFORM=offscreen python branding/make_icons.py

Reads  branding/logo/out/logo.png and branding/logo/out/mark.png
Writes src/vocalis/assets/icons/vocalis-<size>.png and vocalis-mark.png

The PNGs are committed, so building Vocalis does not need Node.js.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QImage

ROOT = Path(__file__).resolve().parent.parent
LOGO = ROOT / "branding" / "logo" / "out" / "logo.png"
MARK = ROOT / "branding" / "logo" / "out" / "mark.png"
OUT_DIR = ROOT / "src" / "vocalis" / "assets" / "icons"
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256, 512)
MARK_SIZE = 256


def _scaled(source: QImage, size: int) -> QImage:
    """Return ``source`` scaled smoothly to a square of ``size`` pixels."""
    return source.scaled(
        size, size, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation
    )


def _load(path: Path) -> QImage:
    """Load an image, failing with a clear message when it is missing."""
    image = QImage(str(path))
    if image.isNull():
        raise SystemExit(f"Cannot read {path}. Run 'npm run render' in branding/logo first.")
    return image


def main() -> int:
    """Write every icon size plus the bare mark."""
    _app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logo = _load(LOGO)
    for size in ICON_SIZES:
        _scaled(logo, size).save(str(OUT_DIR / f"vocalis-{size}.png"))
    _scaled(_load(MARK), MARK_SIZE).save(str(OUT_DIR / "vocalis-mark.png"))
    print(f"Wrote {len(ICON_SIZES) + 1} icons to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
