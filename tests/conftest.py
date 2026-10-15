"""Shared pytest configuration: run PySide6 without a real display.

Set before any PySide6 import happens anywhere in the test suite (a
conftest.py at the tests/ root is imported by pytest before collecting
any test module), so every test — not just the ones that mention Qt —
can safely construct widgets in CI or on a headless machine.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
