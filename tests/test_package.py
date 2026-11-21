"""Smoke test confirming the vocalis package is importable and versioned."""
import vocalis


def test_package_exposes_version() -> None:
    assert isinstance(vocalis.__version__, str)
    assert vocalis.__version__ == "1.0.1"
