"""Smoke test confirming the vocalis package is importable and versioned."""
import vocalis


def test_package_exposes_version() -> None:
    assert isinstance(vocalis.__version__, str)
    assert vocalis.__version__ == "0.1.0"
