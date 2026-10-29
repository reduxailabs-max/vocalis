import pytest

from vocalis.installer.progress import ProgressEvent, parse_line


def test_parse_line_reads_a_step() -> None:
    assert parse_line("STEP 3/5 Installing models") == ProgressEvent("step", 3, 5, "Installing models")


def test_parse_line_reads_a_note() -> None:
    assert parse_line("NOTE Using the CPU only") == ProgressEvent("note", 0, 0, "Using the CPU only")


@pytest.mark.parametrize(
    "line",
    ["STEP x/y t", "STEP 0/3 t", "STEP 4/3 t", "STEP 1/3", "hello", "", "NOTE", "--- Step A: setup ---"],
)
def test_parse_line_ignores_malformed_lines(line: str) -> None:
    assert parse_line(line) is None


def test_parse_line_keeps_text_with_colons() -> None:
    event = parse_line("STEP 1/2 Download: model a: b")
    assert event is not None
    assert event.text == "Download: model a: b"


def test_parse_line_never_raises_on_huge_numbers() -> None:
    assert parse_line("STEP " + "9" * 5000 + "/3 t") is None
