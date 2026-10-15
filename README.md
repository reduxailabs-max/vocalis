# Vocalis

An offline voice AI workstation: dictation, a quick-reader for anything
you've copied, a document reader (with pause/resume), and an OCR reader
that speaks text straight off your screen. Once installed, it runs on
your own machine — no cloud service and no account. The installer
downloads the voice models, the Piper speech program, `uv`, and a few
system packages once; the app itself makes no network calls.

## Installing

See [`packaging/installer/README.md`](packaging/installer/README.md) for
the one-command installer (works on Debian/Ubuntu-family Linux with an
NVIDIA GPU).

## Using it

The full guide to every feature, the tray icon, the floating widget, and
Settings lives in [`docs/user-docs.md`](docs/user-docs.md).

## Contributing / developing

Setup, running the test suite, and project conventions are in
[`docs/developer-docs.md`](docs/developer-docs.md). The design is in
[`docs/architecture.md`](docs/architecture.md), and the folder layout is in
[`docs/folder-structure.md`](docs/folder-structure.md).
