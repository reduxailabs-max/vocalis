"""Lightweight console-script entrypoint for `vocalis`.

Deliberately imports nothing from vocalis.cli (or anything it pulls in --
torch, transformers, PySide6) at module load time, so that `vocalis
--settings` (asking an already-running Vocalis to show its window) stays
fast even though the full app is heavy. The heavy vocalis.cli.main is
imported lazily, only when no fast-path flag was given.
"""
import argparse
import sys

from vocalis.remote import default_socket_path, open_settings


def entrypoint(argv: "list[str] | None" = None) -> None:
    """The `vocalis` console command: run the app, or with --settings open its window."""
    parser = argparse.ArgumentParser(prog="vocalis", description="Offline voice AI workstation.")
    parser.add_argument(
        "--settings", action="store_true",
        help="open the Settings window of the running Vocalis (starting it if needed)",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.settings:
        sys.exit(open_settings(default_socket_path()))
    from vocalis.cli import main  # heavy import, deferred until actually needed
    main()
