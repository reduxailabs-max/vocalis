# Vocalis installer

Run `bash install.sh` (or double-click/run `install.run` if you have
that file instead) from inside this folder. It will:

1. Check your machine is Debian/Ubuntu-based, 64-bit, and has a working
   NVIDIA GPU driver.
2. Install a few small system packages if they're missing (one password
   prompt).
3. Set up a private Python environment for Vocalis under
   `~/.local/share/vocalis/venv` (does not touch your system Python).
4. Download the voice/dictation models if they aren't already on your
   machine (a few gigabytes — this can take a while).
5. Set Vocalis up to start automatically whenever you log in, and start
   it right now.
6. Add a "Vocalis" entry to your desktop's applications menu (if this one
   step fails for some reason, it just prints a note and keeps going —
   Vocalis still works fine without it).

Once it's done, there are three ways to open Vocalis: click the small
round icon it puts in your system tray, choose "Vocalis" in your
applications menu, or run `vocalis --settings` from a terminal.

It's safe to run more than once — anything already correctly set up is
left alone, so re-running it also works as a repair command.

To remove Vocalis, run `bash uninstall.sh` (see its own `--help`). This
also removes the applications-menu entry added above.

If you're using the self-extracting `uninstall.run` instead, put `--`
before any flag meant for the uninstaller itself, because the `.run`
wrapper (makeself) otherwise treats the flag as its own and fails:
`./uninstall.run -- --purge` (not `./uninstall.run --purge`).

## Rebuilding the `.run` files (developers only)

```
bash build-run.sh
```

This builds the Vocalis wheel from the repo and produces
`install.run`/`uninstall.run` next to `install.sh`/`uninstall.sh`. Needs
`uv` and `makeself` installed. The `.run` files are generated artifacts —
never edit them directly or commit them; re-run this script instead.
