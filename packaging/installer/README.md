# Vocalis installer

Run `bash install.sh` (or double-click/run `install.run` if you have
that file instead) from inside this folder. It will:

1. Check your machine is Debian/Ubuntu-based and 64-bit. An NVIDIA GPU
   driver is optional: without one the installer prints a `NOTE`, installs
   the processor-only `onnxruntime`, and skips the big Qwen3-TTS voice
   (see "Phases and options" below).
2. Install a few small system packages if they're missing (one password
   prompt).
3. Set up a private Python environment for Vocalis under
   `~/.local/share/vocalis/venv` (does not touch your system Python).
4. Download the voice/dictation models if they aren't already on your
   machine (a few gigabytes — this can take a while). This includes the
   Kokoro voice, two files in `~/.local/share/vocalis/models/kokoro/`
   (`kokoro-v1.0.onnx`, 325,532,387 bytes, and `voices-v1.0.bin`,
   28,214,398 bytes — about 350 MB together).

   Downloads are **resumable and fast**: every file is fetched in 16 MB
   pieces over four connections at once, into `*.part` files (with a
   `*.part.json` record of finished pieces). If the network drops, you
   click Stop, or the PC restarts, run the installer again (or click
   "Try again") and it continues where it stopped — only the piece being
   fetched is repeated. Failed pieces are retried automatically (8 tries,
   waiting 1, 2, 4… up to 30 seconds), and a stalled connection is given
   up after 30 seconds and retried. Finished files are checked (size, and
   SHA-256 where Hugging Face provides one) before being moved into
   place. Before starting, the installer prints the total size, how much
   is already downloaded, and checks there is enough free disk space.
   The setup window shows a real progress bar with live speed and time
   left (the script prints `PROGRESS done total bytes_per_s eta_s`
   lines); in a terminal you get a single live line. The Piper program's
   archive is cached in `~/.cache/vocalis/downloads/` while downloading
   (removed after unpacking; `uninstall.sh --purge` deletes the folder).
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

## Phases and options

The installer has three phases. `prepare` checks the computer, installs
system packages and sets up the private Python environment. `models`
downloads the voices and the dictation model plus the Piper program.
`finish` starts the background service and adds the menu entry.

```
bash install.sh                       # prepare, then the setup window (or the terminal if there is no screen)
bash install.sh --no-wizard           # everything in the terminal, no window
./install.run -- --no-wizard          # the same, from the .run file (note the extra --)
bash install.sh --phase prepare       # run one phase: prepare | models | finish | all
```

With no flags, `install.sh` runs `prepare` and then opens the setup
window when `DISPLAY` or `WAYLAND_DISPLAY` is set; otherwise it does
everything in the terminal. `--no-wizard` alone means `--phase all`.

The window is started with `vocalis-setup --installer-dir DIR`. That
command is internal: `install.sh` runs it for you, so you normally never
type it. It runs the `models` phase, a microphone test, then the `finish`
phase, and finally opens Vocalis Settings. Its exit code is `0` when
finished, `1` when stopped or a step failed, and `3` when no window could
be opened. If it exits with `3`, `126`, `127` or `128` or more (it could
not start, or crashed), `install.sh` finishes in the terminal instead.

While running, each phase prints progress lines the window understands:
`STEP i/n text` (step *i* of *n*) and `NOTE text` (an information
message, for example that no NVIDIA driver was found).

**No NVIDIA driver:** the installer prints a `NOTE`, installs the plain
(CPU) `onnxruntime` and carries on. The large Qwen3-TTS voice is also
skipped unless the graphics card has more memory than
`AutoPolicy.qwen3_min_gpu_total_mb` (4096 MB, so a 4 GB card skips it, like the app's Auto rule); Piper and Kokoro are still
installed.

**If setup stops or fails:** nothing is rolled back. Run the installer
again (`bash install.sh`, or `./install.run`); finished work is skipped. Wizard progress lines are also saved to
`~/.local/share/vocalis/install.log`.

To remove Vocalis, run `bash uninstall.sh` (see its own `--help`). This
also removes the applications-menu entry added above. `--purge`
additionally removes your config, the setup log, downloaded models, Piper and the Kokoro
voice folder.

The Kokoro Python package (`kokoro-onnx==0.4.7`) is installed with
`--no-deps` on purpose. Both `qwen-tts` and `kokoro-onnx` require plain
`onnxruntime`, which would replace the graphics-card version Parakeet needs.
Its other small dependencies come from Vocalis's own `pyproject.toml`, and
the installer then re-installs the GPU build last
(`pyenv_prefer_gpu_onnxruntime`) so it always wins.

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
