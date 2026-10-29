# Vocalis Stage 8 — Easy installer with a setup wizard (design)

Status: draft for approval. Parent design: `2026-09-11-vocalis-design.md`
(Stage 8 lines: ~1185, 1229, 1259, 1434, 1622). Things the user said are
marked **(user)**. Everything else is a call made by the designer and is
open to change.

## Intent and the why

Today Vocalis installs with a terminal script (`packaging/installer/install.sh`,
wrapped as `install.run`). It only works on Debian/Ubuntu-family Linux with an
NVIDIA card, prints text, and a non-programmer cannot easily tell what
happened or what to do on failure. The user wants: **"simple, interactive
install, then use" (user)**. A person double-clicks the installer, a window
guides them with a few Next-style pages, and when it finishes the normal
Vocalis window opens so they can configure everything there **(user: configure
after setup by opening the GUI)**.

## Goals

1. A GUI setup wizard, in the red-and-black look from `ui/theme.py` (reuse it).
2. Asks (almost) nothing. It detects the computer and chooses by itself:
   GPU vs no GPU, which engines and models to fetch, per the existing
   automatic-choice rules (strong GPU → Qwen3-TTS, GPU over 2 GB → Kokoro,
   no GPU → Kokoro or Piper).
3. **CPU-only computers are allowed** (no NVIDIA card must not block install).
4. Plain-English progress (progress bar + one-line status), no terminal needed
   to follow it; clear message and a "Try again" button when something fails.
5. A **microphone test page**: record about two seconds, show whether sound
   was heard (level meter + pass/warn text). Planned in Stage 5 as a wizard
   page; reuse the existing input-device pick (`pick_input_device`) and the
   clearer microphone problem messages.
6. Finish page: starts Vocalis (as today: autostart at login + started now)
   and opens the Settings window. A "Finish" button; nothing else to answer.

## Non-goals

- Windows, macOS, or non-Debian-family Linux. **(user: just Linux)**
- New voice engines, new models, or changing which models each hardware tier
  gets (6a decided that).
- A graphical uninstaller or updater. Re-running the installer stays the
  repair path; `uninstall.sh` stays as is.
- Asking the user to choose voices, speed, shortcuts, folders inside the
  wizard. They do that later in Settings. **(user)**

## Anti-goals (a failure even if it technically works)

- The wizard window needs a terminal to be understood.
- A failed download leaves a half-installed machine with no explanation.
- Re-running makes things worse (it must stay safe to run twice).
- The existing terminal installer stops working (scripts and CI users keep it).

## Constraints

- Keep `install.sh` / `install.run` / `build-run.sh` working; the wizard is a
  front end over the same steps, not a rewrite of what gets installed. Shell
  steps in `packaging/installer/lib/` (apt, pyenv, models, service, desktop,
  preflight) keep doing the real work; the Python parts (`fetch_models.py`) are
  reused.
- No hardcoded secrets. The one admin-password step (apt packages) must use a
  proper graphical password prompt, not store or echo the password.
- App makes no network calls after install (existing rule); only the installer
  downloads.
- Project rules: tests for new logic, ≥85% coverage, strict mypy, docstrings,
  docs updated in `docs/`, changelog in `.claude/`.

## Designer's calls (how it gets made — please check these)

- **Chicken-and-egg:** a Qt window needs PySide6 before it can show. So
  `install.sh` first does a short prepare phase (system packages, private
  Python environment, install the Vocalis package), printing one-line friendly
  messages, then starts the Qt wizard from that environment, which does the
  rest (models, service, desktop entry, mic test, finish). Alternative the
  builder may prefer if it is simpler: show a tiny `zenity`/`yad` progress
  during the prepare phase.
- **Wizard is a Python module in the package** (e.g. `vocalis.installer`,
  entry `vocalis-setup`) that runs the existing shell steps as subprocesses and
  reads their status lines for the progress bar. Keeping the logic (hardware
  decision, status parsing) in plain testable Python; the Qt pages stay thin.
- **CPU-only path:** relax `preflight.sh` (a missing NVIDIA driver becomes a
  note, not a stop) and install the CPU onnxruntime instead of the GPU one;
  skip the Qwen3-TTS model on machines with no suitable GPU.
- **Password prompt:** `pkexec` (graphical) for the apt step.
- **Where it launches from:** `install.run` still works from a terminal, and
  the wizard window opens from it. A double-click launcher path is nice to have,
  not required.
- **Pages (about 5):** Welcome → What I found (read-only summary of the chosen
  setup) → Installing (progress/log, Try again) → Microphone test → Done.

## Left to the builder

Exact page wording, progress parsing format, how the prepare phase reports
progress, how the hardware decision is shared with the app's own selector so the
two never disagree, and test strategy for the Qt pages (pytest-qt is already
used).

## Real-PC checks left for the user (after build, not on the NTFS dev drive)

Fresh install on a clean user account; a no-GPU computer (or forced CPU mode);
failure messages with the network unplugged; microphone test with a muted mic.

## Settled after the builder check (designer's calls)

1. **Hardware decision at install time** uses `nvidia-smi` (name and total
   memory), not torch (torch does not exist yet). Qwen3-TTS is fetched only if
   total GPU memory is at least `AutoPolicy.qwen3_min_gpu_total_mb` (4096 MB,
   `engines/auto_select.py`); Kokoro, Piper and Parakeet are always fetched.
   `fetch_models.py` gets flags (e.g. `--skip-qwen3`) so the decision is passed
   in; the app's own selector is not refactored. If a user later forces Qwen3 on
   a machine that skipped it, the existing engine fallback chain handles the
   missing model. CPU-only installs use the CPU onnxruntime and skip the CUDA
   torch/Qwen3 pieces where the packaging allows it.
2. **"Strong GPU" = 4096 MB total** (see above); nothing new is invented.
3. **`install.sh` stays the single source of truth.** It gains a
   machine-readable mode that prints lines such as `STEP 3/7 Downloading voices`
   and `PROGRESS 42`; the wizard runs `install.sh` in that mode and shows the
   lines. The plain terminal mode keeps today's text.
4. **Launch:** the prepare phase runs in the terminal that `install.run` opens
   (the self-extracting wrapper already opens one when double-clicked); no new
   `zenity`/`yad` dependency. The password prompt is `pkexec`. *This is the one
   call the user may want to change: the very first moments are a terminal, the
   wizard window appears after the prepare phase.*
5. **Order:** the Vocalis service is started on the Done page, after the
   microphone test, so the daemon does not hold the microphone during the test.
   "Open settings" runs the existing `vocalis --settings` path.
6. **Try again** re-runs the whole idempotent install (finished steps are
   skipped; downloads already complete are skipped by size). Closing the window
   mid-install leaves no rollback; the next page of the next run says "run the
   installer again to finish". No half state is hidden: the log path is shown.
7. **Microphone test is a warning, never a blocker.** No device or silence shows
   the existing plain-English microphone message and the user can still Finish.
8. PySide6 is already a hard dependency of the package. The builder verifies the
   Qt platform libraries (xcb) are in the apt list and adds them if missing.
9. `preflight.sh` changes: missing NVIDIA driver becomes a note; non-Debian
   family or non-64-bit PCs still stop with a plain message. Its tests are
   updated to the new behaviour.
