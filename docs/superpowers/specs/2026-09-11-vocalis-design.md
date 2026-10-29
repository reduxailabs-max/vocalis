# Vocalis — Offline Voice AI Workstation — Design Spec

**Status:** Approved by user, 2026-09-11.
**Produced via:** superpowers:brainstorming (architectural path).

## Context

The user runs a personal Voice-AI setup on Kali Linux (XFCE, X11, PipeWire,
RTX 4060 Laptop 8GB) made of a compiled `voxtype` STT binary plus a handful of
loose bash scripts (`read-selected.sh`, `ocr-read-screen.sh`, etc.), all now
moved into `old/` as a read-only behavioral reference. These worked but were
fragmented, hard to maintain, and not packageable for other machines.

**Outcome:** one maintainable Python application named **Vocalis**, built
fresh in this repo root, that replaces the old binary and scripts entirely,
unifying four voice features into a single stable background app with a
tray icon, floating widget, and settings window. Fully offline/local, GPU
used only while actively working (idle-released), starts on boot and
self-heals, and ships as a `.deb` + AppImage so it can be installed on any
Linux machine without the end user touching Python.

## Hard Constraints (non-negotiable)

1. **Fully offline / local** — no cloud APIs. All models run on-device.
2. **GPU load-on-use / release-on-idle** — a model loads to VRAM only at the
   moment of transcription or speech generation and is unloaded a few seconds
   after work stops. Ceiling ~5–6GB during active work; **~0 GPU when idle**.
   Only **one** AI model resident in VRAM at any time.
   **Amendment (confirmed during Phase 2 research, 2026-09-11):** once the
   process has used the GPU at all, CUDA/cuDNN keep a small, fixed context
   footprint alive for the life of the process — measured ~136MB on this
   machine (15MB idle baseline → 151MB after loading and fully "unloading" a
   model) — that unloading the model cannot release; only exiting the
   process does. This is an ONNX Runtime/CUDA characteristic, not a gap in
   Vocalis's own idle-unload logic (which still correctly frees the bulk of
   each model's memory, e.g. ~260MB of a ~400MB resident Parakeet session in
   the same measurement). User-approved: accept this fixed ~150MB floor
   rather than adding subprocess-per-engine isolation to chase true zero —
   still ~2% of the 8GB budget. Every future phase that wires in a real GPU
   engine (Qwen3-TTS, fish-speech) inherits this same floor; do not re-flag
   it as a bug.
3. **Stable, persistent, starts on boot** — runs as a systemd `--user` service
   with auto-restart + backoff, watchdog health checks, and `journalctl`
   logging. Per-feature error isolation so one bad request never crashes the
   daemon.
4. **Notifications on by default** — Dictation and both Readers must show
   desktop notifications for **start**, **live text**, and **done** states
   (toggleable in Settings).
5. **`old/` is read-only reference** — never modified; the new app lives in
   the repo root and supersedes it.
6. **Governance rules apply** (project `CLAUDE.md`): docstrings on every
   function, structured error handling, input validation, structured logging
   (no sensitive values), `docs/` kept in sync, ≥85% test coverage, KISS/YAGNI/
   SOLID, no hardcoded secrets, changelog per major change at
   `.claude/<timestamp>_CHANGELOG.md`, one-sub-task-at-a-time with user
   approval, conventional-commit messages.

## Product Summary

**Vocalis** — one always-running background app with four shortcut-driven
features:

| # | Feature | Trigger (customizable) | Engine | Direction |
|---|---------|------------------------|--------|-----------|
| 1 | **Dictation (STT)** | Hold **NUMLOCK** | **Parakeet** (parakeet-tdt-0.6b, ONNX/CUDA) | mic → text at cursor (clipboard fallback if typing fails) |
| 2 | **Quick Reader (TTS)** | **Ctrl+Alt+R** | **Qwen3-TTS** | selected/copied text → speech |
| 3 | **Document Reader (TTS)** | **Ctrl+Alt+D** → popup | **Qwen3-TTS** ("Ryan"), Piper fallback (see Phase 4 amendment below) | file (.md/.txt/.pdf/.docx/.html) or pasted text → speech |
| 4 | **OCR Reader** | **Ctrl+Alt+O** → area select | **Tesseract** → Qwen3-TTS | screen region → text → speech |

Three UI surfaces: **tray icon** (status + controls + Settings), **floating
mini-widget** (appears only while reading; Pause/Stop/progress), **Settings
window** (config management + widget on/off only — deliberately minimal).

## Architecture

Single long-lived **daemon process** (the systemd service) containing:

- **Hotkey layer** — `NUMLOCK` push-to-talk via **python-evdev** (needs
  `input` group, already granted). `Ctrl+Alt+R/D/O` registered as
  desktop-environment shortcuts (XFCE now; GNOME/KDE supported) that invoke a
  thin `vocalis` CLI, forwarding to the running daemon over a local Unix
  socket. All bindings editable in Settings.
- **Model Manager** — the heart of the GPU policy. Lazy-loads exactly one
  model on demand, serves the request, starts an idle timer, and unloads
  after the idle window (default 30s, configurable). Guarantees single-model
  residency and idle-time VRAM release.
- **Feature handlers** (one module each, single responsibility): dictation,
  quick-reader, document-reader, ocr. Each isolates its own errors and
  reports human-readable failures.
- **Engine wrappers** behind a common interface: `ParakeetSTT`, `Qwen3TTS`,
  `PiperTTS`, `TesseractOCR` — swappable, individually testable. (No
  `FishSpeechTTS`: dropped for Document Reader, see the Phase 4 amendment
  below.)
- **Text pipeline** — extractors (pdf/docx/html/md/txt) + normalizer
  (markdown → clean speakable text).
- **Audio** — mic capture via `sounddevice`; playback via
  PipeWire/PulseAudio (streaming/chunked).
- **Output** — typing at cursor via **ydotool**, clipboard fallback via
  `xclip`/`wl-copy`.
- **Notifier** — desktop notifications for start/live/done.
- **UI** — **PySide6 (Qt6)** tray icon, floating always-on-top widget,
  settings window, and the Document Reader popup.
- **Config** — `pydantic`-validated file at `~/.config/vocalis/config.toml`.

## Tech Stack

- **Python 3.13**, **PySide6 (Qt6)**, **mypy** (strict) + **pydantic** +
  **pytest** (≥85%), **onnxruntime-gpu** (Parakeet), **Qwen3-TTS**,
  **Piper**, **PyTorch/CUDA**, **Tesseract** (`pytesseract`),
  **python-evdev**, **sounddevice**, **ydotool**, **xclip/xsel/wl-clipboard**,
  **systemd --user**. (**fish-speech-1.5** from the original plan is
  dropped for Document Reader — see the Phase 4 amendment below.)

## Project Structure (repo root)

```
src/vocalis/
├── daemon.py              # main service loop, socket server, watchdog
├── config.py              # pydantic config models + load/save
├── model_manager.py       # lazy load / idle-unload / single-residency
├── hotkeys/                # evdev NUMLOCK + DE-shortcut registration
├── features/               # dictation.py, quick_reader.py, doc_reader.py, ocr.py
├── engines/                # parakeet.py, qwen3_tts.py, fish_speech.py, tesseract_ocr.py
├── text/                   # extractors/, normalizer.py
├── audio/                  # capture.py, playback.py
├── output/                  # typer.py (ydotool), clipboard.py
├── notify.py               # desktop notifications
├── ui/                      # tray.py, floating_widget.py, settings_window.py, doc_popup.py
└── cli.py                  # `vocalis` command → talks to daemon socket
tests/                      # pytest suite (unit + integration), ≥85% coverage
docs/                        # architecture, folder-structure, file-purpose, workflow,
                             #   algorithms, user-docs, developer-docs (governance)
packaging/                   # deb/ + appimage/ build scripts, systemd unit, .desktop
config/                       # default config.toml, model manifest (download URLs/hashes)
pyproject.toml
README.md
old/                          # untouched legacy reference (do not modify)
```

## Implementation Milestones

Each milestone gets its own detailed implementation plan (see
`docs/superpowers/plans/`), written just before that milestone starts:

- **Phase 0 — Scaffold & governance skeleton**
- **Phase 1 — Model Manager + GPU policy** (built first: every feature
  depends on it, and it enforces the core safety rule)
- **Phase 2 — Dictation (STT)**. Engine confirmed via direct feasibility
  testing (2026-09-11): the `onnx-asr` PyPI package
  (`onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", <dir>,
  quantization="int8", providers=[...])`) loads the exact existing Voxtype
  Parakeet ONNX files at `~/.local/share/voxtype/models/parakeet-tdt-0.6b-v3-int8/`
  directly — no conversion or re-download needed. GPU (CUDA) inference
  requires the `nvidia-cudnn-cu12` pip package with its `libcudnn*.so.9`
  files preloaded via `ctypes.CDLL(..., mode=RTLD_GLOBAL)` before the CUDA
  session is created (cuDNN is not bundled with `onnxruntime-gpu` and isn't
  present as a system package on this machine). Verified end-to-end on both
  CPU and GPU providers.
- **Phase 3 — Quick Reader (TTS)**. Design confirmed via
  superpowers:brainstorming, 2026-09-24 (see the Phase 3 amendment below).
- **Phase 4 — Text pipeline + Document Reader**
- **Phase 5 — OCR Reader**
- **Phase 6 — UI** (tray, floating widget, settings window)
- **Phase 7 — Service & autostart** (systemd, cross-DE shortcuts)
- **Phase 8 — Packaging** (`.deb` + AppImage)
- **Phase 9 — Testing & docs hardening**

## Verification

- **Per feature (manual)**: NUMLOCK dictation types correctly (clipboard
  fallback on typing failure); Ctrl+Alt+R reads a selection; Ctrl+Alt+D popup
  reads a file/pasted text; Ctrl+Alt+O reads a screen region; all show
  start/live/done notifications.
- **GPU policy (measured)**: `nvidia-smi -l 1` shows VRAM rising only during
  active work, returning to idle within the idle window; never two models
  loaded at once; peak ≤ ~6GB.
- **Stability**: `systemctl --user status vocalis` active; kill process →
  auto-restart confirmed; bad-input cases (corrupt PDF, no mic, empty
  selection) produce a notification, not a crash.
- **Automated**: `pytest` green at ≥85% coverage; `mypy` clean.
- **Packaging**: `.deb` installs cleanly in a fresh container/VM; AppImage
  runs on a non-Debian distro; first-run model download + first successful
  read both work.

## Risks & Mitigations

- **STT regression vs. current Voxtype** → reuse the exact Parakeet model
  files, validate side-by-side before retiring the old binary, keep `old/`
  as instant rollback.
- **VRAM pressure on 8GB** → single-residency + idle-unload enforced by
  Model Manager, verified with `nvidia-smi` in Phase 1.
- **Global-shortcut portability** → evdev-based detection for both NUMLOCK
  and Ctrl+Alt+R (see Phase 3 amendment below — superseded the original
  DE-specific-registration plan for Ctrl+Alt+R specifically); all editable
  in Settings.
- **fish-speech license (CC-BY-NC-SA)** → dropped for Document Reader in
  favor of the already-installed Qwen3-TTS/Piper pair (see Phase 4
  amendment below); no longer a live risk unless a future phase reconsiders
  fish-speech.

## Phase 3 amendment (confirmed via superpowers:brainstorming, 2026-09-24)

Design for **Quick Reader (TTS)**, refined during brainstorming after
checking `old/read-selected.sh` (the actual working legacy behavior) and
researching Qwen3-TTS's real installation requirements:

1. **Primary engine: Qwen3-TTS, smaller (0.6B / ~1.8GB) CustomVoice
   variant, preset voice `"Ryan"`.** Corrects an earlier assumption made
   during brainstorming: Qwen3-TTS's "Base" variant is actually a
   voice-*cloning* model — it requires a reference audio clip (and
   normally a transcript of it) and has no plain "just read this in a
   default voice" mode. **CustomVoice** is the variant that actually
   fits Quick Reader: it ships 9 built-in preset voices selectable by
   name, no reference audio needed, via
   `model.generate_custom_voice(text=..., language="English",
   speaker="Ryan")`. None of the 9 presets (Ryan, Aiden, Vivian, Serena,
   Uncle_Fu, Dylan, Eric, Ono_Anna, Sohee) are confirmed to have a
   British accent — the set leans American/Chinese-accented — but
   `"Ryan"` is the one documented as suited to "a clear male read," the
   closest match to a narrator-style voice among the two English
   presets (the other, `"Aiden"`, is explicitly American). User
   confirmed this choice after checking the two closest options; a
   different preset (or a future Piper British-English voice model, as
   Piper does have proper UK voices available) can be swapped in later
   via config without any code change, since the speaker name is a
   config value, not hardcoded. Confirmed as a real, open-source,
   pip-installable package (`qwen-tts`, from Alibaba's Qwen team).
   CUDA-only — no CPU/MPS support, consistent
   with the existing single-GPU-residency policy. The smaller variant was
   chosen over the 3.9GB one to keep load time and VRAM footprint down for
   a feature meant to feel instant. Borrowed through the existing
   `ModelManager` exactly like Parakeet — no changes needed to that policy.

2. **Fallback engine: Piper.** `old/read-selected.sh` reveals the actual
   legacy tool used **Piper**, not Qwen3-TTS/fish-speech as originally
   assumed in this spec's Product Summary table — and Piper (binary +
   `en_US-lessac-medium.onnx` voice) is already installed at
   `~/.local/bin/piper/` and `~/.local/share/piper-voices/`, proven working.
   Piper runs on CPU only (no GPU/VRAM involvement at all), making it a
   genuinely safe fallback that can never violate the single-GPU-residency
   rule. `PiperTTS` wraps the existing `piper` binary via subprocess, the
   same pattern `TextOutput`/`Notifier` already use for `ydotool`/`xclip`/
   `notify-send`. If Qwen3-TTS fails to load or generate for any reason,
   Vocalis automatically falls back to Piper and notifies the user, the
   same pattern `TextOutput` already uses for ydotool→clipboard.

3. **Text acquisition: clipboard only, Ctrl+C required first** (kept
   identical to the legacy script's behavior, by explicit user choice —
   not the more "automatic" PRIMARY-selection approach that was also
   considered). Empty clipboard → notify "Nothing copied" and stop.

4. **Hotkey: evdev, not DE-shortcut registration.** Supersedes this spec's
   original plan (Architecture section, Risks & Mitigations) to register
   Ctrl+Alt+R as an XFCE/GNOME/KDE desktop shortcut invoking the `vocalis`
   CLI over a socket. Instead, a new listener (parallel to
   `NumlockListener`) watches raw evdev key events for Ctrl+Alt+R directly
   — same cross-desktop-portable, no-manual-setup, non-suppressing
   approach already built and proven for NUMLOCK in Phase 2, including
   reusing the ydotoold-virtual-device exclusion Task 10 added. Chosen for
   consistency (one hotkey-detection mechanism in the codebase, not two)
   over the original plan's Settings-native feel.

5. **Playback: `sounddevice`, not a `pw-play` subprocess.** Supersedes the
   legacy script's approach of piping raw PCM into the `pw-play` CLI tool.
   `sounddevice` already talks to PipeWire (it's how `MicRecorder` captures
   audio in Phase 2) and is far easier to interrupt cleanly and to unit
   test with an injectable fake, matching `MicRecorder`'s existing
   `stream_factory` pattern.

6. **Press-again-to-stop.** Pressing Ctrl+Alt+R while a reading is already
   in progress stops that playback immediately (notifying "Stopped")
   instead of starting a second one on top or being ignored — the user
   explicitly chose this over Dictation's existing "ignore the duplicate"
   pattern, since interrupting a long or wrongly-selected reading is a
   real, valuable use case here that has no equivalent in Dictation.

**New files (mirrors Phase 2's layout):**
`src/vocalis/engines/qwen3_tts.py`, `src/vocalis/engines/piper_tts.py`,
a new evdev-based Ctrl+Alt+R listener under `src/vocalis/hotkeys/`,
`src/vocalis/audio/playback.py`, `src/vocalis/features/quick_reader.py`,
plus the same `cli.py`/`config.py` wiring pattern Phase 2's Task 7 used.

**Testing:** same fakes-everywhere approach as Phase 2 — fake clipboard
reader, fake Qwen3-TTS/Piper engines, fake playback stream, fake evdev
device for the new listener. ≥85% coverage, strict mypy, no real
hardware needed for the automated suite. Real speech output is the one
thing only a manual check (holding nothing, just listening) can confirm,
same as Phase 2's real-microphone caveat.

## Phase 4 amendment (confirmed via superpowers:brainstorming, 2026-09-26)

Design for **Document Reader (TTS)**, refined during brainstorming after
Phase 3's merge to master. The user chose to continue the roadmap in order
rather than jump ahead to the GUI/installer phases.

1. **Voice: reuse Qwen3-TTS ("Ryan") with Piper fallback — fish-speech is
   dropped.** Supersedes the Product Summary table's original
   `fish-speech-1.5` entry for this feature. fish-speech is not installed
   on this machine, would pull in a large new dependency tree with the same
   kind of packaging-conflict risk already hit once with `onnxruntime` vs
   `onnxruntime-gpu` during Phase 3's merge, and is CC-BY-NC-SA
   (non-commercial only). Reusing the engines already built in Phase 3
   (`src/vocalis/engines/qwen3_tts.py`, `piper_tts.py`) means nothing new to
   download and a proven fallback path
   (`QuickReaderFeature._synthesize_with_fallback`,
   `src/vocalis/features/quick_reader.py:190`) to copy. The user chose this
   over installing fish-speech for an A/B listening comparison, since the
   existing voice already tested well in Phase 3.

2. **Popup: a real PySide6 (Qt6) window, not a throwaway `zenity` dialog.**
   This deliberately brings the PySide6 dependency forward from Phase 6
   (UI) rather than adding a second, disposable UI technology. The window
   offers "Open file…", a paste-text box, a Read button, Pause/Resume and
   Stop buttons, and a "piece N of M" progress label. It is deliberately
   scoped to Document Reader's own needs — becoming the tray icon, the
   floating mini-widget, and the Settings window remains Phase 6's job, not
   this one's.

3. **Controls: Stop plus Pause/Resume (no seek/skip in this phase).**
   Pause stops audio immediately; Resume replays the current chunk from
   its start (simpler than tracking a mid-chunk offset, and each chunk is
   short — see chunking below). Stop, closing the popup window, or
   pressing Ctrl+Alt+D again while a reading is in progress all stop it,
   mirroring Quick Reader's Task 6 press-again-to-stop behavior
   (`quick_reader.py:83`). Pressing Ctrl+Alt+D while idle (nothing being
   read) raises/focuses the existing popup instead of opening a duplicate
   window. Paragraph-level skip/seek is deferred — YAGNI until a real need
   for it shows up.

4. **Architecture: Qt event loop replaces the current blocking wait in the
   daemon process (Approach A) — no second process.** Today,
   `src/vocalis/cli.py:main` blocks in `_block_until_interrupted()` while
   the hotkey listener threads do the real work. That function is replaced
   by running the PySide6 `QApplication` event loop on the main thread
   instead; the daemon socket server and both hotkey listeners keep
   running on their own background threads exactly as today. Hotkey
   threads never touch Qt widgets directly — they emit Qt signals (which
   are queue-and-redeliver-safe across threads by Qt's own design) that the
   popup connects to. This was chosen over running the popup in a second
   process talking to the daemon over the existing Unix socket: a second
   process would mean two things to start, watch, and eventually package,
   for no benefit the user needs in this phase — the daemon process is
   already the trusted single owner of hotkeys, the Model Manager, and
   playback.

5. **One reading at a time, shared with Quick Reader.** A small
   `ReadingCoordinator` (`src/vocalis/features/reading_coordinator.py`) is
   introduced so starting a Document Reader reading stops a Quick Reader
   reading in progress, and vice versa — today they're independent and
   could otherwise play over each other. It follows the same lock-only,
   never-block-while-locked discipline already hard-won in Phase 3 Task 6
   (`quick_reader.py:99`, `:119`): the coordinator only ever guards a plain
   flag/reference swap, never a call into `AudioPlayer` or the notifier.

**New components (one job each, mirroring the existing per-feature
module-per-responsibility pattern):**

- `src/vocalis/text/extractors.py` — one function per format: `.txt`/`.md`
  read directly (UTF-8, replacing undecodable bytes rather than raising);
  `.pdf` via `pypdf`; `.docx` via `python-docx`; `.html` via the stdlib
  `html.parser` (script/style tags skipped). Typed errors:
  `UnsupportedFileTypeError`, `NoReadableTextError` (e.g. an image-only
  scanned PDF with no text layer — the error message points at the future
  Phase 5 OCR Reader instead of trying to do OCR itself here), `DocumentOpenError`
  (corrupt/unreadable file), `DocumentTooLargeError` (extracted text over
  ~5MB, a sanity ceiling, not a real limitation of the design).
- `src/vocalis/text/normalizer.py` — turns raw markdown into speakable
  text: strips heading markers/emphasis markers/code fences, replaces a
  markdown link with just its label text, collapses repeated whitespace.
- `src/vocalis/text/chunker.py` — splits normalized text into sentence-
  bounded chunks capped around 300 characters (never mid-word), so
  Document Reader can start speaking, show progress, and pause/resume at
  reasonable granularity without waiting for the whole document to
  synthesize first.
- `src/vocalis/hotkeys/ctrl_alt_key.py` — generalizes the existing
  `CtrlAltRListener` (`src/vocalis/hotkeys/ctrl_alt_r.py`) to watch a
  configurable third key instead of hardcoding `R`, preserving its
  hard-won lessons (per-callback exception isolation, excluding the
  `"ydotoold virtual device"` from enumeration, no key suppression).
  `CtrlAltRListener` becomes a thin instance of this for Ctrl+Alt+R;
  existing Phase 3 tests are expected to keep passing unchanged. A new
  Ctrl+Alt+D binding is added on top of it for Document Reader.
- `src/vocalis/features/reading_coordinator.py` — the one-reading-at-a-time
  guard described in point 5.
- `src/vocalis/features/doc_reader.py` — `DocumentReaderFeature`: owns
  extraction → normalization → chunking → synthesis → playback, chunk by
  chunk, preparing chunk N+1's audio while chunk N is still playing so
  there's no gap between chunks. Reuses the Phase 3 generation-counter
  pattern (`quick_reader.py`'s `_generation`) to keep a stale background
  thread from clobbering a newer reading's state, and reuses
  `AudioPlayer`/`PlaybackTimeoutError` as-is
  (`src/vocalis/audio/playback.py`) so a stuck sound device still surfaces
  as "Playback failed" instead of hanging, exactly like Quick Reader.
  Falls back from Qwen3-TTS to Piper on any synthesis failure — once
  fallback triggers, it stays on Piper for the rest of that document
  (rather than retrying Qwen3-TTS every chunk) and notifies the user once,
  not once per chunk.
- `src/vocalis/ui/doc_popup.py` — the PySide6 popup window described in
  point 2. Talks to `DocumentReaderFeature` only through plain
  callbacks/Qt signals — the feature module has no PySide6 import at all,
  keeping it independently testable exactly like every other feature
  module.
- `src/vocalis/cli.py` / `config.py` — `build_doc_reader(config,
  model_manager, reading_coordinator)` following the existing
  `build_quick_reader` pattern; `main()` gains the `QApplication` event
  loop (point 4) in place of `_block_until_interrupted()`, keeping the
  existing try/finally start/stop-order guarantee for every component.

**Error messages surfaced to the user** (as desktop notifications and/or
the popup's status line, following the existing Notifier pattern):
unsupported file type (names the supported list), no readable text found
(points at the future OCR Reader), file damaged/unreadable, "Nothing to
read" (empty paste + no file chosen), document too large, backup voice in
use (once, per document), "Playback failed" (unchanged from Phase 3).

**Testing:** the same fakes-everywhere approach as Phases 2–3 (fake
engines, fake `AudioPlayer`, fake notifier), plus small real fixture files
under `tests/fixtures/` for each supported format (including one corrupt
file and one image-only PDF to exercise the error paths for real rather
than through a fake). The PySide6 popup is tested with `pytest-qt` under
`QT_QPA_PLATFORM=offscreen` (no real display needed in CI or for the
automated suite). ≥85% coverage and strict mypy stay the bar. The one
thing only the user can verify manually is real audio + a real popup:
Ctrl+Alt+D → open a real PDF → hear it → Pause → Resume → Stop.

**Build order** (becomes the task list for the Phase 4 implementation
plan): (1) normalizer + chunker, (2) extractors, (3) generalized Ctrl+Alt
key listener, (4) reading coordinator wired into the existing Quick
Reader, (5) `DocumentReaderFeature`, (6) PySide6 popup, (7) `cli.py`/
`config.py` wiring including the new Qt main loop, (8) docs + changelog.

**New dependencies:** `PySide6`, `pypdf`, `python-docx`, and (dev-only)
`pytest-qt`. Given the real onnxruntime/onnxruntime-gpu packaging conflict
already hit once during Phase 3's merge (two packages sharing one
site-packages namespace, last-installed wins), every dependency
install/update in this phase is followed by re-checking
`onnxruntime.get_available_providers()` still lists the CUDA provider
before continuing.

## Phase 5 amendment (confirmed via superpowers:brainstorming, 2026-10-04)

Phase 5 adds the OCR Reader: press **Ctrl+Alt+O**, drag a box around any
text on screen, and Vocalis reads that text aloud. The original design
sketched this as "Tesseract -> Qwen3-TTS" with its own reading code; the
decisions below replace that sketch.

### Decisions

1. **Scope: drag-a-box only.** Reading image files or scanned PDFs from
   disk is out of scope (YAGNI). Phase 4's "no text layer" PDF error keeps
   pointing at this feature's existence but the file path is a later
   request. English only: only the `eng` Tesseract language pack is
   installed and extra languages were declined for this phase.
2. **Extras kept:** copy the recognised text to the clipboard; stop/pause/
   resume while reading. Everything else is out.
3. **Models unchanged.** STT: Parakeet TDT 0.6B v3 int8. TTS: Qwen3-TTS
   0.6B CustomVoice ("Ryan"), Piper `en_US-lessac-medium` as CPU fallback.
   OCR: system Tesseract, CPU only, never routed through `ModelManager`
   (it never touches the GPU, so single-residency is unaffected).
4. **Speak by reusing the Document Reader.** The OCR feature only
   captures, recognises and copies. It then calls the existing
   `DocumentReaderFeature.read_text(text)`, which already provides
   synthesis with fallback, chunking, pause/resume/stop, the popup, and
   `ReadingCoordinator` ownership. No new reading-aloud code is written
   and Quick Reader is untouched.
5. **Screen-region capture tries tools in order:** `spectacle -r -b -n -o <temp/file>`
   first, because the box can be resized by dragging its edges and corners
   before confirming (user requirement: the box must be adjustable).
   **Hands-on amendment (2026-10-04):** Spectacle is preferred over Flameshot
   after testing on this machine (X11 XFCE) revealed that Flameshot hangs
   without showing its selection screen, blocking the OCR feature for up to
   120 seconds. Spectacle's picture goes through a private temp directory
   (`prefix="vocalis-ocr-"`, mode 0700) that is deleted on completion; the
   other tools keep the picture in memory (stdout). Then `maim -s -u` (draw
   once, no resize), then `flameshot gui --raw` (draw once, tried third
   because it may hang on some systems), then `slurp` + `grim` (Wayland,
   draw once). The first one installed is used. Escape / closing the tool /
   zero-size selection counts as "cancelled". A custom in-app overlay is
   deferred to Phase 6 (UI). The user is told once, in the docs, that
   resizing needs Spectacle.

### Flow

Ctrl+Alt+O (evdev key code 24, via the existing `CtrlAltKeyListener`) ->
`OcrReaderFeature.on_trigger()`:

- If the Document Reader is currently reading text that **this feature
  started**, the press stops it (`DocumentReaderFeature.stop()`).
- If a capture is already in progress, the press is ignored.
- Otherwise a background thread runs: capture region -> PNG bytes ->
  Tesseract (whitespace collapsed) -> clipboard write -> notification
  preview -> popup shown -> `read_text()` (which normalises the text). The interactive capture
  never runs on the hotkey watcher thread, which must stay responsive and
  stoppable.
- Spectacle's picture uses a private temp directory that is deleted
  immediately; other tools keep the picture in memory (no temporary file),
  so there is nothing to clean up.

Error handling, each a notification and never a daemon crash: capture
cancelled ("Cancelled"), no capture tool installed (names the tools to
install), Tesseract missing or failing (clear message), no text found
("No text detected", also when fewer than 3 characters are recognised, as
the old script did), clipboard write failing (logged, reading continues).

### New code

| File | Purpose |
|---|---|
| `output/screen_capture.py` | `ScreenRegionCapture`: pick a region with the first available tool, return PNG bytes or `None` when cancelled; injectable command runner. |
| `engines/tesseract_ocr.py` | `TesseractOCR.recognize(png_bytes) -> str` via the `tesseract` binary on stdin/stdout (`-l eng`), timeout, injectable runner. |
| `output/clipboard.py` | Gains a write-side `ClipboardWriter` next to `ClipboardReader` (xclip, same runner seam). |
| `features/ocr_reader.py` | `OcrReaderFeature`: orchestrates the flow above. |
| `cli.py` | `build_ocr_reader()`; listener started/stopped with the others; shares the existing `DocumentReaderFeature` and coordinator. |

`ShortcutsConfig.ocr_reader` ("Ctrl+Alt+O") already exists in config.

### Testing

Fakes everywhere, as in Phases 2-4: fake command runners for the capture
tools, Tesseract and xclip; a fake Document Reader recording `read_text`
and `stop` calls. Cover: tool fallback order, cancel, no tool, Tesseract
missing/timeout, empty and too-short text, clipboard failure, the
stop-on-second-press rule, ignore-during-capture, tool command format,
temp directory cleanup on all paths, and that the capture never runs on
the caller's thread. Gates: >=85% coverage, `mypy` zero errors, zero test
warnings. A real-hardware check (drag a box over text and listen) closes
the phase.

### Planning deviations (decided while writing the Phase 5 plan)

The text above was written before planning; the implementation differs in
three places:

1. **Spectacle uses a private temp directory; other tools use memory.**
   Spectacle writes via a file (using a temp directory prefix `vocalis-ocr-`
   and mode 0700, deleted immediately on every path). Other capture tools
   (maim, flameshot, slurp+grim) output their PNG to stdout, piped directly
   to Tesseract, so no temporary image is written and none needs removing.
2. **Normalisation is done inside `DocumentReaderFeature.read_text`.**
   `read_text()` already runs `normalize_markdown()`, so the OCR feature
   only collapses whitespace (like the old script's `tr -s`); there is no
   separate normalise step.
3. **The popup is shown so Pause is reachable.** Whenever an OCR reading
   starts, the Document Reader window opens with the text and Pause/Stop
   enabled (via the popup's `external_reading_started` signal).

## Phase 6 amendment (confirmed via superpowers:brainstorming, 2026-10-04)

Phase 6 adds the three UI surfaces from the original design — a tray icon,
a floating mini-widget and a Settings window — all deliberately simple.
The decisions below were confirmed with the user.

### Decisions

1. **Scope: all three, kept simple.** (User chose this over tray+settings
   only or settings only.) The XFCE panel offers a system tray
   (`QSystemTrayIcon.isSystemTrayAvailable()` is True on the target machine).
2. **Settings window contents:** notifications on/off, floating widget
   on/off, the four keyboard shortcuts, and the TTS voice (the nine
   Qwen3-TTS CustomVoice presets: serena, vivian, uncle_fu, ryan, aiden,
   ono_anna, sohee, eric, dylan). **Not** included now (user deferred):
   audio output device, speaking speed, streaming start, voice quality.
3. **Live vs restart.** The two switches and the voice apply immediately
   (voice: on the next reading; a reading in progress finishes in its old
   voice). Shortcut changes apply after a restart; the window says so.
   Re-creating the evdev listener threads at runtime is rejected as risky.
4. **One shared settings holder** (`settings.py`, `LiveSettings`):
   created from the config file at startup; the Settings window is the only
   writer; features read it at the moment of use (notifier on/off, widget
   on/off, voice) instead of copying values at construction. Saving is
   atomic (write a temp file next to the config, then `os.replace`) so a
   crash cannot leave a broken file. Saving failure shows the error in the
   window and keeps the previous settings.
5. **Shortcuts are really read.** Today the `[shortcuts]` config values are
   ignored (key codes are hard-coded in `cli.py`). A new
   `hotkeys/shortcut_parser.py` turns `"Ctrl+Alt+R"` into an evdev key code
   and `numlock_key` into a key code; the three Ctrl+Alt features accept
   `Ctrl+Alt+<letter or digit>`; the push-to-talk key accepts a short list
   of safe keys (NUMLOCK default). The Settings window uses drop-downs, so
   typing mistakes are impossible there; values typed by hand into the
   config file are validated on load with a clear error. Two features may
   not share one shortcut (refused, with the reason shown).
6. **Status board.** A small Qt object (`ui/status_hub.py`) with signals
   (state: idle / listening / reading; status text; progress n of m;
   reading started with its source and its controls; reading finished).
   Features report into it; the tray and the widget only listen. Document
   Reader (and so OCR) already reports status/progress/finished and gains a
   fan-out to the hub; Quick Reader and dictation gain small optional hooks;
   Quick Reader gets a public `stop()` for the widget.
7. **Tray icon** (`ui/tray.py`): a coloured dot drawn in code (no image
   files): grey idle, red listening, blue reading, plus a tooltip. Menu:
   status line, Pause/Resume, Stop (both greyed unless a reading is active),
   Open Document Reader, Settings, Quit. Quit performs the same clean
   shutdown as Ctrl+C. If no tray exists, log one warning and continue
   without it.
8. **Floating widget** (`ui/floating_widget.py`): frameless, always on top,
   never takes keyboard focus, bottom-right of the primary screen. Shows the
   source ("Document", "Screen text", "Quick read"), status, progress, and
   Pause/Resume + Stop (Quick Reader: Stop only). Appears when a reading
   starts, hides when it ends, follows the on/off switch live.
9. **OCR popup behaviour changes.** With the widget enabled, an OCR reading
   shows the widget instead of opening the Document Reader window; the
   window opens as it does today only when the widget is switched off, so
   Pause stays reachable either way.

### New code

| File | Purpose |
|---|---|
| `settings.py` | `LiveSettings`: thread-safe holder + atomic save + voice list |
| `hotkeys/shortcut_parser.py` | shortcut text <-> evdev key code, validation, duplicate check |
| `ui/status_hub.py` | `StatusHub`: Qt signals reporting state/progress |
| `ui/tray.py` | `TrayIcon` |
| `ui/floating_widget.py` | `FloatingWidget` |
| `ui/settings_window.py` | `SettingsWindow` |
| `config.py`, `notify.py`, engines/readers, `cli.py` | read live values; wire hub, tray, widget, settings; use parsed shortcuts |

### Testing

Fakes and offscreen Qt (`qtbot`) as in earlier phases: parser (valid,
invalid, duplicate, round trip), `LiveSettings` (thread safety, atomic save
failure keeps old settings), hub signal order, tray state/menu enabling,
widget show/hide/controls/never-focus, settings window validation and save
paths, notifier and voice reading live values, OCR popup-vs-widget switch.
Gates: >=85% coverage, `mypy` zero errors, zero test warnings. A
real-hardware check closes the phase (tray icon visible, widget appears on
a reading, switches and voice apply, a shortcut change works after restart,
Ctrl+C / Quit exit cleanly).

### Out of scope

Audio output device selection and the silent ALSA `default` route (separate
fix), speaking speed, start-reading-sooner streaming, voice quality
(user-deferred), autostart and cross-DE shortcut registration (Phase 7),
packaging (Phase 8).

### Planning/implementation deviations

Decided while planning and building Phase 6 (the decisions above stand):

- **Quick Reader reports from state.** Instead of reporting in each code
  path, `QuickReaderFeature` computes "is a reading active?" and reports
  only changes, under a `_report_lock` with a `_last_reported` flag (needed
  to avoid double or stray "finished" reports).
- **`on_started` is announced after the coordinator hand-over.** The plan
  called the Document Reader's hook before `ReadingCoordinator.start_reading`;
  that made the hub ignore the new reading, because starting it ends the
  previous reader's hub reading. It now runs after the hand-over.
- **`TrayIcon.close()` was added** (idempotent: hide, remove the menu, delete
  it later, disconnect hub signals) and is called before `app.quit()` and
  again on shutdown, so the tray cannot outlive the `QApplication`.
- **`QApplication` and the hub are created before the builders** in
  `main()` (the hub is a `QObject`); `quitOnLastWindowClosed` is off.
- **Keepalive timer teardown.** The Ctrl+C keepalive `QTimer` is stopped and
  `deleteLater()`'d in `main()`'s `finally`; left to garbage collection it
  crashed the full test run.
- **The tray menu is not parented to the tray** (`QMenu` is a widget,
  `TrayIcon` a `QObject`); it is held as an attribute and deleted in `close()`.
- **PySide6 stub gaps:** one `cast(QApplication, ...)` and one
  `type: ignore[arg-type]` (`setContextMenu(None)`).
- **A bad shortcut in the hand-edited config** does not stop start-up:
  `resolve_shortcuts()` logs an error and uses the default shortcuts.
- **Saved config is mode 0600** (temp file from `mkstemp`), and `save_config`
  cleans up its temp file on any failure.
- **`Notifier` guards its enabled-function:** a raising function is logged
  and treated as "off", keeping the "never raises" promise.
- **Quick Reader stale-claim fix (pre-existing Phase 3 race):** a stale
  worker no longer releases the coordinator claim while a newer Quick Reader
  reading is active. A tiny window remains where `finish_reading` runs
  outside the lock (deferred).
- **OCR with the widget on:** status, progress and finished reports still go
  to the Document Reader popup though it never receives the OCR text.
- **Switching the widget on mid-reading** shows it at the next reading.

## Completion roadmap and Stage 1 amendment — Sound and voice (confirmed via superpowers:brainstorming, 2026-10-06)

### Roadmap decided with the user (2026-10-06)

The user wants Vocalis **fully developed, including a very easy installer for
non-technical users, before they install and test anything** (they will run the
installer themselves at the very end; Vocalis is never run from the NTFS
development drive). "Fully developed" means the remaining roadmap phases plus
three items the user had postponed. Stages, in order, each with its own
design -> plan -> reviewed build:

1. **Stage 1 — Sound and voice** (this amendment): working sound output
   chosen automatically + a Settings choice; speaking-pace setting; voice
   naturalness controls.
2. **Stage 2 — Phase 7, Service**: systemd `--user` service that starts at
   *desktop login* (not boot: the user account has `Linger=yes`, so a
   `default.target` unit would start without a display; use
   `xdg-desktop-autostart.target`, as this machine's existing
   `sunshine.service` does), auto-restart with backoff, watchdog health
   checks, journald logging, clean SIGTERM handling, single-instance guard,
   friendly start-up self-checks (input-group permission, ydotoold, tesseract,
   xclip, models present).
3. **Stage 3 — Phase 8, Installer and packages**: one easy installer for
   non-technical users (design pending), `.deb` + AppImage.
4. **Stage 4 — Phase 9, Hardening and docs**: README, user guide, whole-system
   automated tests.

Still out of scope (user chose not to include): "start reading sooner"
(streaming after the first sentence).

### Stage 1 decisions

1. **Output device: automatic + a Settings choice.** New config field
   `audio_output_device: str = "auto"`. In "auto", Vocalis probes candidate
   outputs and uses the first that really works. Probe = open a `sounddevice`
   OutputStream on the candidate with a silent buffer and wait (bounded,
   about 1 s) for the audio callback to fire; the failure seen on the target
   machine is "stream starts but the callback never fires" on the ALSA
   `default` device while `pulse` and `pipewire` work. Candidate order: the
   user's explicit choice (if not "auto"), then `pulse`, `pipewire`,
   `default`, then other output devices. The chosen device is remembered for
   the session; if playback later times out (`PlaybackTimeoutError`), the
   player re-probes, switches to the next working device, and notifies once.
2. **Settings window** gains an "Output" drop-down ("Automatic
   (recommended)" + the detected outputs) and a **Test sound** button that
   plays a short, quiet tone through the selected output and reports
   success/failure in the window. The output choice applies on the next
   reading (live, no restart).
3. **Speaking pace.** New config field `speaking_speed: float` (range 0.7 to
   1.6, default **1.15** because the user found the voice "a bit slow";
   1.0 = unchanged). The voice engine has no speed control, so the pace is
   applied to the finished audio with `librosa.effects.time_stretch` (already
   a dependency) in one shared function used by Quick Reader, Document Reader
   (per chunk) and therefore OCR; it applies to both Qwen3-TTS and Piper
   audio. Live: read at the moment of each synthesis. Settings gets a slider
   with the value shown.
4. **Voice naturalness.** Qwen3-TTS `generate_custom_voice` accepts an
   optional `instruct` style text and sampling options (`temperature`,
   `top_p`, `repetition_penalty`). The 0.6B CustomVoice model may ignore
   `instruct`; a first implementation task is a measured spike (generate the
   same sentence with and without a style hint and compare audio length/pitch
   statistics) that decides whether a "Speaking style" setting is offered.
   Only the user's ears can judge quality, so Settings gets a **Preview
   voice** button (speaks a fixed sample sentence with the current voice,
   pace and style) and the user judges during the final test. Deliver:
   `voice_style` setting (only if the spike shows an effect), the sampling
   defaults tuned conservatively, and the preview button. No claim of "more
   human" is made without the user's listening confirmation.
5. **Unchanged:** models, notifications, shortcuts, hub/tray/widget design.

### Testing

Fakes and offscreen Qt as before: device probe with a fake `sounddevice`
(callback fires / never fires / raises), fallback and re-probe logic,
time-stretch correctness (output length ~ input/speed, silence in = silence
out, speed 1.0 = identical, bounds enforced), Settings controls (apply live,
validation), preview and test-sound buttons with a fake player. Gates:
>=85% coverage, mypy zero errors, zero test warnings. No hands-on until the
final installer test.

### Implementation deviations and findings (recorded after Stage 1 build)

- **15s negative-cache TTL + `mark_failed(None)` clearing it.** The design
  above says "nothing worked" is remembered; in practice this is bounded
  to 15 seconds (injectable clock), not remembered forever, because
  Vocalis may be started by the systemd user service at login *before*
  PipeWire/PulseAudio has finished starting — a permanent "nothing works"
  cache would never recover once the sound server came up. `mark_failed
  (None)` (the system default stalling) only clears a cached negative
  answer; no specific device index is marked bad for that call, since
  "the default" is not one fixed device.
- **One retry on an `AudioPlayer` open/start failure.** Beyond the
  originally-scoped stall detection (`PlaybackTimeoutError`), the player
  also retries once when `stream.start()` itself raises (e.g. a headset
  unplugged between probe and play, or a rejected sample rate) — but only
  if asking the device provider again names a *different* device than
  the one that just failed; otherwise the error propagates immediately.
- **Task 7 (voice style setting) skipped.** The Task 6 measurement spike
  found Qwen3-TTS 0.6B CustomVoice's `instruct` parameter has NO
  measurable, direction-consistent effect on duration or pitch: every
  tested condition's delta vs. the no-instruct baseline (−3.8% to −0.42%
  duration, +2.70% to +7.12% pitch) fell inside the noise the same
  no-instruct call produces across its own repeated runs (6.3% duration
  spread, 16.3% pitch spread). Per the plan's conditional, no
  `voice_style` config field, no STYLES dict, and no style control were
  added; the Preview voice button and sampling defaults were kept as
  originally scoped. Full numbers: `reports/task-6-report.md`.
- **Pause/resume restructuring in `doc_reader.py`.** `_read_worker` gained
  a `_play_chunk_until_done()` helper; a chunk is now synthesized and
  paced exactly once, and a pause/resume cycle replays that same
  already-paced audio rather than resynthesizing and re-pacing it on
  resume. **User-visible change:** changing the speaking speed while a
  document reading is *paused* now applies starting from the *next*
  chunk once resumed, not immediately to the paused chunk (the old
  resynthesize-on-resume behavior would have picked up a new speed
  mid-chunk; the new behavior does not).
- **Test-sound exact-device-match fix.** The Settings window's Test
  sound button, for a user-named output (not "Automatic"), resolves and
  probes exactly that one device — never falling back to
  pulse/pipewire/default/another device the way `OutputDeviceChooser.
  current()` normally would — so a failing named device is reported as
  failing rather than silently succeeding on a substitute.

## Stage 2 — Phase 7 amendment: Service and autostart (confirmed via superpowers:brainstorming, 2026-10-08)

Phase 7 makes Vocalis start itself at login and recover from a crash, as a
`systemd --user` service, replacing the old Voxtype's login-start entry
(removed on 2026-10-06; see `.claude/*CHANGELOG.md` for that date).

### Decisions

1. **Starts at desktop login, on any desktop environment.** The unit is
   `WantedBy=` the XDG autostart target (not `default.target`; the user
   account has `Linger=yes`, which would start a `default.target` unit at
   boot with no display present yet — confirmed on this machine in the
   Phase 6 design notes). It also declares `After=`/`Wants=` on
   `pipewire.service`, `pipewire-pulse.service` and `wireplumber.service`
   (the real unit names on this machine, confirmed via
   `systemctl --user list-units`) — required so Stage 1's sound-output
   chooser sees the real outputs on its first run after login, per that
   stage's carried-forward finding that a device list read before the
   sound server is up never refreshes itself for that process's lifetime.
2. **Crash recovery: `Restart=on-failure` with a burst limit.** Restart
   after a short delay (`RestartSec=3`); if it fails 5 times within 60
   seconds (`StartLimitBurst=5`, `StartLimitIntervalSec=60`), systemd stops
   retrying and Vocalis's own install reacts to that by sending ONE desktop
   notification ("Vocalis could not start — check the log") rather than
   crash-looping silently forever. A single, deliberate `vocalis` launch
   from a terminal (not under the service) is unaffected.
3. **Startup health checks, reported as desktop notifications** (same
   `Notifier` the rest of the app already uses): missing Parakeet model
   files, missing Qwen3-TTS model files, `ydotoold` not reachable,
   `tesseract`/`xclip` missing. Each check is non-fatal — Vocalis still
   starts and the affected feature fails per-request as it does today;
   the notification is a one-time heads-up, not a blocking gate.
4. **True single-instance guard.** Today's `VocalisDaemon.start()` only
   checks its own in-process `is_running` flag and unconditionally deletes
   any existing socket file as "stale" — a second real instance (e.g. a
   manual `vocalis` launch while the service is already running, or two
   service starts racing) would delete the first instance's live socket
   out from under it. Fix: before deleting an existing socket file,
   attempt to CONNECT to it; a successful connect means another instance
   is genuinely alive (raise `DaemonAlreadyRunningError` and exit cleanly,
   logged, no notification spam); a connection failure (`ConnectionRefusedError`
   /`FileNotFoundError`) confirms the file is truly stale and may be removed
   as today.
5. **Graceful `SIGTERM` handling.** `systemd stop`/`restart` sends
   `SIGTERM`; today only `SIGINT` (Ctrl+C) is handled (noted as a known gap
   in `cli.py`'s module docstring since Phase 4). Install a `SIGTERM`
   handler alongside the existing `SIGINT` one in `main()`, routed through
   the same `app.quit()` path, so shutdown is identically clean (listeners
   stopped, tray closed, socket removed) regardless of which signal
   triggered it.
6. **Logging:** `journalctl --user -u vocalis` (systemd's own journal
   capture of stdout/stderr); no new logging code required beyond what
   `logging_setup.py` already does, since systemd captures a unit's
   stdout/stderr into the journal automatically.
7. **The `vocalis` CLI gains no new subcommands in this stage** (a
   `vocalis status`/`restart` convenience command is left to Stage 3's
   installer design, where it fits naturally next to the uninstaller).

### New/changed files

| File | Purpose |
|---|---|
| `packaging/systemd/vocalis.service` | the unit file (template; paths filled in by Stage 3's installer, not hand-edited by the user) |
| `src/vocalis/daemon.py` | real single-instance guard (connect-before-delete) |
| `src/vocalis/cli.py` | `SIGTERM` handler; startup health checks + notifications |
| `src/vocalis/startup_checks.py` (new) | the four non-fatal checks, each returning a human message or `None` |

### Testing

Fakes as in every earlier phase: a fake socket/connect for the single-
instance guard (genuinely-alive vs genuinely-stale, and the connect call
itself raising for an unrelated reason); `SIGTERM` delivered to a running
`main()` under test exits exactly like the existing `SIGINT` test; each
startup check with its dependency present/absent produces the right
message or `None`, and a check that itself raises is caught and logged,
never crashes startup. The unit file itself is validated with
`systemd-analyze --user verify` in the real-machine check at the end of
the stage (text content only is unit-testable; starting a real systemd
service is explicitly NOT done by automated tests — see Global
Constraints). Gates: >=85% coverage, mypy zero errors, zero test
warnings.

### Out of scope (stays for later stages)

Packaging the unit file into an installer, substituting real paths into
it, enabling/starting it for the user, and a `vocalis status`/`restart`
CLI convenience — all Stage 3 (Phase 8, the installer). Cross-DE global
*shortcut* registration (as opposed to autostart) was already handled by
evdev in Phase 2/4/5 and needs no further work here.

### Implementation deviations and findings (recorded after Stage 2 build)

- **Single-instance probe has three outcomes, not two.** Decision #4 above
  only names "alive" and "stale". The build added a third, deliberate
  outcome: a probe `connect()` failing with anything *other than*
  `ConnectionRefusedError`/`FileNotFoundError` (e.g. a permission error)
  is **ambiguous**, and is treated the same as "stale" rather than raising
  or blocking startup — an inconclusive probe must never prevent Vocalis
  from starting. See `docs/architecture.md`'s "Stage 2 addition" for the
  full rationale and `docs/developer-docs.md` for how the probe is tested.
- **`shutil.which` early-binding bug, found and fixed (Task 2).**
  `check_capture_tools()`'s `which` parameter originally defaulted to
  `shutil.which` directly. A default parameter value is evaluated once,
  at import time, so a test's `monkeypatch.setattr("vocalis.
  startup_checks.shutil.which", ...)` silently had no effect — the
  silent-run test only passed because the real `tesseract`/`xclip` happen
  to be installed on the development machine. Fixed by having
  `run_startup_checks()` pass `shutil.which` explicitly at the call site,
  looked up fresh at call time, so a test's monkeypatch is observed.
- **Real-`notify-send` test-isolation bug, found and fixed (Task 4).**
  Wiring `run_startup_checks(config)` into `main()` exposed roughly 15
  pre-existing `tests/test_cli.py` tests that call `main()` for unrelated
  reasons and never patched `run_startup_checks` — each could fire a
  genuine desktop notification depending on live machine state (which
  models/tools happen to be installed on whoever runs the suite). Fixed
  with an autouse `_no_real_startup_checks` fixture plus a dedicated
  regression test; see `docs/developer-docs.md` for the full story and
  why it matters to future contributors adding a new `main()`-calling
  test.
- **"Vocalis could not start" is a design intent for a later stage, not
  runtime code shipped here.** Decision #2 above describes Vocalis's own
  install reacting to systemd's restart-limit with one desktop
  notification. That reaction is **not implemented as runtime code in
  this stage** — systemd alone decides "5 crashes in 60 seconds is too
  many, stop retrying" (`StartLimitBurst=5`/`StartLimitIntervalSec=60` in
  the unit file), and by the time it gives up, Vocalis is not running to
  send any notification about it. Detecting that from the outside and
  reacting to it needs separate code, watching the service rather than
  being the service, which has not been built. This is deferred to Stage
  3 (once the service is actually installed, enabled, and can be
  observed running for real) or later, not silently dropped — see
  `docs/user-docs.md`'s "About 'Vocalis could not start'" for the
  user-facing version of this same honesty.

## Stage 3 amendment — Phase 8, Installer (confirmed via superpowers:brainstorming, 2026-10-09)

### Scope decided with the user (2026-10-09)

Target: any Debian/Ubuntu-family Linux machine (apt-based), `x86_64`, with a
working NVIDIA GPU driver already installed by the person (the installer
checks for this and refuses with a plain message if missing — it never
touches GPU drivers itself, since that can break a display and usually
needs a reboot). This supersedes the original Phase 0 spec's "ships as a
`.deb` + AppImage" aspiration, written before the real dependency weight
(CUDA/torch/transformers, ~3.1GB of models) was known:

- **No `.deb`.** A `.deb` can't carry Python/CUDA dependencies as apt
  packages — it would still need the same venv-creation logic inside a
  `postinst` script, while also fighting root-owned-paths conventions
  against a `systemd --user` service. No real benefit over a script.
- **No AppImage.** Can't install a background service, system packages,
  or persistent model folders by itself; a companion installer script
  would still be needed, which would just duplicate this same logic a
  second time for a single-machine personal tool.
- **Shell script + self-extracting `.run`.** One `install.sh`/
  `uninstall.sh` pair is the real implementation; `install.run`/
  `uninstall.run` are the same content wrapped by `makeself` into a
  single double-clickable file. Re-running `install.sh` is always safe
  (idempotent — it only touches what's missing or wrong), so it also
  serves as a repair command.

### Decisions

1. **System packages (tesseract-ocr, xclip, ydotool, …): auto-installed.**
   One `sudo apt-get install -y <missing list>` call, one password prompt
   total, only for packages `dpkg -s` reports as absent.
2. **Python environment: `uv`.** Installer installs `uv` (pinned release,
   not an unpinned `curl | sh`) if missing, uses it to provision Python
   3.13 and create a venv at `~/.local/share/vocalis/venv`, then
   `uv pip install`s the Vocalis wheel (built from this repo at packaging
   time) into it. Chosen over plain `venv`+`pip` for speed on the
   multi-gigabyte ML wheels (torch, cuDNN, transformers) and because `uv`
   can provision the exact Python version itself.
3. **Command on PATH:** symlink
   `~/.local/share/vocalis/venv/bin/vocalis` → `~/.local/bin/vocalis`
   (creating `~/.local/bin` if needed), matching how Piper is already
   laid out on this machine.
4. **Models: reuse if present, else download — no config rewrite needed.**
   `config.py`'s existing defaults already point at the exact folders
   where Parakeet, Qwen3-TTS and the Piper binary/voice already sit on
   this machine (`~/.local/share/voxtype/models/parakeet-tdt-0.6b-v3-int8`,
   `~/.local/share/vocalis/models/qwen3-tts-customvoice-0.6b`,
   `~/.local/bin/piper/piper`, `~/.local/share/piper-voices/
   en_US-lessac-medium.onnx`). The installer checks each of those exact
   paths; if present, it leaves them alone (no download, no config
   change); if absent, it downloads directly into that exact path
   (Hugging Face snapshot download for the two models, a pinned GitHub
   release tarball for Piper) with simple percent/MB progress. A fresh
   machine ends up with files in the same place a hand-set-up one has.
5. **Service wiring.** Fill `packaging/systemd/vocalis.service`'s
   `__VOCALIS_EXEC__`/`__VOCALIS_WORKDIR__` placeholders with the real
   symlink path and `~`, write to `~/.config/systemd/user/vocalis.service`,
   then `systemctl --user daemon-reload && systemctl --user enable --now
   vocalis`.
6. **Done message.** Plain-English summary of what was installed, that it
   starts automatically at next login, that it's running right now, and a
   `systemctl --user status vocalis` hint for later curiosity.
7. **Uninstall:** disables/stops the service, removes the unit file, the
   `~/.local/bin/vocalis` symlink and the venv. Config and downloaded
   models (several GB) are kept by default with a printed note of what
   was kept and where; `--purge` also removes those.
8. **No new `vocalis` CLI subcommands.** `systemctl --user status/restart
   vocalis` already covers this; adding wrapper subcommands would be
   speculative (YAGNI) for a need `systemctl` already meets.

### New/changed files

| File | Purpose |
|---|---|
| `packaging/installer/install.sh` | the real installer (system packages, Python env, models, service) |
| `packaging/installer/uninstall.sh` | reverses it; `--purge` also removes config/models |
| `packaging/installer/lib/*.sh` | small sourced helper functions (apt, models, systemd, uv), kept short and bats-testable |
| `packaging/installer/build-run.sh` | wraps install.sh/uninstall.sh into install.run/uninstall.run via makeself |
| `packaging/installer/README.md` | what each file is, how to rebuild the `.run` files |

### Testing

A real install does `sudo apt install`, real downloads, and real
`systemctl --user enable` — none of that runs inside automated tests,
consistent with every earlier stage deferring all hands-on verification
to the user's own final run, off the NTFS drive. Instead:

- `shellcheck` on every script, zero warnings, same bar as every other
  gate in this project.
- `bats` (Bash Automated Testing System) unit tests on the small `lib/`
  helper functions (e.g. "detects missing apt packages from a faked
  `dpkg -s`", "skips downloading when the model file already exists at
  the expected path", "substitutes the service-file placeholders
  correctly", "idempotent re-run touches nothing already correct") with
  `apt`/`curl`/`systemctl`/`uv` faked out — same philosophy as every
  Python test in this project.
- No automated test calls real `apt`, `curl`, or `systemctl`. The
  genuine end-to-end run is the user's own final install.

### Out of scope (explicitly declined)

Installing/updating GPU drivers; `.deb` and AppImage packaging; an
auto-updater or new `vocalis` CLI subcommands; portability beyond
Debian/Ubuntu-family `apt`-based distros.

### Implementation deviations and findings (recorded after Stage 3 build)

- **A few `*_run_*` wrapper comments are shared, not individual.** The
  plan's own prescribed code for a handful of paired wrappers (for
  example `apt_run_systemctl_list_unit_files`/`apt_run_systemctl_
  enable_now` in `apt.sh`, and the matching pair in `service.sh`) puts one
  comment above the first of the two and lets the second ride along
  under it, rather than giving each its own line. This is purely
  cosmetic — flagged in review, left as-is, non-blocking.
- **`models_ensure_piper_binary` doesn't check `tar`/`mv`'s own exit
  status.** If the download succeeds but the tarball is corrupt or the
  extracted layout is unexpected, `tar`/`mv` could fail silently and the
  function would still return 0 as if the binary were installed. The
  existing `bats` test for this function only exercises the
  *download*-failure path, not an extraction failure. In practice this
  self-heals: since the real binary still wouldn't exist at the end, the
  very next run would detect it as missing and try again — but the gap
  is real and not currently covered by a test.
- **`uninstall.sh --purge` can under-claim what it deleted.** The model
  directories are only looked up by asking the venv's own Python for
  `VocalisConfig()`'s paths, so on a partially-broken prior install
  (venv already missing, but the unit file and `~/.local/bin/vocalis`
  symlink still present) that lookup is skipped entirely — yet `--purge`
  still prints the "removed, including your config and models" message.
  Nothing is lost that wasn't already gone (no data-loss risk), but the
  message overstates what actually happened on that one specific,
  already-degraded starting state.
- **Two test-hermeticity gaps, same pitfall, caught twice during this
  build.** `tests/installer/test_pyenv.bats`'s original "uv already on
  PATH" test asserted on `$status` alone without overriding `HOME`, so it
  passed against the real system `uv` rather than proving the `command -v`
  shadow trick actually worked — a test that would have stayed green even
  if that shadow were broken. The same shape of problem turned up again
  in `install.sh`'s and `build-run.sh`'s `SCRIPT_DIR`/`REPO_ROOT` override
  tests, caught by mutation testing and fixed in a follow-up commit. Worth
  calling out as a pattern for future subagent-driven stages: a test that
  checks only an exit code or a path substring can pass for the wrong
  reason, and is worth a second look whenever it's exercising an override
  or a conditional branch rather than a straight success path.
- **The Piper-voices Hugging Face path was unverified at design time, now
  confirmed.** The original spec draft (and `fetch_models.py`'s own
  comment) flagged that `rhasspy/piper-voices`' exact file path
  (`en/en_US/lessac/medium/en_US-lessac-medium.onnx[.json]`) followed the
  repo's documented layout but had not been checked live, since there was
  no outbound network access while writing the plan. It was verified
  directly against the live Hugging Face page during Task 4's build and
  confirmed correct — no change was needed.

## Stage 4 amendment — Phase 9, Testing & docs hardening (confirmed via superpowers:brainstorming, 2026-10-09)

### Scope decided with the user (2026-10-09)

This is the last of the four stages the user approved on 2026-10-06. Two
deliverables:

1. **A root-level `README.md`.** The project has no top-level front door
   today — only the detailed `docs/*.md` files. The README is short: what
   Vocalis is (one paragraph), a link to installing it
   (`packaging/installer/README.md`), a link to the full user guide
   (`docs/user-docs.md` — which already covers every feature in depth, so
   no separate "user guide" document is written; the README points to it
   instead of duplicating it), and a link to `docs/developer-docs.md` for
   anyone touching the code.
2. **Whole-system tests.** Every existing automated test (602 at the time
   of this amendment) exercises one feature at a time, heavily faked.
   Nothing proves two or more features behave correctly *together* in one
   running process — sharing the single `ModelManager` GPU slot, the four
   keyboard shortcuts, the one `Notifier`, the one `StatusHub`. A new test
   file reuses `cli.py`'s own private builder functions (`_build_dictation`,
   `_build_quick_reader`, `_build_ocr_reader`, etc. — not reinvented
   wiring) to assemble multiple real feature objects against one shared
   `ModelManager`/`StatusHub`/`Notifier`, using the same fakes every other
   test file already uses (fake audio/mic, fake hotkey devices, fake AI
   engines) — never real hardware, never a real model, consistent with
   every earlier phase.
3. **A documentation accuracy sweep.** "Hardening" also means checking the
   existing `docs/*.md` files for drift — three stages have landed since
   they were last fully checked — and fixing anything stale, not only
   adding the two deliverables above.

### Decisions

1. **Whole-system test scenarios (`tests/test_cross_feature_integration.py`,
   new file):**
   1. **One-GPU-model-at-a-time holds across features.** Dictate → Quick
      Reader → OCR Reader → Document Reader, in sequence. At every point,
      the shared `ModelManager` never has more than one engine loaded;
      switching features correctly unloads the previous one.
   2. **Hotkeys don't cross-trigger.** All four shortcut listeners
      registered together; pressing one never fires a different feature's
      handler.
   3. **A Settings change made while idle is picked up by a different
      feature's next reading.** E.g. change `speaking_speed`, then start a
      Document Reader session, confirm the new value is used — proving the
      shared settings path, not just each feature's own already-tested
      local wiring.
   4. **Notifications from two different features in a row both reach the
      one shared `Notifier` correctly**, un-garbled and in order.
   5. **Full lifecycle with multiple features registered**: startup, two
      different features each handle one request, then a clean shutdown —
      confirms no leaked threads/listeners and the GPU is released,
      mirroring Stage 2's shutdown tests but with more than one feature
      active at once.
2. **`README.md`** at the repo root, short, linking out rather than
   duplicating — per Decisions item 1 above.
3. **Docs accuracy sweep** across `docs/user-docs.md`, `docs/architecture.md`,
   `docs/workflow.md`, `docs/file-purpose.md`, `docs/folder-structure.md`,
   `docs/developer-docs.md`, `docs/algorithms.md`: re-read each against the
   current code (not from memory of earlier stages) and fix any stale
   claim, file path, or behavior description found.
4. **No new product features, no new config fields, no new CLI behavior.**
   This stage only adds tests and documentation.

### Testing

The new whole-system test file follows the same fakes-only discipline as
every earlier phase — never a real microphone, real sound device, real
evdev device, real GPU model, or real network call. Gates: >=85% coverage
(project-wide, unchanged), mypy zero errors, zero test warnings, the new
file's own tests all passing. No hands-on testing in this stage either —
per the user's standing instruction, the very first hands-on use of
Vocalis happens only once the user runs the installer themselves, after
this stage merges.

### Out of scope (explicitly declined)

A separate "user guide" document (the existing `docs/user-docs.md` already
serves that purpose); any new product feature, config field, or CLI
behavior; a real hands-on test run (still deferred to the user's own final
install).

## Stage 5 amendment — Foundations and quick wins (confirmed via superpowers:brainstorming, 2026-10-15)

### Why this stage exists

The user installed Vocalis for real for the first time (2026-10-15) and sent a
long list of problems and wishes. The full roadmap (Stages 5-8) is in the
approved plan; this stage is the first, smallest one. It fixes things that
block everything else and adds the small foundations later stages need.

Roadmap (decided with the user, order chosen by the assistant):

- **Stage 5 (this one):** license, hardware probe, clearer microphone
  problems, a way to find and open the app window.
- **Stage 6:** voice engine overhaul (Kokoro engine, automatic engine choice,
  gapless long reading, clearer audio, CPU/RAM/GPU limits).
- **Stage 7:** red-and-black UI and highlighting of the line being read.
- **Stage 8:** easy installer with a GUI setup wizard (CPU-only PCs allowed).

Decisions already made by the user: engines Kokoro + Qwen3-TTS + Piper
(VoxCPM and online voices are NOT built); automatic choice by hardware (strong
GPU -> Qwen3-TTS, GPU over 2 GB -> Kokoro, no GPU -> Kokoro or Piper); MIT
license; commit to local `main` after each stage and push automatically to
`origin/main` only.

### Already done before this stage (committed in the first public commit)

The NumLock "no audio captured" bug has a confirmed root cause: an old
`~/.asoundrc` left by the previous app ("Voxtype") routes the ALSA "default"
microphone to nowhere, so the stream opens but no sound ever arrives. The
PipeWire and Pulse inputs work. `pick_input_device()` in
`src/vocalis/audio/capture.py` now prefers those; `MicRecorder` takes a
`device`. The user still has to confirm on a real microphone at the end.

### Scope (five small pieces)

1. **MIT license.** `LICENSE` file (holder: reduxailabs-max, year 2026),
   `license = "MIT"` and `license-files` in `pyproject.toml`, a License
   section in `README.md`.
2. **Hardware probe.** New `src/vocalis/hardware.py`: a frozen
   `HardwareInfo` (GPU name, GPU total/free MB, RAM total/free MB, CPU core
   count, 1-minute load) and `probe_hardware()` whose four data sources
   (`/proc/meminfo`, `/proc/loadavg`, `os.cpu_count`, a GPU query using
   `torch.cuda`) are injectable so tests never touch real hardware. No new
   dependency (`psutil` is not declared). It must never raise: any source that
   fails yields `None` for those fields. In this stage it is used only to
   write one diagnostic line to the log at startup (in a background thread,
   because importing torch is slow); Stage 6 uses it to choose engines and
   enforce limits.
3. **Clearer microphone problems** (passive, no recording at startup):
   - New startup check `check_stale_asoundrc(path)`: if `~/.asoundrc` exists
     and mentions `voxtype` (case-insensitive), return a plain-English message
     (other apps may have a silent microphone; Vocalis already works around
     it; the file can be renamed to `.asoundrc.old`). Added to
     `run_startup_checks` after the capture-tools check.
   - Dictation's empty-recording notification changes from "No audio
     captured" to a message that tells the user what to check (microphone on,
     not muted, right microphone chosen). The chosen input device index/name
     is written to the log when recording starts.
   - An active "record two seconds and check the level" test is NOT done at
     every startup (it would open the microphone each launch). It becomes a
     page of the Stage 8 setup wizard instead.
4. **Make the window findable.** Today the only entry is a tray icon with no
   click handler, and a second `vocalis` launch exits silently, so the user
   could not find the GUI.
   - Tray: a left-click (Qt `activated` with `Trigger`) opens Settings.
   - `SettingsWindow.present()` (load current values, show, raise, activate)
     replaces the same three lines duplicated in `build_ui`.
   - `VocalisDaemon` gets an optional `command_handler` called with each
     received command line (exceptions are logged, the `ack:` reply is
     unchanged). `cli.main` connects the command `show-settings` to a small
     Qt bridge object whose signal runs `settings_window.present()` on the GUI
     thread.
   - `vocalis --settings` (argparse in `main(argv=None)`): connects to the
     running daemon's socket and sends `show-settings`. If nothing is running
     it starts the user service (`systemctl --user start vocalis`, injectable
     in tests), waits up to 20 seconds for the socket, then sends the command;
     if that fails it prints a short plain-English message and exits 1.
   - Installer installs a launcher `~/.local/share/applications/vocalis.desktop`
     (Name=Vocalis, Exec=`<installed vocalis> --settings`, an icon from the
     system theme) from a template `packaging/desktop/vocalis.desktop`
     with a `__VOCALIS_EXEC__` placeholder; `uninstall.sh` removes it; the
     closing message of `install.sh` says where the window is (the tray icon,
     or the "Vocalis" entry in the application menu).
5. **Docs and changelog** for all of the above (user docs gain a short "How to
   open Vocalis" section).

### Out of scope (and why)

- Speed/voice quality, engines, limits: Stage 6.
- Theme and highlighting: Stage 7.
- Wizard, CPU-only install, mic test page: Stage 8.
- The `gpu_vram_ceiling_mb` / `engines.*` dead settings: Stage 6.

### Testing

Same gates as every stage: tests >= 85% coverage, mypy strict clean, zero
warnings, shellcheck + bats clean for installer changes. All hardware is faked
(`/proc` files read from temp files, fake GPU query, fake socket/systemctl
runner, offscreen Qt). The existing autouse guards stay. New fakes follow the
existing patterns (`FakeStream`, injectable runner functions).

### Rulings the assistant made (not asked of the user)

1. The roadmap order (5 -> 6 -> 7 -> 8).
2. Dropped "record at startup" mic test in favour of passive diagnosis now and
   a wizard page later (privacy, startup time, and a better place for it).
3. Dropped the vague "speaking_speed messaging" item; Stage 6 owns speed.
4. `--settings` starts the service itself when needed, so the application-menu
   launcher works whether or not Vocalis is already running.

## Stage 6 amendment — Voice engine overhaul (confirmed via superpowers:brainstorming, 2026-10-16)

### Why this stage exists

The user's hands-on test of the real installer (2026-10-15) found the reading
voice unclear, slow, and heavy enough to heat the laptop, and found long
documents pause between pieces. Research and code reading (done in the prior
session) found concrete, fixable causes rather than a fundamentally broken
design. This stage fixes those causes and adds a second, lighter voice engine
plus user-visible resource limits, so Vocalis works well on both the user's
RTX 4060 laptop and on weaker or GPU-less machines.

### Decisions already made with the user (2026-10-15, carried in from the
approved roadmap `humming-wobbling-wreath.md`; not re-asked here)

- **Engines to ship:** Qwen3-TTS (kept, highest quality), **Kokoro** (new,
  light, Apache-2.0, native speed control), Piper (kept, emergency fallback
  only). VoxCPM and any online voice are explicitly NOT built now.
- **Automatic engine choice, the user's own rule, refined once for a real
  conflict found in testing:** GPU total > 4 GB AND free VRAM enough for
  Qwen3 AND the PC not under stress -> Qwen3-TTS; PC stressed or too little
  free VRAM -> Kokoro; no usable GPU -> Piper (or Kokoro on CPU if Piper is
  missing). The user can always force an engine per feature in Settings.
- **Root causes identified in the code** (all drive this stage's tasks):
  - Voice quality: non-English-native speaker in the user's own config;
    `librosa` phase-vocoder time-stretch (`audio/pace.py`) applied per chunk
    with no crossfade; Qwen3's own generation defaults (temperature 0.9,
    top_k 50, uncapped `max_new_tokens`); the sentence splitter
    (`text/chunker.py`) breaks on abbreviations and joins unpunctuated
    lines; no number/URL/symbol normalization; a new audio stream opened per
    chunk (`audio/playback.py`) causing clicks; Quick Reader synthesizes an
    entire clipboard in one call.
  - Heat/speed: no CPU/RAM/GPU limit anywhere; `gpu_vram_ceiling_mb` and
    `engines.quick_tts`/`engines.doc_tts` are dead config (declared, never
    read — confirmed again just now: `model_manager.py:26`'s
    `TTS_ENGINE_ID = "qwen3_tts"` is still the only engine id, shared by both
    features via `_ENGINE_ID = TTS_ENGINE_ID` in `quick_reader.py:30` and
    `doc_reader.py:49`); the idle-unload timer (30s default) starts before
    playback ends and forces a 15-45s reload.
  - Gaps in long reading: `doc_reader.py`'s `_read_worker` is strictly
    synthesize -> pace -> play, one chunk at a time; since synthesis
    (~1.0x real time) is slower than playback once sped up, even a
    one-chunk lookahead would still underrun.

### Scope, in the same 4 groups the roadmap already named (6a-6d), each its
own implementation plan, built in order in one Stage 6 worktree

**6a — Engine plumbing.** A single `TTSEngine` protocol replacing the
duplicated `Synthesizer` Protocol in `quick_reader.py`/`doc_reader.py`; a new
`engines/registry.py` factory; `Literal["auto","qwen3_tts","kokoro","piper"]`
validation on `config.engines.quick_tts`/`doc_tts`; `TTS_ENGINE_ID` replaced
by a per-feature, per-selection engine id (still shared when both features
pick the same engine, so `tests/test_shared_tts_engine.py`'s "no reload"
guarantee survives); a new `engines/kokoro_tts.py` using the `kokoro-onnx`
PyPI package (MIT; depends on `onnxruntime`, `espeakng-loader`,
`phonemizer-fork`; a `[gpu]` extra pulls `onnxruntime-gpu`; model weights —
`kokoro-v1.0.onnx` ~310MB and `voices-v1.0.bin` ~27MB — are NOT in the
package and must be fetched by the installer, same as the existing models);
a new pure, fully unit-tested `engines/auto_select.py` implementing the rule
above from a `HardwareInfo` snapshot (`src/vocalis/hardware.py`, built in
Stage 5) plus Vocalis's own current VRAM use; a Settings engine picker
(Auto/Qwen3-TTS/Kokoro/Piper per feature) with fall-down-the-chain-with-a-
notification if a forced engine fails to load.
  **Known risk, to be solved in this group:** installing plain `onnxruntime`
  (pulled in by `kokoro-onnx`) alongside `onnxruntime-gpu` (already required
  by `onnx-asr[gpu]`/Parakeet) can silently disable CUDA for Parakeet — a
  documented ONNX Runtime issue where whichever package installs last wins
  in the same `onnxruntime/` directory. The plan must pin the install order
  or exclude the dependency, and add a startup check that
  `CUDAExecutionProvider` is still available when a GPU is present.

**6b — Gapless long reading.** A shared `ChunkedSpeaker` used by both
features: a producer thread holds ONE `ModelManager.acquire()` for the whole
reading and fills a bounded (2-3 item) queue, synthesizing ahead with
back-pressure; the consumer feeds a persistent `sd.OutputStream` (replacing
today's per-chunk stream in `audio/playback.py`) with a small fade at chunk
joins and zero-fill on underrun instead of a gap; Quick Reader gains the same
chunking Document Reader already has (`text/chunker.py`) instead of
synthesizing its whole input in one call; the chunker becomes sentence-aware
(abbreviations, bare newlines) and gains true `(start, end)` character spans
(needed again in Stage 7 for highlighting); the idle-unload timer must not
fire while a reading is actively in progress, and should start counting from
when playback ends, not when synthesis ends. Pause/resume/stop keep their
current user-visible behaviour; the existing test fakes
(`GatedSynthesizer`, `LengthRecordingPlayer`, etc. in `tests/test_doc_reader.py`)
are rewritten for the new producer/consumer shape, not discarded.

**6c — Clarity.** Each engine's native speed control is used instead of the
lossy `librosa` time-stretch (Kokoro has a `speed` parameter; Piper has
`--length_scale`); Qwen3-TTS keeps a stretch only as its own fallback, capped
near 1.0-1.2x; Qwen3-TTS's generation call gets a lower, steadier
`temperature` (~0.6-0.7), a lower `top_k`, and a `max_new_tokens` scaled to
the text length instead of the fixed 8192; the default voice becomes an
English-native speaker (Ryan or Aiden), with a warning if a mismatched
language/speaker pair is configured; basic text normalization for numbers,
URLs and symbols is added to `text/normalizer.py`.

**6d — Resource limits.** New Settings controls (and matching `LiveSettings`
getters), all optional (`None` = no cap, the current unconstrained
behaviour):

- `max_cpu_threads: int | None`. Applied at each engine load: Qwen3-TTS via
  `torch.set_num_threads(n)` (process-global, re-applied at each load since
  it's cheap and keeps behaviour obviously tied to the current live
  setting); Kokoro via its ONNX `SessionOptions.intra_op_num_threads` at
  session build (already rebuilt fresh per reading). Piper's installed CLI
  binary has **no thread-count flag at all** (verified directly against the
  real binary's `--help` output, not assumed) — its subprocess instead gets
  `OMP_NUM_THREADS`/`ORT_NUM_THREADS` environment variables set to the
  configured value. This is explicitly documented as **best-effort**: Piper
  doesn't document honouring these, so the docs must say so rather than
  imply a guarantee.
- `max_ram_mb: int | None`. Deliberately a **soft/advisory** signal, not an
  OS-level hard cap: feeds into `auto_select.py`'s existing "stressed"
  determination (comparing Vocalis's own current RSS against this ceiling)
  so a low setting triggers a downshift to a lighter engine. A hard
  `resource.setrlimit(RLIMIT_AS, ...)` cap was considered and rejected — it
  can crash Vocalis itself (`MemoryError`) on a conservative user-chosen
  number, which is worse than a slower engine. This is the one place this
  stage deliberately interprets "enforced" as "influences the adaptive
  downshift" rather than "hard-blocks."
- `gpu_vram_ceiling_mb` (already exists, currently a dead setting) is
  finally enforced: `torch.cuda.set_per_process_memory_fraction(ceiling_mb /
  total_mb)` for Qwen3-TTS, and ONNX Runtime's CUDA execution-provider
  `gpu_mem_limit` option (bytes) for Kokoro-on-GPU — both applied at the
  point each engine builds its session/model, reading the live setting
  fresh each time.

**Adaptive downshift with hysteresis.** `EngineSelector.plan()` already
calls `decide()` fresh at the start of every reading and a chain already
only moves forward (never back) within one reading — both already satisfy
"checked at the start of each reading" and "never flip-flopping
mid-reading" with no new code. What's missing is memory **across**
readings: today, a downshift in reading N due to transient stress would be
immediately forgotten by reading N+1, which could upshift right back the
moment hardware noise clears, causing an unwanted reload. A small stateful
wrapper around `EngineSelector` fixes this: it remembers the last
downshift's timestamp and tier, and for at least 60s after a downshift, it
caps the plan returned to new readings at that tier (never lower than what
fresh `decide()` would choose, only capped from above). After 60s, the next
`plan()` call re-evaluates fully, free of the cap.

Also in this stage: the systemd unit template gains a `Nice=10` line; model
unload in `ModelManager._unload_current_locked()` calls `gc.collect()` in
addition to the existing `torch.cuda.empty_cache()`.

### Out of scope (and why)

- Any UI visuals/theming/highlighting: Stage 7. User requirement recorded
  2026-10-16: the redesigned Settings window must expose EVERY configurable
  setting (including the new 6a engine choice and 6d CPU/RAM/GPU limits
  built in this stage), the Document Reader must highlight the line being
  read, and the look is a modern red-and-black theme with a new widget.
  Stage 6 therefore must add each new setting to `config.py`,
  `LiveSettings` and the current Settings window so Stage 7 only restyles.
- Installer wizard, CPU-only install path, the mic test page: Stage 8 (6a's
  Kokoro model download is added to the EXISTING installer scripts now;
  Stage 8 is about making the installer itself easier to use, not about
  which models it fetches).
- VoxCPM, any online/API voice: explicitly declined by the user this session.

### Testing

Same gates as every stage: tests >= 85% coverage, mypy strict clean, zero
warnings. All hardware, GPU, audio devices, and the `kokoro-onnx`/Qwen3/Piper
libraries themselves stay faked in tests — nothing here changes the project's
standing rule that automated tests never touch real hardware. The auto-select
policy (6a) is pure and injectable, so its decision table is fully covered
without any real GPU. The gapless-queue rewrite (6b) is tested with the
existing fake-synthesizer/fake-player patterns, verifying ordering and timing
without real audio.

### Rulings the assistant made (not asked of the user)

1. 6a builds the engine registry and auto-selector FIRST, because 6b-6d all
   depend on "which engine is this reading using right now" being a stable,
   queryable concept.
2. Qwen3-TTS keeps a (capped) time-stretch as ITS OWN fallback speed control,
   since the model has no native speed parameter — only Kokoro and Piper get
   a true native-speed path. This is a deliberate asymmetry, not an oversight.
3. The Kokoro model download is added to the Stage 1-era installer scripts
   now (6a), not deferred to Stage 8, since 6a's Settings engine picker is
   useless to the user if the model it names is never fetched.


## Stage 7 amendment — Red-and-black UI and reading highlight (confirmed via superpowers:brainstorming, 2026-10-29)

### Goal

Give every Vocalis window one consistent, modern red-and-black look, make
the Settings window easy to navigate now that it holds many settings, and
highlight the sentence being spoken in the Document Reader window.

### Decisions (user-confirmed)

- Scope: theme for tray, floating widget, Document Reader window and
  Settings window; Settings reorganized into tabs; highlighting in the
  Document Reader window only (Quick Reader has no text view, so it gets
  no highlighting).
- Colour feel: "Matte Crimson" — the palette already named in the roadmap.
- Settings layout: tabs across the top (chosen over a left sidebar:
  simpler, and works when the window is resized narrow).

### Palette (`ui/theme.py`, named constants)

| Name | Value | Use |
|------|-------|-----|
| `BACKGROUND` | `#0d0d0f` | window backgrounds |
| `SURFACE` | `#16161a` | panels, inputs, tab bar, widget card |
| `BORDER` | `#2a2a2e` | 1 px borders, separators |
| `TEXT` | `#f5f5f5` | normal text |
| `TEXT_MUTED` | `#9e9e9e` | secondary labels, inactive tabs |
| `ACCENT` | `#e53935` | selected tab underline, focus ring, progress bar, slider |
| `ACCENT_BRIGHT` | `#ff1744` | "listening" tray state, hover on accent items |
| `BUTTON_PRIMARY` | `#c62828` | background of primary buttons (Save, Read) |
| `HIGHLIGHT_BG` | `#5c1a1d` | background of the sentence being read |

Contrast rule: white text on `#e53935` is only about 4.2:1, below the
WCAG AA minimum of 4.5:1 for normal-size text, so text never sits on
`ACCENT`; buttons with text use `BUTTON_PRIMARY` (white on `#c62828` is
about 5.6:1). `TEXT` on `HIGHLIGHT_BG` and `TEXT_MUTED` on `BACKGROUND`
also pass 4.5:1. A unit test computes these ratios from the constants so a
later palette edit cannot quietly break readability.

### Components

1. **`ui/theme.py` (new).** The constants above, one Qt stylesheet string
   built from them, and `apply_theme(app: QApplication) -> None`, which sets
   the Fusion style, a dark `QPalette` (so widgets the stylesheet does not
   cover — menus, file dialogs drawn by Qt, tooltips — are also dark) and
   the stylesheet. `cli.main()` calls it once, right after the
   `QApplication` is created. No window styles itself with literal colours;
   everything comes from `theme.py`.
2. **Tray icon (`ui/tray.py`).** Idle: grey filled circle (`TEXT_MUTED`).
   Listening: bright red filled circle (`ACCENT_BRIGHT`). Reading: red ring
   (`ACCENT`) with a white centre dot. The three states differ in shape as
   well as colour, so they are told apart without relying on colour alone.
   The tray menu is styled by the theme. Menu text and tooltips stay as
   they are (tests assert them).
3. **Floating widget (`ui/floating_widget.py`).** A rounded card (8 px
   radius, `SURFACE`, 1 px `BORDER`) drawn in `paintEvent` on a translucent
   window; source name in bold, status in `TEXT_MUTED`, a thin red
   `QProgressBar` (current piece / total pieces) above the existing
   "piece N of M" label, then Pause/Stop. It can be dragged with the mouse;
   a dragged position is kept until Vocalis quits (it is not saved), and
   until the first drag each reading still opens bottom-right. The existing
   window flags (tool, frameless, stay-on-top, never takes focus) are kept.
4. **Settings window (`ui/settings_window.py`).** A `QTabWidget` with five
   tabs; Save, Close and the message line stay below the tabs and apply to
   all tabs at once:
   - *General:* desktop notifications, floating widget, sound output +
     Test sound.
   - *Voice & Engine:* Qwen3-TTS voice, Quick Reader engine, Document
     Reader engine, Kokoro voice, speaking speed, Preview voice.
   - *Performance:* max CPU threads, max RAM, GPU VRAM ceiling, and NEW:
     "Unload the voice after N seconds idle" (`model_idle_timeout_seconds`,
     1-3600).
   - *Shortcuts:* the four shortcut drop-downs.
   - *Model files (NEW):* the six model locations
     (`parakeet_model_dir`, `qwen3_tts_model_dir`, `piper_binary_path`,
     `piper_voice_model_path`, `kokoro_model_path`, `kokoro_voices_path`),
     each a text field with a Browse… button. A location the user CHANGED
     that does not exist (including an emptied field) is refused on Save
     with a plain message naming it ("Not saved: the Kokoro model file was
     not found at …"); nothing is written. Unchanged locations are not
     checked, so a PC where a model was never installed can still save
     other settings.
   `engines.stt` is not shown: it has exactly one possible value.
   The idle timer and model files are read when Vocalis starts, so a new
   `LiveSettings.engines_need_restart()` (same pattern as
   `shortcuts_need_restart()`) adds "Restart Vocalis to use the new model
   settings." to the Saved message when either changed.
5. **Document Reader window (`ui/doc_popup.py`).** Same controls, restyled:
   the text area on `SURFACE`, Read as the primary button, a thin red
   progress bar next to the status line. Plus highlighting (below).

### Highlight data flow

- `DocumentReaderFeature` switches from `chunk_text()` to the existing
  `chunk_spans()` and gains two optional callbacks (defaults `None`, so all
  existing callers and tests keep working):
  - `on_document(text: str)` — called once per reading, after
    normalization and before the first sound, with the normalized text
    the spans refer to.
  - `on_chunk_span(start: int, end: int)` — called at the same moment as
    `on_progress`, with the character span of the piece now being heard.
  Both are guarded by the same "still the current reading" check as
  `on_progress`, and a callback that raises is logged and ignored, like
  `on_started`.
- `cli.build_doc_reader()` wires them to two new popup signals,
  `document_loaded = Signal(str)` and `span_started = Signal(int, int)`;
  the signals carry the call from the reader's background thread to the
  window's own thread (the same safe pattern every other popup signal uses).
- The popup, on `document_loaded`: shows the normalized text, makes the
  box read-only. On `span_started`: sets one `QTextEdit.ExtraSelection`
  (background `HIGHLIGHT_BG`) over the span and scrolls it into view
  (`ensureCursorVisible()` after moving the cursor to the span's start),
  without taking focus. Pause keeps the highlight. On `reading_finished`:
  clears the highlight and makes the box editable again; the text stays.
- Plain-English consequence the user should know: when you paste text or
  open a file, the box shows the cleaned-up text Vocalis actually reads
  (e.g. markdown symbols removed), because the highlight positions belong
  to that cleaned text.
- Qt counts text positions in UTF-16 units, Python in characters, so an
  emoji or other character outside the Basic Multilingual Plane would shift
  the highlight. The popup converts each span from Python positions to Qt
  positions using the shown text before highlighting.
- Highlight is per piece (one or a few sentences). The voice engines give
  no word timings, so word-by-word highlighting is not possible.

### Error handling

- A failing highlight callback never stops a reading (logged, ignored).
- A span outside the shown text (cannot happen by construction, but a
  stale signal could arrive after a new document loaded) is clamped to the
  text length; an empty span clears the highlight.
- Theme failure: `apply_theme()` only calls Qt setters; there is nothing
  that can fail at runtime beyond Qt itself, so no extra handling.

### Performance impact

Negligible. The stylesheet is parsed once at start-up. Highlighting does one
small selection update per piece (every few seconds). The widget repaints
only when its labels change. `setPlainText()` runs once per reading on text
already capped by the extractor's size limit. No new dependencies.

### Testing

All headless (offscreen Qt + qtbot), no real audio or hardware:
theme constants' contrast ratios; `apply_theme` sets style, palette and
stylesheet; tray icon pixels per state; floating widget keeps its flags,
shows the progress bar value, drags; Settings tab names and which control
is on which tab, every config field round-trips through the window, a
missing model path is refused, restart note for engine settings; feature
fires `on_document` with the normalized text and `on_chunk_span` with each
chunk's span in order, never for a stale reading, survives a raising
callback; popup highlight covers exactly the span, scrolls, clears on
finish, read-only while reading. Existing asserted strings are kept.

### Out of scope

Word-level highlighting (engines give no timings); saving the widget
position; light theme or theme choice; text view for Quick Reader;
installer wizard and its styling (Stage 8 reuses `ui/theme.py`).
