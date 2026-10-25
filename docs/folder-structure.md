# Vocalis Folder Structure

```
README.md                 # top-level project README: what Vocalis is, links to the docs below
LICENSE                    # MIT license text
src/vocalis/
├── __init__.py         # package version
├── audio/
│   ├── capture.py        # MicRecorder: sounddevice-based mic capture
│   ├── speech_stream.py   # SpeechStream: one persistent gapless output stream per reading (Stage 6b)
│   ├── output_device.py   # OutputDeviceChooser: probes and caches a working sound output
│   ├── pace.py            # apply_speed/paced_audio: pitch-preserving speaking-pace change (Stage 6c: Qwen3-TTS's own capped 1.0-1.2x stretch only)
│   └── playback.py        # AudioPlayer: interruptible playback + open_stream/close_stream (Stage 6b)
├── cli.py               # main(): the heavy app startup (imported lazily by entry.py)
├── config.py           # VocalisConfig pydantic model + TOML load/save
├── daemon.py            # Unix socket server lifecycle (start/stop)
├── engines/
│   ├── auto_select.py     # pure Auto policy: which engine order for this PC right now (Stage 6a)
│   ├── base.py            # Synthesizer / TTSEngine protocols (Stage 6a)
│   ├── chain.py           # EngineChain: try engines in order, fall down on failure; hold/close a voice for a reading (Stage 6a, 6b)
│   ├── kokoro_tts.py      # KokoroTTS: light, natural voice via kokoro-onnx (Stage 6a; native speed control, Stage 6c)
│   ├── registry.py        # EngineSelector/EnginePlan: user choice + PC state -> engine order (Stage 6a)
│   ├── parakeet.py        # ParakeetSTT: onnx-asr wrapper implementing ModelEngine
│   ├── piper_tts.py        # PiperTTS: CPU-only fallback voice, never GPU-managed (native --length_scale speed, Stage 6c)
│   ├── qwen3_tts.py        # Qwen3TTS: primary voice, GPU, via ModelManager (tuned generation params + capped speed stretch, Stage 6c)
│   └── tesseract_ocr.py    # TesseractOCR: CPU-only text recognition, never GPU-managed
├── entry.py              # `vocalis` console-script entrypoint — stays light so --settings is fast (Stage 5 fix)
├── features/
│   ├── dictation.py       # DictationFeature: press/release orchestration
│   ├── chunked_speaker.py  # ChunkedSpeaker: producer loop shared by both readers (Stage 6b)
│   ├── doc_reader.py       # DocumentReaderFeature: extract/chunk/speak/pause/stop
│   ├── ocr_reader.py       # OcrReaderFeature: capture -> OCR -> clipboard -> read aloud
│   ├── quick_reader.py     # QuickReaderFeature: trigger/stop orchestration
│   └── reading_coordinator.py  # ReadingCoordinator: one reader speaks at a time
├── hotkeys/
│   ├── ctrl_alt_key.py     # CtrlAltKeyListener: evdev Ctrl+Alt+<key> detection
│   ├── ctrl_alt_r.py       # CtrlAltRListener: thin wrapper fixed to R
│   ├── numlock.py         # NumlockListener: evdev push-to-talk (NUMLOCK by default)
│   ├── shortcut_parser.py  # shortcut text <-> evdev key codes, validation
│   └── stoppable_device.py # StoppableEvdevDevice: read loop that stops on close()
├── hardware.py          # probe_hardware/describe/log_hardware_summary (Stage 5)
├── logging_setup.py     # shared "vocalis" logger configuration
├── model_manager.py     # GPU single-residency + idle-unload policy
├── remote.py             # vocalis --settings client: talks to the daemon's socket (Stage 5)
├── settings.py          # LiveSettings: shared live config holder + voice list
├── startup_checks.py    # seven non-fatal checks run once at startup (Stage 2, 5, 6a)
├── notify.py               # Notifier: desktop notifications (notify-send)
├── output/
│   ├── clipboard.py        # ClipboardReader/ClipboardWriter: clipboard text via xclip
│   ├── screen_capture.py   # ScreenRegionCapture: drag a box, get PNG bytes
│   └── typer.py            # TextOutput: ydotool typing + clipboard fallback
├── text/
│   ├── chunker.py          # chunk_spans/chunk_text: sentence-aware chunks with exact spans (Stage 6b)
│   ├── extractors.py       # extract_text: txt/md/pdf/docx/html readers + typed errors
│   └── normalizer.py       # normalize_markdown + normalize_text: markdown/plain text -> speakable text (numbers, URLs, symbols; Stage 6c)
├── ui/
│   ├── doc_popup.py        # DocumentReaderPopup: PySide6 Document Reader window
│   ├── floating_widget.py  # FloatingWidget: small always-on-top reading box
│   ├── settings_window.py  # SettingsWindow: switches, shortcuts, voice
│   ├── status_hub.py       # StatusHub: Qt signals the tray and widget listen to
│   └── tray.py             # TrayIcon: coloured dot + menu
└── py.typed             # marks the package as type-checked (PEP 561)

tests/                    # pytest suite mirroring src/vocalis/ (≥85% coverage);
                          #   conftest.py sets QT_QPA_PLATFORM=offscreen for Qt tests
├── test_auto_select.py    # Auto policy rules (Stage 6a)
├── test_kokoro_tts.py     # Kokoro engine with fake builders (Stage 6a)
├── test_engine_registry.py  # EngineSelector (Stage 6a)
├── test_engine_chain.py   # EngineChain fall-down behaviour (Stage 6a) and holding a voice (Stage 6b)
├── fake_output.py         # fake sound output: ManualStream / PumpStream / FakeAudioPlayer (Stage 6b)
├── test_speech_stream.py  # SpeechStream (Stage 6b)
├── test_chunked_speaker.py  # ChunkedSpeaker (Stage 6b)
├── test_cross_feature_integration.py  # whole-system tests: real build_*() + one
                          #   shared ModelManager, not a single feature in isolation
├── test_hardware.py       # probe_hardware/describe/log_hardware_summary, every source faked
├── test_remote.py         # vocalis --settings client, against a real socket in tmp_path
├── test_entry.py          # vocalis.entry.entrypoint(): --settings vs. the lazy main() import (Stage 5 fix)
├── test_package_metadata.py  # LICENSE + pyproject.toml license metadata
└── installer/            # bats suite mirroring packaging/installer/
    ├── test_preflight.bats
    ├── test_apt.bats
    ├── test_pyenv.bats
    ├── test_models.bats
    ├── test_service.bats
    ├── test_desktop.bats
    ├── test_install.bats
    ├── test_uninstall.bats
    ├── test_build_run.bats
    └── fixtures/          # fake os-release files, fake systemd unit template, fake vocalis.desktop
config/
└── default_config.toml   # first-run config template

packaging/
├── systemd/
│   ├── vocalis.service    # systemd --user unit file template (placeholders filled in by Stage 3's installer)
│   └── README.md          # placeholder and design notes for the unit file; the systemd-analyze verify command
├── desktop/
│   └── vocalis.desktop    # application-menu launcher template (Stage 5); __VOCALIS_EXEC__ filled in by the installer
└── installer/
    ├── install.sh          # end-to-end, idempotent installer (preflight -> apt -> python env -> models -> service -> menu launcher)
    ├── uninstall.sh        # reverses install.sh; --purge also removes config/models/Piper/Kokoro
    ├── build-run.sh        # wraps install.sh/uninstall.sh into install.run/uninstall.run via makeself
    ├── README.md           # how to run/rebuild everything in this folder
    └── lib/
        ├── preflight.sh    # distro/arch/GPU-driver hard-gate checks
        ├── apt.sh          # missing-package detection/install, best-effort ydotoold enable
        ├── pyenv.sh        # uv install, wheel find-or-build, venv creation, wheel install
        ├── fetch_models.py # Parakeet/Qwen3-TTS/Piper-voice reuse-or-download via huggingface_hub; Kokoro files from a GitHub release
        ├── models.sh       # Piper binary reuse-or-download (the one non-Hugging-Face asset)
        ├── service.sh      # systemd unit placeholder substitution + enable/start
        └── desktop.sh      # application-menu launcher placeholder substitution (Stage 5)

docs/
├── superpowers/
│   ├── specs/           # brainstorming design docs
│   └── plans/            # writing-plans implementation plans (this file's home)
├── architecture.md
├── folder-structure.md   # this file
├── file-purpose.md
├── workflow.md
├── algorithms.md
├── user-docs.md
└── developer-docs.md

old/                       # untouched legacy Voxtype scripts/docs (reference only)
```

`hotkeys/`, `features/`, `engines/`, `audio/`, `output/`, and `notify.py`
were added in Phase 2 (see above); each gained a second file in Phase 3
(`ctrl_alt_r.py`, `quick_reader.py`, `piper_tts.py`/`qwen3_tts.py`,
`playback.py`, `clipboard.py`). Phase 4 added `text/`, `ui/`,
`doc_reader.py`, `reading_coordinator.py` and `ctrl_alt_key.py`.
Phase 5 added `ocr_reader.py`, `tesseract_ocr.py`, `screen_capture.py` and
`ClipboardWriter` (in `clipboard.py`).
Phase 6 added `settings.py`, `hotkeys/shortcut_parser.py` and four files in
`ui/` (`status_hub.py`, `tray.py`, `floating_widget.py`, `settings_window.py`),
with a matching test file for each in `tests/`.
Stage 1 (Sound and voice) added `audio/output_device.py` and `audio/pace.py`,
each with a matching test file (`tests/test_output_device.py`,
`tests/test_pace.py`), and extended `audio/playback.py`, `config.py`,
`settings.py`, `ui/settings_window.py`, `cli.py`,
`features/quick_reader.py` and `features/doc_reader.py` in place.
Stage 2 (Service and autostart) added `startup_checks.py` (with
`tests/test_startup_checks.py`), the `packaging/systemd/` folder (the
unit-file template and its README, with `tests/test_systemd_unit.py`), and
extended `daemon.py` and `cli.py` in place.
Stage 3 (Installer) added the whole `packaging/installer/` folder (above)
and `tests/installer/` (its `bats` suite) and
`tests/test_installer_fetch_models.py`; it also renamed two constants in
`startup_checks.py` to public names so the installer and the runtime
checks share one source of truth, and added
`packaging/installer/lib/fetch_models.py` to `pyproject.toml`'s mypy
scope.
Stage 4 (whole-system testing and docs hardening) added
`tests/test_cross_feature_integration.py` (above) and the top-level
`README.md`; it made no changes to `src/vocalis/`.
Stage 5 (license, hardware probe, findable Settings, desktop launcher)
added the top-level `LICENSE` file, `hardware.py` and `remote.py` (each
with a matching test file), the `packaging/desktop/` folder (the launcher
template), `packaging/installer/lib/desktop.sh` (with
`tests/installer/test_desktop.bats`), and `tests/test_package_metadata.py`;
it extended `startup_checks.py`, `daemon.py`, `cli.py`,
`features/dictation.py`, `ui/tray.py`, `pyproject.toml`, `install.sh`,
`uninstall.sh` and `build-run.sh` in place.
Stage 5's final-review fix wave added `entry.py` (with `tests/test_entry.py`)
so `vocalis --settings` no longer pays `cli.py`'s heavy import cost, moved
the `vocalis` console-script target from `vocalis.cli:entrypoint` to
`vocalis.entry:entrypoint`, and extended `remote.py`, `cli.py` and
`ui/settings_window.py` in place (no other new files).

Stage 6a (voice-engine plumbing) added `engines/auto_select.py`,
`engines/base.py`, `engines/chain.py`, `engines/registry.py` and
`engines/kokoro_tts.py` (with `tests/test_auto_select.py`,
`tests/test_kokoro_tts.py`, `tests/test_engine_registry.py` and
`tests/test_engine_chain.py`), and extended `config.py`, `settings.py`,
`startup_checks.py`, `cli.py`, `features/quick_reader.py`,
`features/doc_reader.py`, `ui/settings_window.py`, `fetch_models.py`,
`pyenv.sh`, `install.sh`, `uninstall.sh` and `pyproject.toml` in place.

Stage 6b (gapless long reading) added `audio/speech_stream.py` and
`features/chunked_speaker.py` (with `tests/test_speech_stream.py`,
`tests/test_chunked_speaker.py` and the shared test helper
`tests/fake_output.py`), and extended `text/chunker.py`, `audio/playback.py`,
`engines/chain.py`, `model_manager.py`, `features/quick_reader.py` and
`features/doc_reader.py` in place.
