# Vocalis Folder Structure

```
README.md                 # top-level project README: what Vocalis is, links to the docs below
LICENSE                    # MIT license text
src/vocalis/
├── __init__.py         # package version
├── audio/
│   ├── capture.py        # MicRecorder: sounddevice-based mic capture
│   ├── output_device.py   # OutputDeviceChooser: probes and caches a working sound output
│   ├── pace.py            # apply_speed/paced_audio: pitch-preserving speaking-pace change
│   └── playback.py        # AudioPlayer: sounddevice-based interruptible playback
├── cli.py               # `vocalis` console-script entrypoint
├── config.py           # VocalisConfig pydantic model + TOML load/save
├── daemon.py            # Unix socket server lifecycle (start/stop)
├── engines/
│   ├── parakeet.py        # ParakeetSTT: onnx-asr wrapper implementing ModelEngine
│   ├── piper_tts.py        # PiperTTS: CPU-only fallback voice, never GPU-managed
│   ├── qwen3_tts.py        # Qwen3TTS: primary voice, GPU, via ModelManager
│   └── tesseract_ocr.py    # TesseractOCR: CPU-only text recognition, never GPU-managed
├── features/
│   ├── dictation.py       # DictationFeature: press/release orchestration
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
├── startup_checks.py    # five non-fatal checks run once at startup (Stage 2, Stage 5)
├── notify.py               # Notifier: desktop notifications (notify-send)
├── output/
│   ├── clipboard.py        # ClipboardReader/ClipboardWriter: clipboard text via xclip
│   ├── screen_capture.py   # ScreenRegionCapture: drag a box, get PNG bytes
│   └── typer.py            # TextOutput: ydotool typing + clipboard fallback
├── text/
│   ├── chunker.py          # chunk_text: sentence-bounded chunks
│   ├── extractors.py       # extract_text: txt/md/pdf/docx/html readers + typed errors
│   └── normalizer.py       # normalize_markdown: markdown -> speakable text
├── ui/
│   ├── doc_popup.py        # DocumentReaderPopup: PySide6 Document Reader window
│   ├── floating_widget.py  # FloatingWidget: small always-on-top reading box
│   ├── settings_window.py  # SettingsWindow: switches, shortcuts, voice
│   ├── status_hub.py       # StatusHub: Qt signals the tray and widget listen to
│   └── tray.py             # TrayIcon: coloured dot + menu
└── py.typed             # marks the package as type-checked (PEP 561)

tests/                    # pytest suite mirroring src/vocalis/ (≥85% coverage);
                          #   conftest.py sets QT_QPA_PLATFORM=offscreen for Qt tests
├── test_cross_feature_integration.py  # whole-system tests: real build_*() + one
                          #   shared ModelManager, not a single feature in isolation
├── test_hardware.py       # probe_hardware/describe/log_hardware_summary, every source faked
├── test_remote.py         # vocalis --settings client, against a real socket in tmp_path
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
    ├── uninstall.sh        # reverses install.sh; --purge also removes config/models/Piper
    ├── build-run.sh        # wraps install.sh/uninstall.sh into install.run/uninstall.run via makeself
    ├── README.md           # how to run/rebuild everything in this folder
    └── lib/
        ├── preflight.sh    # distro/arch/GPU-driver hard-gate checks
        ├── apt.sh          # missing-package detection/install, best-effort ydotoold enable
        ├── pyenv.sh        # uv install, wheel find-or-build, venv creation, wheel install
        ├── fetch_models.py # Parakeet/Qwen3-TTS/Piper-voice reuse-or-download via huggingface_hub
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
