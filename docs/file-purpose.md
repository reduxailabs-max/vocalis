# Purpose of Each File

| File | Purpose |
|---|---|
| `src/vocalis/__init__.py` | Exposes `__version__` for the package. |
| `src/vocalis/config.py` | `VocalisConfig`/`ShortcutsConfig`/`EnginesConfig` pydantic models; `load_config`/`save_config`/`default_config_path`; raises `ConfigError` on invalid input. |
| `src/vocalis/logging_setup.py` | `configure_logging()` — sets up the shared `"vocalis"` logger (console + optional file handler), idempotent. |
| `src/vocalis/model_manager.py` | `ModelManager` — enforces single-model VRAM residency and idle-unload; defines the `ModelEngine`/`TimerHandle` protocols later engine wrappers implement. |
| `src/vocalis/daemon.py` | `VocalisDaemon` — Unix socket server lifecycle: start/stop, stale-socket cleanup, graceful shutdown (calls `ModelManager.shutdown()`). |
| `src/vocalis/cli.py` | `vocalis` console-script entrypoint: loads config, builds the daemon and the dictation, quick reader, document reader and OCR reader features (`build_ocr_reader()` wires the Ctrl+Alt+O listener), then runs a Qt event loop until Ctrl+C (a SIGINT handler quits it so shutdown runs). |
| `config/default_config.toml` | First-run configuration template. |
| `pyproject.toml` | Package metadata, dependencies, mypy/pytest/coverage configuration. |
| `src/vocalis/engines/parakeet.py` | `ParakeetSTT` — wraps `onnx-asr` running the existing Voxtype Parakeet TDT ONNX model; implements `ModelEngine`. |
| `src/vocalis/audio/capture.py` | `MicRecorder` — sounddevice-based microphone capture with a safety-duration cutoff. |
| `src/vocalis/hotkeys/numlock.py` | `NumlockListener` — evdev NUMLOCK push-to-talk detection; `list_numlock_capable_devices()` is the real device enumerator. |
| `src/vocalis/output/typer.py` | `TextOutput` — types text via ydotool, falls back to clipboard (xclip) on any failure. |
| `src/vocalis/notify.py` | `Notifier` — desktop notifications via notify-send, no-op when disabled. |
| `src/vocalis/features/dictation.py` | `DictationFeature` — wires NUMLOCK press/release to record → transcribe → deliver, with notifications at each stage. |
| `src/vocalis/engines/qwen3_tts.py` | `Qwen3TTS` — wraps the `qwen-tts` package's CustomVoice model (preset voice, no cloning); implements `ModelEngine`. |
| `src/vocalis/engines/piper_tts.py` | `PiperTTS` — wraps the existing local `piper` binary via subprocess; CPU-only, never routed through `ModelManager`. |
| `src/vocalis/audio/playback.py` | `AudioPlayer` — sounddevice-based, interruptible speech playback. |
| `src/vocalis/hotkeys/ctrl_alt_r.py` | `CtrlAltRListener` — thin wrapper over `CtrlAltKeyListener` fixed to the R key; `list_hotkey_capable_devices()` is its real device enumerator. |
| `src/vocalis/output/clipboard.py` | `ClipboardReader` — reads clipboard text via `xclip`, the read-side counterpart to `TextOutput`. |
| `src/vocalis/features/quick_reader.py` | `QuickReaderFeature` — wires Ctrl+Alt+R trigger/stop to clipboard-read → speak (with fallback) → play, with notifications at each stage. |
| `src/vocalis/hotkeys/ctrl_alt_key.py` | `CtrlAltKeyListener` — evdev Ctrl+Alt+<any key> detection (one thread per device); `list_devices_with_key(code)` is the real enumerator. Used for Ctrl+Alt+R, Ctrl+Alt+D and Ctrl+Alt+O. |
| `src/vocalis/hotkeys/stoppable_device.py` | `StoppableEvdevDevice` — wraps a real `evdev.InputDevice` so `close()` wakes a blocked `read_loop()` immediately (via a private wake-up pipe). Applied by both real enumerators to the devices they return. Its read loop ends quietly (no error log) if `close()` happens while the watcher is busy. |
| `src/vocalis/features/reading_coordinator.py` | `ReadingCoordinator` — remembers which of Quick Reader / Document Reader owns audio playback and stops the previous owner when the other starts. |
| `src/vocalis/text/normalizer.py` | `normalize_markdown()` — strips markdown syntax (code fences, links, headings, bold/italic) so text is speakable. |
| `src/vocalis/text/chunker.py` | `chunk_text()` — splits text into sentence-bounded chunks (default 300 characters) for chunk-by-chunk speech and progress. |
| `src/vocalis/text/extractors.py` | `extract_text()` — reads .txt/.md/.pdf/.docx/.html/.htm files; reads merged Word table cells (side by side or across rows) once, warns about tables nested deeper than 10 levels; raises typed errors (`UnsupportedFileTypeError`, `DocumentOpenError`, `NoReadableTextError`, `DocumentTooLargeError`). |
| `src/vocalis/features/doc_reader.py` | `DocumentReaderFeature` — extract → normalize → chunk → synthesize (Qwen3-TTS, Piper fallback) → play, with pause/resume/stop and progress callbacks; control notifications are sent in order from one background thread. No Qt import. |
| `src/vocalis/ui/doc_popup.py` | `DocumentReaderPopup` — the PySide6 window (Open file, text box, Read, Pause/Resume, Stop, status and progress labels). Its signals are the only way other threads reach the window. |
| `src/vocalis/output/screen_capture.py` | `ScreenRegionCapture` — lets the user drag a box (spectacle with temp dir, else maim, else flameshot, else slurp+grim) and returns the picture as PNG bytes in memory, or `None` if cancelled; `ScreenCaptureUnavailable` when no tool is installed. Only accepts an ASCII `x,y WxH` region from slurp. |
| `src/vocalis/engines/tesseract_ocr.py` | `TesseractOCR` — runs the `tesseract` binary (CPU only, never via `ModelManager`) on PNG bytes, English, whitespace collapsed; raises `TesseractError` with a user-readable message. |
| `src/vocalis/output/clipboard.py` (also) | `ClipboardWriter` — writes text to the clipboard via `xclip` (output sent to DEVNULL), returns True/False, never raises. |
| `src/vocalis/features/ocr_reader.py` | `OcrReaderFeature` — Ctrl+Alt+O: capture -> Tesseract -> clipboard -> notify -> popup signal -> `DocumentReaderFeature.read_text`, on a background thread; a second press stops a reading it started. |
| `src/vocalis/features/doc_reader.py` (also) | Gained the read-only `generation` property, which the OCR Reader uses to recognise its own reading. |
| `src/vocalis/ui/doc_popup.py` (also) | Gained the `external_reading_started` signal, which shows a reading started by another feature with Pause/Stop enabled. |

## Phase 6 (UI) — new files

| File | Purpose |
|---|---|
| `src/vocalis/settings.py` | `LiveSettings` — thread-safe holder of the current config; the only writer is the Settings window (`update()` validates shortcuts, saves atomically, then adopts). Also the `VOICES` list (the nine voices). |
| `src/vocalis/hotkeys/shortcut_parser.py` | Turns `Ctrl+Alt+<letter or digit>` and the push-to-talk key (NUMLOCK/SCROLLLOCK/PAUSE) into evdev key codes; rejects bad and duplicate shortcuts with a `ShortcutError`; `DEFAULT_SHORTCUTS`. |
| `src/vocalis/ui/status_hub.py` | `StatusHub` (Qt signals: state, status, progress, reading started/finished, paused) and `ReadingControls` (stop/pause/resume callables). |
| `src/vocalis/ui/tray.py` | `TrayIcon` — coloured dot drawn in code, tooltip, right-click menu (status, Pause/Resume, Stop, Open Document Reader, Settings, Quit); `close()` for safe teardown. |
| `src/vocalis/ui/floating_widget.py` | `FloatingWidget` — frameless always-on-top box shown only while reading, never takes focus, Pause/Resume + Stop. |
| `src/vocalis/ui/settings_window.py` | `SettingsWindow` — two switches, four shortcut drop-downs and the voice drop-down; Save / Close; shows "Saved." or "Not saved: …". |
| `tests/test_settings.py`, `test_shortcut_parser.py`, `test_status_hub.py`, `test_tray.py`, `test_floating_widget.py`, `test_settings_window.py` | Tests for the six new modules above. |
| `tests/test_reading_hooks_integration.py` | Checks that Quick Reader, Document Reader, dictation and OCR report correctly into a real `StatusHub`. |

## Phase 6 (UI) — changed files

| File | Change |
|---|---|
| `src/vocalis/config.py` | `save_config` is atomic (temp file + rename), cleans up on any failure and leaves the file private to the user (mode 0600). |
| `src/vocalis/notify.py` | `Notifier(enabled=...)` accepts a function checked on every notification; a failing function is logged and treated as "off". |
| `src/vocalis/engines/qwen3_tts.py` | `speaker` may be a function, read for each synthesis (live voice). |
| `src/vocalis/hotkeys/numlock.py` | The push-to-talk key code is configurable (default NUMLOCK). |
| `src/vocalis/features/quick_reader.py` | Optional `on_reading_changed` hook (reported from state under a lock), a public `stop()`, and a stale reading no longer releases a newer reading's coordinator claim. |
| `src/vocalis/features/doc_reader.py` | Optional `on_started` hook, called after the coordinator hand-over. |
| `src/vocalis/features/dictation.py` | Optional `on_listening_changed` hook. |
| `src/vocalis/features/ocr_reader.py` | Unchanged; its `on_reading_starting` hook is now wired by `cli.py` to label the reading "Screen text" and to skip the popup when the widget is on. |
| `src/vocalis/cli.py` | `resolve_shortcuts()`, `build_ui()`; builders take optional `settings`, `hub` and `key_code`; `main()` creates `LiveSettings`, `StatusHub`, tray, widget, Settings window, and shuts them down in order. |
| `src/vocalis/ui/doc_popup.py` | Unchanged. |

## Stage 1 (Sound and voice) — new files

| File | Purpose |
|---|---|
| `src/vocalis/audio/pace.py` | `apply_speed()`/`clamp_speed()` — pitch-preserving speed change via `librosa.effects.time_stretch` (imported lazily, inside the function); `paced_audio()` — the one helper Quick Reader and Document Reader both call to apply the current speaking pace, swallowing any pacing failure and returning the original audio unchanged. |
| `src/vocalis/audio/output_device.py` | `OutputDeviceChooser` — probes candidate sound outputs (opens a stream, waits for the callback to fire) and caches the first one that really works; `mark_failed()`/`reset()`; a short-lived negative cache (15s default) so a "nothing works" answer is retried instead of sticking forever. |

## Stage 1 (Sound and voice) — changed files

| File | Change |
|---|---|
| `src/vocalis/audio/playback.py` | `AudioPlayer` gained `device_provider` (asked for a device at the start of every `play()`) and `on_device_failed(device)` (told about a stall or open failure); one retry on an open/start failure if the provider now names a different device. |
| `src/vocalis/config.py` | New fields `audio_output_device` (default `"auto"`, non-blank) and `speaking_speed` (default `1.15`, range `0.7`–`1.6`, bounds taken from `vocalis.audio.pace`). |
| `src/vocalis/settings.py` | `LiveSettings` gained `audio_output_device()` and `speaking_speed()` live accessors. |
| `src/vocalis/ui/settings_window.py` | Gained the Sound output drop-down + **Test sound** button, the Speaking speed slider, and the **Preview voice** button; takes `output_names`/`on_test_sound`/`on_preview_voice` callables. |
| `src/vocalis/cli.py` | Builds one shared `OutputDeviceChooser`, wires it into both readers' `AudioPlayer`s and into `build_ui()`'s Settings window; adds `_make_device_failure_hook()`, `_make_on_test_sound()`, `_make_on_preview_voice()` and the `_test_tone()` helper. |
| `src/vocalis/features/quick_reader.py` | Gained `speed_provider`; pacing applied once per reading via `paced_audio()`. |
| `src/vocalis/features/doc_reader.py` | Gained `speed_provider`; pacing applied once per chunk via `paced_audio()`, before that chunk's pause/resume retry loop (a new `_play_chunk_until_done()` helper replays the already-paced audio on resume instead of resynthesizing). |

## Stage 2 (Service and autostart) — new files

| File | Purpose |
|---|---|
| `src/vocalis/startup_checks.py` | Four independent, non-fatal startup checks (Parakeet model files, Qwen3-TTS model files, `ydotoold`'s socket, OCR's `tesseract`/`xclip`), each a function returning a human-readable message or `None`; `run_startup_checks(config)` runs all four in a fixed order and never lets one raising stop the rest. |
| `packaging/systemd/vocalis.service` | systemd `--user` unit file **template** — `__VOCALIS_EXEC__`/`__VOCALIS_WORKDIR__` placeholders, `Restart=on-failure`/`RestartSec=3`/`StartLimitBurst=5`/`StartLimitIntervalSec=60`, `WantedBy=xdg-desktop-autostart.target`, `After=`/`Wants=` on the three PipeWire user services. Not installed, enabled, or started by this stage. |
| `packaging/systemd/README.md` | Explains the two placeholders and who fills them in (Stage 3's installer), why `xdg-desktop-autostart.target` is used instead of `default.target`, the PipeWire ordering rationale, the crash-recovery settings, and the text-only `systemd-analyze --user verify` command. |
| `tests/test_systemd_unit.py` | Text-only checks on the unit file's content: restart policy, autostart target (not `default.target`), PipeWire `After=`/`Wants=`, the `__VOCALIS_EXEC__` placeholder (never a real path), and that the README documents it. Never starts a real service. |
| `tests/test_startup_checks.py` | 13 tests for the four checks (present/missing/partially-missing) and `run_startup_checks()`'s fixed ordering, silence-when-everything-is-fine, and survival of a check that raises. |

## Stage 2 (Service and autostart) — changed files

| File | Change |
|---|---|
| `src/vocalis/daemon.py` | `VocalisDaemon.start()` gained the connect-before-delete single-instance probe (`_is_another_instance_alive()`, three outcomes: alive/stale/ambiguous); the constructor gained the injectable `socket_factory` keyword (defaults to a real `socket.socket`), used only for this probe. |
| `src/vocalis/cli.py` | `main()` installs a `SIGTERM` handler alongside `SIGINT` (identical `app.quit()` lambda; both restored on exit); right after `load_config()`, builds a `Notifier` and calls `run_startup_checks(config)`, notifying each returned message. |
| `tests/test_daemon.py` | 6 new tests for the three probe outcomes and the existing-file/no-existing-file start paths, using fake socket classes (`_RefusingSocket`, `_AcceptingSocket`, `_OddlyFailingSocket`) instead of real sockets. |
| `tests/test_cli.py` | 2 new tests for the startup-checks notifications (every real problem reported; silent when everything is fine), plus the autouse `_no_real_startup_checks` fixture (patches `run_startup_checks` to `[]` for every test in the file) and a regression test guarding it — see `docs/developer-docs.md`. |

## Stage 3 (Installer) — new files

| File | Purpose |
|---|---|
| `packaging/installer/lib/preflight.sh` | Hard-gate checks run first: `preflight_check_distro` (Debian/Ubuntu via `/etc/os-release`'s `ID`/`ID_LIKE`), `preflight_check_arch` (`x86_64`), `preflight_check_gpu` (`nvidia-smi` succeeds); `preflight_run()` runs all three and stops at the first failure with a plain message. |
| `packaging/installer/lib/apt.sh` | `APT_REQUIRED_PACKAGES` (`tesseract-ocr`, `xclip`, `ydotool`); `apt_is_installed`/`apt_missing_packages`/`apt_install_missing` (one `sudo apt-get install` call for everything missing); `apt_enable_ydotoold` (best-effort `systemctl --user enable --now ydotool.service`, never fails the install). |
| `packaging/installer/lib/pyenv.sh` | `pyenv_ensure_uv` (installs `uv` via a version-pinned script only if missing); `pyenv_find_or_build_wheel` (a bundled `dist/*.whl`, else `uv build`s one from the repo); `pyenv_create_venv`/`pyenv_install_vocalis` (provisions Python 3.13 and installs the wheel via `uv`). |
| `packaging/installer/lib/fetch_models.py` | Reuse-or-download the three Hugging-Face-hosted assets (Parakeet, Qwen3-TTS, the Piper voice) at `VocalisConfig()`'s own default paths, importing `vocalis.startup_checks.PARAKEET_MODEL_FILES`/`QWEN3_TTS_MODEL_FILES` as the single source of truth for "which files make a model complete". |
| `packaging/installer/lib/models.sh` | `models_ensure_piper_binary` — reuse-or-download the one asset not on Hugging Face (the Piper binary itself, from a pinned GitHub release tarball), extracting to a temp directory first and only moving the result into place once complete. |
| `packaging/installer/lib/service.sh` | `service_substitute_unit` (fills `packaging/systemd/vocalis.service`'s `__VOCALIS_EXEC__`/`__VOCALIS_WORKDIR__` placeholders); `service_enable_and_start` (`systemctl --user daemon-reload` then `enable --now`). |
| `packaging/installer/install.sh` | The real end-to-end, idempotent installer: wires preflight → apt → python env → models → service into one `main()`, sourceable for `bats` testing with a run-guard for real use. |
| `packaging/installer/uninstall.sh` | Reverses `install.sh`: disables/stops the service, removes the unit file, the `~/.local/bin/vocalis` symlink and the venv; keeps config, downloaded models and the Piper backup voice by default, `--purge` also removes those (including Piper's binary and voice directories); `-h`/`--help` prints usage and removes nothing. |
| `packaging/installer/build-run.sh` | Builds the Vocalis wheel and wraps `install.sh`/`uninstall.sh` (plus `lib/`, the wheel and the systemd template) into self-extracting `install.run`/`uninstall.run` files via `makeself`. Developer-only; never run as part of a real install. |
| `packaging/installer/README.md` | How to run the installer/uninstaller and how to rebuild the `.run` files. |
| `tests/installer/*.bats` | `bats` unit tests for every `lib/*.sh` file and for `install.sh`/`uninstall.sh`'s `main()`, with every external command (`dpkg`, `apt-get`, `curl`, `uv`, `systemctl`, `nvidia-smi`, `tar`, `makeself`) faked out via the `*_run_*` override idiom. |
| `tests/installer/fixtures/*` | Fixture files used by the `bats` tests: fake `/etc/os-release` files (Debian, Ubuntu, Fedora) and a fake systemd unit template. |
| `tests/test_installer_fetch_models.py` | `pytest` tests for `fetch_models.py`, loaded by file path (it lives outside the `vocalis` package) with `huggingface_hub`'s download functions patched — no real network access. |

## Stage 3 (Installer) — changed files

| File | Change |
|---|---|
| `src/vocalis/startup_checks.py` | `_PARAKEET_FILES`/`_QWEN3_TTS_FILES` renamed to public `PARAKEET_MODEL_FILES`/`QWEN3_TTS_MODEL_FILES` — same values, now the installer's single source of truth as well as the runtime startup checks'. |
| `pyproject.toml` | `[tool.mypy] files` gained `packaging/installer/lib/fetch_models.py`, so it is checked under the same strict settings as the rest of the package. |

## Stage 4 (Whole-system testing and docs hardening) — new files

| File | Purpose |
|---|---|
| `tests/test_cross_feature_integration.py` | Whole-system tests that reuse `cli.py`'s own `build_*` functions against one real, shared `ModelManager` and `LiveSettings` — unlike every other test file, which tests one feature in isolation with its own fakes. Covers: single-GPU-model residency held across features, Quick Reader/Document Reader reusing the already-loaded Qwen3-TTS engine, hotkeys never cross-triggering another feature's handler, a settings change picked up by a different feature's next reading, two features' notifications both reaching the one shared notifier, and a full two-feature lifecycle ending in a clean shutdown. See `docs/developer-docs.md` for how this differs from the rest of the test suite. |
| `README.md` (repo root) | Top-level project README: what Vocalis is, quick links to install/usage/developer docs, and an accurate list of every network download the installer performs (added after an initial privacy-claim fix). |
