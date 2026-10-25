# Purpose of Each File

| File | Purpose |
|---|---|
| `src/vocalis/__init__.py` | Exposes `__version__` for the package. |
| `src/vocalis/config.py` | `VocalisConfig`/`ShortcutsConfig`/`EnginesConfig` pydantic models; `load_config`/`save_config`/`default_config_path`; raises `ConfigError` on invalid input. |
| `src/vocalis/logging_setup.py` | `configure_logging()` — sets up the shared `"vocalis"` logger (console + optional file handler), idempotent. |
| `src/vocalis/model_manager.py` | `ModelManager` — enforces single-model VRAM residency and idle-unload; defines the `ModelEngine`/`TimerHandle` protocols later engine wrappers implement. |
| `src/vocalis/daemon.py` | `VocalisDaemon` — Unix socket server lifecycle: start/stop, stale-socket cleanup, graceful shutdown (calls `ModelManager.shutdown()`). |
| `src/vocalis/cli.py` | `main()` — the heavy app startup: loads config, builds the daemon and the dictation, quick reader, document reader and OCR reader features (`build_ocr_reader()` wires the Ctrl+Alt+O listener), then runs a Qt event loop until Ctrl+C (a SIGINT handler quits it so shutdown runs). Imported lazily by `vocalis.entry.entrypoint()`, the actual `vocalis` console-script target (Stage 5). |
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
| `src/vocalis/features/quick_reader.py` | `QuickReaderFeature` — wires Ctrl+Alt+R trigger/stop to clipboard-read → speak → play (the voice engine is chosen per reading, Automatic or the Settings choice, and falls down the chain Qwen3-TTS → Kokoro → Piper), with notifications at each stage. |
| `src/vocalis/hotkeys/ctrl_alt_key.py` | `CtrlAltKeyListener` — evdev Ctrl+Alt+<any key> detection (one thread per device); `list_devices_with_key(code)` is the real enumerator. Used for Ctrl+Alt+R, Ctrl+Alt+D and Ctrl+Alt+O. |
| `src/vocalis/hotkeys/stoppable_device.py` | `StoppableEvdevDevice` — wraps a real `evdev.InputDevice` so `close()` wakes a blocked `read_loop()` immediately (via a private wake-up pipe). Applied by both real enumerators to the devices they return. Its read loop ends quietly (no error log) if `close()` happens while the watcher is busy. |
| `src/vocalis/features/reading_coordinator.py` | `ReadingCoordinator` — remembers which of Quick Reader / Document Reader owns audio playback and stops the previous owner when the other starts. |
| `src/vocalis/text/normalizer.py` | `normalize_markdown()` — strips markdown syntax (code fences, links, headings, bold/italic) so text is speakable. |
| `src/vocalis/text/chunker.py` | `chunk_text()` — splits text into sentence-bounded chunks (default 300 characters) for chunk-by-chunk speech and progress. |
| `src/vocalis/text/extractors.py` | `extract_text()` — reads .txt/.md/.pdf/.docx/.html/.htm files; reads merged Word table cells (side by side or across rows) once, warns about tables nested deeper than 10 levels; raises typed errors (`UnsupportedFileTypeError`, `DocumentOpenError`, `NoReadableTextError`, `DocumentTooLargeError`). |
| `src/vocalis/features/doc_reader.py` | `DocumentReaderFeature` — extract → normalize → chunk → synthesize (voice engine chosen per reading, Automatic or the Settings choice, falling down the chain Qwen3-TTS → Kokoro → Piper) → play, with pause/resume/stop and progress callbacks; control notifications are sent in order from one background thread. No Qt import. |
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
| `src/vocalis/ui/settings_window.py` | `SettingsWindow` — two switches, four shortcut drop-downs, the voice drop-down, the Quick Reader / Document Reader voice-engine drop-downs and the Kokoro voice drop-down (Stage 6a), the Sound output drop-down with Test sound, the Speaking speed slider and Preview voice (Stage 1); Save / Close; shows "Saved." or "Not saved: …". |
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
| `src/vocalis/audio/pace.py` | `apply_speed()`/`clamp_speed()` — pitch-preserving speed change via `librosa.effects.time_stretch` (imported lazily, inside the function); `paced_audio()` — swallows any pacing failure and returns the original audio unchanged (since Stage 6c, its only caller is `Qwen3TTS.synthesize()`, capped to 1.0-1.2x; Kokoro and Piper use their own native speed controls instead — see Stage 6c below). |
| `src/vocalis/audio/output_device.py` | `OutputDeviceChooser` — probes candidate sound outputs (opens a stream, waits for the callback to fire) and caches the first one that really works; `mark_failed()`/`reset()`; a short-lived negative cache (15s default) so a "nothing works" answer is retried instead of sticking forever. |

## Stage 1 (Sound and voice) — changed files

| File | Change |
|---|---|
| `src/vocalis/audio/playback.py` | `AudioPlayer` gained `device_provider` (asked for a device at the start of every `play()`) and `on_device_failed(device)` (told about a stall or open failure); one retry on an open/start failure if the provider now names a different device. |
| `src/vocalis/config.py` | New fields `audio_output_device` (default `"auto"`, non-blank) and `speaking_speed` (default `1.15`, range `0.7`–`1.6`, bounds taken from `vocalis.audio.pace`). |
| `src/vocalis/settings.py` | `LiveSettings` gained `audio_output_device()` and `speaking_speed()` live accessors. |
| `src/vocalis/ui/settings_window.py` | Gained the Sound output drop-down + **Test sound** button, the Speaking speed slider, and the **Preview voice** button; takes `output_names`/`on_test_sound`/`on_preview_voice` callables. |
| `src/vocalis/cli.py` | Builds one shared `OutputDeviceChooser`, wires it into both readers' `AudioPlayer`s and into `build_ui()`'s Settings window; adds `_make_device_failure_hook()`, `_make_on_test_sound()`, `_make_on_preview_voice()` and the `_test_tone()` helper. |
| `src/vocalis/features/quick_reader.py` | Gained `speed_provider`; pacing applied via `paced_audio()` (since Stage 6b once per chunk, inside `ChunkedSpeaker`; since Stage 6c `speed_provider` was removed again — each voice engine paces its own output now, see Stage 6c below). |
| `src/vocalis/features/doc_reader.py` | Gained `speed_provider`; pacing applied via `paced_audio()` (since Stage 6b pacing happens once per chunk inside `ChunkedSpeaker`, and the old `_play_chunk_until_done()` helper was removed; since Stage 6c `speed_provider` was removed again — each voice engine paces its own output now, see Stage 6c below). |

## Stage 2 (Service and autostart) — new files

| File | Purpose |
|---|---|
| `src/vocalis/startup_checks.py` | Seven independent, non-fatal startup checks (Parakeet model files, Qwen3-TTS model files, the Kokoro voice files (Stage 6a), `ydotoold`'s socket, OCR's `tesseract`/`xclip`, the onnxruntime graphics-card check, and a stale Voxtype `~/.asoundrc` (Stage 5)), each a function returning a human-readable message or `None`; `run_startup_checks(config)` runs all seven in a fixed order and never lets one raising stop the rest. |
| `packaging/systemd/vocalis.service` | systemd `--user` unit file **template** — `__VOCALIS_EXEC__`/`__VOCALIS_WORKDIR__` placeholders, `Restart=on-failure`/`RestartSec=3`/`StartLimitBurst=5`/`StartLimitIntervalSec=60`, `WantedBy=xdg-desktop-autostart.target`, `After=`/`Wants=` on the three PipeWire user services. Not installed, enabled, or started by this stage. |
| `packaging/systemd/README.md` | Explains the two placeholders and who fills them in (Stage 3's installer), why `xdg-desktop-autostart.target` is used instead of `default.target`, the PipeWire ordering rationale, the crash-recovery settings, and the text-only `systemd-analyze --user verify` command. |
| `tests/test_systemd_unit.py` | Text-only checks on the unit file's content: restart policy, autostart target (not `default.target`), PipeWire `After=`/`Wants=`, the `__VOCALIS_EXEC__` placeholder (never a real path), and that the README documents it. Never starts a real service. |
| `tests/test_startup_checks.py` | Tests for each startup check (present/missing/partially-missing) and `run_startup_checks()`'s fixed ordering, silence-when-everything-is-fine, and survival of a check that raises. |

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

## Stage 5 (License, hardware probe, findable Settings, desktop launcher) — new files

| File | Purpose |
|---|---|
| `LICENSE` | The project's MIT license text (copyright reduxailabs-max, 2026). |
| `src/vocalis/hardware.py` | `GpuInfo`/`HardwareInfo` dataclasses, `probe_hardware()` (GPU via `query_gpu_with_torch()`, RAM from `/proc/meminfo`, CPU count/load from `os.cpu_count()`/`/proc/loadavg` — every source independently optional and never raises), `describe()` (one plain-English summary line), and `log_hardware_summary()` (probes and logs that line at INFO, downgrading any failure to a WARNING). |
| `src/vocalis/remote.py` | The client side of `vocalis --settings`: `send_command()` talks to the daemon's Unix socket; `start_user_service()` runs `systemctl --user start vocalis`; `open_settings()` sends `SHOW_SETTINGS`, starting the service and polling for up to 20 seconds first if nothing answered. |
| `packaging/desktop/vocalis.desktop` | Application-menu launcher **template** — `__VOCALIS_EXEC__` placeholder, `Exec=... --settings`, filled in and installed by `install.sh` via `desktop_install_launcher()`. |
| `packaging/installer/lib/desktop.sh` | `desktop_install_launcher()` — fills the desktop template's `__VOCALIS_EXEC__` placeholder with the quoted, sed-escaped exec path and writes it atomically (temp file + `mv`), mirroring `service_substitute_unit()`'s pattern. |
| `tests/test_hardware.py` | Tests for `probe_hardware()`/`describe()`/`log_hardware_summary()` with every source (GPU query, `/proc/meminfo`, `/proc/loadavg`, `cpu_count`) faked — no real `/proc` or GPU touched. |
| `tests/test_remote.py` | Tests for `send_command()`/`open_settings()` against a real Unix socket bound in `tmp_path`, with `start_user_service` faked so no real `systemctl` runs, and a `FakeNotifier` proving the failure paths notify and the success path never does. |
| `tests/test_entry.py` | Tests for `vocalis.entry.entrypoint()`: no flags imports and calls `vocalis.cli.main()`; `--settings` calls `open_settings()` and exits with its code without ever touching `main()`. |
| `tests/test_package_metadata.py` | Confirms `LICENSE` is MIT with the right copyright line and that `pyproject.toml` declares `license = "MIT"` and `license-files = ["LICENSE"]`. |
| `tests/installer/test_desktop.bats` | `bats` tests for `desktop_install_launcher()`: fills in and quotes the exec path, and leaves an existing launcher untouched if the template is missing. |

## Stage 5 — changed files

| File | Change |
|---|---|
| `src/vocalis/startup_checks.py` | Gained `check_stale_asoundrc()`, the fifth and last check in `run_startup_checks()`'s fixed order — warns when `~/.asoundrc` still mentions "voxtype". |
| `src/vocalis/features/dictation.py` | The old "No audio captured" notification text was replaced by `NO_AUDIO_MESSAGE`, a longer, more actionable message telling the user to check their microphone is plugged in, not muted, and selected as the input device. |
| `src/vocalis/ui/tray.py` | `TrayIcon` now also opens Settings on a left-click (`QSystemTrayIcon.ActivationReason.Trigger`), not just from its right-click menu's **Settings…** entry. |
| `src/vocalis/daemon.py` | `VocalisDaemon` gained an optional `command_handler` constructor argument, called with each received command line (on the accept-loop thread, before the ack is sent); a raising handler is logged and never blocks the ack. |
| `src/vocalis/cli.py` | `main()` wires the daemon's `command_handler` to show the Settings window on `SHOW_SETTINGS`, and starts a background thread calling `log_hardware_summary()` right after `build_daemon()` (so `configure_logging()` has already run and the summary line is never silently dropped). |
| `src/vocalis/entry.py` | `entrypoint()`, the actual `vocalis` console-script target — imports nothing from `vocalis.cli` at module load time, so `--settings` stays cheap; it calls `vocalis.remote.open_settings()` directly, or lazily imports and calls `vocalis.cli.main()` otherwise. |
| `pyproject.toml` | `license`/`license-files` metadata added; `[project.scripts] vocalis` now points at `vocalis.entry:entrypoint` instead of `vocalis.cli:main`. |
| `packaging/installer/install.sh` | Sources `lib/desktop.sh` and, right after `service_enable_and_start()`, installs the application-menu launcher via `desktop_install_launcher()` (a failure there is logged but never fails the install); the closing message now explains the three ways to open Vocalis instead of just pointing at `systemctl status`. |
| `packaging/installer/uninstall.sh` | Also removes `~/.local/share/applications/vocalis.desktop` if present. |
| `packaging/installer/build-run.sh` | Also stages `packaging/desktop/vocalis.desktop` into both `.run` files' payload. |

## Stage 6a: voice-engine chain and Kokoro

**New files**

| File | Purpose |
|---|---|
| `src/vocalis/engines/auto_select.py` | Pure Auto policy: `AutoPolicy` thresholds, `is_stressed()`, `decide()` (returns the engine order plus a plain-English reason), `kokoro_should_use_gpu()`. |
| `src/vocalis/engines/base.py` | The shared `Synthesizer` and `TTSEngine` protocols. |
| `src/vocalis/engines/registry.py` | `EngineSelector` (turns the user's choice plus the PC's state into an `EnginePlan`) and `FORCED_CHAINS`. |
| `src/vocalis/engines/chain.py` | `EngineChain` — walks one reading's plan, falling to the next engine on failure; Piper is called directly, others through `ModelManager`. |
| `src/vocalis/engines/kokoro_tts.py` | `KokoroTTS` — the Kokoro engine (`kokoro-onnx`), CPU or graphics card, implements `ModelEngine`. |
| `tests/test_auto_select.py` | Every Auto rule, with fake hardware snapshots. |
| `tests/test_kokoro_tts.py` | Kokoro engine with fake session/synthesizer builders. |
| `tests/test_engine_registry.py` | Selector: forced choices, Auto, probe failure, graphics-card choice for Kokoro. |
| `tests/test_engine_chain.py` | Chain: fall-down order, no retry of dead engines, Piper unmanaged. |

**Changed files**

| File | Change |
|---|---|
| `src/vocalis/config.py` | `quick_tts`/`doc_tts` replaced by `quick_reader_engine`/`document_reader_engine` (`EngineChoice`, default `"auto"`); new `kokoro_model_path`, `kokoro_voices_path`, `kokoro_voice`. |
| `src/vocalis/settings.py` | `LiveSettings` gained `quick_reader_engine()`, `document_reader_engine()`, `kokoro_voice()`; new `KOKORO_VOICES` list (eleven voices). |
| `src/vocalis/features/quick_reader.py`, `doc_reader.py` | Take an `engine_selector`; build an `EngineChain` per reading and send the "Primary voice unavailable — using backup voice" notification. |
| `src/vocalis/ui/settings_window.py` | Three new drop-downs: Quick Reader voice engine, Document Reader voice engine, Kokoro voice. |
| `src/vocalis/cli.py` | Builds the `EngineSelector` and the Kokoro factory (`_make_kokoro_factory`) and passes them to both readers. |
| `src/vocalis/startup_checks.py` | New `check_kokoro_model()`; `run_startup_checks()` now runs seven checks in order: Parakeet, Qwen3-TTS, Kokoro, `ydotoold`, OCR tools, onnxruntime graphics card, stale `~/.asoundrc`. |
| `packaging/installer/lib/fetch_models.py` | `ensure_kokoro()` downloads the two Kokoro files (checked by exact size) with a 60-second timeout per connection. |
| `packaging/installer/lib/pyenv.sh` | `pyenv_install_kokoro()` installs `kokoro-onnx==0.4.7` with `--no-deps`; `pyenv_prefer_gpu_onnxruntime()` then re-installs the GPU onnxruntime last so it always wins (qwen-tts and kokoro-onnx both pull plain onnxruntime). |
| `packaging/installer/install.sh`, `uninstall.sh` | Install calls `pyenv_install_kokoro` and reports the Kokoro check; `uninstall.sh --purge` also removes the Kokoro folder. |
| `pyproject.toml` | Lists Kokoro's three small dependencies (`colorlog`, `espeakng-loader`, `phonemizer-fork`) because `kokoro-onnx` itself is installed with `--no-deps`. |

## Stage 6b: gapless long reading

**New files**

| File | Purpose |
|---|---|
| `src/vocalis/audio/speech_stream.py` | `SpeechStream` — ONE persistent sound stream for a whole reading: bounded clip queue (3), 8 ms fades, silence on underrun, pause/rewind, resampling, stall detection (10 s), per-clip "started" tags. |
| `src/vocalis/features/chunked_speaker.py` | `ChunkedSpeaker` / `SpeechSource` / `SpeakOutcome` — the producer loop shared by both readers: makes each chunk, paces it, pushes it into a `SpeechStream`; pause/resume/stop latch; releases the voice when playback ends. |
| `tests/test_speech_stream.py`, `tests/test_chunked_speaker.py` | The stream and the speaker, driven by fake sound output. |
| `tests/fake_output.py` | Fake sound output for tests: `ManualStream`, `PumpStream`, `OutputRecorder`, `FakeAudioPlayer`, `wait_until`. |

**Changed files**

| File | Change |
|---|---|
| `src/vocalis/text/chunker.py` | Sentence-aware (abbreviations, initials, line-break rules) and linear-time; `chunk_spans()` returns `Chunk(text, start, end)`; `chunk_text()` wraps it. |
| `src/vocalis/audio/playback.py` | `AudioPlayer` gained `open_stream()`, `close_stream()`, `report_device_failed()`; `play()` and friends unchanged. |
| `src/vocalis/engines/chain.py` | `EngineChain(hold_engines=True)` + `close()` hold the voice for a whole reading; new `new_reading_chain()` and `make_downshift_handler()` shared by both readers. |
| `src/vocalis/model_manager.py` | New read-only `current_engine` property. |
| `src/vocalis/features/quick_reader.py` | Reads long clipboard text in chunks through `ChunkedSpeaker`; copied-text notice sent from a side thread. |
| `src/vocalis/features/doc_reader.py` | Speaks through `ChunkedSpeaker`; "piece N of M" when a piece starts being heard; repeat read stops the previous one. |
| `tests/test_chunker.py`, `test_playback.py`, `test_engine_chain.py`, `test_model_manager.py`, `test_doc_reader.py`, `test_quick_reader.py`, `test_cross_feature_integration.py`, `test_shared_tts_engine.py`, `test_reading_hooks_integration.py` | Extended or ported to the fake sound output. |

## Stage 6c: clarity

**Changed files**

| File | Change |
|---|---|
| `src/vocalis/engines/qwen3_tts.py` | Gained a `speed` constructor argument (own capped 1.0-1.2x `librosa` stretch, `paced_audio()`'s only remaining caller); `synthesize()` now tunes `temperature`/`top_k`/`max_new_tokens`. |
| `src/vocalis/engines/kokoro_tts.py` | Gained a `speed` constructor argument, passed straight to Kokoro's own `speed=` parameter. |
| `src/vocalis/engines/piper_tts.py` | Gained a `speed` constructor argument, converted to Piper's own `--length_scale` argument (`1 / speed`). |
| `src/vocalis/features/chunked_speaker.py` | Lost `speed_provider` — pacing is now each engine's own job. |
| `src/vocalis/features/quick_reader.py` | Lost `speed_provider`; clipboard text is now passed through `normalize_text()` before chunking (the desktop notification still shows the original text). |
| `src/vocalis/features/doc_reader.py` | Lost `speed_provider`. |
| `src/vocalis/cli.py` | `_make_qwen_factory()`/`_make_kokoro_factory()` and both `PiperTTS(...)` construction sites now resolve and pass a live `speed`. |
| `src/vocalis/config.py` | `EnginesConfig` gained a `model_validator` that warns (never blocks) when `qwen3_tts_speaker` is not English-native (`Ryan`/`Aiden`). |
| `src/vocalis/text/normalizer.py` | Gained `normalize_text()` (URLs, `$`/`%`/`&`/`@`); `normalize_markdown()` now calls it as its last step. |
| `tests/test_qwen3_tts.py`, `tests/test_kokoro_tts.py`, `tests/test_piper_tts.py`, `tests/test_chunked_speaker.py`, `tests/test_quick_reader.py`, `tests/test_doc_reader.py`, `tests/test_cli.py`, `tests/test_config.py`, `tests/test_normalizer.py` | Extended or ported for the above. |
