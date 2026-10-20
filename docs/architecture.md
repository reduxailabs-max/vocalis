# Vocalis Architecture

## Current state (Phase 0+1 foundation)

Vocalis is a single long-lived daemon process (`VocalisDaemon`) that:

1. Loads validated configuration (`VocalisConfig`) from
   `~/.config/vocalis/config.toml`, falling back to defaults on first run.
2. Owns a `ModelManager`, which enforces the core GPU safety rule: **only
   one AI model may be resident in VRAM at a time**, and it is
   automatically unloaded a few seconds (configurable,
   `model_idle_timeout_seconds`) after the last feature releases it.
3. Listens on a Unix domain socket (`~/.local/state/vocalis/vocalis.sock`).
   The socket never grew real command dispatch — every feature ended up
   wired directly into `cli.py`'s `main()` instead, in the same process,
   with hotkey listeners calling feature methods in-process rather than
   sending anything over the socket. The accept loop still only echoes
   commands back; its real job today is the single-instance guard added
   in Stage 2 (see "Single-instance guard: connect-before-delete" below).
4. Guarantees `ModelManager.shutdown()` always runs when the daemon stops —
   even if closing the socket or removing its file raises, or the socket
   fails to bind at all — via nested `try`/`finally` in both `main()` and
   `VocalisDaemon.stop()`. Each accepted connection also gets its own
   timeout, so a silent or hung client can never block shutdown
   indefinitely.

No AI engines (Parakeet, Qwen3-TTS, fish-speech, Tesseract) are wired in
yet. The `ModelManager` is proven with fake engines in tests; Phase 2
plugs in the first real engine (Parakeet) and verifies actual VRAM
behavior with `nvidia-smi`.

## Design rationale

- **Single process, one shared `ModelManager`**: `cli.py`'s `main()`
  builds every feature (dictation, quick reader, document reader, OCR) in
  the same `vocalis` process and hands them all the same `ModelManager`
  instance, so only one process ever holds a GPU model — this is what
  makes the single-model-residency GPU policy possible. (The socket was
  originally meant to carry feature commands to a separate long-lived
  daemon process; in practice every feature ended up wired directly into
  `main()` instead, and the socket's actual job today is the
  single-instance guard described in the Stage 2 section below.)
- **ModelManager as its own module**: the GPU idle-unload policy is the
  riskiest and most safety-critical piece of this app (an 8GB card can't
  hold multiple large models), so it is built and tested in isolation
  before any real feature touches it.
- **pydantic config**: invalid config is rejected loudly at load time with
  a clear `ConfigError`, rather than causing a confusing failure deep
  inside a feature later.

See `docs/superpowers/specs/2026-09-11-vocalis-design.md` for the full
product design (all four features, UI, packaging) this architecture is
building toward.

## Phase 2 addition: Dictation (STT)

Holding NUMLOCK now works end-to-end:

1. `NumlockListener` (evdev, one background thread per keyboard-capable
   device) detects NUMLOCK press/release without suppressing the OS's own
   NumLock handling.
2. On press, `DictationFeature.on_press()` notifies "Listening..." and
   starts `MicRecorder` (sounddevice, float32 @ 16kHz, 60s safety cutoff).
3. On release, `DictationFeature.on_release()` stops the recorder, notifies
   "Transcribing...", and calls `model_manager.acquire("parakeet", ...)` to
   get a loaded `ParakeetSTT` engine — obeying the same single-residency
   and idle-unload policy every future engine will use.
4. `ParakeetSTT` wraps the `onnx-asr` package running the existing Voxtype
   Parakeet TDT ONNX model, GPU (CUDA) by default.
5. The transcribed text is delivered via `TextOutput` (ydotool typing,
   clipboard fallback on any failure) and the engine is released back to
   `ModelManager` (which starts its idle-unload timer).
6. Notifications fire at each stage: listening, transcribing, the
   transcript itself, and done (or the specific failure reason).

`build_daemon()` changed signature in this phase: it now takes an
already-constructed `ModelManager` instead of building its own from
config, because `main()` needs one shared `ModelManager` instance across
both the daemon and the dictation feature — only one may exist per
process, since it enforces single-model GPU residency app-wide.

See `docs/algorithms.md` for how `ModelManager`'s policy applies here
unchanged, and the design spec's Hard Constraint #2 amendment for the
confirmed small (~150MB) fixed GPU residual once any engine has been used.

## Phase 3 addition: Quick Reader (TTS)

Pressing Ctrl+Alt+R now works end-to-end:

1. `CtrlAltRListener` (evdev, one background thread per keyboard-capable
   device, built the same way as `NumlockListener`) detects Ctrl+Alt+R
   without suppressing normal key behavior, and excludes Vocalis's own
   ydotoold virtual device from watching.
2. On trigger, `QuickReaderFeature.on_trigger()` either starts a new
   reading or stops an in-progress one — both dispatched onto separate
   background threads so the listener itself is never blocked.
3. The background thread reads the clipboard (`ClipboardReader`), then
   calls `model_manager.acquire("qwen3_tts", ...)` to get a loaded
   `Qwen3TTS` engine — obeying the same single-residency and idle-unload
   policy every GPU engine uses. If that fails for any reason, it falls
   back to an always-loaded `PiperTTS` instance, which runs entirely on
   CPU and is never routed through `ModelManager`. (Phase 3 design; since
   Stage 6a the voice is chosen per reading and falls down the chain
   Qwen3-TTS -> Kokoro -> Piper — see "Stage 6a addition" below.)
4. The generated speech is played via `AudioPlayer` (sounddevice-based,
   the same library `MicRecorder` uses for capture), which can be
   interrupted mid-playback. Its `wait_until_done()` is bounded by the
   audio's expected duration plus a safety margin, so a misconfigured
   default audio output device fails fast with `PlaybackTimeoutError`
   (surfaced as a normal "Playback failed" notification) instead of
   hanging the reading forever. Since Stage 1 (see below), `AudioPlayer`
   is paired with an `OutputDeviceChooser` that picks a device known to
   actually work, and a failed device is swapped out automatically.
5. Notifications fire at each stage: reading, the text itself, done (or
   stopped early, or the specific failure reason).

See `docs/superpowers/specs/2026-09-11-vocalis-design.md`'s Phase 3
amendment for the full rationale, including two corrections made during
brainstorming: the legacy tool actually used Piper (not Qwen3-TTS/
fish-speech as originally assumed), and Qwen3-TTS's "Base" variant turned
out to be a voice-cloning model — "CustomVoice" with a preset voice is
what this feature actually uses.

## Phase 4 addition: Document Reader

Pressing Ctrl+Alt+D opens a small PySide6 window for reading whole
documents aloud:

1. `CtrlAltKeyListener` (generalized from the Phase 3 R-only listener;
   `CtrlAltRListener` is now a thin wrapper) watches for Ctrl+Alt+D on its
   evdev thread and only emits the popup's `hotkey_pressed` Qt signal.
2. `DocumentReaderPopup` (`ui/doc_popup.py`) is the only module importing
   PySide6 for this feature. It calls `DocumentReaderFeature`'s public
   methods; the feature has no Qt import and reports back through three
   plain callbacks that `build_doc_reader()` wires to the popup's signals.
3. `DocumentReaderFeature` extracts text (`text/extractors.py`), strips
   markdown (`text/normalizer.py`), splits it into sentence-bounded chunks
   (`text/chunker.py`), then synthesizes and plays one chunk at a time,
   using the same `ModelManager` single-residency policy and the same
   voice-engine chain (Qwen3-TTS -> Kokoro -> Piper since Stage 6a) as
   Quick Reader. Both readers use the one shared engine id
   `TTS_ENGINE_ID` (`"qwen3_tts"`, defined in `model_manager.py`), so
   switching between them reuses the already-loaded Qwen3-TTS model
   instead of reloading it. Its Paused/Resuming.../Stopped notifications
   go through a single ordered background queue, so pause/resume/stop
   never wait on `notify-send`.
4. `ReadingCoordinator` is shared by both readers so only one speaks at a
   time. Quick Reader was changed to register with it, and its
   preemption callback is clear-only (a stale claim from an already-stopped
   Quick Reader must never turn it "active").
5. `main()` now runs a Qt event loop (`app.exec()`) on the main thread
   instead of a sleep loop. A SIGINT handler calls `app.quit()` so Ctrl+C
   makes the loop return and the normal shutdown (listeners, daemon,
   `ModelManager.shutdown()`) runs. Each stop (document reader, quick
   reader, dictation listener, then daemon last) is attempted
   independently: if one raises, it is logged and the rest still run, so
   `daemon.stop()` / `ModelManager.shutdown()` always happen.
6. **Shared TTS engine id — caution.** `QuickReaderFeature` and
   `DocumentReaderFeature` both accept a `language` argument, but it is
   currently only stored and never used. That is why the two readers can
   safely share one `qwen3_tts` engine id (and one loaded model). If
   language ever becomes per-feature, revisit the shared id (e.g. give
   each language its own id), or one reader could speak with the other's
   settings.

Hotkey watcher shutdown: evdev's own `read_loop()` blocks in `select()`
on the device file and closing that file from another thread does not wake
it on Linux. The real enumerators (`list_numlock_capable_devices`,
`list_devices_with_key`) therefore wrap each device to be watched in
`StoppableEvdevDevice`, whose read loop also waits on a private pipe.
`stop()` -> `close()` writes to that pipe, the loop ends at once and the
watcher thread exits, so the listeners' 1 s `join` never times out. Devices
that don't qualify are still closed directly, and test fakes are unaffected.

Quiet shutdown: evdev's `close()` closes the file and only then sets the
device's `fd` to -1. A watcher that is busy running a callback (or between
two loop passes) when `stop()` runs would then call `select()` on -1 and get
a `ValueError`. `StoppableEvdevDevice.read_loop()` therefore checks its
"closed" flag at the top of every pass and again after `select()` returns,
and treats `ValueError`/`EBADF` as a normal exit only when it has been
closed; the same errors on an open device still propagate. If building the
wrapper fails during enumeration (for example "too many open files"), the
enumerators close the device that failed and every device already wrapped
before re-raising.

Unplugged keyboards: each listener sets a `_stopping` flag at the start of
`stop()`. An `OSError` that ends a watcher while the flag is set is silent
(it is just `close()` at work). An `OSError` while the flag is not set means
the device vanished, so the listener logs one WARNING ("Input device <name>
disappeared; its hotkey watcher has stopped"). The message contains the
device name only, never any key data.

Threading rule: widgets are only touched on the GUI thread; every other
thread reaches the popup through a Qt signal (queued by Qt automatically).

## Phase 5 addition: OCR Reader

Pressing Ctrl+Alt+O lets you drag a box around text on screen and hear it
read. The feature (`features/ocr_reader.py`) only captures, recognises and
copies; speaking is handed to the existing `DocumentReaderFeature`, so
pause/resume/stop, voice fallback and `ReadingCoordinator` ownership are
reused, not duplicated.

Tesseract runs on the CPU only, so it is **not** routed through
`ModelManager` (single-residency applies only to GPU engines).

```
Ctrl+Alt+O pressed (evdev thread, CtrlAltKeyListener, key code 24)
  -> OcrReaderFeature.on_trigger()           # never blocks the watcher thread
       -> capture already running: ignored
       -> Document Reader is playing a reading THIS feature started
          (reader.generation == remembered generation): spawn thread -> reader.stop()
       -> else: busy = True, spawn background thread

Background thread (_capture_and_read):
  -> ScreenRegionCapture.capture()           # spectacle (temp dir) / maim / flameshot / slurp+grim -> PNG bytes
       no tool -> notify the install hint; cancelled -> notify "Cancelled"
  -> TesseractOCR.recognize(png)             # tesseract stdin -> stdout, English, whitespace collapsed
       failure -> notify the message; under 3 characters -> "No text detected"
  -> ClipboardWriter.write(text)             # xclip; failure is only logged
  -> notify(first 80 characters)
  -> hub.set_next_source("Screen text"); popup.external_reading_started.emit(text)
                                             # popup shown only if the floating widget is OFF (Phase 6)
  -> DocumentReaderFeature.read_text(text)   # normalise, chunk, speak, ReadingCoordinator
  -> remember reader.generation              # so the next press can stop this reading
  -> finally: busy = False
```

`DocumentReaderFeature.generation` (a read-only property) lets the OCR
feature tell "my reading" from a reading someone else started.
`DocumentReaderPopup.external_reading_started` is a signal, so emitting
it from the OCR thread is safe; the slot runs on the GUI thread, fills the
text box, enables Pause/Stop and shows the window without taking focus.
`main()` starts the OCR listener last and stops it first on shutdown.

## Phase 6 addition: tray icon, floating widget, Settings window

Phase 6 adds three small windows and two shared objects that connect them
to the features.

**`LiveSettings` (`settings.py`)** is the one shared, thread-safe holder of
the current `VocalisConfig`. It is created in `main()` from the config file.
The Settings window is its only writer (`update()`): it validates the
shortcuts, saves the file atomically (temp file in the same folder, then
`os.replace`; the file ends up mode 0600), and only then swaps the in-memory
config. If anything fails, both the file and the running settings stay as
they were. Features do **not** copy values at construction; they hold a
function (`settings.notifications_enabled`, `settings.voice`,
`settings.floating_widget_enabled`) and call it at the moment of use, so
those three settings apply immediately (voice: on the next reading).
Shortcuts are read once at startup (`resolve_shortcuts()`), so
`shortcuts_need_restart()` compares the saved shortcuts with those the
program started with.

**`StatusHub` (`ui/status_hub.py`)** is a small Qt object with signals:
`state_changed` (idle / listening / reading), `status_changed`,
`progress_changed`, `reading_started(source, can_pause)`, `reading_finished`
and `paused_changed`. Features *report* into it from any thread; the tray
and the widget only *listen* (Qt delivers the signals on the GUI thread)
and send commands back (`pause`, `resume`, `toggle_pause`, `stop`) through
the `ReadingControls` the feature handed over in `begin_reading()`. State
changes happen under a short lock and signals are emitted after the lock is
released. A second `begin_reading()` while one is active is ignored, so each
feature must call begin/end for its own readings in order, from one thread
or otherwise consistently.

```
Dictation (on_listening_changed) --+
Quick Reader (on_reading_changed) -+--> StatusHub --signals--> TrayIcon
Document Reader (on_started/status/ |                      \--> FloatingWidget
  progress/finished)                |
OCR Reader (set_next_source label,  |
  reads through Document Reader) ---+

Settings window --update()--> LiveSettings --read at the moment of use--> Notifier,
                                                     Qwen3TTS (voice), FloatingWidget (on/off)
```

How each feature reports:

- **Dictation** calls the optional `on_listening_changed(True/False)`
  hook; the hub shows "listening" while the microphone is recording.
- **Quick Reader** reports from its *state* (a `_report_lock` and a
  `_last_reported` flag), not from individual code paths, so a start and a
  stop can never produce two "started" or a stray "finished". It also has a
  public `stop()` (the widget's and tray's Stop button).
- **Document Reader** gained an optional `on_started` hook. It is called
  *after* the reading was handed to the `ReadingCoordinator` (which may stop
  a previous reader and so end its hub reading); calling it earlier would
  make the hub ignore the new reading. Its existing status/progress/finished
  callbacks are also forwarded to the hub; "Paused" and "Resuming..."
  status messages become `set_paused()` confirmations.
- **OCR Reader** reads through the Document Reader, so the hub sees a
  Document Reader reading; OCR first calls `hub.set_next_source("Screen
  text")` so the widget says "Screen text". If the floating widget is on,
  the popup is not opened for OCR; if it is off, the popup opens as in
  Phase 5.

**Tray icon (`ui/tray.py`)** draws a 22x22 coloured circle in code (grey,
red, blue) and keeps its menu in step with the hub. `close()` tears down
the tray safely and is idempotent. **Floating widget
(`ui/floating_widget.py`)** is a frameless, always-on-top tool window that
never accepts keyboard focus and sits bottom-right on the primary screen.
**Settings window (`ui/settings_window.py`)** builds a new config from its
drop-downs and hands it to `LiveSettings.update()`; shortcut choices come
from `hotkeys/shortcut_parser.py`, which turns `Ctrl+Alt+R` or `NUMLOCK`
into evdev key codes and rejects bad or duplicate shortcuts. `Qwen3TTS` and
`Notifier` accept a function instead of a fixed value for the voice and the
on/off switch.

**Start-up and shut-down order in `main()`.** The config is loaded first, then
`LiveSettings` is created and `resolve_shortcuts()` runs (it logs an error and
falls back to the defaults when the config holds a bad shortcut). Only then is
the `QApplication` created (with `quitOnLastWindowClosed` off, so closing a
window never quits), then the `StatusHub`, then the builders for the four
features (each gets its key code from the resolved shortcuts), then `build_ui()` (widget,
Settings window, tray). Shut-down (Ctrl+C or tray Quit), all while the
`QApplication` is still alive: the keepalive timer is stopped and deleted
first; the tray is closed (also before `app.quit()` when Quit is chosen);
then the widget is hidden, the four listeners and the daemon are stopped
(each independently, daemon last so `ModelManager.shutdown()` always runs);
last the Settings window and the widget window are closed.

## Stage 1 addition: Sound output, speaking pace and voice preview

### `OutputDeviceChooser` (`audio/output_device.py`): probe-and-cache

`main()` builds exactly one `OutputDeviceChooser` and passes it to both
`build_quick_reader()` and `build_doc_reader()`, so Quick Reader and
Document Reader agree on, and share, one sound output rather than each
discovering its own.

- **Probe** = open a short-lived `sounddevice.OutputStream` on a candidate
  device with a silent callback and wait (bounded, default 1s) for the
  callback to actually fire. Some PortAudio/ALSA setups open and start a
  stream whose callback never fires (the audio is silently dropped) —
  probing is the only reliable way to tell a device that merely *opens*
  from one that really *plays*.
- **Candidate order**: the user's explicit choice (if not "auto") first,
  then `pulse`, `pipewire`, `default`, then every other output device,
  skipping any device already marked bad. The whole search is bounded by
  a total timeout (default 6s) so a run of unresponsive devices cannot
  hang a reading indefinitely.
- **Caching**: a working device is remembered for the rest of the
  session (`current()` short-circuits straight to it). A "nothing worked"
  answer is cached for only `negative_ttl_seconds` (default 15s), not
  forever — Vocalis can be started by the systemd user service at login
  *before* PipeWire/PulseAudio has finished starting, so the chooser
  retries from scratch a short time after a total failure instead of
  staying stuck on "no device" for the rest of the process's life. This
  recovers a device that was already present but whose sound server was
  not ready yet; it is not hot-plug detection — PortAudio's device list
  is read once per process, so a device plugged in *after* Vocalis starts
  will not appear until Vocalis is restarted. `mark_failed(None)` (the system
  default stalling) only clears that cached negative answer; it does not
  mark any specific device index bad, since "the default" is not one
  fixed device.
- **Locking**: `_probe_lock` serialises probe sequences (so concurrent
  callers produce one sequence, not several overlapping probes);
  `_cache_lock` only ever guards the small cache/bad-device dictionaries.
  `_cache_lock` is never held while waiting on the audio callback, but
  `_probe_lock` is held across the whole probe sequence by design, so a
  second caller waits for the first caller's probe to finish rather than
  starting its own overlapping probe.

### `AudioPlayer` (`audio/playback.py`): stalled-device detection and one retry

`AudioPlayer` takes an optional `device_provider` (called at the start of
every `play()`) and an optional `on_device_failed(device)` hook:

- If `wait_until_done()` times out (the callback never fired) or
  `stream.start()` itself raises (e.g. the device was unplugged, or
  PortAudio rejects its sample rate), the player reports the device that
  was in use to `on_device_failed`, then — only on the *open/start*
  failure path, and only if a `device_provider` exists — asks the
  provider again. If the provider now names a *different* device, the
  player retries that one open/start exactly once before giving up;
  if it names the same device again, or the retry also fails, the
  original error propagates. This covers a `factory raises` style
  failure (e.g. a headset unplugged between probe and play) without
  silently retrying forever.
- `on_device_failed` is never called on a natural finish or a user-
  initiated `stop()` — only on a real stall or open failure.

### Pacing (`audio/pace.py`): one shared helper, applied once per chunk/reading

`apply_speed(audio, sample_rate, speed)` runs the already-synthesized
waveform through `librosa.effects.time_stretch` (pitch-preserving);
`librosa` is imported *inside* the function so merely importing
`vocalis.audio.pace` (which `config.py` does, to get `MIN_SPEED`/
`MAX_SPEED`) does not pay librosa's import cost. `clamp_speed()` keeps
the factor inside `0.7`–`1.6` (NaN treated as `1.0`).

`paced_audio(audio, sample_rate, provider)` is the one function Quick
Reader and Document Reader both call: it reads the caller's current speed
from `provider()` and applies it, or returns the audio unchanged if
`provider` is `None`, or if either call raises (logged as a warning —
a pacing failure must never break a reading). Quick Reader calls it once
per reading; Document Reader calls it once per chunk, *before* that
chunk's inner pause/resume retry loop — a resume after a pause replays
the exact same already-paced audio rather than resynthesizing or
re-pacing it (see the spec amendment below for the user-visible effect of
that choice).

### `main()` wiring and notification rule

`main()` builds one `OutputDeviceChooser` and passes it into both reader
builders; each builder wraps it into an `AudioPlayer` via
`output_chooser.current` (the device provider) and a per-chooser
`on_device_failed` hook built by `_make_device_failure_hook()`. That hook
always calls `chooser.mark_failed(device)` (so the next pick avoids it),
but only sends the desktop notification "Sound output changed because it
stopped working" when the failing device is *different* from the last
device it already notified about — so a device that keeps failing
repeatedly only produces one notification, not a storm, while a *second*,
*different* device failing later produces a fresh one.

## Stage 2 addition: Service and autostart

Phase 7 prepares Vocalis to run as a `systemd --user` service that starts
automatically at desktop login and recovers from a crash. This stage
builds and tests the supporting code (the single-instance guard, the
`SIGTERM` handler, and the startup checks) and a unit-file **template**;
actually installing and turning the service on for a real user is Stage 3's
job (the installer — see `packaging/systemd/README.md`). Nothing in this
stage starts, enables, or sends a real signal to a real systemd service.

### Single-instance guard: connect-before-delete

`VocalisDaemon.start()` used to unconditionally delete any existing socket
file at `_socket_path` as "stale" before binding its own. That was unsafe:
if a second real instance started (a manual `vocalis` launch while the
service was already running, or two service starts racing each other), it
would silently delete the first instance's *live* socket file out from
under it.

`start()` now calls `_is_another_instance_alive()` before deleting
anything. That method builds a short-lived probe socket — via the
constructor's injectable `socket_factory` argument, real
`socket.socket(AF_UNIX, SOCK_STREAM)` by default, swappable in tests — and
tries to `connect()` it to the existing socket file. There are exactly
three outcomes:

1. **Alive.** The connect succeeds: a real process is listening. `start()`
   raises `DaemonAlreadyRunningError` and never touches the file. `cli.py`'s
   `main()` catches that exception around the `daemon.start()` call, logs
   it at INFO level (not a desktop notification), and returns normally
   with exit code 0 instead of letting it propagate as a traceback -- so a
   second launch exits quietly instead of spamming the user, and does not
   count as a crash against systemd's `StartLimitBurst`.
2. **Stale.** The connect raises `ConnectionRefusedError` (the socket
   exists but nothing is listening — exactly what a crashed process
   leaves behind) or `FileNotFoundError` (the file vanished between the
   earlier `exists()` check and the connect attempt). Both are treated the
   same: `start()` logs a warning and deletes the file, exactly as before
   this stage.
3. **Ambiguous.** Any other exception during the probe (for example, a
   permission error). This is deliberately treated the same as "stale" —
   an inconclusive probe must never block Vocalis from starting — and the
   reason is logged as a warning rather than silently swallowed. This
   favors Vocalis actually starting over the rare risk of a wrong "stale"
   verdict: if two processes really are racing, the loser still fails at
   its own `bind()` moments later.

The probe socket is always closed in a `finally`, wrapped in
`contextlib.suppress(Exception)` — a close failure on a socket that
already failed to connect is not worth surfacing.

### `SIGTERM` handled identically to `SIGINT`

`systemctl --user stop`/`restart` sends `SIGTERM`, not `SIGINT` (Ctrl+C).
Before this stage, `main()` only installed a `SIGINT` handler — a known
gap noted in `cli.py`'s own docstring since Phase 4. `main()` now installs
a `SIGTERM` handler right alongside it, pointed at the exact same
`lambda *_: app.quit()`. A `SIGTERM` therefore makes `app.exec()` return
exactly the way Ctrl+C does, and the same `finally` block runs the same
shutdown (listeners stopped, tray closed, daemon stopped — which removes
the socket file and calls `ModelManager.shutdown()`). Both the previous
`SIGINT` and `SIGTERM` handlers are restored on the way out, in all cases.

### Startup checks: where they run and how they're reported

Right after `load_config()` returns — before the `QApplication`, the
`StatusHub`, `LiveSettings`, or any listener/feature is built, because
nothing checked here should depend on anything built later in `main()` —
`main()` builds a plain `Notifier` from `config.notifications_enabled` and
calls `run_startup_checks(config)`. Every message it returns is sent
through that notifier, one at a time. See `src/vocalis/startup_checks.py`
for the seven checks themselves (Parakeet model, Qwen3-TTS model,
Kokoro voice files, `ydotoold`, OCR capture tools, onnxruntime
graphics-card check, stale `~/.asoundrc`) and "Running automatically" in
`docs/user-docs.md` for what each message means in plain language. Every
check is independent and strictly non-fatal: a missing dependency only
means the one affected feature fails per-request later, exactly as it did
before this stage — the checks exist purely to say so up front instead of
leaving the user to discover it the first time they use that feature. A
check that itself raises is caught, logged as a warning, and skipped; it
can never stop the other checks or crash startup.

The Settings window's **Test sound** button is wired by
`_make_on_test_sound()`: for "Automatic" it uses the shared chooser's
`current()` (falling through its normal candidate order); for a *named*
device it builds a temporary, throwaway `OutputDeviceChooser(preference=
lambda: selected)` and calls `.probe()` on exactly that one device index —
deliberately never `.current()` for a named choice, since `.current()`
would silently fall through to pulse/pipewire/default if the requested
device failed, masking the very failure the user is asking to test. The
busy check (`document_reader.is_reading`) happens before any device
lookup, so a test tone never interrupts, or is blocked by, an existing
Quick Reader or Document Reader playback.

## Stage 3 addition: Installer

Phase 8 adds `packaging/installer/`, a small library of bash functions
plus `install.sh`/`uninstall.sh`, that takes a Debian/Ubuntu-family
`x86_64` machine from "nothing" to "Vocalis installed, models in place,
background service running and set to autostart" with one command and
one password prompt. `build-run.sh` wraps the two scripts into
self-extracting `install.run`/`uninstall.run` files via `makeself`, for
people who would rather double-click a file than run a shell command.

**Flow (`install.sh`'s `main()`), in order:**

```
preflight_run()                         # lib/preflight.sh — hard gate
  -> distro (Debian/Ubuntu via /etc/os-release) -> arch (x86_64)
     -> GPU driver (nvidia-smi succeeds); first failure stops everything
apt_install_missing(...)                # lib/apt.sh — the ONE sudo prompt
apt_enable_ydotoold()                   # best-effort, never fails the install
pyenv_ensure_uv() -> pyenv_find_or_build_wheel() -> pyenv_create_venv()
  -> pyenv_install_vocalis()            # lib/pyenv.sh
symlink venv/bin/vocalis -> ~/.local/bin/vocalis
fetch_models.py (Parakeet, Qwen3-TTS, Piper voice — via huggingface_hub)
models_ensure_piper_binary()            # lib/models.sh — the one non-HF asset
service_substitute_unit() -> service_enable_and_start()   # lib/service.sh
```

The apt step (the only slow-but-cheap, password-gated step) always runs
**before** the slow, multi-gigabyte Python-environment and model-download
steps, so a declined or wrong password fails in seconds rather than after
several minutes of downloading.

**Three robustness fixes from final review.** `install.sh` handles three
edge cases that were fixed late in Stage 3's own review pass:

1. **Unit-template path fallback.** `service_substitute_unit()` needs
   `packaging/systemd/vocalis.service` as its source template. Inside an
   extracted `.run` file that template is staged right next to
   `install.sh` (`build-run.sh` puts it there); in a plain dev checkout
   it only exists at the repo root. `main()` tries the script-relative
   path first and falls back to the repo-root path when that file is
   missing, so both ways of running the installer work.
2. **`PATH` export after installing `uv`.** `pyenv_ensure_uv()` puts
   `~/.local/bin` (uv's default install location) on `PATH` *before*
   checking whether `uv` is already available, and does so
   unconditionally — so a `uv` that was only just installed in this same
   run is still found by every later `uv` call, not just a pre-existing one.
3. **Idempotent venv creation.** `pyenv_create_venv()` first checks
   whether `venv_dir/bin/python` already runs; if so it leaves the venv
   alone. Otherwise it creates one, using `uv venv --clear` only when the
   target directory already exists (a broken or partial venv) and a plain
   `uv venv` for a brand-new one — so re-running `install.sh` as a repair
   command never fails on, or needlessly rebuilds, a venv that already works.

**Reuse-or-download model policy.** `config.py`'s existing defaults
already point at the exact folders where Parakeet, Qwen3-TTS, and the
Piper binary/voice belong. The installer checks each of those exact
paths: if a model's files are already there (for example, carried over
from the legacy Voxtype setup), it is left untouched — no download, no
config rewrite; if absent, it is downloaded straight into that same path.
A freshly-installed machine therefore ends up with files in exactly the
same place a hand-set-up one has. `fetch_models.py` imports
`vocalis.config.VocalisConfig` and `vocalis.startup_checks`'s public
`PARAKEET_MODEL_FILES`/`QWEN3_TTS_MODEL_FILES` constants directly from the
just-installed package, so the installer can never drift out of sync
with the file lists Vocalis's own runtime startup checks expect.

**Testing idiom.** Every call to an external command (`dpkg`, `apt-get`,
`curl`, `uv`, `systemctl`, `nvidia-smi`, `tar`, `makeself`) is wrapped in
its own tiny `*_run_*` function (e.g. `apt_run_install`,
`service_run_systemctl_enable_now`), so a `bats` test can redefine that
one function after `source`-ing the library and never invoke a real
subprocess — the same philosophy as the Python tests faking
`sounddevice`/`socket`/`shutil.which` elsewhere in this project.
`install.sh` and `uninstall.sh` are themselves written as a sourceable
`main()` plus a run-guard (`if [[ "${BASH_SOURCE[0]}" == "${0}" ]]`), so
`bats` can source the whole script, override every lib function, call
`main`, and assert the right calls happened in the right order — exactly
how `vocalis.cli.main()` is tested in Python.

**Why no `.deb` or AppImage.** A `.deb` can't carry Python/CUDA
dependencies (torch, transformers, ~3.1GB of models) as apt packages —
it would still need the same venv-creation logic inside a `postinst`
script, while fighting root-owned-path conventions against a
`systemd --user` service. An AppImage can't install a background
service, system packages, or persistent model folders by itself, so a
companion installer script would still be needed — duplicating this same
logic a second time for a single-machine personal tool. A plain shell
script (optionally wrapped as a self-extracting `.run` file) is the
simplest thing that actually does the whole job once.

**Never run for real by any automated test.** No `bats` or `pytest` test
invokes real `sudo`/`apt-get`, downloads a real model, creates the real
venv at `~/.local/share/vocalis/venv`, or enables the real `vocalis.
service` — exactly as every earlier stage deferred all hands-on
verification. The user's own first real run of `install.sh` is the
genuine end-to-end check.

## Stage 5 addition: License, hardware probe, and a findable Settings window

Stage 5 is a grab-bag of small, independent polish items rather than one
feature: an MIT license, a one-line hardware summary at startup, a
leftover-`.asoundrc` warning, a clearer "no audio" message, and — the
largest piece — three new ways to reach the Settings window, plus the
application-menu launcher this task adds on top of them.

**Hardware probe (`hardware.py`).** `probe_hardware()` takes a snapshot of
the machine — the first CUDA GPU's name and free/total memory (via
`torch.cuda`, imported lazily inside `query_gpu_with_torch()`), RAM
total/available from `/proc/meminfo`, and CPU count/1-minute load from
`os.cpu_count()`/`/proc/loadavg`. Every source is independently optional
(a missing GPU, an unreadable `/proc` file, or a raising `cpu_count()` each
degrade that one field to `None` rather than failing the whole probe), and
`log_hardware_summary()` wraps the whole thing in a `try`/`except` so a
hardware-probing bug can never stop Vocalis from starting — at worst it
logs a `WARNING` instead of its usual `INFO` summary line. `main()` runs it
on its own background thread, the same pattern as `_warm_up_pace_changer()`.

**Daemon command hook and `remote.py`.** `VocalisDaemon` gained an optional
`command_handler` constructor argument: a callable invoked with each
received command line, on the accept-loop thread, before the ack is sent.
A raising handler is logged and never blocks the ack from going out.
Because that callable runs on the daemon's accept-loop thread, not the Qt
thread, `main()` cannot have it call `settings_window.present()` directly
— Qt widgets may only be touched from the GUI thread. Instead it wires the
handler to `_SettingsBridge.handle_command()`, a tiny `QObject` whose only
job is to `emit()` a Qt signal (`show_settings`) when it sees
`vocalis.remote.SHOW_SETTINGS`; the signal is connected to
`settings_window.present()`, and Qt's queued cross-thread delivery moves
the actual call onto the GUI thread for it. `remote.py` is the client side, used by the
new `vocalis --settings` command: `send_command()` talks to the daemon's
existing Unix socket exactly like any other client; `open_settings()` tries
that first, and if nothing answers (the service isn't running yet) it runs
`systemctl --user start vocalis` and polls the socket for up to 20 seconds
before giving up with a plain error on stderr.

**Three ways to reach Settings.** Before this stage, Settings only opened
from the tray's right-click menu. Now: (1) a left-click on the tray icon
(`QSystemTrayIcon.ActivationReason.Trigger`) opens it directly, the same
`settings_window.present()` call the menu entry already used; (2)
`vocalis --settings` (the `entrypoint()` function in the small, separate
`vocalis.entry` module, now the package's console-script target instead
of `cli.py`'s `main()` directly — kept apart from `cli.py` specifically so
this fast path never pays `cli.py`'s heavy import cost) opens it in the
already-running process via the socket, starting the service first if
needed; (3) the application-menu launcher this task adds runs the exact
same `vocalis --settings` command from a normal desktop menu click.
`present()` is safe to call more than once — a second request while the
window is already open just raises it, rather than opening a duplicate.

**Application-menu launcher (`packaging/desktop/vocalis.desktop`,
`packaging/installer/lib/desktop.sh`).** The template follows the same
single-placeholder pattern as `packaging/systemd/vocalis.service`:
`Exec=__VOCALIS_EXEC__ --settings`. `desktop_install_launcher()` mirrors
`service_substitute_unit()`'s atomic temp-file-then-`mv` pattern exactly,
with one addition: the exec path is wrapped in double quotes (so a path
containing spaces, like a username with a space in it, still works) before
being sed-escaped (backslash, pipe and ampersand, since `|` is the `sed`
delimiter used here). `install.sh` installs it to
`~/.local/share/applications/vocalis.desktop` right after
`service_enable_and_start()`, using the same script-relative-else-repo-root
template lookup as the unit file; a failure there is logged but never fails
the install, since the tray icon and `vocalis --settings` still work
without a menu entry. `uninstall.sh` removes it; `build-run.sh` stages the
template into both `.run` files alongside the systemd unit template.

## Stage 6a addition: the voice-engine chain

Before Stage 6a, each reader had its own copy of "try Qwen3-TTS, else
Piper". Stage 6a replaces that with one shared path: **selector -> plan ->
chain**.

- `engines/base.py` defines the shared shapes: `Synthesizer` (anything
  with `synthesize(text) -> (audio, rate)`) and `TTSEngine`
  (a `Synthesizer` that is also a `ModelEngine`, so `ModelManager` can
  load/unload it).
- `engines/auto_select.py` is the pure policy: `decide(hardware, policy,
  held_vram_mb)` returns an `AutoDecision` (an ordered tuple of engine
  ids plus a plain-English reason). The thresholds live in the frozen
  `AutoPolicy`: more than 4096 MB total graphics memory and at least 4096
  MB free for Qwen3-TTS; stress means load per core above 0.85 or free RAM
  below `max(2048 MB, 15% of total)`; below 1024 MB free RAM, or no GPU, the
  order starts with Piper. `held_vram_mb` (memory Vocalis itself holds) is
  added back to "free" so its own loaded Qwen3-TTS never counts as stress.
- `engines/registry.py` has `EngineSelector.plan()`. A forced choice maps
  to a fixed chain (`FORCED_CHAINS`: Qwen3-TTS -> Kokoro -> Piper, Kokoro
  -> Piper, Piper -> Kokoro); "auto" probes the hardware and calls
  `decide()`. If the probe raises, the plan is Kokoro -> Piper.
  `kokoro_use_gpu()` tells Kokoro whether to run on the graphics card
  (at least 1024 MB free).
- `engines/chain.py` has `EngineChain`, built per reading from an
  `EnginePlan`. `synthesize()` walks the plan: managed engines go
  through `ModelManager.acquire/release`; Piper is called directly. On a
  failure it remembers the position (so later pieces of the same reading
  never retry a dead engine) and calls `on_downshift`, which the readers
  turn into the "Primary voice unavailable — using backup voice"
  notification.
- `engines/kokoro_tts.py` is the Kokoro engine (`kokoro-onnx`). The
  library is imported lazily, so importing the module is cheap.
- `QuickReaderFeature` and `DocumentReaderFeature` take an
  `engine_selector`; `cli.py` builds one `EngineSelector` and the factory
  dict `{qwen3_tts: ..., kokoro: ...}` for each reader.

**Piper stays unmanaged.** It is a CPU subprocess, so it never goes through
`ModelManager` (as before); the chain calls the always-ready Piper engine
directly and also uses it as the last resort.

**Why `kokoro-onnx` is installed with `--no-deps`, and why the GPU build is
re-installed last.** Both `qwen-tts` and `kokoro-onnx` require plain
`onnxruntime`, which shares one folder with `onnxruntime-gpu`; whichever is
installed last wins, and the wrong one would silently take the graphics card
away from Parakeet. `pyenv_install_kokoro` installs `kokoro-onnx==0.4.7` with
`--no-deps` (its other small dependencies are listed in Vocalis's own
`pyproject.toml`: `colorlog`, `espeakng-loader`, `phonemizer-fork`), and then
`pyenv_prefer_gpu_onnxruntime` re-installs the GPU build last so it always
wins. The `check_onnx_gpu` startup check still guards against a regression,
and re-running the installer fixes it.

**Not in this stage.** Kokoro is called at speed 1.0 and still gets the
same time-stretch as the other voices (native speed is Stage 6c);
gapless long-text reading is 6b; resource limits and "stay downshifted for
60 seconds" hysteresis are 6d.
