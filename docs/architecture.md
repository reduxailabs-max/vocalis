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
3. (Phase 3 design; since Stage 6b Quick Reader splits the text into chunks
   and speaks them through `ChunkedSpeaker` on one stream — see "Stage 6b
   addition".) The background thread reads the clipboard (`ClipboardReader`), then
   calls `model_manager.acquire("qwen3_tts", ...)` to get a loaded
   `Qwen3TTS` engine — obeying the same single-residency and idle-unload
   policy every GPU engine uses. If that fails for any reason, it falls
   back to an always-loaded `PiperTTS` instance, which runs entirely on
   CPU and is never routed through `ModelManager`. (Phase 3 design; since
   Stage 6a the voice is chosen per reading and falls down the chain
   Qwen3-TTS -> Kokoro -> Piper — see "Stage 6a addition" below.)
4. The generated speech is played via `AudioPlayer` (sounddevice-based,
   the same library `MicRecorder` uses for capture), which can be
   interrupted mid-playback. (Phase 3 design: `wait_until_done()` was bounded by the
   audio's expected duration plus a safety margin. Since Stage 6b a broken device
   is caught by `SpeechStream`'s 10 s stall check instead.) A misconfigured
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
   (`text/chunker.py`), then speaks the chunks through `ChunkedSpeaker` on one continuous
   stream (Stage 6b; before that it played one chunk at a time), using the same `ModelManager` single-residency policy and the same
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

### Pacing (`audio/pace.py`): each engine applies its own

`apply_speed(audio, sample_rate, speed)` runs the already-synthesized
waveform through `librosa.effects.time_stretch` (pitch-preserving);
`librosa` is imported *inside* the function so merely importing
`vocalis.audio.pace` (which `config.py` does, to get `MIN_SPEED`/
`MAX_SPEED`) does not pay librosa's import cost. `clamp_speed()` keeps
the factor inside `0.7`–`1.6` (NaN treated as `1.0`).

`paced_audio(audio, sample_rate, provider)` reads the caller's current speed
from `provider()` and applies it, or returns the audio unchanged if
`provider` is `None`, or if either call raises (logged as a warning — a
pacing failure must never break a reading). Since Stage 6c it has exactly one
caller: `Qwen3TTS.synthesize()`, which has no native speed control of its own
and uses it for a mild, capped (1.0-1.2x) time-stretch — never slower than
natural speech, never sped up past 1.2x, so the stretch never gets heavy
enough to hurt this voice's quality. Kokoro and Piper have real speed
controls and never call `paced_audio` at all: `KokoroTTS.synthesize()`
passes the configured speed straight to `kokoro.create(..., speed=...)`, and
`PiperTTS.synthesize()` converts it to Piper's `--length_scale` argument
(`1 / speed`). A pause still rewinds the heard clip inside `SpeechStream`, so
a resume replays the same already-paced audio.

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

## Stage 6b addition: gapless long reading

Before Stage 6b, Document Reader opened a new sound stream for every chunk
and Quick Reader synthesized the whole clipboard text before it said a
word. Long readings therefore had a click or gap at every join. Stage 6b
makes a long reading one continuous stream, for both readers.

### The producer/consumer picture

```
ChunkedSpeaker.run(chunks)            SpeechStream                 sound card
 (reader's worker thread)        (bounded queue, 3 clips)     (audio callback thread)
   make chunk i  ──push──▶  [clip][clip][clip]  ──────▶  plays clips back-to-back
   (EngineChain, pace)        back-pressure: push waits     fades, silence on underrun
```

- `features/chunked_speaker.py` — `ChunkedSpeaker` is made once per
  reading. Its `run(chunks)` is a producer loop on the caller's worker
  thread: for each chunk it asks a `SpeechSource` (in practice an
  `EngineChain`, whose chosen engine already applied its own pace -- see
  "Stage 6c addition: clarity" below) for audio and pushes it into the
  stream. It returns a `SpeakOutcome`
  (`DONE`, `STOPPED`, `SPEECH_FAILED` when no voice could make a chunk, `PLAYBACK_FAILED`). `pause()`, `resume()` and `stop()`
  are thread-safe; `stop()` is a permanent latch, so a stop that lands
  while the model is still loading means the reading never makes a sound.
- `audio/speech_stream.py` — `SpeechStream` opens ONE sound stream through
  `AudioPlayer.open_stream()` and keeps it open for the whole reading.
  Clips wait in a queue of at most 3 (one playing, two waiting); `push()`
  blocks when the queue is full (back-pressure), so the producer runs only
  a few chunks ahead of the listener instead of heating the machine.
- `audio/playback.py` — `AudioPlayer` gained `open_stream()`,
  `close_stream()` and `report_device_failed()`. All device logic (the
  device chooser, the retry, the failure hook) stays in this one class.
  `AudioPlayer.play()` / `wait_until_done()` / `stop()` still exist
  unchanged for other callers (OCR Reader, voice preview, Test sound).

### What makes the joins inaudible

- **Fades.** Every clip gets an 8 ms fade-in and fade-out.
- **Underrun silence.** If the next clip is not ready when the current one
  ends, the sound card is fed silence (counted as an "underrun"); the
  stream never stops or restarts, so a slow chunk is a short pause, never
  a click.
- **Back-pressure.** The queue bound (3 clips) is why synthesis does not
  race ahead.
- **Rate resampling.** The stream opens at the first clip's sample rate.
  If a later clip has another rate (the voice fell down the chain, e.g.
  Qwen3-TTS 24 kHz -> Piper 22.05 kHz) it is linearly resampled, so the
  voice never plays at the wrong speed or pitch.

### Pause, stop and failure

- **Pause** outputs silence at once and rewinds the clip being heard to
  its start; **resume** replays that piece from the beginning (the
  long-standing behaviour). A pause that arrives before the stream exists
  makes the reading start paused.
- **Stop** wakes everything that could be waiting (a blocked `push()`, the
  end-of-reading wait) and closes the device exactly once.
- **Stall detection -> "Playback failed".** If clips are waiting but the
  sound device has not called back for 10 s, the device is broken (the
  real `~/.asoundrc` failure found in Phase 3). `push()` /
  `wait_until_done()` raise `PlaybackTimeoutError`, the device is reported
  to `AudioPlayer`'s failure hook and the reader says "Playback failed".
  This replaces the old "expected duration + margin" timeout, which cannot
  work for a stream that stays open.
- **Progress** ("piece N of M") is reported when a piece *starts being
  heard* (a tag travels with each clip and a helper thread reports it from
  the audio callback), not when it starts being made.

### The voice is held for the whole reading

`EngineChain(hold_engines=True)` (built through `new_reading_chain()`)
keeps the engine that spoke loaded between pieces and for the whole
reading, including a long pause. `ChunkedSpeaker.run()` calls
`close()` only when it leaves, i.e. when playback has *ended*, so
`ModelManager`'s idle-unload timer starts counting then, not when the last
chunk was merely synthesized. The chain still re-acquires its engine
before every piece, so if another feature (e.g. Dictation) took the
graphics card in the meantime the voice is simply reloaded.
`make_downshift_handler()` builds the "Primary voice unavailable — using
backup voice" callback both readers share.

### The chunker

`text/chunker.py` is now sentence-aware and reports exact positions:
`chunk_spans()` returns `Chunk(text, start, end)` objects and `chunk_text()`
is a thin wrapper that returns just the texts. Breaks: `. ! ?` followed by
whitespace, and line breaks (see ruling 10). It never breaks after a
known abbreviation, an initial, or before a lowercase word. It is linear
in time (5,000,000 characters in under half a second).

### Decisions made for Stage 6b

Made while writing the plan:

1. **Where the persistent stream lives.** `AudioPlayer` keeps all the
   output-device logic (chooser, retry, failure hook) and gains the tiny
   `open_stream` / `close_stream` / `report_device_failed` API; `SpeechStream`
   is built on it, so exactly one place talks to `sounddevice`. The old
   `play()` family stays.
2. **Sample-rate change mid-reading:** the stream opens at the first clip's
   rate; later clips at another rate are linearly resampled.
3. **Fade length** 8 ms at both ends of every clip.
4. **Back-pressure bound** is 3 clips (one playing + two waiting).
5. **Stall detection:** 10 s without a single audio callback while clips are
   waiting fails the reading with "Playback failed". (Refinements below.)
6. **Progress** is reported when a piece starts being heard, not when it
   starts being made, so the first report arrives when the first sound
   starts (after the model has loaded).
7. **Voice held while paused:** the engine stays loaded for the whole
   reading including a long pause; it is released, and the idle timer
   starts, when the reading ends.
8. **Another feature taking the GPU mid-reading:** the chain re-acquires its
   engine before every piece, so a stolen model is reloaded instead of
   crashing the reading.
9. **Quick Reader notification order** (corrected from the plan, which said
   "Done waits for the text notice before the end is reported"): "Reading..."
   is sent first; the copied text is sent from a one-off side thread so it
   can never delay the sound. When playback ends the reading END is
   reported FIRST (`_deactivate`), then the worker waits at most 6 s for
   the copied-text notification, then sends "Done" (or the failure
   message). A stop skips the wait (the stop already said "Stopped").
10. **Chunker newline rule:** a blank line is always a break. A single
    newline is a break unless the line before it looks hard-wrapped (at
    least 70 characters, not ending in `.` `?` `!`, and not ending in `,`
    `;` `-`). Short lines without punctuation (headings, list items) break.
    A single capital letter followed by a full stop ("J. Smith", "U.S.")
    never breaks — a missed break only puts two sentences in one chunk,
    while a false break after "Dr." is audible.
11. **Spans are into the text handed to the chunker** (the normalized text
    for Document Reader). Mapping back to the original file for Stage 7
    highlighting is Stage 7's job.
12. **Obsolete tests are deleted, not ported,** where they only exercised
    the old per-chunk `play()` timing windows; their intent is re-covered
    by the `SpeechStream` / `ChunkedSpeaker` tests.

Made during the build (they correct or refine the plan):

- **"`close()` releases once" was wrong.** `ModelManager` keeps a use count
  that `acquire()` raises on every reuse. So `EngineChain(hold_engines=True)`
  remembers the held engine *object*, gives back the extra count whenever a
  repeat `acquire` returns the same object (so the hold is exactly one
  count), and `close()` releases only if the manager's loaded engine (the new
  `ModelManager.current_engine` property) is still that object. Engine
  factories must therefore build a NEW object on every load. A late
  `close()` from an old reading cannot lower a newer reading's count.
- **The chunker is linear-time.** The first version copied large slices on
  every full stop and was quadratic; it now looks back only a bounded
  window (40 characters for the word before a full stop, 200 for the
  previous line).
- **SpeechStream details.** The stall clock restarts after the device opens
  (a slow device open is not a stall); a paused stream is never "stalled";
  a stop during device open is handled by an `_opening` hand-off (the
  device is closed by `start()`); the device is closed only by the stream
  that opened it (`_device_opened`), so a stream that never opened cannot
  close a shared player's device; after a clean finish the clip-started
  helper thread is joined (bounded 2 s) so the final "piece N of M" is
  delivered; the clip-started hook is not called after stop.
- **Document Reader.** A repeat read on the same feature stops the previous
  reading first. The pause is applied to the not-yet-published speaker
  *inside* the publishing lock — the one documented exception to "never
  call out while holding a lock" (it only flips a flag; without it a resume
  landing in that window was lost). `ChunkedSpeaker` no longer calls
  `stream.stop()` after a clean finish (stop drops queued progress tags, so
  the last "piece N of M" could be lost). An empty chunk list with a stop
  already latched returns `STOPPED`.

### Known limits (accepted)

- The voice stays loaded while a reading is paused (ruling 7), however long.
- A stop landing at the same instant as a repeat read can order the reports
  start-before-end.
- Quick Reader can show both "Stopped" and "Done" if a stop lands between
  `run()` returning and the reading end being reported (13 of 300 stress
  runs; the window existed before Stage 6b).

## Stage 6c addition: clarity

Stage 6b made long readings gapless; Stage 6c makes them sound better.
Four independent changes, each scoped to the engine or layer that owns the
problem.

### Each engine paces itself

Before Stage 6c, every engine's output went through one shared
`paced_audio()` step in `ChunkedSpeaker`, which always used a lossy
`librosa` time-stretch — even for engines that have a real speed knob.
Now each engine applies its own pace during `synthesize()`, and
`ChunkedSpeaker` no longer touches pacing at all:

- `KokoroTTS` and `PiperTTS` gained a `speed` constructor argument (a
  float or a live callable, read fresh on every call, the same pattern
  `voice`/`speaker` already use). Kokoro passes it straight through to
  `kokoro.create(..., speed=...)`; Piper converts it to its
  `--length_scale` command-line argument (`1 / speed`). Neither ever
  calls `apply_speed()`.
- `Qwen3TTS` has no native speed control, so it keeps the `librosa`
  stretch — but only for a mild, capped range (**1.0x-1.2x**): a bigger
  stretch, in either direction, audibly hurts this voice, so a slower or
  more-sped-up request is simply played at the nearest end of that range
  instead. `audio/pace.py` is unchanged; `paced_audio()`'s only caller is
  now `Qwen3TTS.synthesize()`.
- `cli.py`'s `_make_qwen_factory()` / `_make_kokoro_factory()` and both
  `PiperTTS(...)` construction sites now resolve `speed` the same way
  they already resolve `voice`/`speaker` (`settings.speaking_speed` when
  live settings exist, else the config value), and pass it to the engine.
  `QuickReaderFeature` and `DocumentReaderFeature` lost their
  `speed_provider` constructor argument entirely — pacing is no longer
  something a reader orchestrates.

### Qwen3-TTS generation tuning

`Qwen3TTS.synthesize()` now passes three extra arguments to
`generate_custom_voice()`: a lower, steadier `temperature` (`0.65` instead
of the library's own `0.9` default), a lower `top_k` (`20` instead of
`50`), and a `max_new_tokens` scaled to the text's length (`6` tokens per
character, floored at `200`, capped at the library's own default of
`2048`) instead of always requesting the full default budget. A typical
~300-character chunk now asks for about 1800 tokens (roughly 88% of the
library's own 2048 default) instead of always requesting the full 2048
regardless of length; the real benefit of this scaling is mainly at the
floor (very short chunks ask for far less than 2048) and the ceiling (an
unusually long chunk is still capped at the library's own tuned default).

### English-native speaker, with a warning on a mismatch

The default Qwen3-TTS speaker was already `Ryan` (English-native); Stage
6c adds a check that stays useful if that ever changes. Qwen3-TTS always
generates speech with `language="English"` (configuring the spoken
language itself is not yet exposed), but five of its nine preset voices
are native to Chinese, one to Japanese, and one to Korean. `EnginesConfig`
gained a `model_validator` that logs a warning (never raises — a
mismatched choice still works, it just sounds worse) whenever
`qwen3_tts_speaker` is not one of the two English-native presets (`Ryan`,
`Aiden`).

### Basic text normalization

`text/normalizer.py` gained `normalize_text()`: a small, deliberately
basic pass that speaks out URLs (`https://example.com/docs` -> "example
dot com slash docs"), dollar amounts (`$42` -> "42 dollars") and
percentages (`50%` -> "50 percent"), and expands `&`/`@` into "and"/"at".
Plain numbers are left alone — the voice engines already read digit
sequences correctly, so over-normalizing them would do more harm than
good. `normalize_markdown()` now calls it as its last step, so Document
Reader gets it for free; Quick Reader (whose clipboard text is never
markdown) calls it directly, right before chunking, while still notifying
the user with the original, un-normalized copied text.

## Stage 6d addition: resource limits

Stage 6c made the voices sound better; Stage 6d makes Vocalis share the PC
more politely. Three settings control this — `max_cpu_threads` and
`max_ram_mb` (new `VocalisConfig` int fields, default `0`, meaning "no
limit") and the existing `gpu_vram_ceiling_mb` (default 6144, never a `0`
sentinel — it is always a real ceiling). All three are exposed in the
Settings window as spin boxes (`ui/settings_window.py`:
`Max CPU threads (0 = no limit)`, `Max RAM, MB (0 = no limit)`,
`GPU VRAM ceiling, MB`) and read live through three new `LiveSettings`
getters — `max_cpu_threads()` and `max_ram_mb()` translate the config's `0`
into `None`; `gpu_vram_ceiling_mb()` just passes the value through.

**CPU threads and GPU VRAM are applied per engine, at load time:**

- **Qwen3-TTS** (`engines/qwen3_tts.py`) calls `torch.set_num_threads()`
  with the live `max_cpu_threads` value, and, when running on the GPU,
  caps VRAM with `torch.cuda.set_per_process_memory_fraction()`: the
  ceiling in MB is converted to a fraction of the card's total memory
  (via `torch.cuda.mem_get_info()`) and clamped to 1.0 if the configured
  ceiling is larger than the card actually has. Both are re-read fresh on
  every `load()` (the callable form `LiveSettings` hands engines), and the
  VRAM cap is best-effort — a failing GPU query is caught, logged as a
  warning, and never blocks the model from loading.
- **Kokoro** (`engines/kokoro_tts.py`) applies the same two caps through
  ONNX Runtime instead of torch: `max_cpu_threads` becomes
  `SessionOptions.intra_op_num_threads`, and `gpu_vram_ceiling_mb` becomes
  the CUDA execution provider's `gpu_mem_limit` option (bytes). Also
  applied at load time, also best-effort.
- **Piper** has no lever for either — its installed binary's `--help`
  output has no thread-count flag at all, and it has no VRAM to cap
  (CPU-only). Its only hook is `OMP_NUM_THREADS`/`ORT_NUM_THREADS`, set on
  the subprocess environment Piper is launched with. Since Piper has no
  persistent "load" (`engines/piper_tts.py` spawns a fresh process per
  synthesis call), these are re-read fresh on *every* synthesis, not once.
  This is explicitly best-effort: Piper does not document honoring either
  variable, so setting them is a polite request, not a guarantee.

**`max_ram_mb` is deliberately advisory, not an OS-level limit.** Vocalis
never calls `resource.setrlimit()` and never risks a `MemoryError` crash by
enforcing a hard ceiling on itself. Instead it feeds into the existing
Auto-selection "is the PC stressed" check: `HardwareInfo` gained an
`own_process_ram_mb` field (`hardware.py`'s `query_own_process_ram_mb()`,
reading the `VmRSS:` line of `/proc/self/status` fresh on every probe —
Vocalis's own *current* resident memory, not its peak). `auto_select.is_stressed()`
now also returns `True` whenever `own_process_ram_mb` is at or above the
live `max_ram_mb` ceiling, alongside its existing CPU-load and system-wide
free-RAM checks. When that happens, Auto treats the PC as stressed and
downshifts to a lighter engine exactly as it already does for a busy
processor — see `docs/algorithms.md` for the full `decide()` policy.

**Downshift hysteresis.** `EngineSelector` (`engines/registry.py`) now
remembers the tier of its *last* downshift and, for 60 seconds after it,
caps any new "Auto" decision back down to that tier even if the hardware
snapshot would otherwise qualify for something better. This only guards
against reload thrashing when a hardware reading is noisy right around a
threshold (e.g. CPU load bouncing just above and below the stress line
between two back-to-back readings); it has no effect once 60 seconds have
passed since the last downshift, and it never applies to a user-forced
engine choice (Qwen3-TTS/Kokoro/Piper picked explicitly in Settings) —
`EngineSelector.plan()` returns a forced chain before `decide()` or the
hysteresis check ever runs. See `docs/algorithms.md` for the mechanics.

**`systemd` scheduling priority.** `packaging/systemd/vocalis.service`'s
`[Service]` section gained `Nice=10`, lowering Vocalis's CPU scheduling
priority so it competes less aggressively for processor time with other
running programs.

**Model unload now forces a garbage collection.**
`ModelManager._unload_current_locked()` calls `gc.collect()` after the
engine's own `unload()` (which already calls `torch.cuda.empty_cache()`
where relevant), to force the release of any model or tensor memory that a
reference cycle would otherwise keep alive until Python's own cyclic
collector happened to run on its own schedule.
