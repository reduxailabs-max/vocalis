# Workflow

## User perspective

Running `vocalis` starts the background process and its Qt event loop.
A tray icon, a floating widget and a Settings window (Phase 6) show and
control what it does. Features are used through hotkeys: NUMLOCK
(dictation), Ctrl+Alt+R (Quick Reader), Ctrl+Alt+D (Document Reader popup)
and Ctrl+Alt+O (OCR Reader); the key choices are the defaults and can be
changed in Settings (applied after a restart). Step-by-step user
instructions are in `docs/user-docs.md`; the internal flows follow.

## Internal (daemon) perspective

```
main()
  -> load_config(default_config_path())     # ~/.config/vocalis/config.toml, or defaults
  -> startup_notifier = Notifier(enabled=config.notifications_enabled)
  -> run_startup_checks(config) -> notify() each real problem found      # Stage 2
                                              #   (model/ydotoold/OCR-tools checks; never fatal)
  -> settings = LiveSettings(config, default_config_path())
  -> codes = resolve_shortcuts(config)        # key codes; bad shortcut -> log error, use defaults
  -> app = QApplication(...); quitOnLastWindowClosed = False
  -> hub = StatusHub()                        # created after QApplication, before the builders
  -> model_manager = ModelManager(idle_timeout_seconds=config.model_idle_timeout_seconds)
                                              # shared across daemon + every feature
  -> build_daemon(model_manager, socket_path)
       -> configure_logging()
       -> VocalisDaemon(socket_path, logger, model_manager)
  -> reading_coordinator = ReadingCoordinator()  # shared by both readers
  -> build_dictation(...)       -> (DictationFeature, NumlockListener)       # codes.push_to_talk
  -> build_quick_reader(...)    -> (QuickReaderFeature, listener)            # codes.quick_reader
  -> build_doc_reader(...)      -> (DocumentReaderFeature, popup, listener)  # codes.document_reader
  -> build_ocr_reader(...)      -> (OcrReaderFeature, listener)              # codes.ocr_reader
  -> build_ui(settings, hub, popup, quit_app) -> (widget, settings_window, tray)
  -> SIGINT and SIGTERM handlers = app.quit()  # Stage 2: SIGTERM handled identically to SIGINT
                                              # 200 ms keepalive QTimer (lets Python run either handler)
  -> daemon.start()                          # Stage 2: connect-before-delete single-instance probe
  -> dictation/quick/doc/ocr listeners .start()
  -> tray.show()                             # warns once and continues if no tray exists
  -> app.exec()                              # Qt event loop; Ctrl+C (SIGINT) or `systemctl stop` (SIGTERM)
                                              #   -> app.quit(); tray Quit -> tray.close() then app.quit()
  -> finally (each step independent, QApplication still alive):
       keepalive timer stop + deleteLater; restore previous SIGINT and SIGTERM handlers
       tray.close(); widget.hide()
       ocr / doc / quick / dictation listeners .stop()
       daemon.stop()                          # closes socket, removes socket file,
                                              #   calls model_manager.shutdown()
       settings_window.close(); widget.close()
```

## Model acquire/release lifecycle (used by every future feature)

```
feature code:
  engine = model_manager.acquire("parakeet", lambda: ParakeetSTT(...))
  ... use engine ...
  model_manager.release("parakeet")

ModelManager, internally:
  acquire(id, factory):
    cancel any pending idle-unload timer; advance the generation counter
    if id already loaded -> reuse it
    else -> unload whatever else is loaded (see below), then factory().load()
           (if factory()/load() raises, bookkeeping is reset to "nothing
           loaded" and the exception is re-raised)
  release(id):
    use count -= 1 (acquire added one); extra release at zero logs a warning but still falls through to arm the idle timer
    if other users of the same id remain (count > 0) -> stop here, no timer
    cancel any timer left over from a previous release()
    start a new idle timer (config.model_idle_timeout_seconds), scoped to
    the current generation
    if not reacquired before it fires -> unload (unless the generation
    changed since scheduling, in which case the timer is a no-op)
  unloading an engine:
    always resets bookkeeping to "nothing loaded" even if the engine's own
    unload() raises
```

(See `docs/algorithms.md` for the full explanation of the generation counter and why it's needed.)

## Dictation press/release flow (Phase 2)

```
NUMLOCK pressed (evdev event, NumlockListener)
  -> DictationFeature.on_press()
       -> notify("Listening...")
       -> recorder.start()                      # begins capturing mic audio

NUMLOCK released (evdev event, NumlockListener)
  -> DictationFeature.on_release()
       -> audio = recorder.stop()
       -> if no audio captured: notify(NO_AUDIO_MESSAGE), stop   # Stage 5: longer, actionable message
       -> notify("Transcribing...")
       -> engine = model_manager.acquire("parakeet", parakeet_factory)
       -> text = engine.transcribe(audio, sample_rate=16000)
       -> model_manager.release("parakeet")      # starts the idle-unload timer
       -> if blank transcript: notify("(no speech detected)"), stop
       -> notify(text)                            # live transcript notification
       -> result = text_output.deliver(text)      # ydotool, or clipboard on failure
       -> notify("Done" | "Typing failed — copied to clipboard instead")
```

Every failure path (recording start/stop, transcription, clipboard-also-
failing) notifies a specific reason and returns without crashing the
listener thread or the daemon.

## Quick Reader trigger/stop flow (Phase 3)

```
Ctrl+Alt+R pressed (evdev event, CtrlAltRListener)
  -> QuickReaderFeature.on_trigger()
       -> if already reading: flip flag, spawn thread to stop()+notify("Stopped"), return immediately
       -> else: mark active, spawn background thread, return immediately

Background thread:
  -> text = clipboard_reader.read()
  -> if cancelled or empty: notify accordingly, stop
  -> notify("Reading...")
  -> try: engine = model_manager.acquire("qwen3_tts", qwen3_tts_factory)
          audio, sr = engine.synthesize(text)
          model_manager.release("qwen3_tts")
     except: notify("Primary voice unavailable — using backup voice"), audio, sr = piper.synthesize(text)
  -> if cancelled while synthesizing: stop, never start playback
  -> notify(text)                              # live transcript notification
  -> player.play(audio, sr); finished = player.wait_until_done()
  -> if finished naturally: notify("Done")
```

Every failure path (empty clipboard, both engines failing, playback
failure) notifies a specific reason and returns without crashing the
listener thread or the daemon. A second press is only ever noticed
promptly because on_trigger() itself never blocks — all the real work
happens in the spawned background thread.

`wait_until_done()` is itself bounded: if the audio callback never fires
(e.g. a misconfigured default sound device), it raises
`PlaybackTimeoutError` after the audio's expected duration plus a safety
margin, rather than blocking forever — that exception is caught by the
same playback `except` path as any other playback failure, so it reaches
the user as a normal "Playback failed" notification instead of an
unexplained freeze.

## Document Reader flow (Phase 4)

User perspective: press Ctrl+Alt+D → popup → Open file (or paste text and
press Read) → hear it read → Pause/Resume, Stop. Ctrl+Alt+D while reading
stops it. See `docs/user-docs.md` for the error messages.

```
Ctrl+Alt+D pressed (evdev thread, CtrlAltKeyListener)
  -> popup.hotkey_pressed.emit()               # signal only; Qt queues the slot
  -> (GUI thread) popup.on_hotkey_trigger()
       -> if feature.is_reading: feature.stop()
       -> else: popup.show()/raise_()/activateWindow()

Open file / Read clicked (GUI thread)
  -> DocumentReaderFeature.read_file(path) | read_text(text)
       -> _start_new_reading()                 # generation + 1
       -> reading_coordinator.start_reading(self, self.stop)
            # if Quick Reader owned playback, its stopper runs now
       -> spawn worker thread; return immediately

Worker thread (_prepare_and_read, then _read_worker):
  -> extract_text(path)       # file only; typed errors -> notify + status + on_finished
  -> normalize_markdown(text)
  -> chunk_text(...)          # no chunks -> "Nothing to read"
  -> for each chunk (index i of N):
       on_progress(i, N)
       synthesize: Qwen3-TTS via model_manager.acquire("qwen3_tts", ...);
                   on the first failure switch to Piper for the REST of this
                   document and notify once ("Primary voice unavailable...")
       wait if paused, then player.play(); wait_until_done()
       finished naturally -> next chunk
       stopped early by pause() -> wait for resume(), replay the SAME chunk
       stopped by stop()/preemption -> return
  -> all chunks played -> "Done", on_finished()
```

`pause()`, `resume()` and `stop()` return at once: their "Paused",
"Resuming..." and "Stopped" desktop notifications are queued and sent, in
call order, by one background thread. The popup's Pause button tracks its
own paused state and does nothing if no reading is active.

Quick Reader and Document Reader share one `ReadingCoordinator`. Whichever
calls `start_reading()` becomes the owner and the previous owner's stop
callback is invoked, so the two never speak at once. Quick Reader's
callback is clear-only: preempting an idle Quick Reader does nothing.

Thread rule: the hotkey thread and the worker thread never touch widgets.
`on_status`, `on_progress` and `on_finished` are wired (in
`build_doc_reader()`) to emit the popup's `status_changed`,
`progress_changed` and `reading_finished` signals, and the hotkey listener
is wired to `hotkey_pressed.emit`; Qt delivers each on the GUI thread.

## OCR Reader flow (Phase 5)

User perspective: press Ctrl+Alt+O -> drag (and resize) a box over text ->
Enter to confirm (Esc cancels) -> the text is copied and read aloud; the
floating widget shows it with Pause/Resume and Stop (if the widget is
switched off, the Document Reader window opens instead, as in Phase 5). Ctrl+Alt+O while it reads
stops it. If a document reading is playing instead, confirming a new box
stops that document (the "Stopped" message appears) and reads the screen text. See `docs/user-docs.md` for the messages.

Bot logic: see "Phase 5 addition: OCR Reader" in `docs/architecture.md`
for the full diagram. In short: `OcrReaderFeature` runs capture ->
Tesseract -> clipboard -> notification -> popup signal ->
`DocumentReaderFeature.read_text` on a background thread. The Tesseract
step is CPU-only and does not go through `ModelManager`. Errors
(cancelled, no tool, Tesseract failing, no text) each become a
notification and never crash the daemon.

## Tray, widget and Settings flow (Phase 6)

User perspective: a coloured dot sits in the tray (grey idle, red while you
hold the dictation key, blue while reading). When something is read aloud a
small box appears bottom-right with Pause/Resume and Stop (Quick Reader:
Stop only) and disappears at the end. **Settings…** in the tray menu opens
the Settings window; the two switches and the voice apply at once,
shortcuts after a restart. **Quit** in the tray menu is the same as Ctrl+C.
See `docs/user-docs.md`.

Bot logic, reporting side: a feature starts a reading -> it calls the hub
(`begin_reading(source, controls)`) -> the hub updates its state under a
lock, then emits `state_changed` and `reading_started` -> the tray changes
colour and enables Stop/Pause, the widget (if `floating_widget_enabled()`)
fills its labels and shows itself. While reading, `report_status`,
`report_progress` and `set_paused` update the labels/buttons. At the end
`end_reading()` emits `reading_finished`; the widget hides and the tray
returns to grey. Commands go the other way: a button click calls
`hub.pause()/resume()/stop()`, which call the `ReadingControls` the feature
supplied.

Settings side: **Save** -> `SettingsWindow` builds a new `VocalisConfig` from
its widgets -> `LiveSettings.update()` parses the shortcuts (invalid or
duplicate -> `ShortcutError`), `save_config()` writes a temp file and renames
it over the real file (failure -> `ConfigError`), then the in-memory config
is replaced -> the window shows "Saved." (plus the restart note when the
shortcuts differ from those at start-up) or "Not saved: <reason>" and leaves
everything unchanged -> after a successful save the widget re-checks its
on/off switch. Notifier, voice and widget read `LiveSettings` each time they
are used, so no restart is needed for them.

## Sound output, speaking pace and voice preview (Stage 1)

User perspective: open Settings, pick a sound output (or leave it on
"Automatic"), click **Test sound** to confirm it works, drag the
**Speaking speed** slider, and click **Preview voice** to hear a sample
with your saved settings before trusting it on a real reading. See
`docs/user-docs.md`'s "Sound and speaking pace" section for the full
list of messages.

```
main()
  -> output_chooser = OutputDeviceChooser(preference=settings.audio_output_device)
  -> build_quick_reader(..., output_chooser=output_chooser)
       -> player = AudioPlayer(device_provider=output_chooser.current,
                                on_device_failed=<hook tied to output_chooser>)
  -> build_doc_reader(..., output_chooser=output_chooser)
       -> player = AudioPlayer(device_provider=output_chooser.current,
                                on_device_failed=<a separate hook, same chooser>)
  -> build_ui(..., output_chooser=output_chooser)
       -> SettingsWindow(output_names=output_chooser.output_names,
                          on_test_sound=<built by _make_on_test_sound>,
                          on_preview_voice=<built by _make_on_preview_voice>)
```

Reading-time pacing flow (Quick Reader, same idea for Document Reader
per chunk):

```
synthesize text -> audio, sample_rate
-> audio = paced_audio(audio, sample_rate, settings.speaking_speed)
     -> provider() reads the CURRENT speed at the moment of synthesis
     -> apply_speed() time-stretches unless speed is within 1% of 1.0
     -> any failure here is logged and the original audio is played instead
-> player.play(audio, sample_rate)
```

Device-failure flow, shared by both readers through one chooser:

```
player.play() -> stream fails to start, or wait_until_done() times out
  -> player asks device_provider() again
       -> if it now names a DIFFERENT device: retry opening that one, once
       -> else: give up, re-raise (reaches the reader's existing
          "Playback failed" notification path, unchanged since Phase 3)
  -> on_device_failed(device) is called regardless of whether the retry
     will be attempted
       -> chooser.mark_failed(device)   # next current() call skips it
       -> notifier.notify(...) only if `device` differs from the last
          device this hook already notified about
```

Test-sound flow (Settings window, `_make_on_test_sound`):

```
Test sound clicked -> on_test_sound(selected)
  -> if a reading is active: return the busy message, open nothing
  -> if selected == "auto": device = output_chooser.current()
       -> None (nothing works) -> "no working output found" message
  -> else (a named device): build a throwaway OutputDeviceChooser fixed to
     that name, call .probe() on exactly that device index (never
     .current(), which would silently fall through to another device)
       -> probe fails -> "<name> did not respond" message, no fallback tried
  -> play a short, quiet 440Hz tone through the resolved device
  -> success -> "Sound test OK (<real device name>)"
  -> any exception during play -> "Sound test failed: <reason>"
```

Preview-voice flow: `on_preview_voice()` simply calls
`DocumentReaderFeature.read_text(PREVIEW_SENTENCE)` — it goes through the
same `ReadingCoordinator`, pacing and device-chooser machinery as any
other document reading (including preempting one already in progress),
using whatever settings were last **saved**, not whatever is currently
sitting unsaved in the window's widgets.

## Single-instance guard and signal handling (Stage 2)

User perspective: starting Vocalis a second time (e.g. by hand, while the
background service copy is already running) does nothing destructive — it
just exits. Stopping Vocalis the way the background service stops it
(`systemctl --user stop`, which this stage builds the code for but does
not itself run) shuts everything down exactly as cleanly as Ctrl+C does.
See "Running automatically" in `docs/user-docs.md`.

```
VocalisDaemon.start()
  -> socket file already exists at _socket_path?
       -> _is_another_instance_alive()        # connects a short-lived probe
                                              #   socket to the existing file
            -> connect succeeds: ALIVE
                 -> raise DaemonAlreadyRunningError; file left untouched;
                    logged only, no desktop notification
            -> ConnectionRefusedError / FileNotFoundError: STALE
                 -> log a warning, unlink() the file, continue to bind
            -> any other exception: AMBIGUOUS
                 -> treated the same as STALE (never blocks startup);
                    the reason is logged as a warning
  -> bind, chmod 0600, listen(5), start the accept-loop thread

systemctl --user stop/restart vocalis   (or Ctrl+C from a terminal)
  -> SIGTERM (or SIGINT)
  -> both point at the same lambda: app.quit()
  -> app.exec() returns -> the usual finally block runs, unchanged:
       listeners stop, tray/widget close, daemon.stop() removes the
       socket file and calls ModelManager.shutdown()
```

## Startup checks (Stage 2)

```
main(), immediately after load_config():
  -> startup_notifier = Notifier(enabled=config.notifications_enabled)
  -> for message in run_startup_checks(config):
       startup_notifier.notify("Vocalis", message)

run_startup_checks(config):
  -> for each of the five checks, in a fixed order
       (Parakeet model, Qwen3-TTS model, ydotoold, OCR capture tools,
        a stale Voxtype ~/.asoundrc — the last one added in Stage 5):
       -> try: result = check()
          except Exception: log a warning, skip this check, continue
       -> if result is not None: collect it
  -> return every real problem found (possibly empty)
```

Every check only reports a problem or stays silent (`None`); nothing it
finds stops Vocalis from starting, and nothing later in `main()` depends
on these checks having run. See `docs/user-docs.md` for what each message
means and `docs/developer-docs.md` for how to add a new check.

## Installing Vocalis (Stage 3)

User perspective: run `bash install.sh` (or double-click `install.run`),
answer the one password prompt, wait for the first-time model download if
one is needed, and Vocalis is installed, running, and set to start at
every future login. See `docs/user-docs.md`'s "Installing Vocalis"
section for the full step list and `uninstall.sh --purge`'s behavior.

```
install.sh main()
  -> preflight_run()                       # distro, arch, GPU driver
       any check fails -> plain message to stderr, exit 1, nothing else runs
  -> apt_install_missing(APT_REQUIRED_PACKAGES)  # the one sudo prompt
  -> apt_enable_ydotoold()                 # best-effort, never fails the install
  -> pyenv_ensure_uv()
  -> wheel_path = pyenv_find_or_build_wheel(script_dir, repo_root)
       # a bundled dist/*.whl (packaged .run case) or built fresh via `uv build`
  -> pyenv_create_venv(~/.local/share/vocalis/venv)
  -> pyenv_install_vocalis(venv_dir, wheel_path)
  -> symlink venv/bin/vocalis -> ~/.local/bin/vocalis
  -> read parakeet_dir/qwen3_tts_dir/piper_binary/piper_voice from
     the just-installed venv's own VocalisConfig (never hard-coded twice)
  -> venv_python lib/fetch_models.py        # Parakeet, Qwen3-TTS, Piper voice
       each: already present at its path -> skip; else -> download there
  -> models_ensure_piper_binary(piper_binary)   # the one non-Hugging-Face asset
       already present -> skip; else -> download to a temp dir, then move into place
  -> service_substitute_unit(template, exec_path, workdir, unit_output)
  -> service_enable_and_start()             # daemon-reload, enable --now
  -> print the "installed and running" summary
```

Re-running `install.sh` is always safe: every step above only acts when
something is missing or wrong, so a second run (deliberate, or a repair
after an earlier partial failure) leaves everything already-correct
untouched.

```
uninstall.sh main()
  -> -h/--help -> print usage, return 0, nothing removed
  -> unrecognised option -> print usage to stderr, return 1, nothing removed
  -> if venv's python exists and --purge was passed: read the config dir
     and four install paths from it first (Parakeet dir, Qwen3-TTS dir,
     and Piper's binary and voice parent dirs) — needed later, before the
     venv is gone
  -> disable/stop vocalis.service (ignored if it isn't running)
  -> remove the unit file (+ daemon-reload), the ~/.local/bin/vocalis
     symlink, and the venv — only each one that actually exists
  -> nothing was found at all -> "nothing to remove", exit 0
  -> --purge -> also remove ~/.config/vocalis, the Parakeet and Qwen3-TTS
     model directories, and Piper's binary and voice directories read
     above; print what was removed
  -> no --purge -> print what was kept (config + models) and how to purge
```

## Opening Settings three ways (Stage 5)

User perspective: a left-click on the tray icon, choosing "Vocalis" in the
applications menu, or running `vocalis --settings` in a terminal all open
the same Settings window. See `docs/user-docs.md`'s "How to open Vocalis"
section.

```
Tray left-click (TrayIcon._on_activated, reason == Trigger)
  -> on_open_settings() -> settings_window.present()

Applications-menu click (packaging/desktop/vocalis.desktop)
  -> runs: <installed vocalis> --settings     # same command as below

`vocalis --settings` (entrypoint(), the vocalis console-script target)
  -> open_settings(socket_path)                        # remote.py
       -> send_command(socket_path, SHOW_SETTINGS)
            -> if the daemon answers: ack received, exit 0, done
       -> else: start_user_service()                    # systemctl --user start vocalis
            -> if it could not even be started: print error, exit 1
            -> else: poll send_command() every 0.5s for up to 20s
                 -> first ack: exit 0
                 -> still nothing after 20s: print error, exit 1

Inside the now-running daemon, when SHOW_SETTINGS arrives:
  VocalisDaemon._handle_connection()           # on the daemon's own socket thread
    -> command_handler(command)                # == bridge.handle_command
         -> bridge.show_settings.emit()          # _SettingsBridge(QObject)
              -> (Qt queued connection, delivered on the GUI thread)
              -> settings_window.present()
    -> sends ack:show-settings regardless of what the handler did
```

`_SettingsBridge` exists only because the socket thread must never touch a
Qt widget directly; its signal is the one safe way to hand the request
over to the GUI thread that owns `settings_window`.

`present()` is safe to call from any of the three paths more than once in
a row — a second call while the window is already open just raises it to
the front instead of opening a duplicate.

## Choosing a voice engine (Stage 6a)

For every Quick Reader or Document Reader reading:

```
reading starts
  -> EngineSelector.plan()
       setting is qwen3_tts / kokoro / piper -> fixed chain
           (Qwen3-TTS -> Kokoro -> Piper | Kokoro -> Piper | Piper -> Kokoro)
       setting is auto -> probe hardware -> auto_select.decide()
           (Vocalis's own loaded graphics memory is credited back)
  -> EngineChain(plan, factories, ModelManager, Piper)
  -> for each piece of text: chain.synthesize(text)
       try the first engine (Piper directly, others via ModelManager)
       on failure: remember it, notify
           "Primary voice unavailable — using backup voice",
           and try the next engine in the plan
  -> audio is paced and played as before
```

A failed engine is not retried for the rest of that reading. The next
reading builds a fresh plan, so a recovered engine is used again.

At startup there are now seven checks, in this order: Parakeet model,
Qwen3-TTS model, Kokoro voice files, `ydotoold`, OCR tools, the
onnxruntime graphics-card check, stale `~/.asoundrc`. At install time,
`fetch_models.py` also downloads the two Kokoro files with a 60-second
timeout per connection, so a stalled download cannot hang the installer.
