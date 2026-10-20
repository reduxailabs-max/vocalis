# Developer Documentation

## Setup

```bash
python3.13 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Phase 4 added these dependencies (all declared in `pyproject.toml`):
`PySide6` (the popup window and Qt event loop), `pypdf` (PDF text) and
`python-docx` (Word files) at runtime; `pytest-qt` and `reportlab` (builds
test PDFs) for development only.

Phase 5 (OCR Reader) adds no Python dependencies. It needs system
programs: `tesseract` with the English pack (`sudo apt install
tesseract-ocr tesseract-ocr-eng`), `xclip`, and one of `spectacle`
(recommended: resizable box), `maim`, `flameshot`, or `slurp` + `grim`. Tests
use fake command runners, so none of them is needed to run the suite.

## Running tests

```bash
.venv/bin/pytest            # runs full suite, fails if coverage < 85%
```

Qt tests run without a screen: `tests/conftest.py` sets
`QT_QPA_PLATFORM=offscreen` before PySide6 is imported, so widgets can be
created on a headless machine. Tests that need Qt use the `qtbot`/`qapp`
fixtures from `pytest-qt`.

## Whole-system tests: `tests/test_cross_feature_integration.py` (Stage 4)

Every other test file in this project tests one feature (or one module)
in isolation, built with its own small set of fakes. `tests/
test_cross_feature_integration.py` is different: it calls `cli.py`'s own
`build_dictation()`/`build_quick_reader()`/`build_doc_reader()`/
`build_ocr_reader()` functions directly — the exact same builders `main()`
uses — and gives them all one real, shared `ModelManager` (and, where
relevant, one real `LiveSettings`), the same way a real `vocalis` process
would. Every external dependency is faked — not just hardware (the
engines, the recorder, the player, the clipboard) but also typed output
(`TextOutput`, which would otherwise run real `ydotool`/`xclip`) and
desktop notifications (`Notifier.notify`, which would otherwise run real
`notify-send`). An automatic safety-net fixture in the file also makes
any test fail loudly if it ever reaches a real `ydotool`, `xclip` or
`notify-send` call, including from a feature's background thread.

This catches a class of bug a single-feature test cannot: problems that
only appear from two or more features sharing state in the same process
— one GPU model actually staying resident across a feature switch,
hotkeys never cross-triggering each other's handlers, a settings change
made through the shared settings path being picked up by an
already-built reader's next reading, two features' notifications both
arriving complete and in order (each builder makes its own `Notifier`,
but they all send through the same `Notifier.notify` path), and a full
two-feature lifecycle ending in a clean shutdown.
If you add a fifth feature or change how `cli.py` wires features together,
add a test here, not just to that feature's own isolated test file.

## Type checking

```bash
.venv/bin/mypy               # strict mode, must report "no issues found"
```

## Project conventions

- Threads never touch Qt widgets directly. The hotkey thread and the
  Document Reader worker thread reach the popup only by emitting Qt
  signals (`hotkey_pressed`, `status_changed`, `progress_changed`,
  `reading_finished`); Qt then runs the receiving code on the GUI thread.

- Settings that should apply immediately are never copied into a feature
  at construction; the feature gets a function from `LiveSettings` and
  calls it where the value is used (see "Adding a new setting").
- Every function has a docstring explaining what it does.
- Config is validated at the boundary (`vocalis.config`) — never trust raw
  TOML data beyond that module.
- All logging goes through the `"vocalis"` logger
  (`vocalis.logging_setup.configure_logging()`); never log full user text
  content or secrets.
- Commits use conventional-commit prefixes (`feat:`, `fix:`, `test:`,
  `docs:`, `chore:`) and are made only after the relevant test suite passes.
- `old/` is a read-only reference to the legacy bash-script implementation
  and must never be modified.

## Adding a new setting

1. **Config field.** Add it (with a default) to `VocalisConfig` or one of
   its sections in `config.py`; pydantic validates it on load.
2. **Read it live.** Add a small accessor to `LiveSettings`
   (`settings.py`), like `voice()` or `notifications_enabled()`. In the
   code that uses the value, take a `Callable` (for example
   `Notifier(enabled=settings.notifications_enabled)`) and call it at the
   point of use, never at construction. If the value can only be applied at
   startup (like shortcuts), say so and extend `shortcuts_need_restart()`
   style logic instead.
3. **Control in the window.** Add a widget in `ui/settings_window.py`, fill
   it in `load_from_settings()`, and put its value into the new config in
   `_config_from_widgets()`. Validation belongs in `LiveSettings.update()`
   (raise `ShortcutError`/`ConfigError`; the window shows "Not saved: ...").
4. **Tests.** `tests/test_settings.py` (accessor, update, failed save keeps
   the old value), `tests/test_settings_window.py` (load, save, error
   message) and one test where the feature picks up a changed value without
   being rebuilt. Use `tmp_path` for the config file and the injectable
   `saver`; never touch the real `~/.config/vocalis/config.toml`.
5. **Docs.** Update `docs/user-docs.md` and the change log.

Saved config files are private to the user (mode 0600) because
`save_config` uses `tempfile.mkstemp`; keep that if you change saving.

## Reporting into the status hub

A feature that reads aloud or listens reports to `StatusHub`
(`ui/status_hub.py`) through plain optional hooks, so the feature itself
never imports Qt:

- Call `hub.begin_reading(source, ReadingControls(stop=..., pause=...,
  resume=...))` when a reading really starts and `hub.end_reading()` when it
  ends for any reason. Do both for your own readings from one thread, or
  otherwise in order: the hub emits its signals after releasing its lock,
  so a begin and an end from different threads at the same instant could be
  delivered the wrong way round. A second begin while one is active is
  ignored.
- Prefer **reporting from state** (as `QuickReaderFeature._report_reading_state`
  does with `_report_lock` and `_last_reported`): compute "is a reading
  active?" and report only when it changed, so racing start/stop paths can
  never produce a double or stray "finished". The `_report_lock` is not
  re-entrant: a hook must not call back into the same feature.
- Call `begin_reading` only after the reading has really been handed over
  (Document Reader does it after `ReadingCoordinator.start_reading`, which
  may end another reader's hub reading first).
- Send `report_status`, `report_progress` and `set_paused` from any thread;
  `set_paused` is the reader's *confirmation*, not a request.
- `hub.set_next_source("Screen text")` renames only the next reading (OCR
  uses it because it reads through the Document Reader).

## Qt lifetime rules learned in Phase 6

- Create the `QApplication` before any Qt object, and the `StatusHub`
  right after it (before the builders that use it).
- Tear Qt objects down explicitly **while the `QApplication` is alive**.
  Objects whose destruction is left to Python's garbage collection can be
  destroyed at a random moment, even after the application is gone.
- The keepalive `QTimer` in `main()` once caused a crash in the full test
  suite (`QTimerInfoList::activateTimers` on a deleted object): it must be
  stopped and `deleteLater()`'d in `main()`'s `finally`.
- The tray icon must not outlive the application: call `TrayIcon.close()`
  (hides it, removes and schedules deletion of its menu, disconnects the hub
  signals; safe to call twice). `main()` calls it first on shutdown, and the
  Quit entry calls it before `app.quit()`.
- A `QMenu` cannot be parented to the `TrayIcon` (a `QObject`, not a
  widget); it is kept as an attribute and deleted in `close()`.
- `app.setQuitOnLastWindowClosed(False)`: closing the Settings or Document
  Reader window must not quit Vocalis.
- The PySide6 type stubs have two gaps that need a `cast(QApplication, ...)`
  in `main()` and one `# type: ignore[arg-type]` for
  `setContextMenu(None)`; do not remove them without re-running `mypy`.

## Sound output is now chosen automatically (Stage 1)

The ALSA `default` output device being silent (stream opens, callback
never fires) on this machine is no longer a manual workaround: `vocalis.
audio.output_device.OutputDeviceChooser` probes candidates (open a
stream, wait for the callback) and picks the first one that really
plays, caching it for the session and re-probing after a short TTL if
none worked. See `docs/architecture.md`'s "Stage 1 addition" section for
the full design, and "Adding a new 'live, probed' setting" below for how
to extend this pattern. Automated tests are unaffected either way (they
use fakes); this note only concerns hands-on runs.

## Adding a new "live, probed" setting

This generalizes "Adding a new setting" above for a setting whose value
needs to be *verified to actually work* before Vocalis commits to it
(the sound output is the first example; a similar pattern would fit any
future setting backed by hardware or an external service that can be
silently broken):

1. **A chooser/prober, not just a config field.** Keep the "does this
   value actually work" logic in its own small class (see
   `OutputDeviceChooser`) with an injectable clock and injectable
   collaborators (the real device/stream factory by default) — this is
   what makes "nothing works yet, retry later" and "a working choice
   degraded mid-session" testable without real hardware.
2. **Cache success, time-box failure.** Remember a verified-working
   answer for the rest of the session; remember a "nothing worked"
   answer for only a short TTL, since the underlying service (the sound
   server, in this case) may simply not be ready yet.
3. **Report a failure once it is detected, not before.** A consumer
   (like `AudioPlayer`) should tell the chooser about a failure via a
   narrow hook (`mark_failed`/`on_device_failed`), not poll it
   continuously; the chooser is the only thing that decides what to try
   next.
4. **A "Test" button that tests exactly what the user asked, not a
   fallback.** `_make_on_test_sound()`'s named-device branch calls
   `.probe()` directly on the one requested device — never `.current()`,
   which would silently substitute a different working device and mask
   the very failure the user wanted to see.
5. **Follow the rest of "Adding a new setting"** for the config field,
   the `LiveSettings` accessor, the Settings window widget, and the docs.

## Voice style: measured, not offered (Stage 1, Task 6 spike)

Qwen3-TTS CustomVoice (the 0.6B model Vocalis uses) accepts an optional
`instruct` style hint (e.g. "Speak quickly and with energy."). A spike
measured whether it has any reliable effect before building a "Speaking
style" setting around it: the same sentence was generated 3x
deterministically (`do_sample=False`) for each of four conditions (none,
natural, slow, fast) and compared by duration and median pitch.

**Verdict: NOT EFFECTIVE.** Every instruct condition's mean delta vs. the
no-instruct baseline (duration −3.8% to −0.42%/pitch +2.70% to +7.12%)
was smaller than, or comparable to, the spread the *same* no-instruct
call produced across its own three "identical" repeats (duration 6.3%,
pitch 16.3% spread) — i.e. the "effect" of the instruct text could not be
told apart from the model's own run-to-run noise. (Temperature *does*
control variance as expected — 0.9 gives ~3x the duration spread of
0.6 — confirming the measurement approach itself was sound.) Full numbers
and reasoning: `.superpowers/sdd/2026-10-06-vocalis-stage1-sound-and-voice/reports/task-6-report.md`.

**Do not re-add a `voice_style` setting, STYLES dict, or style control in
Settings without re-running a similar measurement first** — this result
is specific to the 0.6B CustomVoice model and this speaker/sentence, so a
different model or a future model upgrade could behave differently, but
the current one does not.

## Known open item: a handful of main() tests still touch the real device list

About 8 older tests outside the `_main_patches` fixture in
`tests/test_cli.py` still call the real `sounddevice.query_devices()`
(reading the OS's device metadata only — never opening a stream, never
playing sound). This predates Stage 1 and is left as a cleanup item
rather than a correctness problem: it means those specific tests are not
fully hardware-isolated, but they cannot make any sound or touch any
real audio stream.

## Adding a new startup check (Stage 2)

Each check in `src/vocalis/startup_checks.py` follows one contract:
**never raise for a normal "this thing is missing" case** — return a
human-readable message (naming what's missing and which feature it
affects) if something looks wrong, or `None` if it's fine. `run_startup_
checks()` already wraps every check call in its own `try`/`except
Exception` and logs-and-skips a check that raises anyway, but that safety
net exists for a genuinely unexpected failure (e.g. a permissions error
reading a path) — it is not a substitute for returning `None` on the
ordinary "not installed yet" case.

To add one:

1. Write `check_<thing>(...) -> str | None` with injectable dependencies
   (a `Path`, or a `Callable` like `check_capture_tools`'s `which`
   parameter) so tests never touch the real filesystem or `PATH`.
2. Add a `lambda: check_<thing>(...)` to the `checks` list in
   `run_startup_checks()`, in the position where its message should
   appear (messages are collected in a fixed order — Parakeet model
   first, then Qwen3-TTS model, then the Kokoro voice, then `ydotoold`,
   then the capture tools, then the onnxruntime check, then a stale
   `~/.asoundrc` last).
3. If the check depends on something that can be monkeypatched at module
   level (like `shutil.which`), pass it explicitly at the call site
   inside `run_startup_checks()` rather than relying on the check
   function's own default parameter — a default binds once at
   import time and silently stops observing a test's
   `monkeypatch.setattr("...shutil.which", ...)`. This exact bug was
   found and fixed during this stage's build (see the spec amendment's
   "Implementation deviations and findings").
4. Add tests to `tests/test_startup_checks.py`: the "present" (silent)
   case and the "missing" (message) case for the new check; the raise
   path is already covered generically by `run_startup_checks()`'s own
   "survives a check that raises" test.
5. Document the new message in `docs/user-docs.md`'s startup-notification
   list, in plain language, and note which one feature it affects.

## Test isolation: the `_no_real_startup_checks` autouse fixture (`tests/test_cli.py`)

When `run_startup_checks(config)` was wired into `main()`, code review
found a real, reproducible bug: roughly 15 pre-existing tests in
`tests/test_cli.py` call `main()` for unrelated reasons (listener wiring,
shutdown order, and so on), and none of them patched `run_startup_checks`.
Since that function reads real machine state (the Parakeet/Qwen3-TTS
model folders, `ydotoold`'s socket, whether `tesseract`/`xclip` are on
`PATH`), any one of those tests could fire a genuine `notify-send` on a
developer's machine depending on what happens to be installed there —
tests silently depending on live environment state, exactly the kind of
flake this project's rules exist to prevent.

The fix is the autouse fixture near the top of `tests/test_cli.py`:

```python
@pytest.fixture(autouse=True)
def _no_real_startup_checks():
    with patch("vocalis.cli.run_startup_checks", return_value=[]):
        yield
```

It patches `run_startup_checks` to return `[]` for every test in the
file, by default, so no test can accidentally send a real notification
just by calling `main()`. A test that specifically wants to exercise real
startup-check messages (like `test_main_notifies_about_every_real_
startup_problem`) opens its own `with patch(...)` block, which overrides
the autouse fixture for the duration of that block. A dedicated
regression test (`test_main_never_attempts_a_real_startup_notification_
without_patching_it`) guards the fixture itself: it patches `Notifier`'s
underlying command runner instead of `run_startup_checks`, and asserts
that runner is never called — reproducing the original bug if the
autouse fixture is ever removed.

**If you add a new test that calls `main()`,** you get this protection
automatically. If your test needs to assert something about a *specific*
startup message, patch `run_startup_checks` yourself inside that test's
own `with` block rather than removing or weakening the autouse fixture.

## Single-instance guard: how `_is_another_instance_alive()` decides (Stage 2)

See `docs/architecture.md`'s "Stage 2 addition" section for the full
three-outcome explanation. In short: it tries to `connect()` a probe
socket (built by the constructor's injectable `socket_factory` argument)
to the existing socket file, and treats a successful connect as "alive"
(raise `DaemonAlreadyRunningError`, leave the file alone), `Connection
RefusedError`/`FileNotFoundError` as "stale" (log a warning, delete the
file), and anything else as "ambiguous — treat as stale" (log a warning;
an inconclusive probe must never block startup). Tests inject fake socket
classes (`_RefusingSocket`, `_AcceptingSocket`, `_OddlyFailingSocket` in
`tests/test_daemon.py`) rather than opening real sockets.

## The installer's test suite (Stage 3)

`packaging/installer/` has its own test tooling, separate from `pytest`:

```bash
shellcheck packaging/installer/*.sh packaging/installer/lib/*.sh   # zero warnings
bats tests/installer/                                              # every .bats file
```

(`shellcheck` and `bats` are dev-only tools installed once with
`sudo apt-get install -y shellcheck bats makeself`; they are never
required to run the regular Python suite.) The one Python helper,
`packaging/installer/lib/fetch_models.py`, is covered by the ordinary
`pytest`/`mypy` commands above — `tests/test_installer_fetch_models.py`
loads it by file path (via `importlib.util`) since it lives outside the
`vocalis` package, and `pyproject.toml`'s `[tool.mypy] files` includes it.

### The `*_run_*` override-after-source testing idiom

Every `packaging/installer/lib/*.sh` function that would otherwise call a
real external command (`dpkg`, `apt-get`, `curl`, `uv`, `systemctl`,
`nvidia-smi`, `tar`) calls a small wrapper named `<module>_run_<thing>`
instead — for example `apt_run_dpkg_query`, `pyenv_run_uv`,
`service_run_systemctl_enable_now`. A `bats` test `source`s the library
file, then simply redefines that wrapper function as a bash function of
its own before calling the real function under test:

```bash
setup() {
  source "${BATS_TEST_DIRNAME}/../../packaging/installer/lib/apt.sh"
}

@test "apt_install_missing calls apt_run_install exactly once with every missing package" {
  apt_run_dpkg_query() { [[ "$1" == "xclip" ]]; }
  apt_run_install() { echo "CALLED: $*" >> "$CALL_LOG"; }
  apt_install_missing "tesseract-ocr" "xclip" "ydotool"
  grep -q "tesseract-ocr" "$CALL_LOG"
}
```

No `bats` test ever invokes a real `sudo`, `apt-get`, `curl`, `uv`,
`systemctl` or `nvidia-smi` — the same philosophy as the Python side of
this project faking `sounddevice`/`socket`/`shutil.which` rather than
touching real hardware or the network. `install.sh` and `uninstall.sh`
apply the same idea one level up: each is written as a sourceable
`main()` function plus a run-guard
(`if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then main "$@"; fi`), so a test
can `source` the whole script, override every `lib/*.sh` function it
calls, invoke `main` directly, and assert on the order calls happened in
— exactly how `vocalis.cli.main()` is tested from Python. A `bats` test
that wants to override where a script looks for itself (for `install.sh`
and `build-run.sh`, which otherwise compute `SCRIPT_DIR`/`REPO_ROOT` from
their own real on-disk location) sets the `SCRIPT_DIR`/`REPO_ROOT`
environment variables before calling `main` — the scripts read an
overridable copy of each inside `main()`, computed fresh on every call,
rather than only the non-overridable one computed once at `source` time.

**A test-design pitfall caught twice during this stage's build:** a test
that only checks `$status` or a path substring in `$output` can stay
green even when the behavior it claims to prove is actually broken (the
assertion doesn't distinguish the real code path from a coincidentally
similar-looking failure path). This was caught and fixed once in
`tests/installer/test_pyenv.bats` (the "uv already on PATH" test did not
override `HOME`, so it passed against the real system `uv` rather than
proving the `command -v` shadow actually worked) and once more in
`tests/installer/test_install.bats`/`test_build_run.bats` (the
`SCRIPT_DIR`/`REPO_ROOT` override tests were strengthened after mutation
testing showed they stayed green even with the override broken). When
writing a `bats` test for an override or a conditional path, make sure
the assertion would actually fail if that one piece of behavior were
deleted — not just that the happy path's final exit code matches.

### Rebuilding the `.run` files

```bash
cd packaging/installer
bash build-run.sh
```

This runs `uv build` against the repo, copies the resulting
`vocalis-*.whl` into a staging directory alongside `install.sh`/
`uninstall.sh`, `lib/`, and `packaging/systemd/vocalis.service`, then
calls `makeself` twice to produce `install.run` and `uninstall.run` next
to `install.sh`. The `.run` files are generated artifacts: never edit or
commit them, just re-run `build-run.sh` whenever the installer or the
package changes. Needs `uv` and `makeself` installed (the same one-time
dev-tooling install as `shellcheck`/`bats` above).

### The `PARAKEET_MODEL_FILES`/`QWEN3_TTS_MODEL_FILES` rename

`src/vocalis/startup_checks.py`'s two private file-list constants
(`_PARAKEET_FILES`, `_QWEN3_TTS_FILES`) were promoted to public names
(`PARAKEET_MODEL_FILES`, `QWEN3_TTS_MODEL_FILES`) during this stage, with
no change in value. They are now imported directly by
`packaging/installer/lib/fetch_models.py`, so the installer's idea of
"which files make a model complete" can never silently drift out of sync
with the runtime startup checks' idea of the same thing — there is
exactly one place that list is written down. If you ever need to change
which files count as "the model is present" for either Parakeet or
Qwen3-TTS, change it here; both the installer and `run_startup_checks()`
pick up the change automatically.

## `vocalis --settings` and the console-script entrypoint (Stage 5)

The `vocalis` console script (`[project.scripts]` in `pyproject.toml`)
points at `vocalis.entry:entrypoint`, a small, deliberately separate
module that imports nothing from `vocalis.cli` (and therefore nothing it
pulls in — `torch`, `transformers`, PySide6, ...) at module load time.
`entrypoint()` parses `sys.argv` for a `--settings` flag: with it, it
calls `vocalis.remote.open_settings()` and exits with that function's
return code *without* ever importing `vocalis.cli`; without it, it
imports `vocalis.cli.main` lazily (right there, inside the `else`
branch) and calls it exactly as before. This split exists so a quick
`vocalis --settings` from a terminal (or the application-menu launcher —
see `packaging/installer/README.md`) is cheap and talks to whichever
Vocalis is already running, instead of starting a second full instance.
If you ever add a new import to the top of `vocalis/entry.py`, make sure
it is not one that drags in `vocalis.cli` or any of its heavy
dependencies — that would silently defeat the whole point of this split.

`open_settings()` (`remote.py`) tries `send_command(socket_path,
SHOW_SETTINGS)` first; everything else (starting the service, the 20
second poll loop) only runs if that first attempt gets no answer. Keep
this order if you ever touch it — `start_user_service()` calls `systemctl
--user start vocalis`, which is a no-op but not free if Vocalis is
already running, and the whole point of trying `send_command()` first is
to skip it in the common case.

On the receiving end, `VocalisDaemon`'s new `command_handler` argument
(see `daemon.py`) is wired in `main()` to `_SettingsBridge.handle_command`,
a one-method `QObject` whose only job is `emit()`-ing a Qt signal
connected to `settings_window.present()`. This indirection is required,
not optional: `command_handler` runs on the daemon's own socket-accept
thread, and Qt widgets may only be touched from the thread that owns the
`QApplication`. A Qt signal's queued connection is the safe way across
that boundary — calling `settings_window.present()` straight from the
socket thread would be a race at best and a crash at worst. If you ever
add a second remote command, route it through the same bridge pattern
(a dedicated signal per command, or one signal carrying the command as
its argument) rather than calling into any Qt object directly from
`handle_command()`.

## Running the UI without a screen

`QT_QPA_PLATFORM=offscreen` creates every widget without a display; use it
for ad-hoc scripts (with `PYTHONPATH=$PWD/src`) and keep any such script and
its temporary config outside the repository.

## Adding a voice engine (Stage 6a)

1. Implement `TTSEngine` (`engines/base.py`): the `ModelEngine` load/unload
   methods plus `synthesize(text) -> (float32 audio, sample_rate)`. Import
   heavy libraries lazily and take builders as parameters so tests can
   pass fakes (see `engines/kokoro_tts.py`).
2. Add its id constant to `engines/auto_select.py` and decide where it sits
   in `decide()`'s chains; add a forced chain to `FORCED_CHAINS` and a label
   to `_LABELS` in `engines/registry.py`.
3. Add the id to `EngineChoice` in `config.py`, plus any paths/voice
   fields on `EnginesConfig`, and keep `config/default_config.toml` in step.
4. Write a factory in `cli.py` (see `_make_kokoro_factory`) and add it to
   the `engine_factories` dict in both reader builders.
5. Add the choice to `_ENGINE_CHOICES` in `ui/settings_window.py`.
6. Installer: download the files in `packaging/installer/lib/fetch_models.py`
   (use the shared timeout helper); if it needs a Python package, check
   whether it pulls in a conflicting dependency before deciding how
   `pyenv.sh` installs it (Kokoro uses `--no-deps`, and `pyenv_prefer_gpu_onnxruntime` re-installs the GPU onnxruntime last because qwen-tts and kokoro-onnx both pull plain onnxruntime); make `uninstall.sh
   --purge` remove the files.
7. Add a startup check (see "Adding a new startup check") so a missing
   model produces a clear notification.
8. Tests with fakes only: policy rules (`tests/test_auto_select.py`),
   selector (`tests/test_engine_registry.py`), chain
   (`tests/test_engine_chain.py`), and the engine itself
   (`tests/test_kokoro_tts.py` is the model to follow).

The old `quick_tts` / `doc_tts` config keys were never read; they were
replaced by `quick_reader_engine` / `document_reader_engine`. Old files
still load because unknown keys are ignored.

## Testing code that feeds a persistent stream (Stage 6b)

`SpeechStream` keeps one sound stream open and the sound card *pulls*
audio through a callback. To test that without hardware, `tests/fake_output.py`
provides fake streams that call the callback exactly like a real card:

- **`ManualStream`** — the test drives the audio clock by calling
  `pull(frames)`. Fully deterministic: use it for ordering, fades, pause,
  back-pressure and stall tests (pair it with a fake clock).
- **`PumpStream`** — a background thread pulls continuously, `speedup`
  times faster than real time (default 50). Use it where a whole reading
  must simply play out (feature-level tests).
- **`OutputRecorder`** collects what was opened, pulled and closed
  (`audio()`, `played()`, `frames_pulled`); **`FakeAudioPlayer`** is an
  `AudioPlayer` wired to a `PumpStream`; **`wait_until(predicate)`**
  replaces fixed sleeps.

Rules that came out of this stage: the audio callback must never block or
raise; no lock is held while calling slow or user code; engine factories used
with `EngineChain(hold_engines=True)` must build a NEW engine object per
call; because of the 8 ms fades, assert lengths / maximum / non-silent
counts, not exact samples.

**Process the user asked for:** write ALL the code (implementation and
tests) for a task first, run the tests ONCE at the end, then fix and commit —
do not run tests after every edit. Never run pytest in a worktree while
another agent may also be testing there (it causes false "hangs").
