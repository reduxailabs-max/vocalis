# Vocalis Stage 8 — Installer Wizard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the terminal installer into a short, interactive setup wizard (PySide6, red-and-black theme) that works on Debian/Ubuntu-family Linux with or without an NVIDIA card, tests the microphone, then starts Vocalis and opens its Settings window.

**Architecture:** `install.sh` stays the single installer and is split into three phases (`prepare`, `models`, `finish`). With no flags it runs `prepare` in the terminal, then launches a new Python wizard (`vocalis-setup`, from the freshly installed venv) which runs the `models` phase, a microphone test, and the `finish` phase by calling `install.sh --phase …` and reading its `STEP i/N …` lines. The hardware decision (GPU or not, fetch Qwen3-TTS or not) lives in ONE Python module, `vocalis.installer.plan`, which both the shell script and the wizard use.

**Tech Stack:** bash + bats (`tests/installer/*.bats`), Python 3.13, PySide6, pytest + pytest-qt, mypy strict.

**Spec:** `docs/superpowers/specs/2026-10-29-vocalis-stage8-installer-wizard-design.md` (the "Settled after the builder check" section is binding).

## Global Constraints

- Linux only; Debian/Ubuntu family, 64-bit (`x86_64`). Non-Debian or non-64-bit still stops with a plain message.
- No NVIDIA driver must NOT stop the install (it becomes a note).
- Qwen3-TTS is fetched only if GPU total memory ≥ `AutoPolicy.qwen3_min_gpu_total_mb` (4096 MB, `src/vocalis/engines/auto_select.py`). Parakeet, Piper (voice + binary) and Kokoro are always fetched. Never hard-code 4096 anywhere else.
- Existing terminal install (`bash install.sh`, `install.run`, no display) keeps working and stays safe to re-run (repair path). `uninstall.sh` is untouched.
- The wizard asks the person nothing about voices/speed/shortcuts; that is Settings, opened at the end.
- The one admin-password prompt stays the existing `sudo apt-get` in `lib/apt.sh`, which runs in the terminal during the prepare phase; no new password handling, no `pkexec` (spec call 4 said `pkexec`, but the prompt happens in the terminal, where `sudo` is the natural fit — ruled simpler).
- The installed app makes no network calls; only installer code downloads.
- Project rules: tests for all new logic, total coverage ≥ 85% (`pyproject.toml`), `mypy src` strict clean, one-line docstring on every new function, docs in `docs/` and `packaging/installer/README.md` updated, changelog `.claude/<YYYY-MM-DD_HHMM>_CHANGELOG.md` in plain English, conventional commits, never `git add .` (an untracked `.venv` symlink may exist in a worktree). Commit messages end with the line `Claude-Session: https://claude.ai/code/session_0116ii4z7tzRCRhVAxecpkYC`.
- Implementers write all code and tests of their task first, then run the tests ONCE at the end; do not run pytest/mypy in a worktree while another agent may be testing there.
- Bats tests are run with `bats tests/installer` from the repo root.

## Review Focus

1. `nvidia-smi` present but exits non-zero, prints garbage, is missing, or hangs → treated as "no GPU" (Task 3).
2. Wizard window closed or install fails mid-way → the running `install.sh` child is stopped, the person is told to run the installer again, the log path is shown, nothing is rolled back (Task 7).
3. No display (SSH, TTY) → `install.sh` with no flags falls back to doing all phases in the terminal, never tries to open a window (Task 4).
4. No microphone / silent microphone → a plain warning, and Finish still works (Tasks 6, 7).
5. Re-running on an already-installed machine, or on a PC that skips Qwen3 while a Qwen3 folder already exists → nothing is deleted or re-downloaded; garbled or non-UTF-8 lines from the installer never crash the wizard (Tasks 4, 5).

## File Structure

- Modify `packaging/installer/lib/preflight.sh` — GPU check becomes a note.
- Modify `packaging/installer/lib/fetch_models.py` — `--skip-qwen3`.
- Create `src/vocalis/installer/__init__.py`, `plan.py`, `progress.py`, `runner.py`, `mic_test.py`, `wizard.py`, `app.py`.
- Modify `packaging/installer/lib/pyenv.sh` — CPU onnxruntime helper.
- Modify `packaging/installer/install.sh` — phases, plan use, wizard launch.
- Modify `pyproject.toml` — script `vocalis-setup`.
- Tests: `tests/installer/test_preflight.bats`, `tests/installer/test_pyenv.bats`, `tests/installer/test_install.bats`, `tests/test_installer_fetch_models.py`, `tests/test_installer_plan.py`, `tests/test_installer_progress.py`, `tests/test_installer_runner.py`, `tests/test_installer_mic_test.py`, `tests/test_installer_wizard.py`.
- Docs: `packaging/installer/README.md`, `docs/user-docs.md`, `docs/architecture.md`, `docs/file-purpose.md`, `docs/folder-structure.md`, `docs/workflow.md`, changelog.

**Shared protocol (all tasks):** the installer prints structured lines, one per line, always (terminal included): `STEP <i>/<n> <plain-English text>` at the start of each step, and `NOTE <plain-English text>` for something worth showing. Everything else is free-form log text.

---

### Task 1: Missing NVIDIA driver becomes a note

**Files:**
- Modify: `packaging/installer/lib/preflight.sh` (`preflight_run`, header comment)
- Test: `tests/installer/test_preflight.bats`

**Interfaces:**
- Produces: `preflight_run` returns 0 on a Debian-family x86_64 machine even when `preflight_check_gpu` fails, and prints `NOTE No NVIDIA graphics card driver was found. Vocalis will use its lighter voices on your processor.` to stdout in that case. `preflight_check_gpu` itself is unchanged.

- [x] **Step 1: Tests.** In `test_preflight.bats` replace the "fails when no GPU driver" test with `preflight_run succeeds without an NVIDIA driver and prints the NOTE line` (stub `preflight_run_nvidia_smi` to return 1; assert status 0 and output contains `NOTE No NVIDIA graphics card driver was found`). Keep/adjust: wrong distro → status 1; wrong arch → status 1; GPU present → status 0 and no `NOTE`.
- [x] **Step 2: Implement** the change in `preflight_run`; update the file's header comment so it no longer says the GPU is a hard requirement.
- [x] **Step 3: Run** `bats tests/installer/test_preflight.bats` → all pass. Commit `feat: let the installer continue without an NVIDIA driver`.

---

### Task 2: `fetch_models.py --skip-qwen3`

**Files:**
- Modify: `packaging/installer/lib/fetch_models.py` (`main`)
- Test: `tests/test_installer_fetch_models.py`

**Interfaces:**
- Produces: `main(argv: list[str] | None = None) -> int`; `--skip-qwen3` skips `ensure_qwen3_tts` and prints `NOTE Skipping the Qwen3-TTS voice (needs a graphics card with 4 GB or more).` Without the flag behaviour is unchanged. `if __name__ == "__main__": sys.exit(main())` stays (argv from `sys.argv[1:]`).

- [x] **Step 1: Tests** (patch the four `ensure_*` functions, as the file's existing tests do): `test_main_fetches_all_four_by_default`, `test_main_skip_qwen3_does_not_call_ensure_qwen3_tts_but_fetches_the_rest`, `test_main_skip_qwen3_prints_the_note` (capsys contains `NOTE Skipping the Qwen3-TTS voice`).
- [x] **Step 2: Implement** with `argparse`; unknown flags exit non-zero (argparse default).
- [x] **Step 3: Run** `pytest tests/test_installer_fetch_models.py` → pass. Commit `feat: let the model downloader skip Qwen3-TTS`.

---

### Task 3: Hardware plan module (`vocalis.installer.plan`)

**Files:**
- Create: `src/vocalis/installer/__init__.py` (docstring only), `src/vocalis/installer/plan.py`
- Test: `tests/test_installer_plan.py`

**Interfaces:**
- Consumes: `vocalis.engines.auto_select.AutoPolicy` (field `qwen3_min_gpu_total_mb`).
- Produces (all in `plan.py`):
  - `@dataclass(frozen=True) class InstalledGpu: name: str; total_mb: int`
  - `parse_nvidia_smi(output: str) -> InstalledGpu | None` — input is the stdout of `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits`; first non-empty line `"<name>, <digits>"`; anything else → `None`.
  - `read_gpu(run: Callable[[list[str]], str] = _run_nvidia_smi) -> InstalledGpu | None` — returns `None` on `FileNotFoundError`, non-zero exit, `subprocess.TimeoutExpired` (10 s timeout) or unparsable output. `_run_nvidia_smi` runs the exact command above and raises on non-zero exit.
  - `@dataclass(frozen=True) class InstallPlan: gpu: InstalledGpu | None; gpu_runtime: bool; fetch_qwen3: bool` with `summary_lines(self) -> list[str]` (plain English: GPU name and memory or "No NVIDIA graphics card found – using your processor"; which voices will be installed; one line per voice group).
  - `make_plan(gpu: InstalledGpu | None, policy: AutoPolicy = AutoPolicy()) -> InstallPlan` — `gpu_runtime = gpu is not None`; `fetch_qwen3 = gpu is not None and gpu.total_mb >= policy.qwen3_min_gpu_total_mb`.
  - `main(argv: list[str] | None = None) -> int` — one positional arg: `gpu-runtime` prints `yes`/`no`; `fetch-flags` prints `--skip-qwen3` or an empty line. Callable as `python -m vocalis.installer.plan <what>` (add `if __name__ == "__main__"`).

- [x] **Step 1: Tests:** `test_parse_nvidia_smi_reads_name_and_memory` ("NVIDIA GeForce RTX 3060, 12288" → name/12288); `test_parse_nvidia_smi_garbage_is_none` (parametrize "", "oops", "a, b", "x, -5"); `test_read_gpu_missing_binary_is_none`, `test_read_gpu_nonzero_exit_is_none`, `test_read_gpu_timeout_is_none` (inject `run` raising each); `test_plan_4096mb_fetches_qwen3`, `test_plan_4095mb_skips_qwen3`, `test_plan_no_gpu_uses_cpu_runtime_and_skips_qwen3`; `test_summary_mentions_gpu_name_or_processor`; `test_main_gpu_runtime_prints_yes_or_no` and `test_main_fetch_flags_prints_skip_flag_only_when_needed` (monkeypatch `read_gpu`).
- [x] **Step 2: Implement.**
- [x] **Step 3: Run** `pytest tests/test_installer_plan.py` + `mypy src`. Commit `feat: add the installer hardware plan`.

---

### Task 4: `install.sh` phases, CPU path, wizard launch

**Files:**
- Modify: `packaging/installer/install.sh`, `packaging/installer/lib/pyenv.sh`
- Test: `tests/installer/test_install.bats`, `tests/installer/test_pyenv.bats`

**Interfaces:**
- Consumes: Task 1 `preflight_run`; Task 2 `--skip-qwen3`; Task 3 `python -m vocalis.installer.plan gpu-runtime|fetch-flags`.
- Produces in `pyenv.sh`: `pyenv_use_cpu_onnxruntime <venv_dir>` — if `onnxruntime-gpu` is installed, read its version, `uv pip uninstall onnxruntime-gpu`, then `uv pip install --reinstall --no-deps onnxruntime==<same version>`; no-op when `onnxruntime-gpu` is not installed (mirror `pyenv_prefer_gpu_onnxruntime`).
- Produces in `install.sh`: functions `install_phase_prepare <script_dir> <repo_root>`, `install_phase_models <script_dir>`, `install_phase_finish <script_dir> <repo_root>` and `main "$@"` parsing `--phase prepare|models|finish|all`, `--no-wizard`, `--installer-dir` is NOT a flag (the dir is `$script_dir`).
  - `prepare`: the present preflight → apt → uv → wheel → venv → `pyenv_install_vocalis` → `pyenv_install_kokoro` → then if `python -m vocalis.installer.plan gpu-runtime` prints `yes` run `pyenv_prefer_gpu_onnxruntime`, else `pyenv_use_cpu_onnxruntime` → `~/.local/bin/vocalis` symlink. Prints `STEP 1/5 …` … `STEP 5/5 …`.
  - `models`: present path queries + `fetch_models.py $(plan fetch-flags)` + `models_ensure_piper_binary`. Prints `STEP 1/2 Downloading voices and the dictation model (a few gigabytes – this can take a while)` and `STEP 2/2 Downloading the Piper speech program`.
  - `finish`: present service unit + `service_enable_and_start` + desktop launcher + final "installed and running" text. Prints `STEP 1/2 Starting Vocalis` and `STEP 2/2 Adding Vocalis to your applications menu`.
  - `all`: prepare, models, finish in order (today's behaviour). `--no-wizard` ⇒ `all`.
  - No flags: run `prepare`; then if `DISPLAY` or `WAYLAND_DISPLAY` is non-empty, `exec "${venv_dir}/bin/vocalis-setup" --installer-dir "$script_dir"`; otherwise run `models` then `finish` in the terminal (display-less fallback). If `vocalis-setup` is missing or fails to start, fall back to the terminal path with a one-line `NOTE`.

- [x] **Step 1: Tests (bats).** Update the existing `setup()` stubs so the new calls are stubbed (`pyenv_use_cpu_onnxruntime`, and make the fake venv python answer `yes`/`no` and flags from a per-test variable); keep every existing assertion that still applies and set `DISPLAY=` and `WAYLAND_DISPLAY=` empty in `setup`. New tests: `phase prepare runs only the prepare steps` (call log has preflight…symlink, no `models_ensure_piper_binary`, no `service_enable_and_start`); `phase models runs only the downloads`; `phase finish runs only service and desktop`; `phase all keeps the old order`; `gpu machine prefers the GPU onnxruntime`; `cpu machine switches to the CPU onnxruntime and not the GPU one`; `models phase passes --skip-qwen3 when the plan says so` (fake python prints `--skip-qwen3`, assert the fetch call received it) and `does not delete or touch an existing Qwen3 folder` (create the folder, run, assert it still exists); `no flags and no display runs prepare, models and finish in the terminal`; `no flags with a display launches vocalis-setup with --installer-dir` (stub the launcher via a function `install_launch_wizard`, assert args); `missing vocalis-setup falls back to the terminal path`; `every phase prints STEP i/n lines` (grep `^STEP [0-9]+/[0-9]+ `). In `test_pyenv.bats`: `pyenv_use_cpu_onnxruntime swaps the GPU build for the CPU build at the same version` and `…does nothing when there is no GPU build`.
- [x] **Step 2: Implement** `pyenv_use_cpu_onnxruntime`, restructure `install.sh` (keep `BASH_SOURCE` guard and the overridable `SCRIPT_DIR`/`REPO_ROOT` mechanism; add small function `install_launch_wizard <venv_dir> <script_dir>` that does the `exec`, so tests can stub it).
- [x] **Step 3: Run** `bats tests/installer` → all pass (including `test_build_run.bats`; check `build-run.sh` still stages `lib/` and `install.sh`, and fix only if the test shows otherwise). Commit `feat: split the installer into phases and add a CPU-only path`.

---

### Task 5: Reading the installer's progress lines

**Files:**
- Create: `src/vocalis/installer/progress.py`, `src/vocalis/installer/runner.py`
- Test: `tests/test_installer_progress.py`, `tests/test_installer_runner.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class ProgressEvent: kind: Literal["step", "note"]; index: int; total: int; text: str` (`index`/`total` are 0 for notes).
  - `parse_line(line: str) -> ProgressEvent | None` — `STEP 3/5 Some text` → step(3,5,"Some text"); `NOTE text` → note; anything else, `STEP` with a bad format, `i > n`, `i < 1`, or empty text → `None`.
  - `class PhaseRunner` — `__init__(self, installer_dir: Path, phase: str, on_line: Callable[[str], None], popen: Callable[..., Any] = subprocess.Popen)`; `run(self) -> int` runs `["bash", str(installer_dir / "install.sh"), "--phase", phase]` with `stdout=PIPE, stderr=STDOUT, text=True, errors="replace"`, calls `on_line` for each stripped non-empty line, returns the exit code (non-zero if it could not start: `OSError` → call `on_line` with a plain message and return 127); `terminate(self) -> None` terminates the child if running (safe to call twice or before `run`).

- [x] **Step 1: Tests:** parse: valid step, valid note, `test_parse_line_ignores_malformed_lines` (parametrize "STEP x/y t", "STEP 0/3 t", "STEP 4/3 t", "STEP 1/3", "hello", ""), `test_parse_line_keeps_text_with_colons`. Runner with a fake `popen` object (stdout iterable, `wait()`, `poll()`, `terminate()`): `test_run_forwards_each_line_and_returns_exit_code`, `test_run_survives_undecodable_bytes` (feed a str containing "�"), `test_run_reports_a_missing_bash_as_127`, `test_terminate_stops_a_running_child_and_is_idempotent`, `test_terminate_before_run_is_harmless`.
- [x] **Step 2: Implement.** [ ] **Step 3: Run** both test files + `mypy src`. Commit `feat: read the installer's progress lines`.

---

### Task 6: Microphone test

**Files:**
- Create: `src/vocalis/installer/mic_test.py`
- Test: `tests/test_installer_mic_test.py`

**Interfaces:**
- Consumes: `vocalis.audio.capture.MicRecorder` and `pick_input_device`; the existing "no sound / muted" wording constant `NO_AUDIO_MESSAGE` (find it with `grep -rn NO_AUDIO_MESSAGE src/vocalis` and import it; do not duplicate the text).
- Produces: `@dataclass(frozen=True) class MicResult: verdict: Literal["heard", "quiet", "no_device"]; peak: float; message: str` and `run_mic_test(seconds: float = 2.0, recorder_factory: Callable[[int | None], MicRecorder] = ..., device_picker: Callable[[], int | None] = pick_input_device, sleep: Callable[[float], None] = time.sleep) -> MicResult`. Peak = max absolute sample; `heard` when peak ≥ `HEARD_PEAK = 0.02`, else `quiet`. `recorder.start()` raising any exception (incl. `sounddevice.PortAudioError`) → `no_device`. The recorder is always stopped if started. Messages: heard → "Great – Vocalis can hear you."; quiet → `NO_AUDIO_MESSAGE`; no_device → "Vocalis could not find a microphone. You can still finish setup and fix this later in your sound settings."

- [x] **Step 1: Tests** with a fake recorder: `test_loud_audio_is_heard`, `test_silence_is_quiet_and_uses_the_existing_message`, `test_empty_capture_is_quiet`, `test_start_failure_is_no_device`, `test_recorder_is_stopped_after_a_successful_start`, `test_uses_the_picked_device` (factory receives the picker's value), `test_waits_for_the_requested_seconds` (fake sleep records 2.0).
- [x] **Step 2: Implement.** [ ] **Step 3: Run** the test file + `mypy src`. Commit `feat: add the installer microphone test`.

---

### Task 7: The setup wizard window and `vocalis-setup`

**Files:**
- Create: `src/vocalis/installer/wizard.py`, `src/vocalis/installer/app.py`
- Modify: `pyproject.toml` (`[project.scripts]` add `vocalis-setup = "vocalis.installer.app:main"`)
- Test: `tests/test_installer_wizard.py`

**Interfaces:**
- Consumes: Task 3 `read_gpu`, `make_plan`, `InstallPlan.summary_lines`; Task 5 `PhaseRunner`, `parse_line`; Task 6 `run_mic_test`, `MicResult`; `vocalis.ui.theme.apply_theme`; `vocalis.remote.open_settings(default_socket_path()) -> int` (0 = opened).
- Produces:
  - `class SetupWizard(QWizard)` — `__init__(self, installer_dir: Path, plan_reader: Callable[[], InstallPlan] = ..., runner_factory: Callable[[Path, str, Callable[[str], None]], PhaseRunner] = ..., mic_tester: Callable[[], MicResult] = run_mic_test, settings_opener: Callable[[], int] = ..., parent: QWidget | None = None)`. Five pages in this order and with these object names: `welcome` (Start), `found` (read-only summary from `plan_reader().summary_lines()`), `installing` (runs phase `models`), `microphone` (button "Test my microphone" → shows `MicResult.message`; never blocks Next), `done` (runs phase `finish`, then shows "Vocalis is running"; the Finish button opens Settings via `settings_opener`, retrying up to 10 times 0.5 s apart until it returns 0, then closes).
  - Installing/Done pages run the runner on a `QThread` worker; STEP events set the label text and the progress bar value `(index - 1) * 100 // total` (and 100 on success); NOTE events and plain lines go into a read-only log pane. Non-zero exit → red message "Something went wrong. Check your internet connection, then press Try again." plus a "Try again" button that re-runs the same phase, and the log file path `~/.local/share/vocalis/install.log` is written/shown (append every line to it; create the folder if needed).
  - Closing the window (or Cancel) while a phase runs calls `runner.terminate()` and shows a plain message box text "Setup was stopped. Run the installer again to finish." (do not use blocking dialogs in tests: inject through a method `_show_stopped_message()` that tests can patch).
  - `app.main(argv: list[str] | None = None) -> int` — parses required `--installer-dir`, creates `QApplication`, calls `apply_theme`, shows the wizard, returns the exit code of `exec()`.

- [x] **Step 1: Tests (pytest-qt, all with injected fakes, no real subprocess):** `test_pages_come_in_the_expected_order`; `test_found_page_shows_the_plan_summary`; `test_installing_page_shows_step_text_and_progress` (fake runner emits `STEP 1/2 …` then `STEP 2/2 …`); `test_installing_success_enables_next`; `test_installing_failure_shows_try_again_and_reruns_the_same_phase`; `test_lines_are_appended_to_the_log_file` (use `tmp_path` via monkeypatched log path); `test_closing_mid_install_terminates_the_runner_and_shows_the_stopped_message`; `test_microphone_warning_does_not_block_next_or_finish` (parametrize `quiet`, `no_device`); `test_done_page_runs_the_finish_phase_after_the_microphone_page`; `test_finish_opens_settings_and_retries_until_it_succeeds` (opener returns 1,1,0 → called 3 times; patch the sleep); `test_finish_gives_up_after_ten_tries_without_crashing`; `test_wizard_uses_the_theme_in_app_main` (smoke: `app.main` with `--installer-dir` and a patched `SetupWizard` shown once, returns 0); `test_app_main_requires_installer_dir`.
- [x] **Step 2: Implement** (keep each file under ~300 lines; pages may live as small private classes in `wizard.py`; the worker thread class holds only the runner and emits `line = Signal(str)` and `finished_with = Signal(int)`).
- [x] **Step 3: Run** `pytest tests/test_installer_wizard.py`, then the full suite once (`pytest`), then `mypy src`; coverage must stay ≥ 85%. Commit `feat: add the installer setup wizard`.

---

### Task 8: Docs and changelog

**Files:**
- Modify: `packaging/installer/README.md`, `docs/user-docs.md` (installing section), `docs/architecture.md` (new "Stage 8: installer wizard" section), `docs/file-purpose.md` (new `vocalis/installer/` rows + changed shell files), `docs/folder-structure.md`, `docs/workflow.md` (install flow, user and installer views)
- Create: `.claude/<YYYY-MM-DD_HHMM>_CHANGELOG.md`

- [x] **Step 1:** Update each doc: the three phases and `--phase`/`--no-wizard`; the STEP/NOTE protocol; CPU-only behaviour (no NVIDIA driver → note, CPU onnxruntime, Qwen3 skipped below 4096 MB); the five wizard pages; where the install log is; what to do after a failed or stopped install (run the installer again); `vocalis-setup --installer-dir DIR` is internal. Mention real-PC checks left for the user (fresh account, no-GPU PC, network unplugged, muted mic) — not on the NTFS dev drive.
- [x] **Step 2:** Changelog in plain English (what changed, why, how to roll back: `git revert` the Stage 8 commits; old terminal install still available via `--no-wizard`).
- [x] **Step 3:** Full `pytest`, `mypy src`, `bats tests/installer`; commit `docs: document Stage 8 installer wizard`.
