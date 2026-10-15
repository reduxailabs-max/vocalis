# Vocalis Stage 3 — Phase 8 Installer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A non-technical person on a Debian/Ubuntu-family `x86_64` Linux machine with a working NVIDIA GPU driver can run one script (`install.sh`, or its double-clickable `install.run` form) and end up with Vocalis fully installed, its models in place, and its background service running and set to start automatically at every login — with a matching `uninstall.sh`/`uninstall.run` to remove it.

**Architecture:** A small library of focused, independently testable bash functions under `packaging/installer/lib/` (`preflight.sh`, `apt.sh`, `pyenv.sh`, `models.sh`, `service.sh`), each following the same testing idiom already used throughout this Python codebase — every call to an external command (`nvidia-smi`, `dpkg`, `apt-get`, `curl`, `uv`, `systemctl`) is wrapped in its own tiny `*_run_*` function so a test can redefine it after `source`-ing the library, with no real subprocess ever invoked during automated tests. `install.sh` and `uninstall.sh` are themselves written as a sourceable `main()` function plus a run-guard, so they can be tested the same way `vocalis.cli.main()` is tested in Python: source the file, override every lib function, call `main`, assert the right calls happened in the right order. One Python helper (`packaging/installer/lib/fetch_models.py`) handles the three Hugging-Face-hosted assets by importing `vocalis.config.VocalisConfig` and `vocalis.startup_checks`'s file-list constants directly from the just-installed package, so the installer can never drift out of sync with the paths/files Vocalis itself expects. `build-run.sh` wraps the finished scripts into self-extracting `.run` files with `makeself`.

**Tech Stack:** bash (targeting bash 5, the Debian/Ubuntu default), `bats` (Bash Automated Testing System) for unit tests, `shellcheck` for static analysis, `uv` for the Python environment, `huggingface_hub` (already a project dependency) for model downloads, `makeself` for the `.run` wrapper, Python 3.13/pytest/mypy for the one `.py` helper.

**Spec:** `docs/superpowers/specs/2026-09-11-vocalis-design.md` — section "Stage 3 amendment — Phase 8, Installer".

## Global Constraints

- Work in a fresh git worktree on branch `worktree-vocalis-stage3-installer`; run Python tests with `export PYTHONPATH=$PWD/src` and `.venv/bin/pytest` / `.venv/bin/mypy` (link `.venv` to the main checkout's `.venv`; never `git add` it).
- **One-time dev-tooling install, ask the user first:** `shellcheck`, `bats`, and `makeself` are not installed on this machine. Before Task 1 begins, ask the user for permission to run `sudo apt-get install -y shellcheck bats makeself` (standard password-prompt rule from the user's global instructions) — this installs **dev/test tooling on the development machine only**; it is not the Vocalis installer and does not touch Vocalis, systemd, or any model/config files.
- Conventional commits, **no Co-Authored-By or attribution lines**, explicit `git add` paths, never touch `old/`.
- Every bash function has a one-line comment above it saying what it does (bash has no docstrings); every Python function/method/class keeps the project's existing docstring convention. `mypy` strict **zero errors** on `src/vocalis` and `packaging/installer/lib/fetch_models.py` (added to `[tool.mypy] files` in Task 4). `shellcheck` **zero warnings** on every file under `packaging/installer/`. Full Python suite **zero warnings**, coverage **>= 85%** (baseline at plan time: 594 tests — confirm the exact current baseline by running the suite once before Task 1). All new `bats` tests pass.
- **Every external command a script might run (`sudo`, `apt-get`, `dpkg`, `curl`, `uv`, `systemctl`, `nvidia-smi`, `tar`) is wrapped in its own `*_run_*` bash function.** No automated test ever invokes a real one of these commands — `bats` tests always redefine the wrapper function after sourcing the library, exactly as the Python tests fake `sounddevice`/`socket`/`shutil.which`.
- **Never actually run the real installer end-to-end, never actually install apt packages for Vocalis's own dependencies, never actually create the real venv at `~/.local/share/vocalis/venv`, never actually enable/start the real `vocalis` service, never actually download the real multi-gigabyte models.** The user runs the genuine, complete `install.sh` themselves, for the first time, only once this stage is finished and merged — exactly as every prior stage has deferred all hands-on verification.
- Record the change log in `.claude/<YYYY-MM-DD_HHMM>_CHANGELOG.md`; keep `docs/` in sync.

## Review Focus

1. Re-running `install.sh` after an earlier **partial** failure (e.g. the apt step succeeded but the Python-environment step failed) must not redo the apt install or re-download an already-fully-downloaded model — idempotency holds across the *whole* flow, not just inside one helper (Task 6).
2. A declined or wrong `sudo` password during the apt step must produce one clear plain-English line and stop the whole install there — never a raw `apt`/`dpkg` stack trace, and never proceed to the (slow, multi-gigabyte) model-download step after a failed package install (Task 2, Task 6 ordering).
3. `uninstall.sh` run on a machine where Vocalis was never installed (no venv, no unit file, no symlink) must exit 0 with a plain "nothing to remove" message, never a crash from a missing file/path (Task 7).
4. An interrupted download must never be mistaken for a complete one on the next run. `huggingface_hub`'s `hf_hub_download`/`snapshot_download` already download to a temp location and atomically rename into place on success, so a killed download leaves no file at the final path — the installer's own "does the final path already exist" check is therefore already safe for the three Hugging-Face assets (Task 4). The Piper **binary** download (a plain `curl`+`tar`, not through `huggingface_hub`) does not get that guarantee for free, so Task 4 must extract to a temporary directory first and only `mv` the completed result into its final place.
5. The one `sudo` prompt (apt packages) must run **before** any slow or multi-gigabyte step (Python environment setup, model downloads) so a declined/incorrect password fails in seconds, not after ten minutes of downloading (Task 6's ordering of calls inside `main()`).

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `packaging/installer/lib/preflight.sh` | create | distro / architecture / GPU-driver checks (hard gate) |
| `packaging/installer/lib/apt.sh` | create | detect + install missing system packages, enable `ydotool.service` |
| `packaging/installer/lib/pyenv.sh` | create | install `uv`, build/find the Vocalis wheel, create venv, install it |
| `packaging/installer/lib/models.sh` | create | Piper binary download/extraction (the one non-Hugging-Face asset) |
| `packaging/installer/lib/fetch_models.py` | create | Parakeet / Qwen3-TTS / Piper-voice reuse-or-download via `huggingface_hub` |
| `packaging/installer/lib/service.sh` | create | substitute the systemd unit template, enable/start it |
| `packaging/installer/install.sh` | create | wires everything above into one end-to-end, idempotent install |
| `packaging/installer/uninstall.sh` | create | reverses the install; `--purge` also removes models/config |
| `packaging/installer/build-run.sh` | create | builds the wheel and wraps `install.sh`/`uninstall.sh` into `.run` files via `makeself` |
| `packaging/installer/README.md` | create | how to run/rebuild everything in this folder |
| `src/vocalis/startup_checks.py` | modify | promote `_PARAKEET_FILES`/`_QWEN3_TTS_FILES` to public constants |
| `pyproject.toml` | modify | add `packaging/installer/lib` to `[tool.mypy] files` |
| `tests/installer/*.bats` | create | bats tests for every `lib/*.sh` file and for `install.sh`/`uninstall.sh` |
| `tests/installer/fixtures/*` | create | small fixture files (fake `os-release`, unit template) used by the bats tests |
| `tests/test_installer_fetch_models.py` | create | pytest tests for `fetch_models.py` |
| docs + change log | modify/create | matching documentation |

---

### Task 1: Preflight checks (distro, architecture, GPU driver)

**Files:**
- Create: `packaging/installer/lib/preflight.sh`
- Test: `tests/installer/test_preflight.bats`
- Test fixtures: `tests/installer/fixtures/os-release-debian`, `tests/installer/fixtures/os-release-ubuntu`, `tests/installer/fixtures/os-release-fedora`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces (used by Task 6's `install.sh`):
  - `preflight_check_distro(os_release_file="/etc/os-release")` → exit 0 if `ID=` or `ID_LIKE=` contains `debian` or `ubuntu` (case-insensitive), exit 1 otherwise.
  - `preflight_check_arch()` → exit 0 if `preflight_run_uname_m` prints `x86_64`, exit 1 otherwise.
  - `preflight_check_gpu()` → exit 0 if `preflight_run_nvidia_smi` exits 0, exit 1 otherwise.
  - `preflight_run()` → runs all three in order, printing one plain-English line to stderr and returning 1 on the FIRST failure (does not run the remaining checks once one has failed), returns 0 if all pass.

- [ ] **Step 1: Ask the user for the one dev-tooling sudo install, then run it**

Before writing any code, ask: "I need to install three small developer tools on this machine to build and test Stage 3 (`shellcheck`, `bats`, `makeself` — none of these touch Vocalis itself, they're just for testing the installer scripts). OK to run `sudo apt-get install -y shellcheck bats makeself` (ask the user for their password)?" Once approved:

```bash
sudo apt-get install -y shellcheck bats makeself
shellcheck --version
bats --version
makeself --version
```

- [ ] **Step 2: Write the failing tests** — create `tests/installer/fixtures/os-release-debian`:

```
PRETTY_NAME="Debian GNU/Linux 12 (bookworm)"
NAME="Debian GNU/Linux"
VERSION_ID="12"
ID=debian
```

Create `tests/installer/fixtures/os-release-ubuntu`:

```
PRETTY_NAME="Ubuntu 24.04 LTS"
NAME="Ubuntu"
ID=ubuntu
ID_LIKE=debian
```

Create `tests/installer/fixtures/os-release-fedora`:

```
PRETTY_NAME="Fedora Linux 40"
NAME="Fedora Linux"
ID=fedora
```

Create `tests/installer/test_preflight.bats`:

```bash
#!/usr/bin/env bats

setup() {
  source "${BATS_TEST_DIRNAME}/../../packaging/installer/lib/preflight.sh"
  FIXTURES="${BATS_TEST_DIRNAME}/fixtures"
}

@test "distro check passes for a debian os-release file" {
  run preflight_check_distro "${FIXTURES}/os-release-debian"
  [ "$status" -eq 0 ]
}

@test "distro check passes for an ubuntu os-release file (ID_LIKE=debian)" {
  run preflight_check_distro "${FIXTURES}/os-release-ubuntu"
  [ "$status" -eq 0 ]
}

@test "distro check fails for a non-debian os-release file" {
  run preflight_check_distro "${FIXTURES}/os-release-fedora"
  [ "$status" -eq 1 ]
}

@test "distro check fails when the file does not exist" {
  run preflight_check_distro "${FIXTURES}/does-not-exist"
  [ "$status" -eq 1 ]
}

@test "arch check passes when uname -m reports x86_64" {
  preflight_run_uname_m() { echo "x86_64"; }
  run preflight_check_arch
  [ "$status" -eq 0 ]
}

@test "arch check fails when uname -m reports something else" {
  preflight_run_uname_m() { echo "aarch64"; }
  run preflight_check_arch
  [ "$status" -eq 1 ]
}

@test "gpu check passes when nvidia-smi succeeds" {
  preflight_run_nvidia_smi() { return 0; }
  run preflight_check_gpu
  [ "$status" -eq 0 ]
}

@test "gpu check fails when nvidia-smi fails" {
  preflight_run_nvidia_smi() { return 1; }
  run preflight_check_gpu
  [ "$status" -eq 1 ]
}

@test "preflight_run passes when every check passes" {
  preflight_check_distro() { return 0; }
  preflight_check_arch() { return 0; }
  preflight_check_gpu() { return 0; }
  run preflight_run
  [ "$status" -eq 0 ]
}

@test "preflight_run stops at the first failing check and prints a plain message" {
  preflight_check_distro() { return 0; }
  preflight_check_arch() { return 1; }
  preflight_check_gpu() { echo "SHOULD_NOT_RUN" >&2; return 0; }
  run preflight_run
  [ "$status" -eq 1 ]
  [[ "$output" != *"SHOULD_NOT_RUN"* ]]
  [[ "$output" == *"64-bit"* ]]
}

@test "preflight_run reports the GPU driver message when only the GPU check fails" {
  preflight_check_distro() { return 0; }
  preflight_check_arch() { return 0; }
  preflight_check_gpu() { return 1; }
  run preflight_run
  [ "$status" -eq 1 ]
  [[ "$output" == *"NVIDIA"* ]]
}
```

- [ ] **Step 3: Run, verify FAIL** — `bats tests/installer/test_preflight.bats` → every test fails (`preflight.sh` does not exist yet, "No such file or directory").

- [ ] **Step 4: Implement** — create `packaging/installer/lib/preflight.sh`:

```bash
#!/usr/bin/env bash
# Hard-gate checks run before anything else: is this a supported distro,
# a supported CPU architecture, and does the machine have a working
# NVIDIA GPU driver? Vocalis never tries to install a GPU driver itself
# (too risky/variable to automate safely) -- it only checks and refuses
# with a plain message if one isn't already working.
set -uo pipefail

# Whether /etc/os-release (or the given file) identifies a Debian- or
# Ubuntu-family distro, via its ID or ID_LIKE field.
preflight_check_distro() {
  local os_release_file="${1:-/etc/os-release}"
  if [[ ! -r "$os_release_file" ]]; then
    return 1
  fi
  local content
  content="$(tr '[:upper:]' '[:lower:]' < "$os_release_file")"
  if [[ "$content" == *"id=debian"* || "$content" == *"id=ubuntu"* || "$content" == *"id_like=debian"* || "$content" == *"id_like=ubuntu"* ]]; then
    return 0
  fi
  return 1
}

# Wrapper around `uname -m`, overridable by tests.
preflight_run_uname_m() {
  uname -m
}

# Whether this machine is 64-bit Intel/AMD (x86_64).
preflight_check_arch() {
  [[ "$(preflight_run_uname_m)" == "x86_64" ]]
}

# Wrapper around running `nvidia-smi`, overridable by tests.
preflight_run_nvidia_smi() {
  nvidia-smi >/dev/null 2>&1
}

# Whether a working NVIDIA GPU driver is present (nvidia-smi runs and
# succeeds). Vocalis needs CUDA for its voice engine and never installs
# a driver itself.
preflight_check_gpu() {
  preflight_run_nvidia_smi
}

# Run every preflight check in order; stop and report at the first
# failure (each is a hard requirement, unlike vocalis's own runtime
# startup_checks, which are all independent and non-fatal).
preflight_run() {
  if ! preflight_check_distro; then
    echo "Vocalis only supports Debian- or Ubuntu-based Linux systems. This machine's /etc/os-release doesn't look like either." >&2
    return 1
  fi
  if ! preflight_check_arch; then
    echo "Vocalis only supports 64-bit Intel/AMD machines (x86_64). This machine reports $(preflight_run_uname_m)." >&2
    return 1
  fi
  if ! preflight_check_gpu; then
    echo "Vocalis needs a working NVIDIA graphics card driver. Please install your GPU driver first (running 'nvidia-smi' should work without an error), then run this installer again." >&2
    return 1
  fi
  return 0
}
```

- [ ] **Step 5: Run, verify PASS** — `bats tests/installer/test_preflight.bats` (all green); `shellcheck packaging/installer/lib/preflight.sh` (zero warnings).

- [ ] **Step 6: Commit**

```bash
git add packaging/installer/lib/preflight.sh tests/installer/test_preflight.bats tests/installer/fixtures/os-release-debian tests/installer/fixtures/os-release-ubuntu tests/installer/fixtures/os-release-fedora
git commit -m "feat: add installer preflight checks for distro, arch and GPU driver"
```

---

### Task 2: System packages (apt) and ydotoold enablement

**Files:**
- Create: `packaging/installer/lib/apt.sh`
- Test: `tests/installer/test_apt.bats`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces (used by Task 6):
  - `APT_REQUIRED_PACKAGES=(tesseract-ocr xclip ydotool)` — array constant.
  - `apt_is_installed(package)` → exit 0 if `apt_run_dpkg_query "$package"` exits 0.
  - `apt_missing_packages(package...)` → prints each missing package name on its own line.
  - `apt_install_missing(package...)` → if none are missing, does nothing and returns 0; otherwise calls `apt_run_install` once with the full missing list.
  - `apt_enable_ydotoold()` → if `apt_run_systemctl_list_unit_files ydotool.service` finds the unit, calls `apt_run_systemctl_enable_now ydotool.service`; otherwise does nothing. Never fails the install (best-effort; logs a warning to stderr on failure).

- [ ] **Step 1: Write the failing tests** — create `tests/installer/test_apt.bats`:

```bash
#!/usr/bin/env bats

setup() {
  source "${BATS_TEST_DIRNAME}/../../packaging/installer/lib/apt.sh"
  CALL_LOG="$(mktemp)"
}

teardown() {
  rm -f "$CALL_LOG"
}

@test "apt_is_installed reflects apt_run_dpkg_query's exit code" {
  apt_run_dpkg_query() { [[ "$1" == "xclip" ]]; }
  run apt_is_installed "xclip"
  [ "$status" -eq 0 ]
  run apt_is_installed "tesseract-ocr"
  [ "$status" -eq 1 ]
}

@test "apt_missing_packages lists only the packages that are not installed" {
  apt_run_dpkg_query() { [[ "$1" == "xclip" ]]; }
  run apt_missing_packages "tesseract-ocr" "xclip" "ydotool"
  [ "$status" -eq 0 ]
  [[ "$output" == *"tesseract-ocr"* ]]
  [[ "$output" == *"ydotool"* ]]
  [[ "$output" != *"xclip"* ]]
}

@test "apt_install_missing does nothing and never shells out when nothing is missing" {
  apt_run_dpkg_query() { return 0; }
  apt_run_install() { echo "CALLED: $*" >> "$CALL_LOG"; }
  apt_install_missing "tesseract-ocr" "xclip" "ydotool"
  [ ! -s "$CALL_LOG" ]
}

@test "apt_install_missing calls apt_run_install exactly once with every missing package" {
  apt_run_dpkg_query() { [[ "$1" == "xclip" ]]; }
  apt_run_install() { echo "CALLED: $*" >> "$CALL_LOG"; }
  apt_install_missing "tesseract-ocr" "xclip" "ydotool"
  [ "$(wc -l < "$CALL_LOG")" -eq 1 ]
  grep -q "tesseract-ocr" "$CALL_LOG"
  grep -q "ydotool" "$CALL_LOG"
}

@test "apt_enable_ydotoold enables the unit when it exists" {
  apt_run_systemctl_list_unit_files() { echo "ydotool.service enabled"; }
  apt_run_systemctl_enable_now() { echo "CALLED: $*" >> "$CALL_LOG"; }
  apt_enable_ydotoold
  grep -q "ydotool.service" "$CALL_LOG"
}

@test "apt_enable_ydotoold does nothing when the unit does not exist" {
  apt_run_systemctl_list_unit_files() { echo ""; }
  apt_run_systemctl_enable_now() { echo "CALLED: $*" >> "$CALL_LOG"; }
  apt_enable_ydotoold
  [ ! -s "$CALL_LOG" ]
}

@test "apt_enable_ydotoold never fails the install if systemctl errors" {
  apt_run_systemctl_list_unit_files() { echo "ydotool.service enabled"; }
  apt_run_systemctl_enable_now() { return 1; }
  run apt_enable_ydotoold
  [ "$status" -eq 0 ]
}
```

- [ ] **Step 2: Run, verify FAIL** — `bats tests/installer/test_apt.bats` → fails, `apt.sh` does not exist.

- [ ] **Step 3: Implement** — create `packaging/installer/lib/apt.sh`:

```bash
#!/usr/bin/env bash
# Detects and installs the small system packages Vocalis's non-Python
# features need (OCR, clipboard, typing-at-cursor), and best-effort
# enables the ydotoold background service the ydotool package ships.
set -uo pipefail

APT_REQUIRED_PACKAGES=(tesseract-ocr xclip ydotool)

# Wrapper around `dpkg -s <package>`, overridable by tests.
apt_run_dpkg_query() {
  dpkg -s "$1" >/dev/null 2>&1
}

# Whether a single apt package is installed.
apt_is_installed() {
  apt_run_dpkg_query "$1"
}

# Print the names of every package in the given list that is not
# installed, one per line.
apt_missing_packages() {
  local package
  for package in "$@"; do
    if ! apt_is_installed "$package"; then
      echo "$package"
    fi
  done
}

# Wrapper around the real, one-password-prompt install command,
# overridable by tests.
apt_run_install() {
  sudo apt-get install -y "$@"
}

# Install every package in the given list that is not already
# installed, in one call (one sudo prompt). Does nothing, and never
# shells out, if everything is already present.
apt_install_missing() {
  local missing=()
  while IFS= read -r package; do
    [[ -n "$package" ]] && missing+=("$package")
  done < <(apt_missing_packages "$@")
  if [[ "${#missing[@]}" -eq 0 ]]; then
    return 0
  fi
  echo "Installing missing system packages: ${missing[*]}"
  apt_run_install "${missing[@]}"
}

# Wrappers around the two systemctl calls, overridable by tests.
apt_run_systemctl_list_unit_files() {
  systemctl --user list-unit-files "$1" 2>/dev/null
}

apt_run_systemctl_enable_now() {
  systemctl --user enable --now "$1"
}

# Best-effort: if the ydotool package's systemd user unit exists,
# enable and start it (dictation needs ydotoold to type at the
# cursor). Never fails the install -- a failure here just means the
# existing startup-check notification tells the user later.
apt_enable_ydotoold() {
  local listing
  listing="$(apt_run_systemctl_list_unit_files ydotool.service)"
  if [[ "$listing" != *"ydotool.service"* ]]; then
    return 0
  fi
  if ! apt_run_systemctl_enable_now ydotool.service; then
    echo "Could not enable ydotoold automatically; dictation will fall back to copying text to the clipboard." >&2
  fi
  return 0
}
```

- [ ] **Step 4: Run, verify PASS** — `bats tests/installer/test_apt.bats`; `shellcheck packaging/installer/lib/apt.sh`.

- [ ] **Step 5: Commit**

```bash
git add packaging/installer/lib/apt.sh tests/installer/test_apt.bats
git commit -m "feat: add installer system-package detection/install and ydotoold enablement"
```

---

### Task 3: Python environment (uv, venv, the Vocalis wheel)

**Files:**
- Create: `packaging/installer/lib/pyenv.sh`
- Test: `tests/installer/test_pyenv.bats`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces (used by Task 6):
  - `pyenv_ensure_uv()` → installs `uv` via `pyenv_run_install_uv` only if neither `$HOME/.local/bin/uv` nor `uv` on PATH already exists; otherwise does nothing.
  - `pyenv_find_or_build_wheel(script_dir, repo_root)` → prints the path to a `vocalis-*.whl` file: a bundled one at `script_dir/dist/` if present, else builds one from `repo_root` via `pyenv_run_uv_build` and prints the newest `repo_root/dist/vocalis-*.whl`.
  - `pyenv_create_venv(venv_dir, python_version="3.13")` → calls `pyenv_run_uv python install "$python_version"` then `pyenv_run_uv venv --python "$python_version" "$venv_dir"`.
  - `pyenv_install_vocalis(venv_dir, wheel_path)` → calls `pyenv_run_uv pip install --python "$venv_dir/bin/python" "$wheel_path"`.

- [ ] **Step 1: Write the failing tests** — create `tests/installer/test_pyenv.bats`:

```bash
#!/usr/bin/env bats

setup() {
  source "${BATS_TEST_DIRNAME}/../../packaging/installer/lib/pyenv.sh"
  WORKDIR="$(mktemp -d)"
  CALL_LOG="$(mktemp)"
}

teardown() {
  rm -rf "$WORKDIR"
  rm -f "$CALL_LOG"
}

@test "pyenv_ensure_uv does nothing when uv is already on PATH" {
  command() { [[ "$1" == "-v" && "$2" == "uv" ]] && return 0; builtin command "$@"; }
  pyenv_run_install_uv() { echo "CALLED" >> "$CALL_LOG"; }
  pyenv_ensure_uv
  [ ! -s "$CALL_LOG" ]
}

@test "pyenv_ensure_uv installs uv when it is missing" {
  command() { [[ "$1" == "-v" && "$2" == "uv" ]] && return 1; builtin command "$@"; }
  HOME="$WORKDIR"
  pyenv_run_install_uv() { echo "CALLED" >> "$CALL_LOG"; }
  pyenv_ensure_uv
  [ -s "$CALL_LOG" ]
}

@test "pyenv_find_or_build_wheel returns a bundled wheel without building" {
  mkdir -p "$WORKDIR/script/dist" "$WORKDIR/repo"
  touch "$WORKDIR/script/dist/vocalis-0.1.0-py3-none-any.whl"
  pyenv_run_uv_build() { echo "CALLED" >> "$CALL_LOG"; }
  result="$(pyenv_find_or_build_wheel "$WORKDIR/script" "$WORKDIR/repo")"
  [ "$result" = "$WORKDIR/script/dist/vocalis-0.1.0-py3-none-any.whl" ]
  [ ! -s "$CALL_LOG" ]
}

@test "pyenv_find_or_build_wheel builds from the repo when nothing is bundled" {
  mkdir -p "$WORKDIR/script" "$WORKDIR/repo/dist"
  pyenv_run_uv_build() {
    touch "$WORKDIR/repo/dist/vocalis-0.1.0-py3-none-any.whl"
    echo "CALLED: $1" >> "$CALL_LOG"
  }
  result="$(pyenv_find_or_build_wheel "$WORKDIR/script" "$WORKDIR/repo")"
  [ "$result" = "$WORKDIR/repo/dist/vocalis-0.1.0-py3-none-any.whl" ]
  grep -q "$WORKDIR/repo" "$CALL_LOG"
}

@test "pyenv_create_venv installs the right python version then creates the venv" {
  pyenv_run_uv() { echo "CALLED: $*" >> "$CALL_LOG"; }
  pyenv_create_venv "$WORKDIR/venv" "3.13"
  grep -q "python install 3.13" "$CALL_LOG"
  grep -q "venv --python 3.13 $WORKDIR/venv" "$CALL_LOG"
}

@test "pyenv_install_vocalis installs the wheel into the venv's own python" {
  pyenv_run_uv() { echo "CALLED: $*" >> "$CALL_LOG"; }
  pyenv_install_vocalis "$WORKDIR/venv" "$WORKDIR/vocalis-0.1.0-py3-none-any.whl"
  grep -q "pip install --python $WORKDIR/venv/bin/python $WORKDIR/vocalis-0.1.0-py3-none-any.whl" "$CALL_LOG"
}
```

(The `command()` override trick in the first two tests shadows the bash builtin inside the test's subshell only — this is a standard bats technique and does not affect any other test.)

- [ ] **Step 2: Run, verify FAIL** — `bats tests/installer/test_pyenv.bats` → fails, `pyenv.sh` does not exist.

- [ ] **Step 3: Implement** — create `packaging/installer/lib/pyenv.sh`:

```bash
#!/usr/bin/env bash
# Provisions the private Python environment Vocalis runs in: installs
# `uv` if needed, finds or builds the Vocalis wheel, creates a venv
# with the right Python version, and installs the wheel into it.
set -uo pipefail

UV_PINNED_VERSION="0.10.8"

# Wrapper around the official, version-pinned uv installer, overridable
# by tests. Pinned (not a bare "latest" URL) for reproducibility.
pyenv_run_install_uv() {
  curl -LsSf "https://astral.sh/uv/${UV_PINNED_VERSION}/install.sh" | sh
}

# Install uv if it is not already available, either on PATH or at its
# default install location.
pyenv_ensure_uv() {
  if command -v uv >/dev/null 2>&1 || [[ -x "${HOME}/.local/bin/uv" ]]; then
    return 0
  fi
  echo "Installing uv (a fast Python package manager) ..."
  pyenv_run_install_uv
}

# Wrapper around `uv build`, overridable by tests.
pyenv_run_uv_build() {
  local repo_root="$1"
  (cd "$repo_root" && uv build)
}

# Return the path to a Vocalis wheel: one already bundled alongside the
# installer script (the normal case when run from a packaged .run
# file), or else one freshly built from the given repo checkout (the
# normal case when running install.sh straight from a dev checkout).
pyenv_find_or_build_wheel() {
  local script_dir="$1"
  local repo_root="$2"
  local bundled
  bundled="$(find "${script_dir}/dist" -maxdepth 1 -name 'vocalis-*.whl' 2>/dev/null | sort | tail -n1)"
  if [[ -n "$bundled" ]]; then
    echo "$bundled"
    return 0
  fi
  pyenv_run_uv_build "$repo_root" >&2
  find "${repo_root}/dist" -maxdepth 1 -name 'vocalis-*.whl' | sort | tail -n1
}

# Wrapper around every `uv` invocation below, overridable by tests.
pyenv_run_uv() {
  uv "$@"
}

# Ensure the given Python version is available to uv, then create a
# venv at venv_dir using it.
pyenv_create_venv() {
  local venv_dir="$1"
  local python_version="${2:-3.13}"
  pyenv_run_uv python install "$python_version"
  pyenv_run_uv venv --python "$python_version" "$venv_dir"
}

# Install the given wheel into the given venv's own interpreter
# (no "activate" step needed).
pyenv_install_vocalis() {
  local venv_dir="$1"
  local wheel_path="$2"
  pyenv_run_uv pip install --python "${venv_dir}/bin/python" "$wheel_path"
}
```

- [ ] **Step 4: Run, verify PASS** — `bats tests/installer/test_pyenv.bats`; `shellcheck packaging/installer/lib/pyenv.sh`.

- [ ] **Step 5: Commit**

```bash
git add packaging/installer/lib/pyenv.sh tests/installer/test_pyenv.bats
git commit -m "feat: add installer python-environment provisioning via uv"
```

---

### Task 4: Model and voice asset acquisition

**Files:**
- Modify: `src/vocalis/startup_checks.py` (promote two private constants to public)
- Modify: `pyproject.toml` (`[tool.mypy] files`)
- Create: `packaging/installer/lib/fetch_models.py`
- Create: `packaging/installer/lib/models.sh`
- Test: `tests/test_installer_fetch_models.py`
- Test: `tests/installer/test_models.bats`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces (used by Task 6):
  - `vocalis.startup_checks.PARAKEET_MODEL_FILES` / `QWEN3_TTS_MODEL_FILES` — same tuples, now public, same values.
  - `fetch_models.py`'s `main() -> int`, runnable as `"$VENV_PYTHON" packaging/installer/lib/fetch_models.py` — downloads whatever of the three Hugging-Face-hosted assets (Parakeet, Qwen3-TTS, Piper voice) is missing at `VocalisConfig()`'s own default paths, leaves anything already present untouched, prints one line per asset either way.
  - `models.sh`'s `models_ensure_piper_binary(binary_path)` — reuse-or-download the Piper binary (the one asset not on Hugging Face), via an atomic temp-dir-then-move extraction.

- [ ] **Step 1: Promote the two constants** — in `src/vocalis/startup_checks.py`, rename:

```python
_PARAKEET_FILES = ("config.json", "encoder-model.int8.onnx", "decoder_joint-model.int8.onnx", "vocab.txt")
_QWEN3_TTS_FILES = ("config.json", "model.safetensors")
```

to:

```python
PARAKEET_MODEL_FILES = ("config.json", "encoder-model.int8.onnx", "decoder_joint-model.int8.onnx", "vocab.txt")
QWEN3_TTS_MODEL_FILES = ("config.json", "model.safetensors")
```

and update the two usages inside `check_parakeet_model`/`check_qwen3_tts_model` (`_PARAKEET_FILES` → `PARAKEET_MODEL_FILES`, `_QWEN3_TTS_FILES` → `QWEN3_TTS_MODEL_FILES`). These are now the installer's single source of truth for "which files make a model complete" — no existing test references the old private names directly (they use literal tuples inline), so this rename is safe.

- [ ] **Step 2: Run, verify still PASS** — `.venv/bin/pytest tests/test_startup_checks.py -q --no-cov && .venv/bin/mypy` (unchanged behaviour, just a rename).

- [ ] **Step 3: Add the mypy scope** — in `pyproject.toml`, change:

```toml
[tool.mypy]
python_version = "3.13"
strict = true
mypy_path = "src"
files = ["src/vocalis", "tests"]
```

to:

```toml
[tool.mypy]
python_version = "3.13"
strict = true
mypy_path = "src"
files = ["src/vocalis", "tests", "packaging/installer/lib/fetch_models.py"]
```

- [ ] **Step 4: Write the failing pytest tests** — create `tests/test_installer_fetch_models.py`:

```python
"""Tests for packaging/installer/lib/fetch_models.py.

Loaded by file path (it lives outside the vocalis package, under the
installer's own tooling folder) with huggingface_hub's two download
functions patched -- no real network access.
"""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_MODULE_PATH = Path(__file__).resolve().parent.parent / "packaging" / "installer" / "lib" / "fetch_models.py"


def _load_module():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("fetch_models", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def fetch_models():  # type: ignore[no-untyped-def]
    return _load_module()


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_ensure_parakeet_skips_download_when_all_files_present(fetch_models, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    model_dir = tmp_path / "parakeet"
    for name in fetch_models.PARAKEET_MODEL_FILES:
        _touch(model_dir / name)
    with patch.object(fetch_models, "hf_hub_download") as mock_download:
        fetch_models.ensure_parakeet(model_dir)
    mock_download.assert_not_called()


def test_ensure_parakeet_downloads_every_file_when_missing(fetch_models, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    model_dir = tmp_path / "parakeet"
    with patch.object(fetch_models, "hf_hub_download") as mock_download:
        fetch_models.ensure_parakeet(model_dir)
    assert mock_download.call_count == len(fetch_models.PARAKEET_MODEL_FILES)
    for call in mock_download.call_args_list:
        assert call.kwargs["repo_id"] == fetch_models.PARAKEET_REPO
        assert call.kwargs["local_dir"] == str(model_dir)


def test_ensure_qwen3_tts_skips_download_when_all_files_present(fetch_models, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    model_dir = tmp_path / "qwen"
    for name in fetch_models.QWEN3_TTS_MODEL_FILES:
        _touch(model_dir / name)
    with patch.object(fetch_models, "snapshot_download") as mock_snapshot:
        fetch_models.ensure_qwen3_tts(model_dir)
    mock_snapshot.assert_not_called()


def test_ensure_qwen3_tts_downloads_the_full_snapshot_when_missing(fetch_models, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    model_dir = tmp_path / "qwen"
    with patch.object(fetch_models, "snapshot_download") as mock_snapshot:
        fetch_models.ensure_qwen3_tts(model_dir)
    mock_snapshot.assert_called_once_with(repo_id=fetch_models.QWEN3_TTS_REPO, local_dir=str(model_dir))


def test_ensure_piper_voice_skips_download_when_both_files_present(fetch_models, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    voice_path = tmp_path / "piper-voices" / "en_US-lessac-medium.onnx"
    _touch(voice_path)
    _touch(Path(str(voice_path) + ".json"))
    with patch.object(fetch_models, "hf_hub_download") as mock_download:
        fetch_models.ensure_piper_voice(voice_path)
    mock_download.assert_not_called()


def test_ensure_piper_voice_downloads_and_flattens_both_files_when_missing(fetch_models, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    voice_path = tmp_path / "piper-voices" / "en_US-lessac-medium.onnx"

    def fake_download(repo_id: str, filename: str, local_dir: str) -> str:  # noqa: ANN001
        nested = Path(local_dir) / filename
        _touch(nested)
        return str(nested)

    with patch.object(fetch_models, "hf_hub_download", side_effect=fake_download) as mock_download:
        fetch_models.ensure_piper_voice(voice_path)

    assert mock_download.call_count == len(fetch_models.PIPER_VOICE_FILES)
    assert voice_path.exists()
    assert Path(str(voice_path) + ".json").exists()


def test_main_runs_all_three_ensures_in_order(fetch_models, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    calls = []
    fake_config = MagicMock()
    fake_config.engines.parakeet_model_dir = str(tmp_path / "parakeet")
    fake_config.engines.qwen3_tts_model_dir = str(tmp_path / "qwen")
    fake_config.engines.piper_voice_model_path = str(tmp_path / "piper-voices" / "en_US-lessac-medium.onnx")
    with (
        patch.object(fetch_models, "VocalisConfig", return_value=fake_config),
        patch.object(fetch_models, "ensure_parakeet", side_effect=lambda p: calls.append(("parakeet", p))),
        patch.object(fetch_models, "ensure_qwen3_tts", side_effect=lambda p: calls.append(("qwen3_tts", p))),
        patch.object(fetch_models, "ensure_piper_voice", side_effect=lambda p: calls.append(("piper_voice", p))),
    ):
        assert fetch_models.main() == 0
    assert [name for name, _ in calls] == ["parakeet", "qwen3_tts", "piper_voice"]
```

- [ ] **Step 5: Run, verify FAIL** — `.venv/bin/pytest tests/test_installer_fetch_models.py -q --no-cov` → `FileNotFoundError`/import failure, `fetch_models.py` does not exist.

- [ ] **Step 6: Implement** — create `packaging/installer/lib/fetch_models.py`:

```python
"""Reuse-or-download the three Hugging-Face-hosted Vocalis assets:
the Parakeet dictation model, the Qwen3-TTS voice model, and the
Piper fallback voice. Run with the just-installed vocalis venv's
python so it can import vocalis's own config defaults and file lists
as the single source of truth -- the installer must never hard-code
a second copy of these paths/filenames and risk drifting out of sync.

The Piper BINARY (not hosted on Hugging Face) is handled separately,
in models.sh, via a plain curl+tar download.
"""
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from vocalis.config import VocalisConfig
from vocalis.startup_checks import PARAKEET_MODEL_FILES, QWEN3_TTS_MODEL_FILES

# PARAKEET_REPO and QWEN3_TTS_REPO were confirmed directly from this
# machine's own Hugging Face cache folder names
# (~/.cache/huggingface/hub/models--*). PIPER_VOICE_FILES' path follows
# rhasspy/piper-voices' well-documented, standard layout but was NOT
# independently re-confirmed over the network while writing this plan
# (no outbound network access at plan-writing time) -- if it's wrong,
# hf_hub_download raises a loud, clear "not found" error rather than
# silently installing the wrong file, so double-check this one path
# against https://huggingface.co/rhasspy/piper-voices during this
# task if network access is available, and fix here if it's changed.
PARAKEET_REPO = "istupakov/parakeet-tdt-0.6b-v3-onnx"
QWEN3_TTS_REPO = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
PIPER_VOICE_REPO = "rhasspy/piper-voices"
PIPER_VOICE_FILES = (
    "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
    "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
)


def ensure_parakeet(model_dir: Path) -> None:
    """Download the Parakeet model's files into model_dir, unless already present.

    huggingface_hub downloads to a temp location and only renames into
    place on success, so a half-finished earlier download never looks
    like a complete one here.
    """
    if all((model_dir / name).exists() for name in PARAKEET_MODEL_FILES):
        print(f"Parakeet model already present at {model_dir}")
        return
    print(f"Downloading Parakeet model to {model_dir} (about 650MB) ...")
    for name in PARAKEET_MODEL_FILES:
        hf_hub_download(repo_id=PARAKEET_REPO, filename=name, local_dir=str(model_dir))


def ensure_qwen3_tts(model_dir: Path) -> None:
    """Download the full Qwen3-TTS snapshot into model_dir, unless already present."""
    if all((model_dir / name).exists() for name in QWEN3_TTS_MODEL_FILES):
        print(f"Qwen3-TTS model already present at {model_dir}")
        return
    print(f"Downloading Qwen3-TTS model to {model_dir} (about 2.4GB) ...")
    snapshot_download(repo_id=QWEN3_TTS_REPO, local_dir=str(model_dir))


def ensure_piper_voice(voice_path: Path) -> None:
    """Download the Piper fallback voice to voice_path, unless already present.

    hf_hub_download preserves the repo's nested folder layout under
    local_dir; the two downloaded files are moved up to the flat path
    vocalis's config expects.
    """
    companion = Path(str(voice_path) + ".json")
    if voice_path.exists() and companion.exists():
        print(f"Piper voice already present at {voice_path}")
        return
    print(f"Downloading Piper voice to {voice_path.parent} ...")
    voice_path.parent.mkdir(parents=True, exist_ok=True)
    for filename in PIPER_VOICE_FILES:
        downloaded = Path(hf_hub_download(repo_id=PIPER_VOICE_REPO, filename=filename, local_dir=str(voice_path.parent)))
        target = voice_path.parent / Path(filename).name
        if downloaded != target:
            downloaded.rename(target)


def main() -> int:
    """Ensure all three Hugging-Face-hosted assets are present, at vocalis's own default paths."""
    config = VocalisConfig()
    ensure_parakeet(Path(config.engines.parakeet_model_dir))
    ensure_qwen3_tts(Path(config.engines.qwen3_tts_model_dir))
    ensure_piper_voice(Path(config.engines.piper_voice_model_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 7: Run, verify PASS** — `.venv/bin/pytest tests/test_installer_fetch_models.py -q --no-cov && .venv/bin/mypy`.

- [ ] **Step 8: Write the failing bats tests for the Piper binary** — create `tests/installer/test_models.bats`:

```bash
#!/usr/bin/env bats

setup() {
  source "${BATS_TEST_DIRNAME}/../../packaging/installer/lib/models.sh"
  WORKDIR="$(mktemp -d)"
  CALL_LOG="$(mktemp)"
}

teardown() {
  rm -rf "$WORKDIR"
  rm -f "$CALL_LOG"
}

@test "models_ensure_piper_binary does nothing when the binary already exists and is executable" {
  local binary_path="$WORKDIR/piper/piper"
  mkdir -p "$WORKDIR/piper"
  install -m 755 /bin/true "$binary_path"
  models_run_download_piper() { echo "CALLED" >> "$CALL_LOG"; }
  models_ensure_piper_binary "$binary_path"
  [ ! -s "$CALL_LOG" ]
}

@test "models_ensure_piper_binary downloads and extracts when the binary is missing" {
  local binary_path="$WORKDIR/bin/piper/piper"
  models_run_download_piper() {
    local dest="$1"
    local staging="$(mktemp -d)"
    mkdir -p "$staging/piper"
    install -m 755 /bin/true "$staging/piper/piper"
    (cd "$staging" && tar czf "$dest" piper)
    rm -rf "$staging"
  }
  models_ensure_piper_binary "$binary_path"
  [ -x "$binary_path" ]
}

@test "models_ensure_piper_binary never leaves a partial result on the final path if extraction fails" {
  local binary_path="$WORKDIR/bin/piper/piper"
  models_run_download_piper() { return 1; }
  run models_ensure_piper_binary "$binary_path"
  [ "$status" -ne 0 ]
  [ ! -e "$binary_path" ]
}
```

- [ ] **Step 9: Run, verify FAIL** — `bats tests/installer/test_models.bats` → fails, `models.sh` does not exist.

- [ ] **Step 10: Implement** — create `packaging/installer/lib/models.sh`:

```bash
#!/usr/bin/env bash
# Handles the one model/voice asset not hosted on Hugging Face: the
# Piper binary itself, downloaded from its pinned GitHub release.
# (Parakeet, Qwen3-TTS and the Piper voice are handled by
# fetch_models.py, since they're all on Hugging Face.)
set -uo pipefail

PIPER_RELEASE_URL="https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz"

# Wrapper around downloading the Piper release tarball to the given
# destination path, overridable by tests.
models_run_download_piper() {
  local dest="$1"
  curl -fsSL -o "$dest" "$PIPER_RELEASE_URL"
}

# Reuse-or-download the Piper binary at binary_path (e.g.
# ~/.local/bin/piper/piper). Extracts to a temporary staging directory
# first and only moves the completed result into place, so a failed or
# interrupted download/extraction never leaves a partial binary at the
# final path.
models_ensure_piper_binary() {
  local binary_path="$1"
  if [[ -x "$binary_path" ]]; then
    echo "Piper binary already present at $binary_path"
    return 0
  fi

  local bin_dir
  bin_dir="$(dirname "$(dirname "$binary_path")")"  # .../bin/piper/piper -> .../bin
  local tmp_tarball
  tmp_tarball="$(mktemp --suffix=.tar.gz)"
  local staging_dir
  staging_dir="$(mktemp -d)"

  echo "Downloading Piper to $bin_dir ..."
  if ! models_run_download_piper "$tmp_tarball"; then
    rm -f "$tmp_tarball"
    rm -rf "$staging_dir"
    echo "Could not download Piper from $PIPER_RELEASE_URL" >&2
    return 1
  fi

  tar xzf "$tmp_tarball" -C "$staging_dir"
  rm -f "$tmp_tarball"
  mkdir -p "$bin_dir"
  rm -rf "${bin_dir}/piper"
  mv "${staging_dir}/piper" "$bin_dir/"
  rm -rf "$staging_dir"
}
```

- [ ] **Step 11: Run, verify PASS** — `bats tests/installer/test_models.bats`; `shellcheck packaging/installer/lib/models.sh`.

- [ ] **Step 12: Run the full existing suite once** — `.venv/bin/pytest -q && .venv/bin/mypy` (the Task-1 rename must not have broken anything else).

- [ ] **Step 13: Commit**

```bash
git add src/vocalis/startup_checks.py pyproject.toml packaging/installer/lib/fetch_models.py packaging/installer/lib/models.sh tests/test_installer_fetch_models.py tests/installer/test_models.bats
git commit -m "feat: add installer model/voice asset acquisition"
```

---

### Task 5: Systemd service wiring

**Files:**
- Create: `packaging/installer/lib/service.sh`
- Test: `tests/installer/test_service.bats`
- Test fixture: `tests/installer/fixtures/unit-template.service`

**Interfaces:**
- Consumes: nothing from other tasks (reads `packaging/systemd/vocalis.service` at real-install time, but tests use a fixture).
- Produces (used by Task 6):
  - `service_substitute_unit(template_path, exec_path, workdir, output_path)` — writes `output_path` with `__VOCALIS_EXEC__`/`__VOCALIS_WORKDIR__` replaced, creating `output_path`'s parent directory if needed.
  - `service_enable_and_start(unit_name="vocalis.service")` — calls `service_run_systemctl_daemon_reload` then `service_run_systemctl_enable_now "$unit_name"`.

- [ ] **Step 1: Write the failing tests** — create `tests/installer/fixtures/unit-template.service`:

```
[Unit]
Description=Vocalis voice AI workstation

[Service]
Type=simple
ExecStart=__VOCALIS_EXEC__
WorkingDirectory=__VOCALIS_WORKDIR__

[Install]
WantedBy=xdg-desktop-autostart.target
```

Create `tests/installer/test_service.bats`:

```bash
#!/usr/bin/env bats

setup() {
  source "${BATS_TEST_DIRNAME}/../../packaging/installer/lib/service.sh"
  WORKDIR="$(mktemp -d)"
  CALL_LOG="$(mktemp)"
  FIXTURE="${BATS_TEST_DIRNAME}/fixtures/unit-template.service"
}

teardown() {
  rm -rf "$WORKDIR"
  rm -f "$CALL_LOG"
}

@test "service_substitute_unit replaces both placeholders and writes the output file" {
  local output="$WORKDIR/systemd/user/vocalis.service"
  service_substitute_unit "$FIXTURE" "/home/example/.local/share/vocalis/venv/bin/vocalis" "/home/example" "$output"
  [ -f "$output" ]
  grep -q "ExecStart=/home/example/.local/share/vocalis/venv/bin/vocalis" "$output"
  grep -q "WorkingDirectory=/home/example" "$output"
  ! grep -q "__VOCALIS_EXEC__" "$output"
  ! grep -q "__VOCALIS_WORKDIR__" "$output"
}

@test "service_substitute_unit creates the output directory if it doesn't exist" {
  local output="$WORKDIR/does/not/exist/yet/vocalis.service"
  service_substitute_unit "$FIXTURE" "/x/vocalis" "/x" "$output"
  [ -f "$output" ]
}

@test "service_enable_and_start reloads then enables the given unit, in that order" {
  service_run_systemctl_daemon_reload() { echo "daemon-reload" >> "$CALL_LOG"; }
  service_run_systemctl_enable_now() { echo "enable-now: $1" >> "$CALL_LOG"; }
  service_enable_and_start "vocalis.service"
  [ "$(sed -n '1p' "$CALL_LOG")" = "daemon-reload" ]
  [ "$(sed -n '2p' "$CALL_LOG")" = "enable-now: vocalis.service" ]
}

@test "service_enable_and_start defaults to vocalis.service" {
  service_run_systemctl_daemon_reload() { :; }
  service_run_systemctl_enable_now() { echo "enable-now: $1" >> "$CALL_LOG"; }
  service_enable_and_start
  grep -q "enable-now: vocalis.service" "$CALL_LOG"
}
```

- [ ] **Step 2: Run, verify FAIL** — `bats tests/installer/test_service.bats` → fails, `service.sh` does not exist.

- [ ] **Step 3: Implement** — create `packaging/installer/lib/service.sh`:

```bash
#!/usr/bin/env bash
# Fills in the systemd user unit template with real, installed paths
# and enables/starts it.
set -uo pipefail

# Write output_path with the template's two placeholders replaced by
# real values, creating output_path's parent directory if needed.
service_substitute_unit() {
  local template_path="$1"
  local exec_path="$2"
  local workdir="$3"
  local output_path="$4"
  mkdir -p "$(dirname "$output_path")"
  sed -e "s|__VOCALIS_EXEC__|${exec_path}|g" -e "s|__VOCALIS_WORKDIR__|${workdir}|g" "$template_path" > "$output_path"
}

# Wrappers around the two systemctl calls, overridable by tests.
service_run_systemctl_daemon_reload() {
  systemctl --user daemon-reload
}

service_run_systemctl_enable_now() {
  systemctl --user enable --now "$1"
}

# Reload the user systemd manager, then enable and start the given
# unit (defaulting to vocalis.service).
service_enable_and_start() {
  local unit_name="${1:-vocalis.service}"
  service_run_systemctl_daemon_reload
  service_run_systemctl_enable_now "$unit_name"
}
```

- [ ] **Step 4: Run, verify PASS** — `bats tests/installer/test_service.bats`; `shellcheck packaging/installer/lib/service.sh`.

- [ ] **Step 5: Commit**

```bash
git add packaging/installer/lib/service.sh tests/installer/test_service.bats tests/installer/fixtures/unit-template.service
git commit -m "feat: add installer systemd unit substitution and enablement"
```

---

### Task 6: `install.sh` — wiring it all together

**Files:**
- Create: `packaging/installer/install.sh`
- Create: `packaging/installer/README.md`
- Test: `tests/installer/test_install.bats`

**Interfaces:**
- Consumes: every function produced by Tasks 1-5.
- Produces: `install.sh` is both runnable directly (`bash install.sh`) and sourceable for tests (a `main()` function plus a run-guard at the bottom: `if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then main "$@"; fi`).

Behaviour of `main()`, in order (ordering matters — see Review Focus #5: the one sudo prompt happens before any slow step):

1. Compute `SCRIPT_DIR` and `REPO_ROOT` (`SCRIPT_DIR/../..`).
2. Source all five `lib/*.sh` files.
3. `preflight_run` — on failure, `exit 1` immediately (its own message already went to stderr).
4. `apt_install_missing "${APT_REQUIRED_PACKAGES[@]}"` then `apt_enable_ydotoold`.
5. `pyenv_ensure_uv`.
6. `wheel_path="$(pyenv_find_or_build_wheel "$SCRIPT_DIR" "$REPO_ROOT")"`.
7. `venv_dir="$HOME/.local/share/vocalis/venv"`; `pyenv_create_venv "$venv_dir"`; `pyenv_install_vocalis "$venv_dir" "$wheel_path"`.
8. `mkdir -p "$HOME/.local/bin"`; symlink `$venv_dir/bin/vocalis` to `$HOME/.local/bin/vocalis` (remove any existing symlink/file at that path first, so re-running is idempotent).
9. Query the four model/voice/binary paths from the just-installed package via four `"$venv_dir/bin/python" -c '...'` one-liners (see code below).
10. `"$venv_dir/bin/python" "$SCRIPT_DIR/lib/fetch_models.py"` (handles Parakeet, Qwen3-TTS, Piper voice).
11. `models_ensure_piper_binary "$piper_binary_path"` (Piper binary itself).
12. `service_substitute_unit "$REPO_ROOT/packaging/systemd/vocalis.service" "$HOME/.local/bin/vocalis" "$HOME" "$HOME/.config/systemd/user/vocalis.service"`; `service_enable_and_start`.
13. Print the plain-English success message.

- [ ] **Step 1: Write the failing tests** — create `tests/installer/test_install.bats`:

```bash
#!/usr/bin/env bats

setup() {
  WORKDIR="$(mktemp -d)"
  CALL_LOG="$(mktemp)"
  source "${BATS_TEST_DIRNAME}/../../packaging/installer/install.sh"

  # Replace every real external effect with a call-logging stub.
  preflight_run() { echo "preflight_run" >> "$CALL_LOG"; return 0; }
  apt_install_missing() { echo "apt_install_missing: $*" >> "$CALL_LOG"; }
  apt_enable_ydotoold() { echo "apt_enable_ydotoold" >> "$CALL_LOG"; }
  pyenv_ensure_uv() { echo "pyenv_ensure_uv" >> "$CALL_LOG"; }
  pyenv_find_or_build_wheel() { echo "pyenv_find_or_build_wheel" >> "$CALL_LOG"; echo "$WORKDIR/vocalis-0.1.0-py3-none-any.whl"; }
  pyenv_create_venv() { echo "pyenv_create_venv: $*" >> "$CALL_LOG"; mkdir -p "$1/bin"; echo '#!/bin/sh' > "$1/bin/python"; chmod +x "$1/bin/python"; }
  pyenv_install_vocalis() { echo "pyenv_install_vocalis: $*" >> "$CALL_LOG"; }
  models_ensure_piper_binary() { echo "models_ensure_piper_binary: $*" >> "$CALL_LOG"; }
  service_substitute_unit() { echo "service_substitute_unit: $*" >> "$CALL_LOG"; }
  service_enable_and_start() { echo "service_enable_and_start" >> "$CALL_LOG"; }

  # install.sh queries four paths and runs fetch_models.py via the
  # venv's python -- stub that interpreter itself so no real python runs.
  cat > "$WORKDIR/fake_python" <<'EOF'
#!/usr/bin/env bash
echo "/fake/path"
EOF
  chmod +x "$WORKDIR/fake_python"
}

teardown() {
  rm -rf "$WORKDIR"
  rm -f "$CALL_LOG"
}

@test "main stops immediately when preflight fails, before touching anything else" {
  preflight_run() { echo "preflight_run" >> "$CALL_LOG"; return 1; }
  run main
  [ "$status" -ne 0 ]
  [ "$(wc -l < "$CALL_LOG")" -eq 1 ]
}

@test "main installs system packages before provisioning the python environment" {
  SCRIPT_DIR="$WORKDIR" REPO_ROOT="$WORKDIR" HOME="$WORKDIR" run main
  [ "$status" -eq 0 ]
  apt_line="$(grep -n "apt_install_missing" "$CALL_LOG" | cut -d: -f1)"
  pyenv_line="$(grep -n "pyenv_ensure_uv" "$CALL_LOG" | cut -d: -f1)"
  [ -n "$apt_line" ]
  [ -n "$pyenv_line" ]
  [ "$apt_line" -lt "$pyenv_line" ]
}

@test "main is idempotent: running it twice never fails" {
  SCRIPT_DIR="$WORKDIR" REPO_ROOT="$WORKDIR" HOME="$WORKDIR" run main
  [ "$status" -eq 0 ]
  SCRIPT_DIR="$WORKDIR" REPO_ROOT="$WORKDIR" HOME="$WORKDIR" run main
  [ "$status" -eq 0 ]
}

@test "main creates the vocalis command symlink" {
  SCRIPT_DIR="$WORKDIR" REPO_ROOT="$WORKDIR" HOME="$WORKDIR" run main
  [ "$status" -eq 0 ]
  [ -L "$WORKDIR/.local/bin/vocalis" ]
}
```

- [ ] **Step 2: Run, verify FAIL** — `bats tests/installer/test_install.bats` → fails, `install.sh` does not exist.

- [ ] **Step 3: Implement** — create `packaging/installer/install.sh`:

```bash
#!/usr/bin/env bash
# Installs Vocalis end-to-end on a Debian/Ubuntu-family x86_64 machine:
# system packages, a private Python environment, the voice/dictation
# models, and the background service, all idempotent (safe to re-run
# as a repair command).
set -euo pipefail

# Real, non-overridable location of this script and the repo it lives
# in -- used ONLY to find the lib/ files to source below. main()
# below reads its OWN overridable copies of these two paths, computed
# fresh on every call, so a test can redirect them (by setting
# SCRIPT_DIR/REPO_ROOT before calling main) without this top-level
# sourcing ever needing to change.
_INSTALL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_INSTALL_REPO_ROOT="$(cd "${_INSTALL_SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=lib/preflight.sh
source "${_INSTALL_SCRIPT_DIR}/lib/preflight.sh"
# shellcheck source=lib/apt.sh
source "${_INSTALL_SCRIPT_DIR}/lib/apt.sh"
# shellcheck source=lib/pyenv.sh
source "${_INSTALL_SCRIPT_DIR}/lib/pyenv.sh"
# shellcheck source=lib/models.sh
source "${_INSTALL_SCRIPT_DIR}/lib/models.sh"
# shellcheck source=lib/service.sh
source "${_INSTALL_SCRIPT_DIR}/lib/service.sh"

main() {
  # Read fresh on every call (not computed once at source time) so a
  # test can override SCRIPT_DIR/REPO_ROOT in its own environment
  # right before calling main, without touching the real sourcing above.
  local script_dir="${SCRIPT_DIR:-$_INSTALL_SCRIPT_DIR}"
  local repo_root="${REPO_ROOT:-$_INSTALL_REPO_ROOT}"

  if ! preflight_run; then
    exit 1
  fi

  apt_install_missing "${APT_REQUIRED_PACKAGES[@]}"
  apt_enable_ydotoold

  pyenv_ensure_uv

  local wheel_path
  wheel_path="$(pyenv_find_or_build_wheel "$script_dir" "$repo_root")"

  local venv_dir="${HOME}/.local/share/vocalis/venv"
  pyenv_create_venv "$venv_dir"
  pyenv_install_vocalis "$venv_dir" "$wheel_path"

  mkdir -p "${HOME}/.local/bin"
  rm -f "${HOME}/.local/bin/vocalis"
  ln -s "${venv_dir}/bin/vocalis" "${HOME}/.local/bin/vocalis"

  local venv_python="${venv_dir}/bin/python"
  local parakeet_dir qwen3_tts_dir piper_binary piper_voice
  parakeet_dir="$("$venv_python" -c 'from vocalis.config import VocalisConfig; print(VocalisConfig().engines.parakeet_model_dir)')"
  qwen3_tts_dir="$("$venv_python" -c 'from vocalis.config import VocalisConfig; print(VocalisConfig().engines.qwen3_tts_model_dir)')"
  piper_binary="$("$venv_python" -c 'from vocalis.config import VocalisConfig; print(VocalisConfig().engines.piper_binary_path)')"
  piper_voice="$("$venv_python" -c 'from vocalis.config import VocalisConfig; print(VocalisConfig().engines.piper_voice_model_path)')"

  echo "Checking Parakeet model at ${parakeet_dir} ..."
  echo "Checking Qwen3-TTS model at ${qwen3_tts_dir} ..."
  echo "Checking Piper voice at ${piper_voice} ..."
  "$venv_python" "${script_dir}/lib/fetch_models.py"

  echo "Checking Piper binary at ${piper_binary} ..."
  models_ensure_piper_binary "$piper_binary"

  local unit_output="${HOME}/.config/systemd/user/vocalis.service"
  service_substitute_unit "${repo_root}/packaging/systemd/vocalis.service" "${HOME}/.local/bin/vocalis" "$HOME" "$unit_output"
  service_enable_and_start

  echo ""
  echo "Vocalis is installed and running."
  echo "It will start automatically the next time you log in."
  echo "To check on it later: systemctl --user status vocalis"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
```

Create `packaging/installer/README.md`:

```markdown
# Vocalis installer

Run `bash install.sh` (or double-click/run `install.run` if you have
that file instead) from inside this folder. It will:

1. Check your machine is Debian/Ubuntu-based, 64-bit, and has a working
   NVIDIA GPU driver.
2. Install a few small system packages if they're missing (one password
   prompt).
3. Set up a private Python environment for Vocalis under
   `~/.local/share/vocalis/venv` (does not touch your system Python).
4. Download the voice/dictation models if they aren't already on your
   machine (a few gigabytes — this can take a while).
5. Set Vocalis up to start automatically whenever you log in, and start
   it right now.

It's safe to run more than once — anything already correctly set up is
left alone, so re-running it also works as a repair command.

To remove Vocalis, run `bash uninstall.sh` (see its own `--help`).

## Rebuilding the `.run` files (developers only)

See `build-run.sh`.
```

- [ ] **Step 4: Run, verify PASS** — `bats tests/installer/test_install.bats`; `shellcheck packaging/installer/install.sh`.

- [ ] **Step 5: Commit**

```bash
git add packaging/installer/install.sh packaging/installer/README.md tests/installer/test_install.bats
git commit -m "feat: add the end-to-end install.sh installer"
```

---

### Task 7: `uninstall.sh`

**Files:**
- Create: `packaging/installer/uninstall.sh`
- Test: `tests/installer/test_uninstall.bats`

**Interfaces:**
- Consumes: `service_run_systemctl_daemon_reload`/`apt_run_systemctl_enable_now`-style wrappers are re-declared locally in this file (uninstall needs `disable`, which the other tasks didn't need) rather than importing `service.sh`/`apt.sh`, to keep this file self-contained and short.
- Produces: `uninstall.sh`, same `main "$@"`/run-guard structure as `install.sh`. Accepts an optional `--purge` flag.

Behaviour of `main()`:

1. `uninstall_run_systemctl_disable_now vocalis.service` (wrapped, ignores failure — "not installed" is not an error here).
2. Remove `~/.config/systemd/user/vocalis.service` if present, then `uninstall_run_systemctl_daemon_reload`.
3. Remove `~/.local/bin/vocalis` if present.
4. If `~/.local/share/vocalis/venv/bin/python` exists AND `--purge` was passed: query the four model/config paths the same way `install.sh` does, remember them.
5. Remove `~/.local/share/vocalis/venv` if present.
6. If nothing existed at any of steps 1-3/5 to begin with, print "Vocalis does not appear to be installed; nothing to remove." instead of the normal summary — but still exit 0.
7. If `--purge` was passed and step 4 found paths, remove `~/.config/vocalis`, the parakeet dir, and the qwen3-tts dir (never removes the Piper binary/voice, since those are commonly reused by other tools on the machine — print exactly what was removed either way); otherwise print what was deliberately kept and where.

- [ ] **Step 1: Write the failing tests** — create `tests/installer/test_uninstall.bats`:

```bash
#!/usr/bin/env bats

setup() {
  WORKDIR="$(mktemp -d)"
  CALL_LOG="$(mktemp)"
  source "${BATS_TEST_DIRNAME}/../../packaging/installer/uninstall.sh"
  uninstall_run_systemctl_disable_now() { echo "disable-now: $*" >> "$CALL_LOG"; return 0; }
  uninstall_run_systemctl_daemon_reload() { echo "daemon-reload" >> "$CALL_LOG"; }
}

teardown() {
  rm -rf "$WORKDIR"
  rm -f "$CALL_LOG"
}

@test "main on a machine where vocalis was never installed exits 0 and says so" {
  HOME="$WORKDIR" run main
  [ "$status" -eq 0 ]
  [[ "$output" == *"nothing to remove"* ]]
}

@test "main removes the unit file, symlink and venv when they exist" {
  mkdir -p "$WORKDIR/.config/systemd/user" "$WORKDIR/.local/share/vocalis/venv/bin" "$WORKDIR/.local/bin"
  touch "$WORKDIR/.config/systemd/user/vocalis.service"
  touch "$WORKDIR/.local/share/vocalis/venv/bin/python"
  ln -s "$WORKDIR/.local/share/vocalis/venv/bin/vocalis" "$WORKDIR/.local/bin/vocalis"
  HOME="$WORKDIR" run main
  [ "$status" -eq 0 ]
  [ ! -e "$WORKDIR/.config/systemd/user/vocalis.service" ]
  [ ! -e "$WORKDIR/.local/bin/vocalis" ]
  [ ! -e "$WORKDIR/.local/share/vocalis/venv" ]
}

@test "main without --purge keeps config and models and says so" {
  mkdir -p "$WORKDIR/.config/vocalis" "$WORKDIR/.local/share/vocalis/venv/bin"
  touch "$WORKDIR/.local/share/vocalis/venv/bin/python"
  HOME="$WORKDIR" run main
  [ "$status" -eq 0 ]
  [ -d "$WORKDIR/.config/vocalis" ]
  [[ "$output" == *"kept"* ]]
}

@test "main --purge also removes the config directory" {
  mkdir -p "$WORKDIR/.config/vocalis" "$WORKDIR/.local/share/vocalis/venv/bin"
  cat > "$WORKDIR/.local/share/vocalis/venv/bin/python" <<'EOF'
#!/usr/bin/env bash
echo "/fake/parakeet/dir"
EOF
  chmod +x "$WORKDIR/.local/share/vocalis/venv/bin/python"
  HOME="$WORKDIR" run main --purge
  [ "$status" -eq 0 ]
  [ ! -d "$WORKDIR/.config/vocalis" ]
}
```

- [ ] **Step 2: Run, verify FAIL** — `bats tests/installer/test_uninstall.bats` → fails, `uninstall.sh` does not exist.

- [ ] **Step 3: Implement** — create `packaging/installer/uninstall.sh`:

```bash
#!/usr/bin/env bash
# Reverses install.sh. By default keeps your config and downloaded
# models (several GB); pass --purge to remove those too.
set -uo pipefail

uninstall_run_systemctl_disable_now() {
  systemctl --user disable --now "$1" >/dev/null 2>&1
}

uninstall_run_systemctl_daemon_reload() {
  systemctl --user daemon-reload
}

main() {
  local purge=0
  if [[ "${1:-}" == "--purge" ]]; then
    purge=1
  fi

  local found_anything=0
  local unit_file="${HOME}/.config/systemd/user/vocalis.service"
  local symlink="${HOME}/.local/bin/vocalis"
  local venv_dir="${HOME}/.local/share/vocalis/venv"
  local venv_python="${venv_dir}/bin/python"

  local config_dir="" parakeet_dir="" qwen3_tts_dir=""
  if [[ -x "$venv_python" ]]; then
    found_anything=1
    if [[ "$purge" -eq 1 ]]; then
      config_dir="${HOME}/.config/vocalis"
      parakeet_dir="$("$venv_python" -c 'from vocalis.config import VocalisConfig; print(VocalisConfig().engines.parakeet_model_dir)' 2>/dev/null || true)"
      qwen3_tts_dir="$("$venv_python" -c 'from vocalis.config import VocalisConfig; print(VocalisConfig().engines.qwen3_tts_model_dir)' 2>/dev/null || true)"
    fi
  fi

  uninstall_run_systemctl_disable_now vocalis.service

  if [[ -f "$unit_file" ]]; then
    found_anything=1
    rm -f "$unit_file"
    uninstall_run_systemctl_daemon_reload
  fi

  if [[ -e "$symlink" ]]; then
    found_anything=1
    rm -f "$symlink"
  fi

  if [[ -d "$venv_dir" ]]; then
    found_anything=1
    rm -rf "$venv_dir"
  fi

  if [[ "$found_anything" -eq 0 ]]; then
    echo "Vocalis does not appear to be installed; nothing to remove."
    return 0
  fi

  if [[ "$purge" -eq 1 ]]; then
    [[ -n "$config_dir" && -d "$config_dir" ]] && rm -rf "$config_dir"
    [[ -n "$parakeet_dir" && -d "$parakeet_dir" ]] && rm -rf "$parakeet_dir"
    [[ -n "$qwen3_tts_dir" && -d "$qwen3_tts_dir" ]] && rm -rf "$qwen3_tts_dir"
    echo "Vocalis has been removed, including your config and downloaded models."
  else
    echo "Vocalis has been removed. Your config (~/.config/vocalis) and downloaded models were kept."
    echo "Run with --purge to remove those too."
  fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
```

- [ ] **Step 4: Run, verify PASS** — `bats tests/installer/test_uninstall.bats`; `shellcheck packaging/installer/uninstall.sh`.

- [ ] **Step 5: Commit**

```bash
git add packaging/installer/uninstall.sh tests/installer/test_uninstall.bats
git commit -m "feat: add uninstall.sh with an optional --purge flag"
```

---

### Task 8: `build-run.sh` — self-extracting `.run` files

**Files:**
- Create: `packaging/installer/build-run.sh`
- Modify: `packaging/installer/README.md` (fill in the "Rebuilding" section)
- Test: `tests/installer/test_build_run.bats`

**Interfaces:**
- Consumes: `pyenv_run_uv_build` is reused by sourcing `lib/pyenv.sh`.
- Produces: `build-run.sh`'s `main()` — builds the wheel, stages it alongside `install.sh`/`uninstall.sh`/`lib/` in a temp directory, and calls `build_run_run_makeself` twice (once per script) to produce `install.run` and `uninstall.run` next to this file.

- [ ] **Step 1: Write the failing tests** — create `tests/installer/test_build_run.bats`:

```bash
#!/usr/bin/env bats

setup() {
  WORKDIR="$(mktemp -d)"
  CALL_LOG="$(mktemp)"
  source "${BATS_TEST_DIRNAME}/../../packaging/installer/build-run.sh"
  build_run_run_makeself() { echo "CALLED: $*" >> "$CALL_LOG"; touch "$4"; }
  pyenv_run_uv_build() {
    mkdir -p "$1/dist"
    touch "$1/dist/vocalis-0.1.0-py3-none-any.whl"
  }

  # main()'s real (unstubbed) `cp` calls need real source files to
  # copy, even though SCRIPT_DIR below points at an isolated fixture
  # directory rather than the real packaging/installer/ -- fake but
  # present install.sh/uninstall.sh/lib/ are enough.
  mkdir -p "$WORKDIR/packaging/installer/lib" "$WORKDIR/packaging/systemd"
  touch "$WORKDIR/packaging/installer/install.sh" "$WORKDIR/packaging/installer/uninstall.sh"
  touch "$WORKDIR/packaging/installer/lib/preflight.sh"
  touch "$WORKDIR/packaging/systemd/vocalis.service"
}

teardown() {
  rm -rf "$WORKDIR"
  rm -f "$CALL_LOG"
}

@test "main stages the wheel and produces both .run files" {
  SCRIPT_DIR="$WORKDIR/packaging/installer" REPO_ROOT="$WORKDIR" run main
  [ "$status" -eq 0 ]
  [ "$(grep -c "CALLED" "$CALL_LOG")" -eq 2 ]
  [[ "$(cat "$CALL_LOG")" == *"install.run"* ]]
  [[ "$(cat "$CALL_LOG")" == *"uninstall.run"* ]]
}
```

- [ ] **Step 2: Run, verify FAIL** — `bats tests/installer/test_build_run.bats` → fails, `build-run.sh` does not exist.

- [ ] **Step 3: Implement** — create `packaging/installer/build-run.sh`:

```bash
#!/usr/bin/env bash
# Builds the Vocalis wheel and wraps install.sh/uninstall.sh (plus the
# wheel and lib/ helpers they need) into two self-extracting .run files
# via makeself, for people who'd rather double-click one file than run
# a shell script by hand. Developer-only tool; never run as part of a
# real install.
set -euo pipefail

# Real, non-overridable location, used only to find lib/pyenv.sh.
# main() below reads its own overridable copies, the same pattern
# install.sh uses and for the same reason (see its comment).
_BUILD_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_BUILD_REPO_ROOT="$(cd "${_BUILD_SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=lib/pyenv.sh
source "${_BUILD_SCRIPT_DIR}/lib/pyenv.sh"

# Wrapper around the real makeself invocation, overridable by tests.
# label: human label shown when the .run file is extracted.
# staging_dir: the directory whose contents become the payload.
# entry_script: which file inside staging_dir makeself should run.
# output_path: where to write the finished .run file.
build_run_run_makeself() {
  local label="$1"
  local staging_dir="$2"
  local entry_script="$3"
  local output_path="$4"
  makeself "$staging_dir" "$output_path" "$label" "./${entry_script}"
}

main() {
  local script_dir="${SCRIPT_DIR:-$_BUILD_SCRIPT_DIR}"
  local repo_root="${REPO_ROOT:-$_BUILD_REPO_ROOT}"

  local wheel_dist_dir="${script_dir}/dist"
  mkdir -p "$wheel_dist_dir"
  pyenv_run_uv_build "$repo_root"
  cp "${repo_root}"/dist/vocalis-*.whl "$wheel_dist_dir/"

  local staging_dir
  staging_dir="$(mktemp -d)"
  cp -r "${script_dir}/lib" "$staging_dir/"
  cp "${script_dir}/install.sh" "${script_dir}/uninstall.sh" "$staging_dir/"
  mkdir -p "$staging_dir/dist"
  cp "$wheel_dist_dir"/vocalis-*.whl "$staging_dir/dist/"
  mkdir -p "$staging_dir/packaging/systemd"
  cp "${repo_root}/packaging/systemd/vocalis.service" "$staging_dir/packaging/systemd/"

  build_run_run_makeself "Vocalis installer" "$staging_dir" "install.sh" "${script_dir}/install.run"
  build_run_run_makeself "Vocalis uninstaller" "$staging_dir" "uninstall.sh" "${script_dir}/uninstall.run"

  rm -rf "$staging_dir"
  echo "Built ${script_dir}/install.run and ${script_dir}/uninstall.run"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
```

Update the "Rebuilding the `.run` files" section of `packaging/installer/README.md`:

```markdown
## Rebuilding the `.run` files (developers only)

```
bash build-run.sh
```

This builds the Vocalis wheel from the repo and produces
`install.run`/`uninstall.run` next to `install.sh`/`uninstall.sh`. Needs
`uv` and `makeself` installed. The `.run` files are generated artifacts —
never edit them directly or commit them; re-run this script instead.
```

- [ ] **Step 4: Run, verify PASS** — `bats tests/installer/test_build_run.bats`; `shellcheck packaging/installer/build-run.sh`.

- [ ] **Step 5: Add a `.gitignore` entry** — append to the repo's `.gitignore` (create it if it doesn't already cover this):

```
packaging/installer/*.run
packaging/installer/dist/
```

- [ ] **Step 6: Commit**

```bash
git add packaging/installer/build-run.sh packaging/installer/README.md tests/installer/test_build_run.bats .gitignore
git commit -m "feat: add build-run.sh to produce self-extracting .run installers"
```

---

### Task 9: Docs, change log, full verification sweep

**Files:**
- Modify: `docs/user-docs.md`, `docs/architecture.md`, `docs/workflow.md`, `docs/file-purpose.md`, `docs/folder-structure.md`, `docs/developer-docs.md`, the Stage 3 amendment in the spec (add an "Implementation deviations and findings" subsection)
- Create: `.claude/<YYYY-MM-DD_HHMM>_CHANGELOG.md`

- [ ] **Step 1: Docs** (verify every sentence against the actual code/scripts):
  - `docs/user-docs.md` — a "Installing Vocalis" section: run `install.sh`/`install.run`, what it asks for (one password prompt), roughly how long it takes (models can take a while to download the first time), and that `uninstall.sh --purge` fully removes everything including models.
  - `docs/architecture.md`/`docs/workflow.md` — the installer's overall flow (preflight → packages → python env → models → service), the "reuse if present at vocalis's own default paths, else download" model policy, and why there's no `.deb`/AppImage (same reasoning as the spec amendment, summarized).
  - `docs/file-purpose.md`/`docs/folder-structure.md` — every new file under `packaging/installer/`.
  - `docs/developer-docs.md` — how to run the bats/shellcheck suite locally, the `*_run_*` override-after-source testing idiom, how to rebuild the `.run` files, and the `PARAKEET_MODEL_FILES`/`QWEN3_TTS_MODEL_FILES` rename.
- [ ] **Step 2: Spec deviations** — add an "Implementation deviations and findings (recorded after Stage 3 build)" subsection under the Stage 3 amendment in `docs/superpowers/specs/2026-09-11-vocalis-design.md`, covering anything that changed from the design during the build (e.g. the exact Hugging Face repo IDs and pinned Piper/`uv` release versions discovered via live lookups, which were not yet known at design time), following the same style as Stage 1/2's deviation sections.
- [ ] **Step 3: Change log** — files touched, bats/pytest/mypy/shellcheck results from your own final run, and an explicit `- [ ] Real-world check pending: the user's own first real run of install.sh — apt install, model downloads, venv creation, enabling the service, and uninstall.sh — none of which any automated test in this stage actually performed.`
- [ ] **Step 4: Full verification**:

```bash
export PYTHONPATH=$PWD/src
.venv/bin/pytest -q    # three times sequentially; zero warnings, coverage >= 85%
.venv/bin/mypy         # Success
shellcheck packaging/installer/*.sh packaging/installer/lib/*.sh   # zero warnings
bats tests/installer/   # every .bats file, all green
```

- [ ] **Step 5: Commit**

```bash
git add docs .claude/*CHANGELOG.md docs/superpowers/specs/2026-09-11-vocalis-design.md
git commit -m "docs: document the installer, its testing approach and how to rebuild it"
```
