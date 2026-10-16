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
  pyenv_find_or_build_wheel() { echo "pyenv_find_or_build_wheel: $*" >> "$CALL_LOG"; echo "$WORKDIR/vocalis-0.1.0-py3-none-any.whl"; }
  pyenv_create_venv() { echo "pyenv_create_venv: $*" >> "$CALL_LOG"; mkdir -p "$1/bin"; echo '#!/bin/sh' > "$1/bin/python"; chmod +x "$1/bin/python"; }
  pyenv_install_vocalis() { echo "pyenv_install_vocalis: $*" >> "$CALL_LOG"; }
  pyenv_install_kokoro() { echo "pyenv_install_kokoro: $*" >> "$CALL_LOG"; }
  models_ensure_piper_binary() { echo "models_ensure_piper_binary: $*" >> "$CALL_LOG"; }
  service_substitute_unit() { echo "service_substitute_unit: $*" >> "$CALL_LOG"; }
  service_enable_and_start() { echo "service_enable_and_start" >> "$CALL_LOG"; }
  desktop_install_launcher() { echo "desktop_install_launcher: $*" >> "$CALL_LOG"; }

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
  grep -q "pyenv_find_or_build_wheel: $WORKDIR $WORKDIR" "$CALL_LOG"
}

@test "main is idempotent: running it twice never fails" {
  SCRIPT_DIR="$WORKDIR" REPO_ROOT="$WORKDIR" HOME="$WORKDIR" run main
  [ "$status" -eq 0 ]
  [ "$(readlink "$WORKDIR/.local/bin/vocalis")" = "$WORKDIR/.local/share/vocalis/venv/bin/vocalis" ]
  # bats' `run` disables errexit for the duration of the call, so a mid-function
  # failure (e.g. a stale symlink making `ln -s` error out) would NOT change
  # $status here -- main() would just keep going and still exit via its last
  # command. Since the symlink's target is identical on both runs, a broken
  # `rm -f`/`ln -s` step also wouldn't change the *final* readlink result (the
  # already-correct symlink from run 1 is simply left alone when `ln -s` fails
  # on run 2). The only observable sign of that regression is the swallowed
  # "ln: failed to create symbolic link" error text in the captured output, so
  # assert it is absent to actually catch it.
  [[ "$output" != *"ln: failed to create symbolic link"* ]]
  SCRIPT_DIR="$WORKDIR" REPO_ROOT="$WORKDIR" HOME="$WORKDIR" run main
  [ "$status" -eq 0 ]
  [ "$(readlink "$WORKDIR/.local/bin/vocalis")" = "$WORKDIR/.local/share/vocalis/venv/bin/vocalis" ]
  [[ "$output" != *"ln: failed to create symbolic link"* ]]
}

@test "main creates the vocalis command symlink" {
  SCRIPT_DIR="$WORKDIR" REPO_ROOT="$WORKDIR" HOME="$WORKDIR" run main
  [ "$status" -eq 0 ]
  [ -L "$WORKDIR/.local/bin/vocalis" ]
}

@test "main uses the unit template staged next to the script when run from an extracted .run" {
  # Inside a makeself extraction, SCRIPT_DIR is the extracted temp dir and
  # REPO_ROOT (SCRIPT_DIR/../..) is unrelated -- the template only exists
  # under SCRIPT_DIR/packaging/systemd/.
  mkdir -p "$WORKDIR/extracted/packaging/systemd" "$WORKDIR/elsewhere" "$WORKDIR/home"
  touch "$WORKDIR/extracted/packaging/systemd/vocalis.service"
  SCRIPT_DIR="$WORKDIR/extracted" REPO_ROOT="$WORKDIR/elsewhere" HOME="$WORKDIR/home" run main
  [ "$status" -eq 0 ]
  grep -q "service_substitute_unit: $WORKDIR/extracted/packaging/systemd/vocalis.service " "$CALL_LOG"
}

@test "main falls back to the repo's unit template in a plain dev checkout" {
  mkdir -p "$WORKDIR/repo/packaging/systemd" "$WORKDIR/repo/packaging/installer" "$WORKDIR/home"
  touch "$WORKDIR/repo/packaging/systemd/vocalis.service"
  SCRIPT_DIR="$WORKDIR/repo/packaging/installer" REPO_ROOT="$WORKDIR/repo" HOME="$WORKDIR/home" run main
  [ "$status" -eq 0 ]
  grep -q "service_substitute_unit: $WORKDIR/repo/packaging/systemd/vocalis.service " "$CALL_LOG"
}

@test "main explains in plain English when system packages cannot be installed" {
  apt_install_missing() { echo "apt_install_missing: $*" >> "$CALL_LOG"; return 1; }
  SCRIPT_DIR="$WORKDIR" REPO_ROOT="$WORKDIR" HOME="$WORKDIR" run main
  [ "$status" -ne 0 ]
  [[ "$output" == *"Could not install required system packages"* ]]
  ! grep -q "pyenv_ensure_uv" "$CALL_LOG"
}

@test "main explains in plain English when the ydotoold service cannot be enabled" {
  apt_enable_ydotoold() { echo "apt_enable_ydotoold" >> "$CALL_LOG"; return 1; }
  SCRIPT_DIR="$WORKDIR" REPO_ROOT="$WORKDIR" HOME="$WORKDIR" run main
  [ "$status" -ne 0 ]
  [[ "$output" == *"Could not install required system packages"* ]]
  ! grep -q "pyenv_ensure_uv" "$CALL_LOG"
}

@test "main installs the menu launcher and tells the user how to open Vocalis" {
  SCRIPT_DIR="$WORKDIR" REPO_ROOT="$WORKDIR" HOME="$WORKDIR" run main
  [ "$status" -eq 0 ]
  grep -q "desktop_install_launcher: .* $WORKDIR/.local/bin/vocalis $WORKDIR/.local/share/applications/vocalis.desktop" "$CALL_LOG"
  [[ "$output" == *"vocalis --settings"* ]]
}

@test "a failing menu launcher does not fail the whole install" {
  desktop_install_launcher() { return 1; }
  SCRIPT_DIR="$WORKDIR" REPO_ROOT="$WORKDIR" HOME="$WORKDIR" run main
  [ "$status" -eq 0 ]
  [[ "$output" == *"menu entry could not be added"* ]]
}

@test "main installs kokoro-onnx right after the Vocalis wheel" {
  SCRIPT_DIR="$WORKDIR" REPO_ROOT="$WORKDIR" HOME="$WORKDIR" run main
  [ "$status" -eq 0 ]
  wheel_line="$(grep -n "pyenv_install_vocalis" "$CALL_LOG" | cut -d: -f1)"
  kokoro_line="$(grep -n "pyenv_install_kokoro" "$CALL_LOG" | cut -d: -f1)"
  [ -n "$kokoro_line" ]
  [ "$wheel_line" -lt "$kokoro_line" ]
}
