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
  # Fully hermetic: HOME points at an empty temp dir (so no real
  # ~/.local/bin/uv can be found) and PATH holds only system dirs plus
  # a fake uv we control -- dispositive on any machine, with or without
  # a real uv installed.
  mkdir -p "$WORKDIR/fakebin"
  printf '#!/bin/sh\necho fake-uv\n' > "$WORKDIR/fakebin/uv"
  chmod +x "$WORKDIR/fakebin/uv"
  HOME="$WORKDIR/home"
  PATH="$WORKDIR/fakebin:/usr/bin:/bin"
  pyenv_run_install_uv() { echo "CALLED" >> "$CALL_LOG"; }
  pyenv_ensure_uv
  [ ! -s "$CALL_LOG" ]
}

@test "pyenv_ensure_uv installs uv when it is missing" {
  HOME="$WORKDIR/home"
  PATH="/usr/bin:/bin"
  ! command -v uv >/dev/null 2>&1 || skip "a real uv exists in /usr/bin or /bin"
  pyenv_run_install_uv() { echo "CALLED" >> "$CALL_LOG"; }
  pyenv_ensure_uv
  [ -s "$CALL_LOG" ]
}

@test "pyenv_ensure_uv makes a freshly installed uv usable by later calls in the same session" {
  HOME="$WORKDIR/home"
  PATH="/usr/bin:/bin"
  ! command -v uv >/dev/null 2>&1 || skip "a real uv exists in /usr/bin or /bin"
  # Simulate the real installer: drop a working uv into ~/.local/bin,
  # which is NOT on this stripped PATH.
  pyenv_run_install_uv() {
    mkdir -p "$HOME/.local/bin"
    printf '#!/bin/sh\necho "fake-uv $*"\n' > "$HOME/.local/bin/uv"
    chmod +x "$HOME/.local/bin/uv"
  }
  pyenv_ensure_uv
  run pyenv_run_uv --version
  [ "$status" -eq 0 ]
  [ "$output" = "fake-uv --version" ]
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

@test "pyenv_find_or_build_wheel fails clearly when the build fails" {
  mkdir -p "$WORKDIR/script" "$WORKDIR/repo/dist"
  pyenv_run_uv_build() { return 1; }
  run pyenv_find_or_build_wheel "$WORKDIR/script" "$WORKDIR/repo"
  [ "$status" -ne 0 ]
  [[ "$output" == *"Could not build"* ]]
}

@test "pyenv_find_or_build_wheel fails clearly when the build produces no wheel" {
  mkdir -p "$WORKDIR/script" "$WORKDIR/repo/dist"
  pyenv_run_uv_build() { return 0; }
  run pyenv_find_or_build_wheel "$WORKDIR/script" "$WORKDIR/repo"
  [ "$status" -ne 0 ]
  [[ "$output" == *"No Vocalis wheel"* ]]
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

@test "pyenv_create_venv leaves an existing, working venv alone" {
  mkdir -p "$WORKDIR/venv/bin"
  printf '#!/bin/sh\n[ "$1" = "--version" ] && echo "Python 3.13.0"\n' > "$WORKDIR/venv/bin/python"
  chmod +x "$WORKDIR/venv/bin/python"
  pyenv_run_uv() { echo "CALLED: $*" >> "$CALL_LOG"; }
  run pyenv_create_venv "$WORKDIR/venv" "3.13"
  [ "$status" -eq 0 ]
  ! grep -q "venv" "$CALL_LOG"
}

@test "pyenv_create_venv replaces a venv folder whose python is broken" {
  mkdir -p "$WORKDIR/venv/bin"
  printf '#!/bin/sh\nexit 1\n' > "$WORKDIR/venv/bin/python"
  chmod +x "$WORKDIR/venv/bin/python"
  pyenv_run_uv() { echo "CALLED: $*" >> "$CALL_LOG"; }
  pyenv_create_venv "$WORKDIR/venv" "3.13"
  grep -q "venv --clear --python 3.13 $WORKDIR/venv" "$CALL_LOG"
}
