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
  # Prove the SCRIPT_DIR override actually took effect: the logged calls
  # must reference WORKDIR's install.run/uninstall.run paths, not the real
  # packaging/installer/ directory (both end in "install.run" / "uninstall.run",
  # so the asserts above alone can't tell the two apart).
  [[ "$(cat "$CALL_LOG")" == *"${WORKDIR}/packaging/installer/install.run"* ]]
  [[ "$(cat "$CALL_LOG")" == *"${WORKDIR}/packaging/installer/uninstall.run"* ]]
}

@test "main marks the staged install.sh and uninstall.sh as executable" {
  # makeself runs "./install.sh" inside the extracted payload, so the
  # staged copies must be executable even if the source files lost
  # their exec bits (the fixture files from setup() have none).
  build_run_run_makeself() {
    if [[ -x "$2/install.sh" && -x "$2/uninstall.sh" ]]; then
      echo "EXEC_OK" >> "$CALL_LOG"
    else
      echo "EXEC_MISSING" >> "$CALL_LOG"
    fi
    touch "$4"
  }
  SCRIPT_DIR="$WORKDIR/packaging/installer" REPO_ROOT="$WORKDIR" run main
  [ "$status" -eq 0 ]
  [ "$(grep -c "EXEC_OK" "$CALL_LOG")" -eq 2 ]
  ! grep -q "EXEC_MISSING" "$CALL_LOG"
}
