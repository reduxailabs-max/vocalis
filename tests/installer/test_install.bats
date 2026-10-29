#!/usr/bin/env bats

# Builds a fake venv: its python hands every call to $WORKDIR/fake_python.
make_fake_venv() {
  mkdir -p "$1/bin"
  printf '#!/bin/sh\nexec "%s/fake_python" "$@"\n' "$WORKDIR" > "$1/bin/python"
  chmod +x "$1/bin/python"
  if [ -z "${FAKE_NO_SETUP:-}" ]; then
    printf '#!/bin/sh\necho "vocalis-setup: $*" >> "%s"\nexit ${FAKE_SETUP_EXIT:-0}\n' "$CALL_LOG" > "$1/bin/vocalis-setup"
    chmod +x "$1/bin/vocalis-setup"
  fi
}

setup() {
  WORKDIR="$(mktemp -d)"
  CALL_LOG="$(mktemp)"
  export CALL_LOG WORKDIR
  # No screen by default: main with no flags then runs every phase.
  export DISPLAY= WAYLAND_DISPLAY=
  unset FAKE_GPU_ANSWER FAKE_GPU_FAIL FAKE_FETCH_FLAGS FAKE_FLAGS_FAIL FAKE_NO_SETUP FAKE_SETUP_EXIT
  source "${BATS_TEST_DIRNAME}/../../packaging/installer/install.sh"

  # Replace every real external effect with a call-logging stub.
  preflight_run() { echo "preflight_run" >> "$CALL_LOG"; return 0; }
  apt_install_missing() { echo "apt_install_missing: $*" >> "$CALL_LOG"; }
  apt_enable_ydotoold() { echo "apt_enable_ydotoold" >> "$CALL_LOG"; }
  pyenv_ensure_uv() { echo "pyenv_ensure_uv" >> "$CALL_LOG"; }
  pyenv_find_or_build_wheel() { echo "pyenv_find_or_build_wheel: $*" >> "$CALL_LOG"; echo "$WORKDIR/vocalis-0.1.0-py3-none-any.whl"; }
  pyenv_create_venv() {
    echo "pyenv_create_venv: $*" >> "$CALL_LOG"
    make_fake_venv "$1"
  }
  pyenv_install_vocalis() { echo "pyenv_install_vocalis: $*" >> "$CALL_LOG"; }
  pyenv_install_kokoro() { echo "pyenv_install_kokoro: $*" >> "$CALL_LOG"; }
  pyenv_prefer_gpu_onnxruntime() { echo "pyenv_prefer_gpu_onnxruntime: $*" >> "$CALL_LOG"; }
  pyenv_use_cpu_onnxruntime() { echo "pyenv_use_cpu_onnxruntime: $*" >> "$CALL_LOG"; }
  models_ensure_piper_binary() { echo "models_ensure_piper_binary: $*" >> "$CALL_LOG"; }
  service_substitute_unit() { echo "service_substitute_unit: $*" >> "$CALL_LOG"; }
  service_enable_and_start() { echo "service_enable_and_start" >> "$CALL_LOG"; }
  desktop_install_launcher() { echo "desktop_install_launcher: $*" >> "$CALL_LOG"; }

  # A fake interpreter that answers by looking at its arguments:
  #   plan gpu-runtime  -> $FAKE_GPU_ANSWER (default "yes"), fails if FAKE_GPU_FAIL is set
  #   plan fetch-flags  -> $FAKE_FETCH_FLAGS (default empty), fails if FAKE_FLAGS_FAIL is set
  #   fetch_models.py   -> logs its arguments to the call log
  #   anything else     -> a path ($WORKDIR/qwen3 for the Qwen3 folder query)
  cat > "$WORKDIR/fake_python" <<'FAKE_EOF'
#!/usr/bin/env bash
case "$*" in
  *"installer.plan gpu-runtime"*)
    [ -z "${FAKE_GPU_FAIL:-}" ] || exit 3
    echo "${FAKE_GPU_ANSWER:-yes}" ;;
  *"installer.plan fetch-flags"*)
    [ -z "${FAKE_FLAGS_FAIL:-}" ] || exit 3
    echo "${FAKE_FETCH_FLAGS:-}" ;;
  *fetch_models.py*)
    echo "fetch_models: $*" >> "$CALL_LOG" ;;
  *qwen3_tts_model_dir*)
    echo "$WORKDIR/qwen3" ;;
  *)
    echo "/fake/path" ;;
esac
FAKE_EOF
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

@test "main puts the GPU onnxruntime back after installing kokoro-onnx" {
  SCRIPT_DIR="$WORKDIR" REPO_ROOT="$WORKDIR" HOME="$WORKDIR" run main
  [ "$status" -eq 0 ]
  local kokoro_line gpu_line
  kokoro_line="$(grep -n "pyenv_install_kokoro" "$CALL_LOG" | cut -d: -f1)"
  gpu_line="$(grep -n "pyenv_prefer_gpu_onnxruntime" "$CALL_LOG" | cut -d: -f1)"
  [ -n "$gpu_line" ]
  [ "$kokoro_line" -lt "$gpu_line" ]
}

# Runs main with the standard test environment.
# Phases other than "prepare" expect the venv to exist already, so make one.
run_main() {
  make_fake_venv "$WORKDIR/.local/share/vocalis/venv"
  SCRIPT_DIR="$WORKDIR" REPO_ROOT="$WORKDIR" HOME="$WORKDIR" run main "$@"
}

@test "phase prepare runs only the prepare steps" {
  run_main --phase prepare
  [ "$status" -eq 0 ]
  grep -q "^preflight_run$" "$CALL_LOG"
  grep -q "pyenv_install_kokoro" "$CALL_LOG"
  [ -L "$WORKDIR/.local/bin/vocalis" ]
  ! grep -q "models_ensure_piper_binary" "$CALL_LOG"
  ! grep -q "fetch_models" "$CALL_LOG"
  ! grep -q "service_enable_and_start" "$CALL_LOG"
}

@test "phase models runs only the downloads" {
  run_main --phase models
  [ "$status" -eq 0 ]
  grep -q "fetch_models:" "$CALL_LOG"
  grep -q "models_ensure_piper_binary" "$CALL_LOG"
  ! grep -q "preflight_run" "$CALL_LOG"
  ! grep -q "apt_install_missing" "$CALL_LOG"
  ! grep -q "service_enable_and_start" "$CALL_LOG"
}

@test "phase finish runs only service and desktop" {
  run_main --phase finish
  [ "$status" -eq 0 ]
  grep -q "service_substitute_unit" "$CALL_LOG"
  grep -q "service_enable_and_start" "$CALL_LOG"
  grep -q "desktop_install_launcher" "$CALL_LOG"
  ! grep -q "preflight_run" "$CALL_LOG"
  ! grep -q "fetch_models" "$CALL_LOG"
  ! grep -q "models_ensure_piper_binary" "$CALL_LOG"
}

@test "phase all keeps the old order" {
  run_main --phase all
  [ "$status" -eq 0 ]
  local pre fetch piper svc
  pre="$(grep -n "^preflight_run$" "$CALL_LOG" | cut -d: -f1)"
  fetch="$(grep -n "fetch_models:" "$CALL_LOG" | cut -d: -f1)"
  piper="$(grep -n "models_ensure_piper_binary" "$CALL_LOG" | cut -d: -f1)"
  svc="$(grep -n "service_enable_and_start" "$CALL_LOG" | cut -d: -f1)"
  [ "$pre" -lt "$fetch" ]
  [ "$fetch" -lt "$piper" ]
  [ "$piper" -lt "$svc" ]
}

@test "--no-wizard behaves like phase all even with a display" {
  DISPLAY=:0 run_main --no-wizard
  [ "$status" -eq 0 ]
  grep -q "service_enable_and_start" "$CALL_LOG"
}

@test "an unknown option or phase is refused" {
  run_main --bogus
  [ "$status" -eq 2 ]
  run_main --phase nonsense
  [ "$status" -eq 2 ]
}

@test "gpu machine prefers the GPU onnxruntime" {
  run_main --phase prepare
  [ "$status" -eq 0 ]
  grep -q "pyenv_prefer_gpu_onnxruntime" "$CALL_LOG"
  ! grep -q "pyenv_use_cpu_onnxruntime" "$CALL_LOG"
}

@test "cpu machine switches to the CPU onnxruntime and not the GPU one" {
  FAKE_GPU_ANSWER=no run_main --phase prepare
  [ "$status" -eq 0 ]
  grep -q "pyenv_use_cpu_onnxruntime" "$CALL_LOG"
  ! grep -q "pyenv_prefer_gpu_onnxruntime" "$CALL_LOG"
}

@test "a failing hardware plan means the CPU onnxruntime, not the GPU one" {
  FAKE_GPU_FAIL=1 run_main --phase prepare
  [ "$status" -eq 0 ]
  grep -q "pyenv_use_cpu_onnxruntime" "$CALL_LOG"
  ! grep -q "pyenv_prefer_gpu_onnxruntime" "$CALL_LOG"
}

@test "an odd hardware plan answer means the CPU onnxruntime" {
  FAKE_GPU_ANSWER="yes please" run_main --phase prepare
  [ "$status" -eq 0 ]
  grep -q "pyenv_use_cpu_onnxruntime" "$CALL_LOG"
  ! grep -q "pyenv_prefer_gpu_onnxruntime" "$CALL_LOG"
}

@test "models phase passes --skip-qwen3 when the plan says so" {
  FAKE_FETCH_FLAGS=--skip-qwen3 run_main --phase models
  [ "$status" -eq 0 ]
  grep -q "fetch_models: .*-u .*fetch_models.py --skip-qwen3" "$CALL_LOG"
}

@test "models phase passes no extra flag when the plan prints nothing" {
  run_main --phase models
  [ "$status" -eq 0 ]
  grep -q "fetch_models.py$" "$CALL_LOG"
}

@test "a failing fetch-flags plan still skips Qwen3" {
  FAKE_FLAGS_FAIL=1 run_main --phase models
  [ "$status" -eq 0 ]
  grep -q "fetch_models.py --skip-qwen3" "$CALL_LOG"
}

@test "an odd fetch-flags answer still skips Qwen3" {
  FAKE_FETCH_FLAGS="--something-else" run_main --phase models
  [ "$status" -eq 0 ]
  grep -q "fetch_models.py --skip-qwen3$" "$CALL_LOG"
}

@test "models phase does not delete or touch an existing Qwen3 folder" {
  mkdir -p "$WORKDIR/qwen3"
  echo keep > "$WORKDIR/qwen3/model.bin"
  FAKE_FETCH_FLAGS=--skip-qwen3 run_main --phase models
  [ "$status" -eq 0 ]
  [ "$(cat "$WORKDIR/qwen3/model.bin")" = "keep" ]
}

@test "no flags and no display runs prepare, models and finish in the terminal" {
  run_main
  [ "$status" -eq 0 ]
  grep -q "pyenv_install_kokoro" "$CALL_LOG"
  grep -q "fetch_models:" "$CALL_LOG"
  grep -q "service_enable_and_start" "$CALL_LOG"
}

@test "no flags with a display launches vocalis-setup with --installer-dir" {
  install_launch_wizard() { echo "install_launch_wizard: $*" >> "$CALL_LOG"; }
  DISPLAY=:0 run_main
  [ "$status" -eq 0 ]
  grep -q "install_launch_wizard: $WORKDIR/.local/share/vocalis/venv $WORKDIR" "$CALL_LOG"
  ! grep -q "fetch_models" "$CALL_LOG"
  ! grep -q "service_enable_and_start" "$CALL_LOG"
}

@test "a Wayland-only display also launches the wizard" {
  install_launch_wizard() { echo "install_launch_wizard: $*" >> "$CALL_LOG"; }
  WAYLAND_DISPLAY=wayland-0 run_main
  [ "$status" -eq 0 ]
  grep -q "install_launch_wizard" "$CALL_LOG"
}

@test "missing vocalis-setup falls back to the terminal path" {
  install_launch_wizard() { echo "install_launch_wizard: $*" >> "$CALL_LOG"; }
  FAKE_NO_SETUP=1 DISPLAY=:0 run_main
  [ "$status" -eq 0 ]
  [[ "$output" == *"NOTE The setup window could not open"* ]]
  ! grep -q "install_launch_wizard" "$CALL_LOG"
  grep -q "fetch_models:" "$CALL_LOG"
  grep -q "service_enable_and_start" "$CALL_LOG"
}

@test "every phase prints STEP i/n lines" {
  run_main --phase prepare
  [ "$(grep -cE '^STEP [0-9]+/[0-9]+ ' <<<"$output")" -eq 5 ]
  [[ "$output" == *"STEP 1/5 "* && "$output" == *"STEP 5/5 "* ]]
  run_main --phase models
  [ "$(grep -cE '^STEP [0-9]+/[0-9]+ ' <<<"$output")" -eq 2 ]
  [[ "$output" == *"STEP 1/2 Downloading voices and the dictation model (a few gigabytes – this can take a while)"* ]]
  [[ "$output" == *"STEP 2/2 Downloading the Piper speech program"* ]]
  run_main --phase finish
  [ "$(grep -cE '^STEP [0-9]+/[0-9]+ ' <<<"$output")" -eq 2 ]
  [[ "$output" == *"STEP 1/2 Starting Vocalis"* ]]
  [[ "$output" == *"STEP 2/2 Adding Vocalis to your applications menu"* ]]
}

@test "the real launcher passes --installer-dir to vocalis-setup and runs nothing else when it exits 0" {
  DISPLAY=:0 run_main
  [ "$status" -eq 0 ]
  grep -q "^vocalis-setup: --installer-dir $WORKDIR$" "$CALL_LOG"
  ! grep -q "fetch_models" "$CALL_LOG"
  ! grep -q "service_enable_and_start" "$CALL_LOG"
  [[ "$output" == *"NOTE If no setup window opens, run: bash install.sh --no-wizard"* ]]
}

@test "wizard exit 1 is returned without any terminal fallback" {
  FAKE_SETUP_EXIT=1 DISPLAY=:0 run_main
  [ "$status" -eq 1 ]
  ! grep -q "fetch_models" "$CALL_LOG"
  ! grep -q "service_enable_and_start" "$CALL_LOG"
  [[ "$output" != *"could not open"* ]]
}

@test "wizard exit 3, 127 or 134 falls back to models and finish in the terminal" {
  local code
  for code in 3 127 134; do
    : > "$CALL_LOG"
    FAKE_SETUP_EXIT=$code DISPLAY=:0 run_main
    [ "$status" -eq 0 ]
    [[ "$output" == *"NOTE The setup window could not open, so I will finish the installation here in the terminal."* ]]
    grep -q "fetch_models:" "$CALL_LOG"
    grep -q "service_enable_and_start" "$CALL_LOG"
  done
}

@test "the terminal run prints plain step headings" {
  run_main
  [[ "$output" == *"--- Step A: preparing your computer ---"* ]]
  [[ "$output" == *"--- Step B: "* ]]
  [[ "$output" == *"--- Step C: "* ]]
}

@test "a failing hardware plan prints a NOTE about the safe choice" {
  FAKE_GPU_FAIL=1 FAKE_FLAGS_FAIL=1 run_main
  [ "$status" -eq 0 ]
  [[ "$output" == *"NOTE The hardware check did not work, so I will use the processor-only speech runtime"* ]]
  [[ "$output" == *"NOTE The hardware check did not work, so I will skip the large Qwen3 voice"* ]]
}

@test "the Qwen3 checking line is not printed when Qwen3 is skipped" {
  FAKE_FETCH_FLAGS=--skip-qwen3 run_main --phase models
  [[ "$output" != *"Checking Qwen3-TTS"* ]]
  run_main --phase models
  [[ "$output" == *"Checking Qwen3-TTS"* ]]
}
