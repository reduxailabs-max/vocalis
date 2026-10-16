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

@test "main removes the application-menu launcher" {
  mkdir -p "$WORKDIR/.local/share/applications"
  touch "$WORKDIR/.local/share/applications/vocalis.desktop"
  HOME="$WORKDIR" run main
  [ "$status" -eq 0 ]
  [ ! -e "$WORKDIR/.local/share/applications/vocalis.desktop" ]
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

make_installed_fixtures() {
  mkdir -p "$WORKDIR/.config/systemd/user" "$WORKDIR/.local/share/vocalis/venv/bin" "$WORKDIR/.local/bin" "$WORKDIR/.config/vocalis"
  touch "$WORKDIR/.config/systemd/user/vocalis.service"
  touch "$WORKDIR/.local/share/vocalis/venv/bin/python"
  ln -s "$WORKDIR/.local/share/vocalis/venv/bin/vocalis" "$WORKDIR/.local/bin/vocalis"
}

assert_installed_fixtures_untouched() {
  [ -e "$WORKDIR/.config/systemd/user/vocalis.service" ]
  [ -L "$WORKDIR/.local/bin/vocalis" ]
  [ -d "$WORKDIR/.local/share/vocalis/venv" ]
  [ -d "$WORKDIR/.config/vocalis" ]
  [ ! -s "$CALL_LOG" ]
}

@test "main --help prints usage, exits 0 and removes nothing" {
  make_installed_fixtures
  HOME="$WORKDIR" run main --help
  [ "$status" -eq 0 ]
  [[ "$output" == *"Usage"* ]]
  [[ "$output" == *"--purge"* ]]
  assert_installed_fixtures_untouched
}

@test "main -h prints usage, exits 0 and removes nothing" {
  make_installed_fixtures
  HOME="$WORKDIR" run main -h
  [ "$status" -eq 0 ]
  [[ "$output" == *"Usage"* ]]
  assert_installed_fixtures_untouched
}

@test "main with an unknown flag names it, exits 1 and removes nothing" {
  make_installed_fixtures
  HOME="$WORKDIR" run main --bogus
  [ "$status" -eq 1 ]
  [[ "$output" == *"--bogus"* ]]
  assert_installed_fixtures_untouched
}

write_purge_python_stub() {
  # Fake venv python that answers each config-path query uninstall.sh makes.
  mkdir -p "$WORKDIR/.local/share/vocalis/venv/bin"
  cat > "$WORKDIR/.local/share/vocalis/venv/bin/python" <<STUB
#!/usr/bin/env bash
case "\$*" in
  *parakeet_model_dir*) echo "$WORKDIR/models/parakeet" ;;
  *qwen3_tts_model_dir*) echo "$WORKDIR/models/qwen3" ;;
  *piper_binary_path*) echo "$WORKDIR/.local/bin/piper/piper" ;;
  *piper_voice_model_path*) echo "$WORKDIR/.local/share/piper-voices/en_US-lessac-medium.onnx" ;;
  *kokoro_model_path*) echo "$WORKDIR/kokoro-dir/kokoro-v1.0.onnx" ;;
esac
STUB
  chmod +x "$WORKDIR/.local/share/vocalis/venv/bin/python"
}

@test "main --purge removes the Parakeet, Qwen3-TTS, Piper binary and Piper voice directories" {
  mkdir -p "$WORKDIR/.config/vocalis" "$WORKDIR/models/parakeet" "$WORKDIR/models/qwen3" \
    "$WORKDIR/.local/bin/piper" "$WORKDIR/.local/share/piper-voices"
  touch "$WORKDIR/.local/bin/piper/piper" "$WORKDIR/.local/share/piper-voices/en_US-lessac-medium.onnx"
  write_purge_python_stub
  HOME="$WORKDIR" run main --purge
  [ "$status" -eq 0 ]
  [ ! -e "$WORKDIR/.config/vocalis" ]
  [ ! -e "$WORKDIR/models/parakeet" ]
  [ ! -e "$WORKDIR/models/qwen3" ]
  [ ! -e "$WORKDIR/.local/bin/piper" ]
  [ ! -e "$WORKDIR/.local/share/piper-voices" ]
  [ -d "$WORKDIR/.local/bin" ]
  [ -d "$WORKDIR/.local/share" ]
}

@test "main --purge leaves Piper alone when the venv cannot report its paths" {
  mkdir -p "$WORKDIR/.local/share/vocalis/venv/bin" "$WORKDIR/.local/bin/piper"
  touch "$WORKDIR/.local/bin/piper/piper"
  printf '#!/usr/bin/env bash\nexit 1\n' > "$WORKDIR/.local/share/vocalis/venv/bin/python"
  chmod +x "$WORKDIR/.local/share/vocalis/venv/bin/python"
  HOME="$WORKDIR" run main --purge
  [ "$status" -eq 0 ]
  [ -d "$WORKDIR/.local/bin/piper" ]
}

@test "main --purge also removes the Kokoro model directory" {
  mkdir -p "$WORKDIR/kokoro-dir"
  touch "$WORKDIR/kokoro-dir/kokoro-v1.0.onnx"
  write_purge_python_stub
  HOME="$WORKDIR" run main --purge
  [ "$status" -eq 0 ]
  [ ! -d "$WORKDIR/kokoro-dir" ]
}
