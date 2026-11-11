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
  service_run_systemctl_is_active() { return 1; }
  service_run_systemctl_daemon_reload() { echo "daemon-reload" >> "$CALL_LOG"; }
  service_run_systemctl_enable_now() { echo "enable-now: $1" >> "$CALL_LOG"; }
  service_enable_and_start "vocalis.service"
  [ "$(sed -n '1p' "$CALL_LOG")" = "daemon-reload" ]
  [ "$(sed -n '2p' "$CALL_LOG")" = "enable-now: vocalis.service" ]
}

@test "service_enable_and_start defaults to vocalis.service" {
  service_run_systemctl_is_active() { return 1; }
  service_run_systemctl_daemon_reload() { :; }
  service_run_systemctl_enable_now() { echo "enable-now: $1" >> "$CALL_LOG"; }
  service_enable_and_start
  grep -q "enable-now: vocalis.service" "$CALL_LOG"
}

@test "service_substitute_unit leaves an existing unit file untouched when substitution fails" {
  local unit_path="$WORKDIR/systemd/user/vocalis.service"
  mkdir -p "$(dirname "$unit_path")"
  echo "ExecStart=/previous/good/vocalis" > "$unit_path"
  run service_substitute_unit "$WORKDIR/no-such-template.service" "/x/vocalis" "/x" "$unit_path"
  [ "$status" -ne 0 ]
  [ "$(cat "$unit_path")" = "ExecStart=/previous/good/vocalis" ]
  # No half-written temp files left beside it.
  [ "$(find "$(dirname "$unit_path")" -mindepth 1 | wc -l)" -eq 1 ]
}

@test "service_enable_and_start does not restart a service that was not running" {
  service_run_systemctl_is_active() { return 1; }
  service_run_systemctl_daemon_reload() { :; }
  service_run_systemctl_enable_now() { :; }
  service_run_systemctl_restart() { echo "restart: $1" >> "$CALL_LOG"; }
  service_enable_and_start
  [ ! -e "$CALL_LOG" ] || ! grep -q restart "$CALL_LOG"
}

@test "service_enable_and_start restarts a service that was already running" {
  service_run_systemctl_is_active() { return 0; }
  service_run_systemctl_daemon_reload() { :; }
  service_run_systemctl_enable_now() { echo "enable-now: $1" >> "$CALL_LOG"; }
  service_run_systemctl_restart() { echo "restart: $1" >> "$CALL_LOG"; }
  service_enable_and_start
  [ "$(sed -n '1p' "$CALL_LOG")" = "enable-now: vocalis.service" ]
  [ "$(sed -n '2p' "$CALL_LOG")" = "restart: vocalis.service" ]
}
