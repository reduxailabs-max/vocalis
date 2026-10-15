#!/usr/bin/env bats

setup() {
  source "${BATS_TEST_DIRNAME}/../../packaging/installer/lib/desktop.sh"
  WORKDIR="$(mktemp -d)"
  FIXTURE="${BATS_TEST_DIRNAME}/fixtures/vocalis.desktop"
}

teardown() { rm -rf "$WORKDIR"; }

@test "desktop_install_launcher fills in the command and creates the folder" {
  local out="$WORKDIR/share/applications/vocalis.desktop"
  desktop_install_launcher "$FIXTURE" "/home/ex/.local/bin/vocalis" "$out"
  grep -q '^Exec="/home/ex/.local/bin/vocalis" --settings$' "$out"
  ! grep -q "__VOCALIS_EXEC__" "$out"
}

@test "desktop_install_launcher quotes a path that contains spaces" {
  local out="$WORKDIR/vocalis.desktop"
  desktop_install_launcher "$FIXTURE" "/home/my user/.local/bin/vocalis" "$out"
  grep -q '^Exec="/home/my user/.local/bin/vocalis" --settings$' "$out"
}

@test "desktop_install_launcher leaves an existing launcher untouched on failure" {
  local out="$WORKDIR/vocalis.desktop"
  echo "old" > "$out"
  run desktop_install_launcher "$WORKDIR/missing.desktop" "/x/vocalis" "$out"
  [ "$status" -ne 0 ]
  [ "$(cat "$out")" = "old" ]
  [ "$(find "$WORKDIR" -mindepth 1 | wc -l)" -eq 1 ]
}
