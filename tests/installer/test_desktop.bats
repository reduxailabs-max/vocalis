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

@test "desktop_install_icons copies every size into the icon theme" {
  mkdir -p "$WORKDIR/src"
  touch "$WORKDIR/src/vocalis-16.png" "$WORKDIR/src/vocalis-256.png" "$WORKDIR/src/vocalis-mark.png"
  desktop_install_icons "$WORKDIR/src" "$WORKDIR/hicolor"
  [ -f "$WORKDIR/hicolor/16x16/apps/vocalis.png" ]
  [ -f "$WORKDIR/hicolor/256x256/apps/vocalis.png" ]
  [ ! -e "$WORKDIR/hicolor/mark" ]
}

@test "desktop_install_icons fails clearly when there are no icons" {
  mkdir -p "$WORKDIR/empty"
  run desktop_install_icons "$WORKDIR/empty" "$WORKDIR/hicolor"
  [ "$status" -ne 0 ]
}

@test "desktop_remove_icons removes only the Vocalis icons" {
  mkdir -p "$WORKDIR/hicolor/16x16/apps"
  touch "$WORKDIR/hicolor/16x16/apps/vocalis.png" "$WORKDIR/hicolor/16x16/apps/other.png"
  desktop_remove_icons "$WORKDIR/hicolor"
  [ ! -e "$WORKDIR/hicolor/16x16/apps/vocalis.png" ]
  [ -f "$WORKDIR/hicolor/16x16/apps/other.png" ]
}
