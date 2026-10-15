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
