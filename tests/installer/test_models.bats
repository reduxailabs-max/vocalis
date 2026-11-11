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

@test "models_ensure_piper_binary never leaves a partial result on the final path if the download fails" {
  local binary_path="$WORKDIR/bin/piper/piper"
  models_run_download_piper() { return 1; }
  run models_ensure_piper_binary "$binary_path"
  [ "$status" -ne 0 ]
  [ ! -e "$binary_path" ]
}

@test "models_ensure_piper_binary fails cleanly when the download is not a valid tarball" {
  local binary_path="$WORKDIR/bin/piper/piper"
  models_run_download_piper() { echo "this is not a tarball" > "$1"; }
  run models_ensure_piper_binary "$binary_path"
  [ "$status" -ne 0 ]
  [[ "$output" == *"Could not unpack"* ]]
  [ ! -e "$binary_path" ]
  [ ! -e "$WORKDIR/bin/piper" ]
}

@test "models_ensure_piper_binary fails cleanly when the tarball has no piper folder" {
  local binary_path="$WORKDIR/bin/piper/piper"
  models_run_download_piper() {
    local dest="$1"
    local staging
    staging="$(mktemp -d)"
    mkdir -p "$staging/something-else"
    (cd "$staging" && tar czf "$dest" something-else)
    rm -rf "$staging"
  }
  run models_ensure_piper_binary "$binary_path"
  [ "$status" -ne 0 ]
  [ ! -e "$binary_path" ]
  [ ! -e "$WORKDIR/bin/piper" ]
}

make_piper_tarball() {
  local dest="$1" staging
  staging="$(mktemp -d)"
  mkdir -p "$staging/piper"
  install -m 755 /bin/true "$staging/piper/piper"
  tar czf "$dest" -C "$staging" piper
  rm -rf "$staging"
}

@test "models_ensure_piper_binary unpacks a given tarball without downloading, then removes it" {
  local binary_path="$WORKDIR/bin/piper/piper"
  make_piper_tarball "$WORKDIR/given.tar.gz"
  models_run_download_piper() { echo "CALLED" >> "$CALL_LOG"; }
  run models_ensure_piper_binary "$binary_path" "$WORKDIR/given.tar.gz"
  [ "$status" -eq 0 ]
  [ -x "$binary_path" ]
  [ ! -s "$CALL_LOG" ]
  [ ! -e "$WORKDIR/given.tar.gz" ]
  [[ "$output" == *"Unpacking Piper"* ]]
}

@test "models_ensure_piper_binary downloads when the given tarball does not exist" {
  local binary_path="$WORKDIR/bin/piper/piper"
  models_run_download_piper() { make_piper_tarball "$1"; }
  models_ensure_piper_binary "$binary_path" "$WORKDIR/missing.tar.gz"
  [ -x "$binary_path" ]
}

@test "the curl fallback resumes and retries" {
  run grep -c -e "-C - --retry 5 --retry-delay 2 --retry-all-errors" "${BATS_TEST_DIRNAME}/../../packaging/installer/lib/models.sh"
  [ "$output" = "1" ]
}
