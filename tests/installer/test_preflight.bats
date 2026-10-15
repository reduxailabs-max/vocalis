#!/usr/bin/env bats

setup() {
  source "${BATS_TEST_DIRNAME}/../../packaging/installer/lib/preflight.sh"
  FIXTURES="${BATS_TEST_DIRNAME}/fixtures"
}

@test "distro check passes for a debian os-release file" {
  run preflight_check_distro "${FIXTURES}/os-release-debian"
  [ "$status" -eq 0 ]
}

@test "distro check passes for an ubuntu os-release file (ID_LIKE=debian)" {
  run preflight_check_distro "${FIXTURES}/os-release-ubuntu"
  [ "$status" -eq 0 ]
}

@test "distro check fails for a non-debian os-release file" {
  run preflight_check_distro "${FIXTURES}/os-release-fedora"
  [ "$status" -eq 1 ]
}

@test "distro check fails when the file does not exist" {
  run preflight_check_distro "${FIXTURES}/does-not-exist"
  [ "$status" -eq 1 ]
}

@test "arch check passes when uname -m reports x86_64" {
  preflight_run_uname_m() { echo "x86_64"; }
  run preflight_check_arch
  [ "$status" -eq 0 ]
}

@test "arch check fails when uname -m reports something else" {
  preflight_run_uname_m() { echo "aarch64"; }
  run preflight_check_arch
  [ "$status" -eq 1 ]
}

@test "gpu check passes when nvidia-smi succeeds" {
  preflight_run_nvidia_smi() { return 0; }
  run preflight_check_gpu
  [ "$status" -eq 0 ]
}

@test "gpu check fails when nvidia-smi fails" {
  preflight_run_nvidia_smi() { return 1; }
  run preflight_check_gpu
  [ "$status" -eq 1 ]
}

@test "preflight_run passes when every check passes" {
  preflight_check_distro() { return 0; }
  preflight_check_arch() { return 0; }
  preflight_check_gpu() { return 0; }
  run preflight_run
  [ "$status" -eq 0 ]
}

@test "preflight_run stops at the first failing check and prints a plain message" {
  preflight_check_distro() { return 0; }
  preflight_check_arch() { return 1; }
  preflight_check_gpu() { echo "SHOULD_NOT_RUN" >&2; return 0; }
  run preflight_run
  [ "$status" -eq 1 ]
  [[ "$output" != *"SHOULD_NOT_RUN"* ]]
  [[ "$output" == *"64-bit"* ]]
}

@test "preflight_run reports the GPU driver message when only the GPU check fails" {
  preflight_check_distro() { return 0; }
  preflight_check_arch() { return 0; }
  preflight_check_gpu() { return 1; }
  run preflight_run
  [ "$status" -eq 1 ]
  [[ "$output" == *"NVIDIA"* ]]
}
