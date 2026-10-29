#!/usr/bin/env bash
# Checks run before anything else: is this a supported distro and a
# supported CPU architecture (both hard requirements)? A working NVIDIA
# GPU driver is optional: without one the install continues with a NOTE
# and Vocalis uses its lighter voices on the processor. Vocalis never
# tries to install a GPU driver itself (too risky/variable to automate).
set -uo pipefail

# Whether /etc/os-release (or the given file) identifies a Debian- or
# Ubuntu-family distro, via its ID or ID_LIKE field.
preflight_check_distro() {
  local os_release_file="${1:-/etc/os-release}"
  if [[ ! -r "$os_release_file" ]]; then
    return 1
  fi
  local content
  content="$(tr '[:upper:]' '[:lower:]' < "$os_release_file")"
  if [[ "$content" == *"id=debian"* || "$content" == *"id=ubuntu"* || "$content" == *"id_like=debian"* || "$content" == *"id_like=ubuntu"* ]]; then
    return 0
  fi
  return 1
}

# Wrapper around `uname -m`, overridable by tests.
preflight_run_uname_m() {
  uname -m
}

# Whether this machine is 64-bit Intel/AMD (x86_64).
preflight_check_arch() {
  [[ "$(preflight_run_uname_m)" == "x86_64" ]]
}

# Wrapper around running `nvidia-smi`, overridable by tests.
preflight_run_nvidia_smi() {
  nvidia-smi >/dev/null 2>&1
}

# Whether a working NVIDIA GPU driver is present (nvidia-smi runs and
# succeeds). Vocalis needs CUDA for its voice engine and never installs
# a driver itself.
preflight_check_gpu() {
  preflight_run_nvidia_smi
}

# Run the preflight checks in order; distro and arch failures stop with
# a message, a missing NVIDIA driver only prints a NOTE line.
preflight_run() {
  if ! preflight_check_distro; then
    echo "Vocalis only supports Debian- or Ubuntu-based Linux systems. This machine's /etc/os-release doesn't look like either." >&2
    return 1
  fi
  if ! preflight_check_arch; then
    echo "Vocalis only supports 64-bit Intel/AMD machines (x86_64). This machine reports $(preflight_run_uname_m)." >&2
    return 1
  fi
  if ! preflight_check_gpu; then
    echo "NOTE No NVIDIA graphics card driver was found. Vocalis will use its lighter voices on your processor."
  fi
  return 0
}
