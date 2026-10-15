#!/usr/bin/env bash
# Hard-gate checks run before anything else: is this a supported distro,
# a supported CPU architecture, and does the machine have a working
# NVIDIA GPU driver? Vocalis never tries to install a GPU driver itself
# (too risky/variable to automate safely) -- it only checks and refuses
# with a plain message if one isn't already working.
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

# Run every preflight check in order; stop and report at the first
# failure (each is a hard requirement, unlike vocalis's own runtime
# startup_checks, which are all independent and non-fatal).
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
    echo "Vocalis needs a working NVIDIA graphics card driver. Please install your GPU driver first (running 'nvidia-smi' should work without an error), then run this installer again." >&2
    return 1
  fi
  return 0
}
