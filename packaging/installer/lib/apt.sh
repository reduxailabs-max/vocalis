#!/usr/bin/env bash
# Detects and installs the small system packages Vocalis's non-Python
# features need (OCR, clipboard, typing-at-cursor), and best-effort
# enables the ydotoold background service the ydotool package ships.
set -uo pipefail

export APT_REQUIRED_PACKAGES=(tesseract-ocr xclip ydotool)

# Wrapper around `dpkg -s <package>`, overridable by tests.
apt_run_dpkg_query() {
  dpkg -s "$1" >/dev/null 2>&1
}

# Whether a single apt package is installed.
apt_is_installed() {
  apt_run_dpkg_query "$1"
}

# Print the names of every package in the given list that is not
# installed, one per line.
apt_missing_packages() {
  local package
  for package in "$@"; do
    if ! apt_is_installed "$package"; then
      echo "$package"
    fi
  done
}

# Wrapper around the real, one-password-prompt install command,
# overridable by tests.
apt_run_install() {
  sudo apt-get install -y "$@"
}

# Install every package in the given list that is not already
# installed, in one call (one sudo prompt). Does nothing, and never
# shells out, if everything is already present.
apt_install_missing() {
  local missing=()
  while IFS= read -r package; do
    [[ -n "$package" ]] && missing+=("$package")
  done < <(apt_missing_packages "$@")
  if [[ "${#missing[@]}" -eq 0 ]]; then
    return 0
  fi
  echo "Installing missing system packages: ${missing[*]}"
  apt_run_install "${missing[@]}"
}

# Wrappers around the two systemctl calls, overridable by tests.
apt_run_systemctl_list_unit_files() {
  systemctl --user list-unit-files "$1" 2>/dev/null
}

apt_run_systemctl_enable_now() {
  systemctl --user enable --now "$1"
}

# Best-effort: if the ydotool package's systemd user unit exists,
# enable and start it (dictation needs ydotoold to type at the
# cursor). Never fails the install -- a failure here just means the
# existing startup-check notification tells the user later.
apt_enable_ydotoold() {
  local listing
  listing="$(apt_run_systemctl_list_unit_files ydotool.service)"
  if [[ "$listing" != *"ydotool.service"* ]]; then
    return 0
  fi
  if ! apt_run_systemctl_enable_now ydotool.service; then
    echo "Could not enable ydotoold automatically; dictation will fall back to copying text to the clipboard." >&2
  fi
  return 0
}
