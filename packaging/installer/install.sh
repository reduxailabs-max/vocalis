#!/usr/bin/env bash
# Installs Vocalis end-to-end on a Debian/Ubuntu-family x86_64 machine:
# system packages, a private Python environment, the voice/dictation
# models, and the background service, all idempotent (safe to re-run
# as a repair command).
set -euo pipefail

# Real, non-overridable location of this script and the repo it lives
# in -- used ONLY to find the lib/ files to source below. main()
# below reads its OWN overridable copies of these two paths, computed
# fresh on every call, so a test can redirect them (by setting
# SCRIPT_DIR/REPO_ROOT before calling main) without this top-level
# sourcing ever needing to change.
_INSTALL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_INSTALL_REPO_ROOT="$(cd "${_INSTALL_SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=lib/preflight.sh
# shellcheck disable=SC1091
source "${_INSTALL_SCRIPT_DIR}/lib/preflight.sh"
# shellcheck source=lib/apt.sh
# shellcheck disable=SC1091
source "${_INSTALL_SCRIPT_DIR}/lib/apt.sh"
# shellcheck source=lib/pyenv.sh
# shellcheck disable=SC1091
source "${_INSTALL_SCRIPT_DIR}/lib/pyenv.sh"
# shellcheck source=lib/models.sh
# shellcheck disable=SC1091
source "${_INSTALL_SCRIPT_DIR}/lib/models.sh"
# shellcheck source=lib/service.sh
# shellcheck disable=SC1091
source "${_INSTALL_SCRIPT_DIR}/lib/service.sh"

main() {
  # Read fresh on every call (not computed once at source time) so a
  # test can override SCRIPT_DIR/REPO_ROOT in its own environment
  # right before calling main, without touching the real sourcing above.
  local script_dir="${SCRIPT_DIR:-$_INSTALL_SCRIPT_DIR}"
  local repo_root="${REPO_ROOT:-$_INSTALL_REPO_ROOT}"

  if ! preflight_run; then
    exit 1
  fi

  if ! apt_install_missing "${APT_REQUIRED_PACKAGES[@]}" || ! apt_enable_ydotoold; then
    echo "Could not install required system packages. Please check your internet connection and password, then try again." >&2
    exit 1
  fi

  pyenv_ensure_uv

  local wheel_path
  wheel_path="$(pyenv_find_or_build_wheel "$script_dir" "$repo_root")"

  local venv_dir="${HOME}/.local/share/vocalis/venv"
  pyenv_create_venv "$venv_dir"
  pyenv_install_vocalis "$venv_dir" "$wheel_path"

  mkdir -p "${HOME}/.local/bin"
  rm -f "${HOME}/.local/bin/vocalis"
  ln -s "${venv_dir}/bin/vocalis" "${HOME}/.local/bin/vocalis"

  local venv_python="${venv_dir}/bin/python"
  local parakeet_dir qwen3_tts_dir piper_binary piper_voice
  parakeet_dir="$("$venv_python" -c 'from vocalis.config import VocalisConfig; print(VocalisConfig().engines.parakeet_model_dir)')"
  qwen3_tts_dir="$("$venv_python" -c 'from vocalis.config import VocalisConfig; print(VocalisConfig().engines.qwen3_tts_model_dir)')"
  piper_binary="$("$venv_python" -c 'from vocalis.config import VocalisConfig; print(VocalisConfig().engines.piper_binary_path)')"
  piper_voice="$("$venv_python" -c 'from vocalis.config import VocalisConfig; print(VocalisConfig().engines.piper_voice_model_path)')"

  echo "Checking Parakeet model at ${parakeet_dir} ..."
  echo "Checking Qwen3-TTS model at ${qwen3_tts_dir} ..."
  echo "Checking Piper voice at ${piper_voice} ..."
  "$venv_python" "${script_dir}/lib/fetch_models.py"

  echo "Checking Piper binary at ${piper_binary} ..."
  models_ensure_piper_binary "$piper_binary"

  # Inside an extracted .run file the unit template is staged right
  # next to this script (build-run.sh puts it there); in a plain dev
  # checkout it lives at the repo root instead.
  local unit_template="${script_dir}/packaging/systemd/vocalis.service"
  if [[ ! -f "$unit_template" ]]; then
    unit_template="${repo_root}/packaging/systemd/vocalis.service"
  fi
  local unit_output="${HOME}/.config/systemd/user/vocalis.service"
  service_substitute_unit "$unit_template" "${HOME}/.local/bin/vocalis" "$HOME" "$unit_output"
  service_enable_and_start

  echo ""
  echo "Vocalis is installed and running."
  echo "It will start automatically the next time you log in."
  echo "To check on it later: systemctl --user status vocalis"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
