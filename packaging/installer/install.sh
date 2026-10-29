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
# shellcheck source=lib/desktop.sh
# shellcheck disable=SC1091
source "${_INSTALL_SCRIPT_DIR}/lib/desktop.sh"

# Where the private Python environment lives.
install_venv_dir() {
  echo "${HOME}/.local/share/vocalis/venv"
}

# Ask the hardware plan whether to use the graphics-card build of
# onnxruntime. ANY failure (the command errors, or prints anything other
# than exactly "yes") means "no graphics card": the safe CPU path.
install_plan_wants_gpu() {
  local venv_dir="$1"
  local answer
  answer="$("${venv_dir}/bin/python" -m vocalis.installer.plan gpu-runtime 2>/dev/null)" || return 1
  [[ "$answer" == "yes" ]]
}

# Print the extra flags for fetch_models.py (one per line). Only the exact
# output "--skip-qwen3" or an empty line is trusted; a failing or odd
# answer means "--skip-qwen3", so a failure never causes the huge
# Qwen3-TTS download.
install_fetch_flags() {
  local venv_dir="$1"
  local flags
  flags="$("${venv_dir}/bin/python" -m vocalis.installer.plan fetch-flags 2>/dev/null)" || flags="--skip-qwen3"
  if [[ -n "$flags" && "$flags" != "--skip-qwen3" ]]; then
    flags="--skip-qwen3"
  fi
  if [[ -n "$flags" ]]; then
    echo "$flags"
  fi
  return 0
}

# Hand the terminal over to the setup wizard. Kept as its own function so
# tests can replace it (the real one never returns).
install_launch_wizard() {
  local venv_dir="$1"
  local script_dir="$2"
  exec "${venv_dir}/bin/vocalis-setup" --installer-dir "$script_dir"
}

# Phase 1: system packages, Python environment, Vocalis itself.
install_phase_prepare() {
  local script_dir="$1"
  local repo_root="$2"
  local venv_dir
  venv_dir="$(install_venv_dir)"

  echo "STEP 1/5 Checking your computer"
  if ! preflight_run; then
    exit 1
  fi

  echo "STEP 2/5 Installing system packages"
  if ! apt_install_missing "${APT_REQUIRED_PACKAGES[@]}" || ! apt_enable_ydotoold; then
    echo "Could not install required system packages. Please check your internet connection and password, then try again." >&2
    exit 1
  fi

  echo "STEP 3/5 Preparing the Python tools"
  pyenv_ensure_uv
  local wheel_path
  wheel_path="$(pyenv_find_or_build_wheel "$script_dir" "$repo_root")"

  echo "STEP 4/5 Installing Vocalis"
  pyenv_create_venv "$venv_dir"
  pyenv_install_vocalis "$venv_dir" "$wheel_path"
  pyenv_install_kokoro "$venv_dir"
  if install_plan_wants_gpu "$venv_dir"; then
    pyenv_prefer_gpu_onnxruntime "$venv_dir"
  else
    pyenv_use_cpu_onnxruntime "$venv_dir"
  fi

  echo "STEP 5/5 Adding the vocalis command"
  mkdir -p "${HOME}/.local/bin"
  rm -f "${HOME}/.local/bin/vocalis"
  ln -s "${venv_dir}/bin/vocalis" "${HOME}/.local/bin/vocalis"
}

# Phase 2: voice and dictation models, plus the Piper program.
install_phase_models() {
  local script_dir="$1"
  local venv_dir venv_python
  venv_dir="$(install_venv_dir)"
  venv_python="${venv_dir}/bin/python"

  local parakeet_dir qwen3_tts_dir piper_binary piper_voice kokoro_model
  parakeet_dir="$("$venv_python" -c 'from vocalis.config import VocalisConfig; print(VocalisConfig().engines.parakeet_model_dir)')"
  qwen3_tts_dir="$("$venv_python" -c 'from vocalis.config import VocalisConfig; print(VocalisConfig().engines.qwen3_tts_model_dir)')"
  piper_binary="$("$venv_python" -c 'from vocalis.config import VocalisConfig; print(VocalisConfig().engines.piper_binary_path)')"
  piper_voice="$("$venv_python" -c 'from vocalis.config import VocalisConfig; print(VocalisConfig().engines.piper_voice_model_path)')"
  kokoro_model="$("$venv_python" -c 'from vocalis.config import VocalisConfig; print(VocalisConfig().engines.kokoro_model_path)')"

  local fetch_flags=()
  local flag
  while IFS= read -r flag; do
    if [[ -n "$flag" ]]; then
      fetch_flags+=("$flag")
    fi
  done < <(install_fetch_flags "$venv_dir")

  echo "STEP 1/2 Downloading voices and the dictation model (a few gigabytes – this can take a while)"
  echo "Checking Parakeet model at ${parakeet_dir} ..."
  echo "Checking Qwen3-TTS model at ${qwen3_tts_dir} ..."
  echo "Checking Piper voice at ${piper_voice} ..."
  echo "Checking Kokoro voice at ${kokoro_model} ..."
  "$venv_python" "${script_dir}/lib/fetch_models.py" ${fetch_flags[@]+"${fetch_flags[@]}"}

  echo "STEP 2/2 Downloading the Piper speech program"
  echo "Checking Piper binary at ${piper_binary} ..."
  models_ensure_piper_binary "$piper_binary"
}

# Phase 3: background service, menu entry, final message.
install_phase_finish() {
  local script_dir="$1"
  local repo_root="$2"

  echo "STEP 1/2 Starting Vocalis"
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

  echo "STEP 2/2 Adding Vocalis to your applications menu"
  # Same script-relative-else-repo-root lookup as the unit template above.
  local desktop_template="${script_dir}/packaging/desktop/vocalis.desktop"
  if [[ ! -f "$desktop_template" ]]; then
    desktop_template="${repo_root}/packaging/desktop/vocalis.desktop"
  fi
  desktop_install_launcher "$desktop_template" "${HOME}/.local/bin/vocalis" "${HOME}/.local/share/applications/vocalis.desktop" \
    || echo "(The menu entry could not be added; Vocalis still works.)"

  echo ""
  echo "Vocalis is installed and running."
  echo "It will start automatically the next time you log in."
  echo "To open it: click the small round Vocalis icon in your panel's system tray,"
  echo "or choose \"Vocalis\" in your applications menu, or run: vocalis --settings"
}

# Usage: main [--phase prepare|models|finish|all] [--no-wizard]
main() {
  # Read fresh on every call (not computed once at source time) so a
  # test can override SCRIPT_DIR/REPO_ROOT in its own environment
  # right before calling main, without touching the real sourcing above.
  local script_dir="${SCRIPT_DIR:-$_INSTALL_SCRIPT_DIR}"
  local repo_root="${REPO_ROOT:-$_INSTALL_REPO_ROOT}"

  local phase=""
  local no_wizard=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --phase)
        if [[ $# -lt 2 ]]; then
          echo "--phase needs a value: prepare, models, finish or all." >&2
          return 2
        fi
        phase="$2"
        shift 2
        ;;
      --no-wizard)
        no_wizard=1
        shift
        ;;
      *)
        echo "Unknown option: $1" >&2
        echo "Usage: install.sh [--phase prepare|models|finish|all] [--no-wizard]" >&2
        return 2
        ;;
    esac
  done

  if [[ "$no_wizard" -eq 1 && -z "$phase" ]]; then
    phase="all"
  fi

  case "$phase" in
    prepare) install_phase_prepare "$script_dir" "$repo_root" ;;
    models) install_phase_models "$script_dir" ;;
    finish) install_phase_finish "$script_dir" "$repo_root" ;;
    all) install_run_all "$script_dir" "$repo_root" ;;
    "") install_run_default "$script_dir" "$repo_root" ;;
    *)
      echo "Unknown phase: ${phase} (use prepare, models, finish or all)." >&2
      return 2
      ;;
  esac
}

install_run_all() {
  install_phase_prepare "$1" "$2"
  install_phase_models "$1"
  install_phase_finish "$1" "$2"
}

# No flags: prepare, then open the setup wizard when there is a screen;
# otherwise (or if the wizard is missing) finish in the terminal.
install_run_default() {
  local script_dir="$1"
  local repo_root="$2"
  install_phase_prepare "$script_dir" "$repo_root"

  local venv_dir
  venv_dir="$(install_venv_dir)"
  if [[ -n "${DISPLAY:-}" || -n "${WAYLAND_DISPLAY:-}" ]]; then
    if [[ -x "${venv_dir}/bin/vocalis-setup" ]]; then
      install_launch_wizard "$venv_dir" "$script_dir"
      return 0
    fi
    echo "NOTE: the setup window could not be started, so the installer will finish here in the terminal."
  fi
  install_phase_models "$script_dir"
  install_phase_finish "$script_dir" "$repo_root"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
