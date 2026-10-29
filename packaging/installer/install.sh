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
# than exactly "yes" or "no") means "no graphics card": the safe CPU path,
# and a NOTE line says so. The plan's own error text is kept out of the way.
install_plan_wants_gpu() {
  local venv_dir="$1"
  local answer
  if ! answer="$("${venv_dir}/bin/python" -m vocalis.installer.plan gpu-runtime 2>/dev/null)"; then
    echo "NOTE The hardware check did not work, so I will use the processor-only speech runtime (the safe choice)."
    return 1
  fi
  if [[ "$answer" == "yes" ]]; then
    return 0
  fi
  if [[ "$answer" != "no" ]]; then
    echo "NOTE The hardware check gave an unexpected answer, so I will use the processor-only speech runtime (the safe choice)."
  fi
  return 1
}

# Fill the global array INSTALL_FETCH_FLAGS with the extra flags for
# fetch_models.py. Only empty output or exactly "--skip-qwen3" is trusted;
# a failing or odd answer means "--skip-qwen3" (plus a NOTE), so a failure
# never causes the huge Qwen3-TTS download. Runs in the caller's shell (not
# a subshell) so the array survives.
install_fetch_flags() {
  local venv_dir="$1"
  local flags
  INSTALL_FETCH_FLAGS=()
  if ! flags="$("${venv_dir}/bin/python" -m vocalis.installer.plan fetch-flags 2>/dev/null)"; then
    echo "NOTE The hardware check did not work, so I will skip the large Qwen3 voice (the safe choice)."
    INSTALL_FETCH_FLAGS=("--skip-qwen3")
    return 0
  fi
  if [[ -z "$flags" ]]; then
    return 0
  fi
  if [[ "$flags" != "--skip-qwen3" ]]; then
    echo "NOTE The hardware check gave an unexpected answer, so I will skip the large Qwen3 voice (the safe choice)."
  fi
  INSTALL_FETCH_FLAGS=("--skip-qwen3")
}

# Run the setup wizard as a CHILD process (not exec) so the installer
# regains control if the window cannot open. Returns the wizard's exit
# status. Contract with vocalis-setup:
#   0   finished, or the user closed it normally
#   1   the user stopped it, or an install step failed (message already shown)
#   3   could not open a window
#   126, 127, >=128   could not start / crashed / killed by a signal;
#       treated like 3 by the caller
# Kept as its own function so tests can replace it.
install_launch_wizard() {
  local venv_dir="$1"
  local script_dir="$2"
  local wizard_status=0
  "${venv_dir}/bin/vocalis-setup" --installer-dir "$script_dir" || wizard_status=$?
  return "$wizard_status"
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

  install_fetch_flags "$venv_dir"

  echo "STEP 1/2 Downloading voices and the dictation model (a few gigabytes – this can take a while)"
  echo "Checking Parakeet model at ${parakeet_dir} ..."
  if [[ ${#INSTALL_FETCH_FLAGS[@]} -eq 0 ]]; then
    echo "Checking Qwen3-TTS model at ${qwen3_tts_dir} ..."
  fi
  echo "Checking Piper voice at ${piper_voice} ..."
  echo "Checking Kokoro voice at ${kokoro_model} ..."
  "$venv_python" -u "${script_dir}/lib/fetch_models.py" ${INSTALL_FETCH_FLAGS[@]+"${INSTALL_FETCH_FLAGS[@]}"}

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

# Run every phase in order, with a plain heading before each one
# (used by --phase all, --no-wizard and the no-window fallback).
install_run_all() {
  echo "--- Step A: preparing your computer ---"
  install_phase_prepare "$1" "$2"
  install_run_terminal_rest "$1" "$2"
}

# The second and third phases in the terminal, each with a plain heading.
install_run_terminal_rest() {
  echo "--- Step B: downloading voices and models ---"
  install_phase_models "$1"
  echo "--- Step C: starting Vocalis ---"
  install_phase_finish "$1" "$2"
}

# No flags given: prepare, then open the setup wizard when there is a
# screen. Without a screen, or when the wizard cannot open (missing, exit
# 3, 126, 127 or >=128), finish here in the terminal. Exit 0 means the
# wizard did everything; any other status (such as 1) is returned as is.
install_run_default() {
  local script_dir="$1"
  local repo_root="$2"
  echo "--- Step A: preparing your computer ---"
  install_phase_prepare "$script_dir" "$repo_root"

  local venv_dir
  venv_dir="$(install_venv_dir)"
  if [[ -n "${DISPLAY:-}" || -n "${WAYLAND_DISPLAY:-}" ]]; then
    if [[ -x "${venv_dir}/bin/vocalis-setup" ]]; then
      echo "NOTE If no setup window opens, run: bash install.sh --no-wizard"
      local wizard_status=0
      install_launch_wizard "$venv_dir" "$script_dir" || wizard_status=$?
      if [[ "$wizard_status" -eq 0 ]]; then
        return 0
      fi
      if [[ "$wizard_status" -ne 3 && "$wizard_status" -ne 126 && "$wizard_status" -lt 128 && "$wizard_status" -ne 127 ]]; then
        return "$wizard_status"
      fi
    fi
    echo "NOTE The setup window could not open, so I will finish the installation here in the terminal."
  fi
  install_run_terminal_rest "$script_dir" "$repo_root"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
