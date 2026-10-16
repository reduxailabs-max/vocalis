#!/usr/bin/env bash
# Reverses install.sh. By default keeps your config and downloaded
# models (several GB); pass --purge to remove those too.
set -uo pipefail

uninstall_run_systemctl_disable_now() {
  systemctl --user disable --now "$1" >/dev/null 2>&1
}

uninstall_run_systemctl_daemon_reload() {
  systemctl --user daemon-reload
}

uninstall_print_usage() {
  echo "Usage: uninstall.sh [--purge]"
  echo "  (no flag)  Remove Vocalis but keep your config and downloaded models."
  echo "  --purge    Also remove your config, downloaded models, Piper and its voices, and the Kokoro voice."
  echo "  -h, --help Show this help and exit without removing anything."
}

main() {
  local purge=0
  case "${1:-}" in
    "") ;;
    --purge) purge=1 ;;
    -h|--help)
      uninstall_print_usage
      return 0
      ;;
    *)
      echo "Unrecognised option: $1 (nothing was removed)" >&2
      uninstall_print_usage >&2
      return 1
      ;;
  esac

  local found_anything=0
  local unit_file="${HOME}/.config/systemd/user/vocalis.service"
  local symlink="${HOME}/.local/bin/vocalis"
  local venv_dir="${HOME}/.local/share/vocalis/venv"
  local venv_python="${venv_dir}/bin/python"
  local launcher="${HOME}/.local/share/applications/vocalis.desktop"

  local config_dir="" parakeet_dir="" qwen3_tts_dir="" piper_binary="" piper_voice=""
  local piper_binary_dir="" piper_voice_dir="" kokoro_model="" kokoro_dir=""
  if [[ -x "$venv_python" ]]; then
    found_anything=1
    if [[ "$purge" -eq 1 ]]; then
      config_dir="${HOME}/.config/vocalis"
      parakeet_dir="$("$venv_python" -c 'from vocalis.config import VocalisConfig; print(VocalisConfig().engines.parakeet_model_dir)' 2>/dev/null || true)"
      qwen3_tts_dir="$("$venv_python" -c 'from vocalis.config import VocalisConfig; print(VocalisConfig().engines.qwen3_tts_model_dir)' 2>/dev/null || true)"
      piper_binary="$("$venv_python" -c 'from vocalis.config import VocalisConfig; print(VocalisConfig().engines.piper_binary_path)' 2>/dev/null || true)"
      piper_voice="$("$venv_python" -c 'from vocalis.config import VocalisConfig; print(VocalisConfig().engines.piper_voice_model_path)' 2>/dev/null || true)"
      kokoro_model="$("$venv_python" -c 'from vocalis.config import VocalisConfig; print(VocalisConfig().engines.kokoro_model_path)' 2>/dev/null || true)"
      # Only derive a parent dir from a real answer: dirname "" is ".",
      # which must never be passed to rm -rf.
      [[ -n "$piper_binary" ]] && piper_binary_dir="$(dirname "$piper_binary")"
      [[ -n "$piper_voice" ]] && piper_voice_dir="$(dirname "$piper_voice")"
      [[ -n "$kokoro_model" ]] && kokoro_dir="$(dirname "$kokoro_model")"
    fi
  fi

  uninstall_run_systemctl_disable_now vocalis.service

  if [[ -f "$unit_file" ]]; then
    found_anything=1
    rm -f "$unit_file"
    uninstall_run_systemctl_daemon_reload
  fi

  if [[ -e "$symlink" ]]; then
    found_anything=1
    rm -f "$symlink"
  fi

  if [[ -e "$launcher" ]]; then
    found_anything=1
    rm -f "$launcher"
  fi

  if [[ -d "$venv_dir" ]]; then
    found_anything=1
    rm -rf "$venv_dir"
  fi

  if [[ "$found_anything" -eq 0 ]]; then
    echo "Vocalis does not appear to be installed; nothing to remove."
    return 0
  fi

  if [[ "$purge" -eq 1 ]]; then
    [[ -n "$config_dir" && -d "$config_dir" ]] && rm -rf "$config_dir"
    [[ -n "$parakeet_dir" && -d "$parakeet_dir" ]] && rm -rf "$parakeet_dir"
    [[ -n "$qwen3_tts_dir" && -d "$qwen3_tts_dir" ]] && rm -rf "$qwen3_tts_dir"
    [[ -n "$piper_binary_dir" && -d "$piper_binary_dir" ]] && rm -rf "$piper_binary_dir"
    [[ -n "$piper_voice_dir" && -d "$piper_voice_dir" ]] && rm -rf "$piper_voice_dir"
    [[ -n "$kokoro_dir" && -d "$kokoro_dir" ]] && rm -rf "$kokoro_dir"
    echo "Vocalis has been removed, including your config, downloaded models, Piper and its voices, and the Kokoro voice."
  else
    echo "Vocalis has been removed. Your config (~/.config/vocalis) and downloaded models were kept."
    echo "Run with --purge to remove those too."
  fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
