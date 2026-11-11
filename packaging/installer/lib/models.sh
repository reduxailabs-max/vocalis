#!/usr/bin/env bash
# Handles the one model/voice asset not hosted on Hugging Face: the
# Piper binary itself, downloaded from its pinned GitHub release.
# (Parakeet, Qwen3-TTS and the Piper voice are handled by
# fetch_models.py, since they're all on Hugging Face.)
set -uo pipefail

PIPER_RELEASE_URL="https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz"

# Wrapper around downloading the Piper release tarball to the given
# destination path, overridable by tests.
models_run_download_piper() {
  local dest="$1"
  curl -fL -C - --retry 5 --retry-delay 2 --retry-all-errors -sS -o "$dest" "$PIPER_RELEASE_URL"
}

# Reuse-or-download the Piper binary at binary_path (e.g.
# ~/.local/bin/piper/piper). An optional second argument is a tarball that
# fetch_models.py already downloaded (resumably); it is unpacked instead
# of downloading again, and removed once it has been used. Extracts to a temporary staging directory
# first and only moves the completed result into place, so a failed or
# interrupted download/extraction never leaves a partial binary at the
# final path.
models_ensure_piper_binary() {
  local binary_path="$1"
  if [[ -x "$binary_path" ]]; then
    echo "Piper binary already present at $binary_path"
    return 0
  fi

  local bin_dir
  bin_dir="$(dirname "$(dirname "$binary_path")")"  # .../bin/piper/piper -> .../bin
  local given_tarball="${2:-}"
  local tmp_tarball
  if [[ -n "$given_tarball" && -s "$given_tarball" ]]; then
    tmp_tarball="$given_tarball"
  else
    tmp_tarball="$(mktemp --suffix=.tar.gz)"
    given_tarball=""
  fi
  local staging_dir
  staging_dir="$(mktemp -d)"

  if [[ -n "$given_tarball" ]]; then
    echo "Unpacking Piper to $bin_dir ..."
  else
    echo "Downloading Piper to $bin_dir ..."
  fi
  if [[ -z "$given_tarball" ]] && ! models_run_download_piper "$tmp_tarball"; then
    rm -f "$tmp_tarball"
    rm -rf "$staging_dir"
    echo "Could not download Piper from $PIPER_RELEASE_URL" >&2
    return 1
  fi

  if ! tar xzf "$tmp_tarball" -C "$staging_dir"; then
    rm -f "$tmp_tarball"
    rm -rf "$staging_dir"
    echo "Could not unpack the Piper download (the file may be damaged). Please run the installer again." >&2
    return 1
  fi
  rm -f "$tmp_tarball"

  if [[ ! -d "${staging_dir}/piper" ]]; then
    rm -rf "$staging_dir"
    echo "The Piper download did not contain the expected piper folder." >&2
    return 1
  fi

  # Only replace the old copy once the new one is fully unpacked. If the
  # move itself fails, remove whatever half-copied folder it left behind.
  if ! mkdir -p "$bin_dir" \
    || ! rm -rf "${bin_dir}/piper" \
    || ! mv "${staging_dir}/piper" "$bin_dir/"; then
    rm -rf "${bin_dir}/piper" "$staging_dir"
    echo "Could not move Piper into $bin_dir" >&2
    return 1
  fi
  rm -rf "$staging_dir"
}
