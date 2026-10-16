#!/usr/bin/env bash
# Provisions the private Python environment Vocalis runs in: installs
# `uv` if needed, finds or builds the Vocalis wheel, creates a venv
# with the right Python version, and installs the wheel into it.
set -uo pipefail

UV_PINNED_VERSION="0.10.8"

# Wrapper around the official, version-pinned uv installer, overridable
# by tests. Pinned (not a bare "latest" URL) for reproducibility.
pyenv_run_install_uv() {
  curl -LsSf "https://astral.sh/uv/${UV_PINNED_VERSION}/install.sh" | sh
}

# Install uv if it is not already available, either on PATH or at its
# default install location. Always puts uv's default install location
# (~/.local/bin) on PATH for the rest of this session, so a uv that was
# only just installed (or that lives there but isn't on PATH yet) is
# found by every later `uv` call in the same run.
pyenv_ensure_uv() {
  export PATH="${HOME}/.local/bin:${PATH}"
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi
  echo "Installing uv (a fast Python package manager) ..."
  pyenv_run_install_uv
}

# Wrapper around `uv build`, overridable by tests.
pyenv_run_uv_build() {
  local repo_root="$1"
  (cd "$repo_root" && uv build)
}

# Return the path to a Vocalis wheel: one already bundled alongside the
# installer script (the normal case when run from a packaged .run
# file), or else one freshly built from the given repo checkout (the
# normal case when running install.sh straight from a dev checkout).
pyenv_find_or_build_wheel() {
  local script_dir="$1"
  local repo_root="$2"
  local bundled
  bundled="$(find "${script_dir}/dist" -maxdepth 1 -name 'vocalis-*.whl' 2>/dev/null | sort | tail -n1)"
  if [[ -n "$bundled" ]]; then
    echo "$bundled"
    return 0
  fi
  if ! pyenv_run_uv_build "$repo_root" >&2; then
    echo "Could not build the Vocalis package from ${repo_root}." >&2
    return 1
  fi
  local built
  built="$(find "${repo_root}/dist" -maxdepth 1 -name 'vocalis-*.whl' 2>/dev/null | sort | tail -n1)"
  if [[ -z "$built" ]]; then
    echo "No Vocalis wheel was found in ${repo_root}/dist after building." >&2
    return 1
  fi
  echo "$built"
}

# Wrapper around every `uv` invocation below, overridable by tests.
pyenv_run_uv() {
  uv "$@"
}

# Ensure the given Python version is available to uv, then create a
# venv at venv_dir using it. Safe to re-run: a venv whose python
# already works is left alone; a venv folder whose python is broken
# or missing is recreated from scratch (--clear, never a y/n prompt).
pyenv_create_venv() {
  local venv_dir="$1"
  local python_version="${2:-3.13}"
  if "${venv_dir}/bin/python" --version >/dev/null 2>&1; then
    echo "Python environment already present at ${venv_dir}"
    return 0
  fi
  pyenv_run_uv python install "$python_version" || return 1
  if [[ -e "$venv_dir" ]]; then
    pyenv_run_uv venv --clear --python "$python_version" "$venv_dir"
  else
    pyenv_run_uv venv --python "$python_version" "$venv_dir"
  fi
}

# Install the given wheel into the given venv's own interpreter
# (no "activate" step needed).
pyenv_install_vocalis() {
  local venv_dir="$1"
  local wheel_path="$2"
  pyenv_run_uv pip install --python "${venv_dir}/bin/python" "$wheel_path"
}

# Install kokoro-onnx WITHOUT its dependencies. It hard-requires plain
# onnxruntime, which would silently remove the graphics card from
# Parakeet if installed next to onnxruntime-gpu; its other dependencies
# are listed in Vocalis's own pyproject.toml.
pyenv_install_kokoro() {
  local venv_dir="$1"
  pyenv_run_uv pip install --python "${venv_dir}/bin/python" --no-deps "kokoro-onnx==0.4.7"
}
