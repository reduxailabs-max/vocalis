#!/usr/bin/env bash
# Builds the Vocalis wheel and wraps install.sh/uninstall.sh (plus the
# wheel and lib/ helpers they need) into two self-extracting .run files
# via makeself, for people who'd rather double-click one file than run
# a shell script by hand. Developer-only tool; never run as part of a
# real install.
set -euo pipefail

# Real, non-overridable location, used only to find lib/pyenv.sh.
# main() below reads its own overridable copies, the same pattern
# install.sh uses and for the same reason (see its comment).
_BUILD_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_BUILD_REPO_ROOT="$(cd "${_BUILD_SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=lib/pyenv.sh
# shellcheck disable=SC1091
source "${_BUILD_SCRIPT_DIR}/lib/pyenv.sh"

# Wrapper around the real makeself invocation, overridable by tests.
# label: human label shown when the .run file is extracted.
# staging_dir: the directory whose contents become the payload.
# entry_script: which file inside staging_dir makeself should run.
# output_path: where to write the finished .run file.
build_run_run_makeself() {
  local label="$1"
  local staging_dir="$2"
  local entry_script="$3"
  local output_path="$4"
  makeself "$staging_dir" "$output_path" "$label" "./${entry_script}"
}

main() {
  local script_dir="${SCRIPT_DIR:-$_BUILD_SCRIPT_DIR}"
  local repo_root="${REPO_ROOT:-$_BUILD_REPO_ROOT}"

  local wheel_dist_dir="${script_dir}/dist"
  mkdir -p "$wheel_dist_dir"
  pyenv_run_uv_build "$repo_root"
  cp "${repo_root}"/dist/vocalis-*.whl "$wheel_dist_dir/"

  local staging_dir
  staging_dir="$(mktemp -d)"
  cp -r "${script_dir}/lib" "$staging_dir/"
  cp "${script_dir}/install.sh" "${script_dir}/uninstall.sh" "$staging_dir/"
  chmod +x "${staging_dir}/install.sh" "${staging_dir}/uninstall.sh"
  mkdir -p "$staging_dir/dist"
  cp "$wheel_dist_dir"/vocalis-*.whl "$staging_dir/dist/"
  mkdir -p "$staging_dir/packaging/systemd"
  cp "${repo_root}/packaging/systemd/vocalis.service" "$staging_dir/packaging/systemd/"
  mkdir -p "$staging_dir/packaging/desktop"
  cp "${repo_root}/packaging/desktop/vocalis.desktop" "$staging_dir/packaging/desktop/"

  build_run_run_makeself "Vocalis installer" "$staging_dir" "install.sh" "${script_dir}/install.run"
  build_run_run_makeself "Vocalis uninstaller" "$staging_dir" "uninstall.sh" "${script_dir}/uninstall.run"

  rm -rf "$staging_dir"
  echo "Built ${script_dir}/install.run and ${script_dir}/uninstall.run"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
