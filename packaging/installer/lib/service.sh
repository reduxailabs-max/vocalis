#!/usr/bin/env bash
# Fills in the systemd user unit template with real, installed paths
# and enables/starts it.
set -uo pipefail

# Write output_path with the template's two placeholders replaced by
# real values, creating output_path's parent directory if needed.
# Writes to a temp file in the same directory and moves it into place
# only on success, so a failure never leaves a truncated unit file.
service_substitute_unit() {
  local template_path="$1"
  local exec_path="$2"
  local workdir="$3"
  local output_path="$4"
  local output_dir tmp_path
  output_dir="$(dirname "$output_path")"
  mkdir -p "$output_dir" || return 1
  tmp_path="$(mktemp "${output_dir}/.vocalis.service.XXXXXX")" || return 1
  if ! sed -e "s|__VOCALIS_EXEC__|${exec_path}|g" -e "s|__VOCALIS_WORKDIR__|${workdir}|g" "$template_path" > "$tmp_path" \
    || ! chmod 644 "$tmp_path" \
    || ! mv -f "$tmp_path" "$output_path"; then
    rm -f "$tmp_path"
    echo "Could not write the Vocalis service file at ${output_path}." >&2
    return 1
  fi
}

# Wrappers around the two systemctl calls, overridable by tests.
service_run_systemctl_daemon_reload() {
  systemctl --user daemon-reload
}

service_run_systemctl_enable_now() {
  systemctl --user enable --now "$1"
}

# Reload the user systemd manager, then enable and start the given
# unit (defaulting to vocalis.service).
service_enable_and_start() {
  local unit_name="${1:-vocalis.service}"
  service_run_systemctl_daemon_reload
  service_run_systemctl_enable_now "$unit_name"
}
