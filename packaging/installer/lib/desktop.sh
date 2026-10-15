#!/usr/bin/env bash
# Installs the "Vocalis" entry in the desktop's application menu.
set -uo pipefail

# Write output_path from the template with __VOCALIS_EXEC__ replaced by the
# quoted exec_path. Atomic: a failure never leaves a broken launcher behind.
desktop_install_launcher() {
  local template_path="$1"
  local exec_path="$2"
  local output_path="$3"
  local output_dir tmp_path escaped
  output_dir="$(dirname "$output_path")"
  mkdir -p "$output_dir" || return 1
  escaped="$(printf '%s' "\"${exec_path}\"" | sed -e 's/[\\|&]/\\&/g')"
  tmp_path="$(mktemp "${output_dir}/.vocalis.desktop.XXXXXX")" || return 1
  if ! sed -e "s|__VOCALIS_EXEC__|${escaped}|g" "$template_path" > "$tmp_path" \
    || ! chmod 644 "$tmp_path" \
    || ! mv -f "$tmp_path" "$output_path"; then
    rm -f "$tmp_path"
    echo "Could not create the Vocalis menu entry at ${output_path}." >&2
    return 1
  fi
}
