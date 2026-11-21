#!/usr/bin/env bash
# Installs the "Vocalis" entry in the desktop's application menu.
set -uo pipefail

# Copy the Vocalis icon PNGs (named vocalis-<size>.png) into the user's icon
# theme so the menu entry and windows show the logo.
desktop_install_icons() {
  local source_dir="$1"
  local icon_root="${2:-${HOME}/.local/share/icons/hicolor}"
  local file size copied=0
  for file in "$source_dir"/vocalis-[0-9]*.png; do
    [[ -f "$file" ]] || continue
    size="$(basename "$file" .png)"
    size="${size#vocalis-}"
    mkdir -p "${icon_root}/${size}x${size}/apps" || return 1
    cp -f "$file" "${icon_root}/${size}x${size}/apps/vocalis.png" || return 1
    copied=1
  done
  [[ "$copied" -eq 1 ]] || return 1
  if command -v gtk-update-icon-cache >/dev/null 2>&1; then
    gtk-update-icon-cache -q -t -f "$icon_root" >/dev/null 2>&1 || true
  fi
  return 0
}

# Remove the icons desktop_install_icons added.
desktop_remove_icons() {
  local icon_root="${1:-${HOME}/.local/share/icons/hicolor}"
  rm -f "${icon_root}"/*x*/apps/vocalis.png
  if command -v gtk-update-icon-cache >/dev/null 2>&1; then
    gtk-update-icon-cache -q -t -f "$icon_root" >/dev/null 2>&1 || true
  fi
}

# Copies the static autostart template to output_path (no placeholders to
# fill in). This is a second, independent way of starting Vocalis at login,
# on top of the systemd unit's own WantedBy=: some desktops don't reliably
# raise the signal systemd needs ("the graphical session is ready"), but
# every desktop (KDE, XFCE, GNOME, ...) reads ~/.config/autostart/*.desktop
# for itself. The command it runs (systemctl --user start vocalis.service)
# does nothing if Vocalis is already running. Atomic, like the others here.
desktop_install_autostart() {
  local template_path="$1"
  local output_path="$2"
  local output_dir tmp_path
  output_dir="$(dirname "$output_path")"
  mkdir -p "$output_dir" || return 1
  tmp_path="$(mktemp "${output_dir}/.vocalis.autostart.XXXXXX")" || return 1
  if ! cp -f "$template_path" "$tmp_path" \
    || ! chmod 644 "$tmp_path" \
    || ! mv -f "$tmp_path" "$output_path"; then
    rm -f "$tmp_path"
    echo "Could not create the Vocalis autostart entry at ${output_path}." >&2
    return 1
  fi
}

# Remove the autostart entry desktop_install_autostart added.
desktop_remove_autostart() {
  rm -f "$1"
}

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
