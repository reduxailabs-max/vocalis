# Vocalis systemd user service (template)

`vocalis.service` is a template, not a ready-to-use unit file. Two
placeholders must be substituted before it can be installed:

- `__VOCALIS_EXEC__` — the absolute path to the installed `vocalis`
  command (e.g. `/opt/vocalis/venv/bin/vocalis` or wherever Stage 3's
  installer puts it).
- `__VOCALIS_WORKDIR__` — the directory `vocalis` should run from.

Substituting these values, copying the result to
`~/.config/systemd/user/vocalis.service`, and running
`systemctl --user enable --now vocalis.service` is Stage 3's job (the
installer), not something this stage does or asks the user to do by hand.

## Design notes

- `WantedBy=xdg-desktop-autostart.target` (not `default.target`): this
  account has `Linger=yes`, so a `default.target` unit would start at
  boot, before any desktop session — and before a display — exists.
  `xdg-desktop-autostart.target` starts once a desktop session is up, on
  any desktop environment (XFCE, GNOME, KDE, ...).
- **Stage 13 addition:** `xdg-desktop-autostart.target` turned out to be
  unreliable in practice — on one real machine it depends on systemd
  thinking the login is "the" graphical session (logged as *"Current
  graphical user session is inactive"*), which does not always happen,
  so the service sometimes never starts after a reboot even though it is
  enabled. The installer now also writes
  `~/.config/autostart/vocalis-autostart.desktop` (from
  `packaging/desktop/vocalis-autostart.desktop`), which every desktop
  environment's own login autostart reads directly, independent of that
  systemd signal. Its command, `systemctl --user start vocalis.service`,
  does nothing if Vocalis is already running, so having both mechanisms
  enabled is harmless.
- `After=`/`Wants=` on PipeWire's three user services: Vocalis's sound
  output chooser (Stage 1) reads the list of sound devices once at
  startup and never refreshes it. Starting before the sound system is
  ready would mean the real outputs (`pulse`, `pipewire`) are invisible
  for that entire run.
- `Restart=on-failure` + `RestartSec=3` + the `StartLimitBurst=5` /
  `StartLimitIntervalSec=60` pair: a crash is retried quickly, but 5
  crashes inside 60 seconds makes systemd give up rather than loop
  forever. Vocalis's own startup-checks notification (`cli.py`) is a
  separate, best-effort heads-up about missing dependencies — it is not
  what detects systemd's restart-limit; systemd decides that on its own.

## Verifying the file (text only — never starts the real service)

```
systemd-analyze --user verify packaging/systemd/vocalis.service
```

This checks the file parses as a valid unit. It still fails on the
placeholders (they are not real paths) until Stage 3 substitutes them —
that failure is expected and fine; it proves the rest of the syntax is
sound.
