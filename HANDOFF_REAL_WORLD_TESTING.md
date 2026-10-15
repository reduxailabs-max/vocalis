# Handoff: Real-World Verification of Vocalis

**Who this is for:** an AI coding agent (e.g. Antigravity) picking up where Claude Code left off, running on the user's real machine.

**Why this handoff exists:** all development is finished and merged, but every automated test in this repo deliberately fakes or blocks real hardware and real system commands (microphone, keyboard typing, GPU inference, `apt`/`systemctl`/`uv`). Nothing has actually been installed or run for real yet. That is the one remaining gap before this project can be called production ready, and it's what you're here to close.

## 1. Project context

- **Vocalis** is an offline voice AI workstation (successor to an earlier project called "Voxtype") for Debian/Ubuntu-based Linux. Features: Dictation (NUMLOCK hold-to-record → transcribe → type/clipboard), Quick Reader (hotkey → TTS), Document Reader (open a file → TTS with pause/resume), OCR Reader (drag-select screen region → OCR → clipboard/TTS).
- Source lives at `/run/media/redx/3A99F6CB722764A7/02_projects/voxtype` on an **NTFS drive used only for development**. The user has explicitly said they will never run or install Vocalis for real from this NTFS drive.
- `git` branch `master`, latest commit `b392085`, clean tree, no remote configured.
- Current state when this handoff was written: 607 automated tests passing, ~96.75% coverage, mypy strict clean, shellcheck clean, 58/58 bats installer tests passing. Installer artifacts already built once at `packaging/installer/install.run` / `uninstall.run` (may be stale — rebuilding is cheap, see step 3).
- No LICENSE file yet, version is `0.1.0`. Neither blocks this testing pass; only matters if the user later publishes/shares the project.

## 2. Your job, in order

### Step 1 — Get a real copy
Copy (not symlink) the whole repository to a real, non-removable drive — e.g. the user's home directory. Do **not** operate on the NTFS path above. If you're unsure which path to use, ask the user.

### Step 2 — Build the installer fresh
```bash
cd <real-copy>/packaging/installer
bash build-run.sh
```
This builds the Vocalis wheel and packages `install.run` / `uninstall.run` via makeself. Confirm both files are produced and executable.

### Step 3 — Install for real
```bash
./install.run
```
- This will ask for `sudo` once (for `apt` packages). **Ask the human user for their password when prompted — do not guess or hardcode a password.**
- It installs apt dependencies, enables `ydotoold`, installs `uv`, creates a Python 3.13 venv at `~/.local/share/vocalis/venv`, installs the Vocalis wheel into it, symlinks `~/.local/bin/vocalis`, downloads/verifies model files, and installs the systemd user service.

### Step 4 — Verify the install actually did what it says
- `systemctl --user status vocalis` — should be enabled and running (unit installed from `packaging/systemd/vocalis.service`, `WantedBy=xdg-desktop-autostart.target`).
- `systemctl --user status ydotoold` (or equivalent) — running.
- `~/.local/share/vocalis/venv/bin/vocalis --version` (or similar) works.
- Model files present at the paths `VocalisConfig()` reports, e.g.:
  ```bash
  ~/.local/share/vocalis/venv/bin/python -c 'from vocalis.config import VocalisConfig; c=VocalisConfig().engines; print(c.parakeet_model_dir, c.qwen3_tts_model_dir, c.piper_binary_path, c.piper_voice_model_path)'
  ```
  Each printed path should exist and be non-empty. Parakeet dir needs `config.json`, `encoder-model.int8.onnx`, `decoder_joint-model.int8.onnx`, `vocab.txt`. Qwen3-TTS dir needs `config.json`, `model.safetensors`.
- Check `journalctl --user -u vocalis -e` for startup errors.

### Step 5 — Functional tests with real hardware
Do these with a real microphone, real keyboard focus, and a real screen:
1. **Dictation** — open a text editor, hold NUMLOCK, say a short sentence, release. Confirm: "Listening..." → "Transcribing..." notifications appear, then the real transcribed text is typed into the editor (or copied to clipboard with a "copied to clipboard instead" notification if typing isn't possible).
2. **Quick Reader** — trigger its hotkey (check `docs/user-docs.md` for the exact binding), confirm real TTS audio plays through speakers.
3. **Document Reader** — open a real text/PDF/etc. file with it, confirm TTS reads it aloud, and pause/resume works.
4. **OCR Reader** — drag-select a screen region containing real text, confirm it's OCR'd and either typed, copied, or read aloud correctly.
5. Confirm desktop notifications appear at each stage for each feature (this is how the user will know what's happening).

### Step 6 — Uninstall tests
```bash
./uninstall.run            # normal: removes app/service, keeps config + downloaded models
./uninstall.run -- --purge # also removes config, Parakeet, Qwen3-TTS, Piper binary + voice
```
Confirm each leaves the filesystem in the state its name promises (check `~/.config/vocalis`, `~/.local/share/vocalis`, `~/.local/bin/vocalis`, the systemd unit).

### Step 7 — Report back
Write a markdown report (save it back into this repo, e.g. `docs/real_world_test_report.md`) covering:
- Pass/fail for every step above, with exact command output / error text / log excerpts for anything that failed.
- Any mismatch between actual behavior and what `docs/user-docs.md` or `docs/superpowers/specs/2026-09-11-vocalis-design.md` describes.
- Whether you'd now call this installer and app genuinely production ready, and why or why not.

## 3. Reference docs already in the repo
- `README.md` — project overview and privacy notes
- `docs/user-docs.md` — end-user usage, hotkeys, expected behavior
- `docs/developer-docs.md`, `docs/architecture.md`, `docs/workflow.md`, `docs/algorithms.md`
- `packaging/installer/README.md` — installer internals
- `docs/superpowers/specs/2026-09-11-vocalis-design.md` — full design spec, including the Stage 3 (installer) and Stage 4 (hardening) amendments

## 4. Rules to respect
- Never hardcode or guess the sudo password — always prompt the human.
- Never run the installer against the NTFS dev drive.
- Don't modify application code to make a test pass; if something real-world fails, report it precisely so it can be fixed deliberately, not papered over.
