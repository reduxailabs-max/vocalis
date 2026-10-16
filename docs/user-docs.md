# User Documentation

## How to open Vocalis

Once Vocalis is installed and running (see "Installing Vocalis" below),
there are three ways to open it:

- **Click the tray icon.** A single left-click on the small round Vocalis
  icon in your system tray opens the Settings window.
- **Use the applications menu.** Vocalis adds a normal "Vocalis" entry to
  your desktop's applications menu, the same as any other installed
  program — look for it there and click it like you would any app.
- **Run `vocalis --settings` from a terminal.** If Vocalis is already
  running, this opens Settings right away. If it isn't running yet, this
  starts it for you first, which can take a few seconds.

Two messages you might see are worth knowing about:

- **"An old sound settings file from Voxtype is still at ~/.asoundrc..."**
  — a leftover file from the previous "Voxtype" app is still changing
  your computer's default sound settings. Vocalis works around it
  automatically, so you don't have to do anything, but if it's causing
  problems in other programs too, you can rename it (for example to
  `.asoundrc.old`) to retire it everywhere.
- **"No sound was recorded. Check that your microphone is plugged in,
  not muted, and chosen as the input device, then try again."** — you
  held the dictation key but nothing was picked up from your microphone.
  Check the microphone is plugged in, not muted, and selected as your
  system's input device, then try dictating again.

## Installing Vocalis (Phase 8)

On a Debian/Ubuntu-family 64-bit machine with a working NVIDIA graphics
card driver, run the installer from the `packaging/installer/` folder:

```bash
bash install.sh
```

(or double-click `install.run` if you were given that single file instead
— it does exactly the same thing).

What it does, in order:

1. Checks your machine qualifies (Debian/Ubuntu-based, 64-bit, NVIDIA
   driver already working). If any of these fail, it stops immediately
   with a plain explanation and changes nothing.
2. Installs a few small system packages if they're missing — this is the
   **one and only password prompt** you'll see, right at the start.
3. Sets up a private copy of Python for Vocalis (it never touches your
   system Python).
4. Downloads the voice and dictation models if they aren't already on
   your machine — several gigabytes, so **the first run can take a
   while** depending on your internet connection. If a model is already
   present, it's left alone and nothing is re-downloaded.
5. Sets Vocalis up to start automatically every time you log in, and
   starts it right now.

When it finishes you'll see a plain "Vocalis is installed and running"
message. It's safe to run `install.sh` again any time — anything already
set up correctly is left alone, so re-running it also works as a repair
command if something ever goes wrong.

To remove Vocalis, run `bash uninstall.sh` from the same folder. By
default this keeps your settings and your downloaded models (in case you
install again later); run `bash uninstall.sh --purge` instead to remove
everything, including your settings, every downloaded model, and the
Piper backup voice (its binary and its voice files) and the Kokoro voice — a full, clean
removal. Run `bash uninstall.sh --help` any time to see this usage
summary without removing anything.

If you were given the single-file `uninstall.run` instead of the folder,
run `./uninstall.run` for a normal removal, or `./uninstall.run -- --purge`
for the full removal. The extra `--` matters: `./uninstall.run --purge`
on its own fails, because the `.run` wrapper tries to read `--purge` as
one of its own options instead of passing it on to the uninstaller.

## Dictation (Phase 2)

1. Hold **NUMLOCK** (or the push-to-talk key you chose in Settings).
2. Speak. A "Listening..." notification confirms recording started.
3. Release NUMLOCK. You'll see "Transcribing...", then the transcribed
   text, then either the text typed at your cursor or (if typing failed —
   e.g. `ydotoold` isn't running) a note that it was copied to your
   clipboard instead.

Your normal NumLock LED/state is unaffected — this only listens for the
key, it doesn't suppress it.

## Quick Reader (Phase 3)

1. Copy some text (**Ctrl+C**).
2. Press **Ctrl+Alt+R** (or the shortcut you chose in Settings). You'll see "Reading...", then the text itself,
   then it plays out loud through your speakers, then "Done".
3. Press **Ctrl+Alt+R** again while it's still talking to stop it
   immediately.
4. If nothing is copied, you'll see "Nothing copied" instead.

Quick Reader picks its voice for each reading (see "Voice engines"
below). If that voice ever fails to load or generate speech for any
reason, it automatically switches to a lighter backup voice and tells you
so — you'll still hear your text read aloud either way. Which backup
voice depends on the chain: usually Kokoro, then Piper as the last
resort.

Vocalis now finds a working sound output automatically — see "Sound and
speaking pace" below for how that works and what to do if you ever see a
"Playback failed" message.

## Document Reader (Phase 4)

1. Press **Ctrl+Alt+D** (or your own shortcut). A "Vocalis — Document Reader" window appears.
2. Either click **Open file...** and pick a document (`.txt`, `.md`,
   `.pdf`, `.docx`, `.html`, `.htm`) — reading starts straight away — or
   paste text into the box and click **Read**. Text inside tables in Word
   files (including tables nested inside other tables) is read too; the
   text of a merged cell is read once, whether it is merged side by side
   or across several rows. Tables nested more than 10 levels
   deep are skipped, and a warning is written to the log.
3. Use **Pause** (the button then says **Resume**) and **Stop** as you
   like. Resuming replays the sentence group that was playing from its
   start. The window shows the status and "piece N of M" progress.
   Pause does nothing when nothing is being read.
4. Pressing **Ctrl+Alt+D** again while something is being read stops it;
   pressing it when nothing is being read just brings the window forward.

Quick Reader and Document Reader never talk at the same time: starting one
stops the other.

What the messages mean:

- **Unsupported file type** — the file is not one of the formats above.
- **Could not open ...** — the file is damaged or cannot be read.
- **... is password-protected. Remove the password and try again.** — the
  PDF is encrypted; remove its password and open it again.
- **No readable text found ...** — the file opened but has no text in it;
  a scanned PDF (just pictures of pages) is the usual cause.
- **... too large** — the file has more than 5,000,000 characters of text.
- **Primary voice unavailable — using backup voice** — the main AI voice
  failed, so the rest of that document is read with a lighter backup voice
  (the next one down the chain: usually Kokoro, then Piper).
- **Nothing to read** — the text had nothing speakable in it.

## Reading text off the screen (OCR Reader)

1. Press **Ctrl+Alt+O**. A selection box appears on screen.
2. Drag a box around the text you want. You can **drag its edges or
   corners to resize** the box before confirming. Press **Enter** (or
   click the on-screen confirm button) to confirm, or **Esc** to cancel.
3. The text is copied to your clipboard and read aloud. While the
   floating widget is switched on (the default) the small widget appears
   (see "The floating widget" below) with **Pause/Resume** and **Stop**.
   If you switched the widget off in Settings, the Document Reader window
   opens instead, showing the text with the same buttons.
4. Press **Ctrl+Alt+O** again while it is reading to stop it. (This only
   stops a reading that the OCR Reader started. If a document you opened
   is being read, the press starts a new capture instead; once you confirm
   the box, the document reading is stopped (you see a "Stopped" message)
   and the screen text is read in its place.)

What the messages mean:

- **Cancelled** — you pressed Esc or closed the capture tool.
- **No text detected** — the box had no readable text (fewer than 3
  characters were found).
- **No screen-capture tool found. Install one of: spectacle, maim,
  flameshot, or slurp + grim** — none of the capture tools is installed.
- **Tesseract is not installed. Install it with: sudo apt install
  tesseract-ocr tesseract-ocr-eng** — the text-recognition program is
  missing.
- **Tesseract took too long to read the picture** / **Tesseract could not
  read the picture** — recognition timed out or failed; try a smaller box.

Only English text is recognised. The box can be resized by dragging its
edges or corners with Spectacle (the preferred tool). If Spectacle isn't
installed, the tools are tried in order: `maim` (draw once, no resize),
`flameshot` (draw once; tried third because it may hang on some systems),
then `slurp` + `grim` (Wayland fallback). If copying to the clipboard
fails, the text is still read aloud.

## The tray icon

Vocalis puts a small round icon in your system tray (the XFCE panel). Its
colour tells you what Vocalis is doing:

- **Grey** — idle.
- **Red** — listening (you are holding the push-to-talk key).
- **Blue** — reading aloud (Quick Reader, Document Reader or OCR Reader).

Hover over the icon to read the tooltip ("Vocalis — idle", "Vocalis —
listening", or "Vocalis — reading: Document"). Right-click it for the menu:

- A status line (the same text as the tooltip; not clickable).
- **Pause** / **Resume** — only available while a Document Reader or OCR
  reading is active; after you pause, the entry changes to **Resume**.
  It stays greyed out for Quick Reader (which can only be stopped) and when
  nothing is being read.
- **Stop** — available while any reading is active.
- **Open Document Reader** — shows the Document Reader window.
- **Settings…** — opens the Settings window.
- **Quit** — shuts Vocalis down cleanly, the same as Ctrl+C.

If your desktop has no system tray, Vocalis writes one warning to the log
and keeps working without the icon (use Ctrl+C to quit).

## The floating widget

A small box that sits in the bottom-right corner of your main screen,
above other windows, **only while something is being read aloud**. It
appears when the reading starts and disappears when it ends. It shows
where the text came from ("Document", "Screen text" or "Quick read"),
the status, and the progress ("piece N of M"), plus two buttons:

- **Pause** (which becomes **Resume**) and **Stop** for Document Reader and
  OCR readings.
- **Stop** only for Quick Reader (it cannot be paused).

The widget never takes keyboard focus, so it will not steal what you are
typing into. If you switch it off in Settings, the OCR Reader opens the
Document Reader window as it did before. Switching it off while it is
showing hides it at once; switching it on takes effect at the next
reading.

## Sound and speaking pace

Vocalis tries to pick a sound output that actually plays audio
automatically, and gives you a slider for how fast it talks and a way to
preview your voice before you commit to it.

- **Sound output** — in the Settings window, a drop-down lets you choose
  **Automatic (recommended)** or one of your computer's named sound
  outputs. "Automatic" tries your usual outputs in a sensible order and
  uses the first one that really plays sound (some sound setups silently
  swallow audio on one output while another works fine — Automatic skips
  the broken one for you). If you pick a named output, Vocalis tries that
  one first and automatically falls back to the usual order if it does not
  work — so a dead named choice does not silently go quiet. (The Test
  sound button below only checks the one you picked, with no fallback, so
  you can tell whether that specific device works.)
- **Test sound** — a button next to the output drop-down. Click it and
  Vocalis plays a short, quiet tone through the selected output and tells
  you what happened. The Settings window can pause for a few seconds
  while it checks (up to about 8-9 seconds in the worst case: checking the
  device, playing the test tone, and waiting for it to finish) — this is
  normal and not a freeze or a crash.
  - **"Sound test OK (<device name>)"** — you should have heard a brief
    tone through that device.
  - **"Sound test failed: no working output found"** — none of your
    outputs responded; check that your speakers/headphones are connected
    and your system's sound is working.
  - **"Sound test failed: <name> did not respond"** — the specific output
    you picked by name did not work (try "Automatic" instead).
  - If a reading is already playing, Test sound tells you to try again
    once it has finished, instead of interrupting it.
- **Speaking speed** — a slider from **0.7x** (slower) to **1.6x**
  (faster), default **1.15x** (a little faster than the voice's natural
  pace). It applies from the next piece of speech it generates — if you
  change it while something is already mid-sentence, that part keeps its
  old speed and the new speed takes over from the next piece onward. This
  is the same for every voice (Qwen3-TTS, Kokoro and Piper).
  **Note:** if you change the speed while a Document Reader reading is
  *paused*, the new speed takes effect from the next chunk once you
  resume — the paused chunk itself resumes at the speed it was already
  playing at.
- **Preview voice** — plays a short sample sentence using your **saved**
  settings (the voice, speed and output you last clicked Save on — not
  whatever you've changed in the window but not yet saved). If something
  is already being read aloud, Preview voice stops it and speaks the
  sample instead.
- **If your chosen output stops working while Vocalis is running** (for
  example, you unplug a headset), Vocalis automatically switches to
  another working output and sends you one desktop notification
  ("Sound output changed because it stopped working") the first time that
  happens — not every single time if the same device keeps failing, but
  it will tell you again if a *different* device fails later.
- **There is no "speaking style" control** (no "calm"/"energetic" toggle).
  Vocalis's installed voice was tested with style hints and the audio it
  produced did not reliably sound any different, so no such control is
  offered — a control that doesn't actually do anything would just be
  confusing.

## Voice engines

Vocalis can read aloud with three different voices, called "engines".
They sound a little different and need different amounts of computer
power:

- **Qwen3-TTS** — the best-sounding voice. It needs a strong graphics
  card and some free memory.
- **Kokoro** — a lighter voice that still sounds natural. It works well on
  an ordinary PC and on smaller graphics cards.
- **Piper** — the lightest voice. It runs on almost any PC and is always
  ready, so it is the safety net.

**Automatic (recommended)** means Vocalis picks for you, each time you
start a reading, based on how your PC is doing at that moment:

- If you have a graphics card with more than 4 GB of memory, your PC is
  not busy, and enough of the card's memory is free, Vocalis uses
  **Qwen3-TTS**.
- If your PC is busy (the processor is working hard, or free memory is
  low), or your graphics card is small or already in use by other
  programs, Vocalis uses **Kokoro**.
- If your PC is very short of free memory, or has no graphics card at all,
  Vocalis uses **Piper**.

Vocalis does not count its own voice against you: if Qwen3-TTS is already
loaded, the memory it holds is not treated as "busy", so Automatic does
not keep flipping between voices.

You can choose a voice yourself instead. In Settings there is a **Quick
Reader voice engine** drop-down and a **Document Reader voice engine**
drop-down, each with **Automatic**, **Qwen3-TTS**, **Kokoro** and
**Piper**. A third drop-down, **Kokoro voice**, picks which Kokoro voice
speaks. The new choices apply from the next reading.

If the voice you chose (or Automatic chose) cannot work, Vocalis moves to
the next lighter one and shows **"Primary voice unavailable — using backup
voice"**. This is not an error and your text is still read aloud. The
order is: Qwen3-TTS, then Kokoro, then Piper; if you chose Kokoro, then
Piper; if you chose Piper, then Kokoro.

The installer downloads the Kokoro voice (about 350 MB) along with the
other models. If it is missing, Vocalis tells you when it starts and uses
another voice until you run the installer again.

Older config files that still contain the old `quick_tts` / `doc_tts`
lines keep working; those lines are ignored and both readers use
Automatic.

## Settings

Open the Settings window from the tray menu (**Settings…**). It has:

- **Show desktop notifications** — applies immediately.
- **Show the floating widget while reading** — applies immediately.
- **Voice** — pick one of nine voices: Serena, Vivian, Uncle_Fu, Ryan,
  Aiden, Ono_Anna, Sohee, Eric, Dylan. It applies from the **next**
  reading; a reading that is already in progress finishes in its old voice.
- **Quick Reader voice engine**, **Document Reader voice engine** and
  **Kokoro voice** — see "Voice engines" above.
- **Sound output**, **Test sound**, **Speaking speed** and **Preview
  voice** — see "Sound and speaking pace" above.
- **Shortcuts** — drop-down lists, so typing mistakes are impossible:
  - *Dictation (hold)*: **NUMLOCK**, **SCROLLLOCK** or **PAUSE**.
  - *Quick Reader*, *Document Reader*, *OCR Reader*: **Ctrl+Alt+** plus any
    letter A–Z or digit 0–9.

Click **Save** to keep your changes. Saving rewrites the whole config file,
so any manual edits made to `config.toml` while Vocalis is running are
overwritten. The message under the settings says what happened:

- **Saved.** — everything is saved. If you changed a shortcut it adds
  **Restart Vocalis to use the new shortcuts.** Shortcuts only apply after
  a restart (the switches and the voice do not need one).
- **Not saved: …** — nothing was changed (neither the file nor the running
  settings), and the reason follows. For example, two features cannot use
  the same shortcut ("quick_reader and document_reader both use the same
  shortcut; each feature needs its own"), or the file could not be written.

**Close** (or the window's X) only hides the window; Vocalis keeps running
in the tray.

## Running the daemon manually (development only)

```bash
.venv/bin/vocalis
```

Press Ctrl+C in that terminal, or choose **Quit** from the tray menu, to stop it cleanly (it quits the Qt event loop, then stops the tray icon, the hotkey listeners, the daemon and the model manager). Closing the Settings or Document Reader window does **not** quit Vocalis. If Vocalis is ever stopped the way the
background service stops it (see "Running automatically" below) instead of
with Ctrl+C, it shuts down exactly the same clean way.

## Running automatically

Running the installer (see "Installing Vocalis" above) sets this up for
you — there is nothing to copy or switch on by hand. Once it has run,
Vocalis will:

- Start on its own a little after you log in to your desktop (not at the
  moment the computer boots — it waits for your desktop session, so it
  never tries to start before a screen exists).
- Restart itself automatically a few seconds after a crash.
- Stop trying to restart if it crashes 5 times within one minute, rather
  than looping forever.

### What the start-up notifications mean

Every time Vocalis starts, it quickly checks several things and tells you
about any problem it finds with a desktop notification. **None of these
notifications is an error, and none of them stops Vocalis from
starting.** Each one is a heads-up about one specific feature — that one
feature won't work right until you fix the thing it mentions, but
everything else keeps working normally:

- **"The Parakeet dictation model is missing or incomplete..."** —
  Dictation (hold NUMLOCK to speak) will not work until this model is
  installed in the folder the message names.
- **"The primary voice model is missing..."** — Reading aloud will use
  a lighter backup voice (Kokoro, or Piper if that is missing too)
  instead of the main AI voice until this model is installed. You will still hear your text read out loud.
- **"The light Kokoro voice is missing..."** — Reading aloud will use
  another voice until the Kokoro voice is installed in the folder the
  message names (re-run the installer).
- **"ydotoold is not running..."** — Dictation still works, but instead
  of typing the words at your cursor, it copies them to your clipboard
  so you can paste them yourself.
- **"The OCR Reader needs tesseract/xclip..."** — Reading text off the
  screen (OCR Reader) won't work until the program(s) the message names
  are installed.
- **"An old sound settings file from Voxtype is still at ~/.asoundrc..."**
  — see "How to open Vocalis" at the top of this document for what this
  means and what to do about it.

### About "Vocalis could not start"

Being honest about where things stand: at this point, Vocalis does
**not** send any notification saying "Vocalis could not start — check
the log" when it gives up retrying after too many crashes. That idea is
part of the plan for this feature, but the part that decides "5 crashes
in 60 seconds is too many, stop retrying" is the background-service
system itself (systemd), not Vocalis's own code — and by the time it
gives up, Vocalis isn't running anymore to send any notification about
it. Watching for that from the outside and reacting to it would need
separate code that has not been built yet. If you ever suspect this has
happened (Vocalis seems to not be running and didn't tell you why), the
way to check — once the installer has set the service up — is to open a
terminal and run `journalctl --user -u vocalis`, which shows everything
the service tried and why it stopped.

## Configuration

Most settings can be changed in the Settings window (see below). You can
also edit `~/.config/vocalis/config.toml` by hand (created on first save,
or copy `config/default_config.toml`). The file is private to you: when
Vocalis saves it, only your user can read it (permission mode 0600).
Relevant settings:

- `notifications_enabled` — set to `false` to silence all notifications.
- `model_idle_timeout_seconds` — how long the Parakeet model stays loaded
  in VRAM after you stop dictating before it's automatically released.
- `[engines] parakeet_model_dir` — path to the Parakeet ONNX model files
  (defaults to the existing Voxtype installation's model directory).
- `[engines] qwen3_tts_speaker` — which of Qwen3-TTS's built-in voices to
  use (default: "Ryan").
- `[engines] quick_reader_engine` and `document_reader_engine` — `"auto"`
  (default), `"qwen3_tts"`, `"kokoro"` or `"piper"` (see "Voice engines").
- `[engines] kokoro_voice` — which Kokoro voice to use (default:
  `"af_heart"`).
- `audio_output_device` — `"auto"` (default) or the exact name of one of
  your sound outputs. Blank is rejected.
- `speaking_speed` — a number from `0.7` to `1.6` (default `1.15`).
  Anything outside that range is rejected when the file is loaded.
- `floating_widget_enabled` — `false` hides the floating widget.
- `[shortcuts]` — `numlock_key`, `quick_reader`, `document_reader` and
  `ocr_reader`. These are really used now (see "Shortcuts" below). Changes
  made while Vocalis is running apply after a restart.

**A mistake in a hand-edited shortcut** (for example `Ctrl+Shift+R`, a
second feature using the same shortcut, or an unsupported push-to-talk
key) does not stop Vocalis from starting: a clear error naming the field
is written to the log and the default shortcuts (NUMLOCK, Ctrl+Alt+R,
Ctrl+Alt+D, Ctrl+Alt+O) are used for that run.
