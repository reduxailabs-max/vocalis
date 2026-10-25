# Algorithms Used

## Single-residency + idle-unload (`ModelManager`)

**Problem:** an 8GB GPU cannot hold multiple large speech models at once,
and models must not sit in VRAM when nothing is using them.

**Approach:** a simple reference-free "last write wins, idle timeout evicts"
policy, hardened with a generation counter to close a race between a firing
idle timer and a concurrent re-acquire:

- `acquire(id, factory)` — O(1). Cancels any pending eviction timer and
  advances a monotonic generation counter. If the requested id is already
  the loaded one, reuse it (no reload cost). If a *different* engine is
  loaded, unload it synchronously before loading the new one, guaranteeing
  at most one engine loaded at any instant. If `factory()`/`engine.load()`
  raises, the manager's bookkeeping is left reflecting "nothing loaded"
  (never a stale reference to an already-unloaded engine) and the
  exception is re-raised to the caller.
- `release(id)` — O(1). Decrements the engine's use count (acquire adds
  one per caller, so Quick Reader and Document Reader sharing the
  `qwen3_tts` id are counted separately). While the count is still above
  zero it only logs and returns: no idle timer is started, so one reader
  finishing cannot arm an unload under the other. An extra release at
  count zero logs a warning and keeps the count at zero. When the count
  reaches zero it cancels any older timer, then starts a new one (wall clock `model_idle_timeout_seconds`,
  default 30s) carrying the generation captured at that moment. When it
  fires, the callback only unloads if the manager's generation is still
  the one it was scheduled under — if something reacquired the engine in
  the meantime (bumping the generation), the callback is a no-op instead
  of pulling the model out from under the new caller.
- Thread safety: a single `threading.Lock` guards all state transitions,
  so hotkey threads and the idle-timer thread can never race each other
  into an inconsistent state.

**Why the generation counter?** `threading.Timer.cancel()` cannot stop a
callback that has already started running. Without the generation check, a
timer that fires just as a hotkey thread reacquires the same engine could
still unload it a moment later, leaving the caller holding a reference to
an unloaded model while the manager reports nothing loaded. The generation
counter closes that window without needing to cancel an in-flight OS
thread (which Python cannot do).

**Complexity:** O(1) per acquire/release; one background timer thread at a
time (daemon thread, does not block process exit).

**Use counting (same engine id only):** the manager keeps one use count
for the loaded engine so that two features sharing an id do not idle-unload
it while the other is still inside it. Single residency still wins across
*different* ids: acquiring another engine unloads the current one even if
its count is above zero (deliberate, conservative: it is the original
rule).

*Known limitation (low risk):* release is tracked per engine id, not per
loaded copy. If a caller is evicted by a different engine and another
reader later reloads the same id, the evicted caller's late `release()` is
counted against the new load and can take its count to zero, arming the
idle-unload timer while the new reader still uses the engine. The timer is
30 s, a reader re-acquires on every chunk, and a re-acquire invalidates the
pending timer (the generation guard), so the practical worst case is one
unnecessary unload and reload. Since Stage 6b the reader's `EngineChain` also
checks `ModelManager.current_engine` (object identity) before releasing, so a
late `close()` from an old reading cannot lower a newer reading's count.

## Auto-selection policy and downshift hysteresis (`auto_select.py`, `EngineSelector._apply_hysteresis`)

**The policy (`decide()`).** For "Auto", `engines/auto_select.py`'s
`decide(hardware, policy, held_vram_mb, max_own_ram_mb)` is a pure function
of a `HardwareInfo` snapshot and a frozen `AutoPolicy` of named thresholds —
no GPU is needed to unit-test it. In order: no graphics card, or system RAM
below 1024 MB free -> Piper first; a graphics card of 4096 MB or less ->
Kokoro first; `is_stressed()` (CPU load per core above 0.85, or free RAM
below `max(2048 MB, 15% of total)`, or — since Stage 6d — Vocalis's own
current memory use at or above the user's advisory `max_ram_mb` ceiling) ->
Kokoro first; otherwise, if the GPU has at least 4096 MB free (after adding
back `held_vram_mb`, the memory Vocalis's own already-loaded model holds,
so it never counts against itself) -> Qwen3-TTS first. Each branch returns
an ordered chain plus a plain-English reason string for the log.

**Problem the hysteresis solves.** `EngineSelector.plan()` calls `decide()`
fresh for every reading. Hardware readings are noisy right around a
threshold — CPU load that bounces just above and just below the stress
line between two readings a few seconds apart would otherwise make Auto
flip back and forth between, say, Qwen3-TTS and Kokoro, reloading the
model chain on every flip.

**The rule (`EngineSelector._apply_hysteresis`).** Engine ids are ranked
into tiers (`_TIER_RANK`: Qwen3-TTS = 0 (best), Kokoro = 1, Piper = 2).
After every `decide()` call, `_apply_hysteresis()` compares the new
decision's tier against the tier and wall-clock time (`time.monotonic()`,
injectable) of the *last* downshift (a move to a numerically higher,
i.e. worse, tier):

- If a downshift happened within the last `downshift_hold_seconds`
  (default 60s) and the new decision would climb back to a *better* tier
  than the one it was held at, the better tier is discarded and the held
  tier's forced chain is returned instead, with the reason string extended
  by "(holding the lighter voice a little longer after a recent
  slowdown.)".
- A decision that stays at or below the held tier passes through
  unchanged, and updates the remembered downshift time if it is itself a
  new downshift.
- Once `downshift_hold_seconds` has elapsed, Auto is free to climb back to
  a better tier again on the next reading that qualifies for one.

**Scope.** This only applies to the "Auto" engine choice. A user-forced
engine (Qwen3-TTS, Kokoro or Piper picked explicitly in Settings) maps
straight to its fixed `FORCED_CHAINS` entry in `EngineSelector.plan()`,
before `decide()` or `_apply_hysteresis()` ever run, so a forced choice is
never held back or delayed.

**Complexity:** O(1) per `plan()` call — a couple of dict lookups and one
monotonic clock read.

## Sentence-bounded chunking (`text/chunker.py`)

**Problem:** speaking a whole document as one block delays the first word
and gives no progress; cutting at fixed lengths would split words.

**Approach:** split on whitespace that follows `.`, `!` or `?`, then pack
whole sentences greedily into a chunk until adding the next one would
exceed `max_chars` (default 300). A single sentence longer than the limit
becomes its own over-limit chunk rather than being cut. Example (tested
with `max_chars=14`): `"One. Two three. Four five six."` →
`["One.", "Two three.", "Four five six."]`.

**Complexity:** O(n) in the text length.

Since Stage 6b the chunker is sentence-aware (never breaks after abbreviations or
initials, has a line-break rule, linear time) and returns exact spans
(`chunk_spans`) — see `docs/architecture.md`.

## Markdown stripping (`text/normalizer.py`)

Applied in a fixed order: remove code fences → images/links become their
label → heading `#` markers removed → inline code is swapped for
placeholders → bold/italic removed → whitespace collapsed → inline code
restored.

- **Inline-code protection:** inline code is hidden behind placeholders
  during emphasis stripping and put back untouched, so `` `__init__` ``
  is never mistaken for bold.
- **Star flanking:** `*`/`**`/`***` emphasis only matches when the
  character after the opener and before the closer is not whitespace, so
  `2 * 3` is left alone.
- **Underscore flanking:** `_`/`__`/`___` follow the same whitespace rule
  and also must not touch a word character outside the markers, so
  `text_input` is never treated as emphasis.

Example (tested):

```
in : # Title\nSee **bold**, *it*, `__init__` and [a link](http://x). 2 * 3 and text_input.
out: Title\nSee bold, it, __init__ and a link. 2 * 3 and text_input.
```

## Speaking-pace time-stretch (`audio/pace.py`)

**Problem:** neither voice engine has its own speed control, so playing
faster or slower must happen to already-synthesized audio, without
changing its pitch (a naive resample would make a faster voice sound
higher-pitched, like a sped-up tape).

**Approach:** `apply_speed(audio, sample_rate, speed)` clamps `speed` to
`MIN_SPEED..MAX_SPEED` (`0.7`-`1.6`; `NaN` is treated as `1.0` by
`clamp_speed()`), then runs the waveform through
`librosa.effects.time_stretch` (a phase-vocoder time stretch, which
preserves pitch). Two short-circuits skip the stretch entirely and return
the original array unchanged: a speed within 1% of neutral (`1.0`), or
audio shorter than 4096 samples (too short to stretch meaningfully).
`librosa` is imported *inside* the function so merely importing
`vocalis.audio.pace` (which `config.py` does, just to get
`MIN_SPEED`/`MAX_SPEED`) never pays librosa's import cost.

`paced_audio(audio, sample_rate, provider)` is the shared entry point: with
`provider` is `None` it returns `audio` unchanged; otherwise it calls
`provider()` for the current speed and applies it, catching any exception
from either call, logging it as a warning, and returning the original
`audio` — a pacing failure must never break a reading. Since Stage 6c it has
exactly one caller, `Qwen3TTS.synthesize()`, capped to a mild 1.0-1.2x (that
engine has no native speed control of its own, and a bigger stretch audibly
hurts its voice). Kokoro and Piper have real speed controls and never call
it: Kokoro passes the speed straight to `kokoro.create(..., speed=...)`, and
Piper converts it to its `--length_scale` argument (`1 / speed`). A pause
still rewinds the heard clip inside `SpeechStream`, so a resume replays the
same already-paced audio.

**Complexity:** O(n) in sample count for the stretch itself (one-time
librosa JIT-compilation cost on first use, pre-paid at startup by
`cli.py`'s `_warm_up_pace_changer()`).

## Single-instance guard: connect-before-delete (`daemon.py`)

**Problem:** `VocalisDaemon.start()` needs to remove a stale Unix socket
file left behind by a crashed previous run before binding its own — but
unconditionally deleting any existing socket file is unsafe if a second
real instance is already running (a manual `vocalis` launch while the
systemd service copy is live, or two launches racing each other): that
would delete the first instance's *live* socket file out from under it.

**Approach:** before deleting anything, `_is_another_instance_alive()`
opens a short-lived probe socket (via the constructor's injectable
`socket_factory`, swappable in tests) and tries to `connect()` it to the
existing socket file path. Exactly three outcomes:

- **Alive** — connect succeeds: a real process is listening. `start()`
  raises `DaemonAlreadyRunningError` and never touches the file.
- **Stale** — `ConnectionRefusedError` (socket file exists, nothing
  listening — exactly what a crashed process leaves behind) or
  `FileNotFoundError` (the file vanished between the earlier `exists()`
  check and the connect attempt). Both: log a warning and delete the file.
- **Ambiguous** — any other exception (e.g. a permission error) is
  deliberately treated the same as "stale", since an inconclusive probe
  must never block Vocalis from starting; the reason is logged as a
  warning rather than swallowed. If two processes really are racing, the
  loser still fails at its own later `bind()`.

The probe socket is always closed in a `finally`, wrapped in
`contextlib.suppress(Exception)` — a close failure on a socket that
already failed to connect is not worth surfacing.

**Complexity:** O(1); one extra socket connect attempt (bounded by a
0.5s timeout) added to daemon startup.

## Generation counter + lock-only-guards-a-swap (readers)

`QuickReaderFeature` and `DocumentReaderFeature` both keep a `_generation`
integer that is bumped each time a reading starts. A background worker
carries the generation it was started with and, at each stage, asks
`_is_current(generation)` (active **and** same generation) before
continuing; a worker from a stopped reading therefore bails out and can
never clobber a newer reading's state.

The feature's lock only guards the flag/counter swap. Anything that can
block — `AudioPlayer.stop()`, `notify-send`, calling a coordinator stopper —
runs after the lock is released, so a hotkey press never waits on audio or
a subprocess. `ReadingCoordinator` uses the same rule: it swaps the owner
under its lock, then calls the previous owner's stopper outside it.

Because `ReadingCoordinator.finish_reading(owner)` compares owner identity
and not generation, only the current reading may call it; Quick Reader's
preemption callback is therefore clear-only (it never starts or flips an
idle reader to active).

**Pause/resume:** (historical per-chunk design replaced in Stage 6b) `pause()`
now tells the reading's `ChunkedSpeaker`, whose `SpeechStream` outputs silence
at once and rewinds the heard clip; `resume()` replays it from its start;
`stop()` is a permanent latch that wakes everything.

**Control notifications:** the "Paused", "Resuming..." and "Stopped"
desktop notifications are put on a per-feature `queue.Queue` and sent by a
single lazily-started daemon thread. `pause()`/`resume()`/`stop()` return
immediately (no lock is held and nothing waits on `notify-send`), the
messages arrive in the order the calls were made, and a `notify()` that
raises is logged without ending the thread.

**Popup pause button:** `DocumentReaderPopup` keeps its own `_paused`
flag (reset when a reading finishes or a new one starts) so the button
alternates Pause/Resume correctly; a Pause click while nothing is being
read is ignored.
