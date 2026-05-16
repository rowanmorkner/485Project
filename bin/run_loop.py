"""
Long-running orchestrator: poll all markets every N minutes, run the daily
training pass once a day, log everything to a rotating file, exit cleanly
on SIGINT/SIGTERM.

This is the "always-on" entry point — start it once on your desktop and
let it run. The recursive train→evaluate→improve loop happens automatically:

  1. Every POLL_INTERVAL seconds, run main.analyze_city for every city.
     - Snapshots, forecasts, and order recommendations are persisted.
     - In paper-trading mode (PAPER_TRADING=1, default in this script),
       each recommended order also gets a synthetic fill at intended price.

  2. At each hour in TRAIN_HOURS_UTC (default every 6h: 00, 06, 12, 18
     UTC), run the training pass:
       a. log_settlements    — pulls newly-settled events from both venues
       b. settle_orders      — scores any filled orders against settlements
     Sweeping 4×/day catches D-1 settlements as soon as either venue
     publishes them, so dashboard P&L stops being a day stale. The strategy
     itself reads the empirical Δ histogram directly from bot.db on every
     call; no separate refresh step is required.

Start (foreground for testing):
  PAPER_TRADING=1 python -m bin.run_loop

Start in background (survives ssh disconnect):
  PAPER_TRADING=1 nohup python -m bin.run_loop > /dev/null 2>&1 &

Or under tmux (preferred — easy to attach later):
  tmux new -d -s arbbot 'PAPER_TRADING=1 python -m bin.run_loop'
  tmux attach -t arbbot   # to view logs live
  Ctrl-b d                # to detach without killing

Stop:
  Ctrl-C if foreground, or `pkill -f bin.run_loop` if backgrounded.
"""

import logging
import logging.handlers
import os
import signal
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Default paper-trading on for this entrypoint — explicit opt-out via
# PAPER_TRADING=0 in the environment. Must be set BEFORE importing main.
os.environ.setdefault("PAPER_TRADING", "1")

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# Import after env is set up so any module-level config sees the right values
import main as bot_main  # noqa: E402
from config import CITIES  # noqa: E402
from clients.kalshi import KalshiClient  # noqa: E402
from clients.polymarket import PolymarketClient  # noqa: E402
from clients.weather import NWSClient  # noqa: E402
from persistence import db  # noqa: E402


# ── Configuration ────────────────────────────────────────────────────────

POLL_INTERVAL_SEC = int(os.environ.get("POLL_INTERVAL_SEC", 60))       # 1 min


def _parse_train_hours(raw: str) -> list[int]:
  hours: set[int] = set()
  for part in raw.split(","):
    part = part.strip()
    if not part:
      continue
    try:
      h = int(part)
    except ValueError:
      continue
    if 0 <= h <= 23:
      hours.add(h)
  return sorted(hours) or [0, 6, 12, 18]


# Comma-separated list of UTC hours to run the training pass. Default is
# every 6 hours (00, 06, 12, 18) so D-1 settlements land in the dashboard
# within a few hours of the venue publishing them.
TRAIN_HOURS_UTC   = _parse_train_hours(os.environ.get("TRAIN_HOURS_UTC", "0,6,12,18"))
LOG_DIR           = PROJECT_ROOT / "logs"
LOG_FILE          = LOG_DIR / "bot.log"
LOG_RETENTION_DAYS = 7

# Marker file format: one line "YYYY-MM-DD H1 H2 ..." where the integers
# are the scheduled training-hours that have already completed today.
# Resets implicitly when the date rolls over. Cheap, persistent, no DB
# schema needed. Old single-date markers ("YYYY-MM-DD") parse cleanly as
# "no hours done yet today" — at worst we re-run training once after an
# upgrade.
TRAIN_MARKER = PROJECT_ROOT / "data" / ".last_training_date"


# ── Logging setup ────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
  """
  Logs go to BOTH stdout (for tmux/foreground monitoring) AND a rotating
  file under logs/. Daily rotation, keep LOG_RETENTION_DAYS days.
  """
  LOG_DIR.mkdir(parents=True, exist_ok=True)
  fmt = logging.Formatter(
    fmt="%(asctime)sZ [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
  )
  fmt.converter = time.gmtime  # log in UTC

  root = logging.getLogger()
  root.setLevel(logging.INFO)
  # Wipe any handlers main.py installed at import time so we don't double-log
  root.handlers.clear()

  stream = logging.StreamHandler(sys.stdout)
  stream.setFormatter(fmt)
  root.addHandler(stream)

  rot = logging.handlers.TimedRotatingFileHandler(
    LOG_FILE, when="midnight", interval=1,
    backupCount=LOG_RETENTION_DAYS, utc=True,
  )
  rot.setFormatter(fmt)
  root.addHandler(rot)

  # Quiet down noisy third-party loggers
  logging.getLogger("urllib3").setLevel(logging.WARNING)
  logging.getLogger("requests").setLevel(logging.WARNING)

  return logging.getLogger("run_loop")


# ── Daily training pass ──────────────────────────────────────────────────

TRAINING_STEPS = [
  ("log_settlements", "bin.log_settlements"),
  ("settle_orders",   "bin.settle_orders"),
]


def run_training_pass(log: logging.Logger) -> None:
  """
  Run each training step in sequence as a subprocess. Each is invoked
  via `python -m <module>` so any sys.path / env quirks match the rest of
  the codebase. A failure in one step is logged but doesn't abort the
  others — better to have partial training than none.
  """
  log.info("=== Training pass starting ===")
  for name, module in TRAINING_STEPS:
    log.info("  → %s", name)
    try:
      result = subprocess.run(
        [sys.executable, "-m", module],
        cwd=str(PROJECT_ROOT),
        capture_output=True, text=True, timeout=60 * 30,  # 30 min hard cap
        env=os.environ.copy(),
      )
      if result.returncode != 0:
        log.warning("  %s exited %d. stderr:\n%s",
                    name, result.returncode, result.stderr[-2000:])
      else:
        # Surface the last few lines of each step so the rotating log has
        # a record of what changed (settlements counts, histogram updates).
        tail = "\n".join(result.stdout.strip().splitlines()[-5:])
        log.info("  %s ok. tail:\n%s", name, tail)
    except subprocess.TimeoutExpired:
      log.error("  %s timed out", name)
    except Exception:
      log.error("  %s crashed:\n%s", name, traceback.format_exc())
  log.info("=== Training pass complete ===")


def _read_marker() -> tuple[str | None, set[int]]:
  if not TRAIN_MARKER.exists():
    return None, set()
  parts = TRAIN_MARKER.read_text().strip().split()
  if not parts:
    return None, set()
  date = parts[0]
  hours: set[int] = set()
  for p in parts[1:]:
    try:
      hours.add(int(p))
    except ValueError:
      continue
  return date, hours


def _write_marker(date: str, hours: set[int]) -> None:
  TRAIN_MARKER.parent.mkdir(parents=True, exist_ok=True)
  TRAIN_MARKER.write_text(
    date + (" " + " ".join(str(h) for h in sorted(hours)) if hours else ""))


def _due_slot(now: datetime) -> int | None:
  """Most-recent scheduled training hour today that has already elapsed,
  or None if we're earlier than the first slot."""
  passed = [h for h in TRAIN_HOURS_UTC if h <= now.hour]
  return max(passed) if passed else None


def should_train_now(log: logging.Logger) -> bool:
  """
  True iff today's most-recently-elapsed scheduled slot hasn't been
  recorded as run yet. Idempotent across restarts via TRAIN_MARKER.
  """
  now = datetime.now(timezone.utc)
  target = _due_slot(now)
  if target is None:
    return False
  date, hours_done = _read_marker()
  if date == now.date().isoformat() and target in hours_done:
    return False
  return True


def mark_trained() -> None:
  now = datetime.now(timezone.utc)
  target = _due_slot(now)
  if target is None:
    return
  today = now.date().isoformat()
  date, hours_done = _read_marker()
  if date != today:
    hours_done = set()
  hours_done.add(target)
  _write_marker(today, hours_done)


# ── Polling pass ─────────────────────────────────────────────────────────

def run_poll_pass(log: logging.Logger,
                  k: KalshiClient, p: PolymarketClient, w: NWSClient) -> None:
  """Sweep every city in parallel. Each city is mostly HTTP I/O, so a
  thread pool wins easily; per-city errors stay isolated. Stdout from
  concurrent cities will interleave but the per-city ### headers in
  main.analyze_city keep each section legible.
  """
  log.info("--- Poll pass starting (%d cities, parallel) ---", len(CITIES))
  with ThreadPoolExecutor(max_workers=len(CITIES)) as pool:
    futures = {
      pool.submit(bot_main.analyze_city, name, cfg, k, p, w): name
      for name, cfg in CITIES.items()
    }
    for fut in as_completed(futures):
      name = futures[fut]
      try:
        fut.result()
      except Exception:
        log.error("[%s] poll failed:\n%s", name, traceback.format_exc())
  log.info("--- Poll pass complete ---")


# ── Main loop ────────────────────────────────────────────────────────────

_shutdown_requested = False


def _handle_signal(signum, frame):
  """Graceful shutdown: finish the current iteration, then exit."""
  global _shutdown_requested
  _shutdown_requested = True


def main():
  log = setup_logging()
  log.info("Bot starting. PAPER_TRADING=%s POLL_INTERVAL_SEC=%d TRAIN_HOURS_UTC=%s",
           os.environ.get("PAPER_TRADING"), POLL_INTERVAL_SEC,
           ",".join(str(h) for h in TRAIN_HOURS_UTC))
  log.info("Cities: %s", list(CITIES.keys()))
  log.info("Logs:   %s", LOG_FILE)

  # Make sure DB schema is in place before anything writes
  db.init()

  signal.signal(signal.SIGINT, _handle_signal)
  signal.signal(signal.SIGTERM, _handle_signal)

  # Build clients once; reuse across all poll iterations
  k = KalshiClient()
  p = PolymarketClient()
  w = NWSClient()

  while not _shutdown_requested:
    iter_start = time.monotonic()

    # Daily training first — newly-refreshed params then apply to today's polls
    if should_train_now(log):
      try:
        run_training_pass(log)
        mark_trained()
      except Exception:
        log.error("Training pass crashed:\n%s", traceback.format_exc())

    # Polling pass
    try:
      run_poll_pass(log, k, p, w)
    except KeyboardInterrupt:
      break
    except Exception:
      log.error("Poll pass crashed:\n%s", traceback.format_exc())

    # Sleep the remainder of the polling interval, in 1s chunks so we can
    # respond to SIGINT/SIGTERM promptly.
    elapsed = time.monotonic() - iter_start
    sleep_left = max(0.0, POLL_INTERVAL_SEC - elapsed)
    log.info("Iteration done in %.1fs; sleeping %.0fs", elapsed, sleep_left)
    while sleep_left > 0 and not _shutdown_requested:
      step = min(1.0, sleep_left)
      time.sleep(step)
      sleep_left -= step

  log.info("Shutdown signal received — exiting cleanly.")


if __name__ == "__main__":
  main()
