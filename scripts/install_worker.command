#!/bin/bash
# Double-click this once. It sets the Mac up to analyze songs for Snoocle in
# the background, forever, starting at login.
#
# What it does: makes a private virtualenv, installs Snoocle into it, asks for
# the server URL and token, and registers a launchd agent that keeps the worker
# running. Nothing listens on a port; the worker only makes outgoing requests.

HERE="$(cd "$(dirname "$0")" && pwd)" || exit 1
REPO="$(cd "$HERE/.." && pwd)" || exit 1

LABEL="com.snoocle.worker"
AGENT_DIR="$HOME/Library/LaunchAgents"
PLIST="$AGENT_DIR/$LABEL.plist"
VENV="$HOME/.snoocle-worker/venv"
ENVFILE="$HOME/.snoocle-worker/env"
LOGDIR="$HOME/Library/Logs/Snoocle"

say() { printf '%s\n' "$*"; }
die() { say ""; say "✗ $*"; say ""; say "Press return to close."; read -r _; exit 1; }

say "Setting up the Snoocle worker."
say ""

# --- python ---------------------------------------------------------------
# The venv must be built on 3.10–3.12: macOS system python3 is 3.9 (too old
# for the server package) and the ML dependencies are not vetted on 3.13+.
# Candidates are checked by ASKING THE INTERPRETER, never guessed from the
# name — a bare `python3` is accepted only if it really is 3.10–3.12, and the
# old silent fall-through to 3.9 (which produced a venv that could never run
# demucs) is now a loud failure instead.
py_ok() { "$1" -c 'import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)' 2>/dev/null; }

PY=""
for cand in python3.12 python3.11 python3.10 \
            /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 \
            /usr/local/bin/python3.12 /usr/local/bin/python3.11 \
            /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
            /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 \
            "$HOME/.snoocle-worker/mm/envs/snoocle/bin/python3.12" \
            "$HOME/.snoocle-worker/mm/envs/snoocle/bin/python3.11" \
            python3; do
  loc="$(command -v "$cand" 2>/dev/null)" || continue
  [ -n "$loc" ] && py_ok "$loc" && PY="$loc" && break
done
[ -n "$PY" ] || die "No Python 3.10–3.12 found (system python3 is too old, and 3.13+ is
  not supported by the analysis dependencies). Install one first:
    brew install python@3.12       # or python.org, or any 3.10–3.12
  then run this again."
say "Using $($PY --version) at $PY"

# --- ffmpeg ---------------------------------------------------------------
# Look beyond the login PATH: Homebrew on both prefixes, MacPorts, and the
# micromamba toolchain some machines use instead of Homebrew. Remember where
# it lives — the launchd agent gets that directory on ITS path below, because
# launchd agents do not inherit a login shell's PATH.
FFMPEG_BIN=""
for cand in ffmpeg /opt/homebrew/bin/ffmpeg /usr/local/bin/ffmpeg \
            /opt/local/bin/ffmpeg "$HOME/.snoocle-worker/mm/envs/snoocle/bin/ffmpeg"; do
  loc="$(command -v "$cand" 2>/dev/null)" || continue
  [ -n "$loc" ] && [ -x "$loc" ] && FFMPEG_BIN="$loc" && break
done
if [ -z "$FFMPEG_BIN" ]; then
  say ""
  say "! ffmpeg isn't installed. Audio conversion needs it:"
  say "    brew install ffmpeg"
  say "  The worker will install anyway, but jobs will fail until ffmpeg is there."
  say ""
fi

# --- credentials ----------------------------------------------------------
mkdir -p "$(dirname "$ENVFILE")" "$LOGDIR" || die "Couldn't create ~/.snoocle-worker"

if [ -f "$ENVFILE" ]; then
  # shellcheck disable=SC1090
  . "$ENVFILE"
fi

if [ -z "$SNOOCLE_SERVER_URL" ]; then
  say "Server URL (e.g. https://snoocle-99287560712.europe-west1.run.app):"
  read -r SNOOCLE_SERVER_URL
fi
[ -n "$SNOOCLE_SERVER_URL" ] || die "A server URL is required."

if [ -z "$SNOOCLE_API_TOKEN" ]; then
  say "API token (leave blank if the server doesn't use one):"
  read -r SNOOCLE_API_TOKEN
fi

# The machine name feeds an HTTP header, and macOS's default naming scheme
# ("<Name>'s MacBook") contains a curly apostrophe that is not ASCII. The
# worker sanitises its own header now, but store an ASCII name anyway so the
# value survives every shell, plist and log that touches it.
WORKER_NAME="$("$PY" -c '
import re, subprocess, unicodedata
raw = subprocess.run(["scutil", "--get", "ComputerName"],
                     capture_output=True, text=True).stdout.strip()
raw = raw or subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip()
raw = raw.translate(str.maketrans({"\\u2018": chr(39), "\\u2019": chr(39), "\\u2013": "-", "\\u2014": "-"}))
raw = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode()
# The value is embedded verbatim in plist XML and in a double-quoted, sourced
# shell file - whitelist, because &, <, $ or a backtick in a machine name
# would corrupt one or the other.
raw = re.sub(r"[^A-Za-z0-9 ._()\x27-]+", " ", raw)
print(" ".join(raw.split()) or "snoocle-mac")')"

cat > "$ENVFILE" <<EOF
SNOOCLE_SERVER_URL="${SNOOCLE_SERVER_URL%/}"
SNOOCLE_API_TOKEN="$SNOOCLE_API_TOKEN"
SNOOCLE_WORKER_NAME="$WORKER_NAME"
EOF
chmod 600 "$ENVFILE"
say "Saved settings to $ENVFILE (readable only by you)."

# --- virtualenv -----------------------------------------------------------
say ""
say "Installing Snoocle into $VENV — this takes a few minutes the first time…"
# A healthy venv on a supported Python is kept, not clobbered — it may hold
# multi-GB ML packages (torch, demucs) that a rebuild would re-download.
# Anything else (missing, broken, or built on the wrong Python) is replaced.
if [ -x "$VENV/bin/python" ] && py_ok "$VENV/bin/python"; then
  say "Existing venv on $("$VENV/bin/python" --version 2>&1) — keeping it."
else
  if [ -e "$VENV" ]; then
    say "Existing venv is unusable (missing or wrong Python) — rebuilding."
    rm -rf "$VENV" || die "Couldn't remove the old virtualenv."
  fi
  "$PY" -m venv "$VENV" || die "Couldn't create the virtualenv."
fi
"$VENV/bin/pip" install --quiet --upgrade pip || die "pip upgrade failed."
"$VENV/bin/pip" install --quiet -e "$REPO[mir]" \
  || die "Install failed. Run this in Terminal to see why:
    $VENV/bin/pip install -e '$REPO[mir]'"

command -v "$VENV/bin/snoocle-worker" >/dev/null 2>&1 || [ -x "$VENV/bin/snoocle-worker" ] \
  || die "snoocle-worker didn't install. Is this script inside the snoocle repo?"

# --- launchd agent --------------------------------------------------------
mkdir -p "$AGENT_DIR"

# launchd agents get a bare PATH, not the login shell's. Build one from where
# the tools were ACTUALLY found (ffmpeg may live in a micromamba env or
# MacPorts, not Homebrew), with the standard prefixes appended as fallback.
AGENT_PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
[ -n "$FFMPEG_BIN" ] && AGENT_PATH="$(dirname "$FFMPEG_BIN"):$AGENT_PATH"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$VENV/bin/snoocle-worker</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>SNOOCLE_SERVER_URL</key><string>${SNOOCLE_SERVER_URL%/}</string>
    <key>SNOOCLE_API_TOKEN</key><string>$SNOOCLE_API_TOKEN</string>
    <key>SNOOCLE_WORKER_NAME</key><string>$WORKER_NAME</string>
    <key>PATH</key><string>$AGENT_PATH</string>
    <!-- launchd gives the agent no locale, so Python falls back to ASCII
         for anything it must encode; and stdout to a log file is
         block-buffered, so worker.log lags by kilobytes without this. -->
    <key>LANG</key><string>en_US.UTF-8</string>
    <key>PYTHONUNBUFFERED</key><string>1</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <!-- Throttle restarts so a misconfigured worker can't spin. -->
  <key>ThrottleInterval</key><integer>30</integer>
  <key>StandardOutPath</key><string>$LOGDIR/worker.log</string>
  <key>StandardErrorPath</key><string>$LOGDIR/worker.log</string>
  <key>ProcessType</key><string>Background</string>
  <!-- Analysis is long and CPU-heavy; don't let macOS deprioritise it into
       uselessness, but do let the machine sleep normally. -->
  <key>LowPriorityIO</key><false/>
</dict>
</plist>
EOF

launchctl bootout "gui/$UID/$LABEL" 2>/dev/null
launchctl bootstrap "gui/$UID" "$PLIST" 2>/dev/null \
  || launchctl load -w "$PLIST" 2>/dev/null \
  || die "Couldn't register the background agent."

sleep 2
say ""
if launchctl print "gui/$UID/$LABEL" >/dev/null 2>&1; then
  say "✓ The worker is running and will start again at every login."
else
  say "! Registered, but it isn't running yet. Check $LOGDIR/worker.log"
fi
say ""
say "It picks up songs you queue in Snoocle. Log: $LOGDIR/worker.log"
say "To stop it:  launchctl bootout gui/$UID/$LABEL"
say ""
say "Press return to close."
read -r _
