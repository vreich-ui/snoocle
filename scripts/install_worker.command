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
PY="$(command -v python3.11 || command -v python3.10 || command -v python3)"
[ -n "$PY" ] || die "No python3 found. Install it from python.org, then run this again."
say "Using $($PY --version) at $PY"

# --- ffmpeg ---------------------------------------------------------------
if ! command -v ffmpeg >/dev/null 2>&1; then
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

cat > "$ENVFILE" <<EOF
SNOOCLE_SERVER_URL="${SNOOCLE_SERVER_URL%/}"
SNOOCLE_API_TOKEN="$SNOOCLE_API_TOKEN"
SNOOCLE_WORKER_NAME="$(scutil --get ComputerName 2>/dev/null || hostname)"
EOF
chmod 600 "$ENVFILE"
say "Saved settings to $ENVFILE (readable only by you)."

# --- virtualenv -----------------------------------------------------------
say ""
say "Installing Snoocle into $VENV — this takes a few minutes the first time…"
"$PY" -m venv "$VENV" || die "Couldn't create the virtualenv."
"$VENV/bin/pip" install --quiet --upgrade pip || die "pip upgrade failed."
"$VENV/bin/pip" install --quiet -e "$REPO[mir]" \
  || die "Install failed. Run this in Terminal to see why:
    $VENV/bin/pip install -e '$REPO[mir]'"

command -v "$VENV/bin/snoocle-worker" >/dev/null 2>&1 || [ -x "$VENV/bin/snoocle-worker" ] \
  || die "snoocle-worker didn't install. Is this script inside the snoocle repo?"

# --- launchd agent --------------------------------------------------------
mkdir -p "$AGENT_DIR"
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
    <key>SNOOCLE_WORKER_NAME</key><string>$(scutil --get ComputerName 2>/dev/null || hostname)</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
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
