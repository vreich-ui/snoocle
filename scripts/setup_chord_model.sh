#!/usr/bin/env bash
# Fetch and wire the real chord-recognition model: Chord-CNN-LSTM
# (ISMIR2019 large-vocabulary chord transcription, music-x-lab).
#
# The pretrained 5-fold checkpoints (~28MB total) ship IN the upstream repo,
# so a plain clone is a complete install — no LFS pull, no external download.
# Our snoocle_runner.py adapter is copied in; snoocle_server invokes it via
# the external-runner contract in snoocle_server/mir/chordrec.py.
#
# Usage: scripts/setup_chord_model.sh [target-dir]   (default models/chord-cnn-lstm)
set -euo pipefail

TARGET="${1:-models/chord-cnn-lstm}"
REPO_URL="https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition.git"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -e "$TARGET/chord_recognition.py" ]; then
    git clone --depth 1 "$REPO_URL" "$TARGET"
fi
cp "$SCRIPT_DIR/snoocle_runner.py" "$TARGET/snoocle_runner.py"

cat <<EOF
Chord-CNN-LSTM ready at: $TARGET

1. Install its Python deps into the SAME environment snoocle_server runs in
   (the runner is invoked with sys.executable). CPU-only torch is fine:
       pip install torch --index-url https://download.pytorch.org/whl/cpu
       pip install h5py pretty_midi mir_eval pydub
2. Point the server at it:
       export SNOOCLE_CHORD_CNN_LSTM_DIR=$(cd "$TARGET" && pwd)

Verify: /healthz should report mirEngines.chords = "chord-cnn-lstm".
EOF
