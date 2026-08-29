#!/bin/bash
# Executive Communication Coach - Ambient Speech Auto-Nudge Listener
# Passively monitors room audio and triggers desktop alerts and continuous recording.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PYTHON_BIN="$DIR/venv/bin/python"

if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

echo "=========================================================================="
echo " 👂 Executive Communication Coach: Ambient Auto-Nudge Mode Active"
echo "=========================================================================="
echo " Passively idling in background (< 2.5% CPU)."
echo " When you or anyone starts speaking, you will get an instant macOS desktop"
echo " notification & terminal alert, and recording will capture the conversation."
echo "=========================================================================="
echo ""

"$PYTHON_BIN" "$DIR/core/record_live_coach.py" "$@"
