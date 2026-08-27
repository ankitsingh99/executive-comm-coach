#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
"$DIR/venv/bin/python" "$DIR/core/record_live_coach.py" "$@"
