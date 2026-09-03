#!/usr/bin/env bash
# Bot vs bot in real Slippi Dolphin on the Mac. Usage: bash bots/play.sh [p1_char] [p2_char] [stage]
# Chars: fox cptfalcon falco marth sheik peach ... (lowercase, see --helpfull). Ctrl-C to stop.
cd "$(dirname "$0")/slippi-ai"
exec .venv/bin/python scripts/eval_two.py \
  --p1.ai.path ../models/medium-v2 --p2.ai.path ../models/medium-v2 \
  --p1.character "${1:-fox}" --p2.character "${2:-cptfalcon}" --dolphin.stage "${3:-BATTLEFIELD}" \
  --dolphin.path "/Applications/Slippi Dolphin.app/Contents/MacOS/Slippi Dolphin" \
  --dolphin.iso "/Users/justincheng/Downloads/Super Smash Bros. Melee (USA) (En,Ja) (v1.02).iso"
