#!/usr/bin/env bash
# Head-to-head stats: player vs opponent in Slippi Dolphin, rendered (Mac has no headless/fast-forward build).
# Usage: bash bots/evaluate.sh <player_model> <opponent_model> [steps=21600 (=6 min at 60fps)] [p1_char] [p2_char]
# Prints "ko diff per minute", "completed games", fps. For 200-game runs use a Linux CPU pod with vladfi1's
# fast-forward Dolphin build (headless), see https://github.com/vladfi1/libmelee#setup-instructions
cd "$(dirname "$0")/slippi-ai"
exec .venv/bin/python scripts/run_evaluator.py \
  --player.ai.path "${1:-../models/medium-v2}" --opponent.ai.path "${2:-../models/medium-v2}" \
  --player.character "${4:-fox}" --opponent.character "${5:-cptfalcon}" --dolphin.stage BATTLEFIELD \
  --num_envs 1 --noswap_ports --nodolphin.headless --rollout_length "${3:-21600}" --nouse_gpu \
  --dolphin.path "/Applications/Slippi Dolphin.app/Contents/MacOS/Slippi Dolphin" \
  --dolphin.iso "/Users/justincheng/Downloads/Super Smash Bros. Melee (USA) (En,Ja) (v1.02).iso"
