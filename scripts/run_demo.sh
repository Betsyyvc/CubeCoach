#!/usr/bin/env bash
set -euo pipefail

# Activate venv (works in WSL / Git Bash)
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
  source .venv/Scripts/activate
fi

python -m cubecoach.cli --demo
