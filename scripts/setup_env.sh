#!/usr/bin/env bash
set -euo pipefail

# Cross-platform Bash helper to create a venv and install requirements
# Works in Git Bash, WSL, or other Bash terminals on Windows

if [ ! -d ".venv" ]; then
  python -m venv .venv
fi

# Try to source the typical activation scripts (POSIX and Windows Git Bash)
if [ -f ".venv/bin/activate" ]; then
  # WSL or normal Unix venv
  source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
  # Git Bash on Windows provides a Scripts/activate that's Bash-compatible
  source .venv/Scripts/activate
else
  echo "Could not find a venv activate script; activate manually: source .venv/bin/activate or .venv/Scripts/activate"
fi

python -m pip install --upgrade pip
pip install -r requirements.txt

echo "Environment set up and dependencies installed. To activate later: source .venv/bin/activate OR source .venv/Scripts/activate"
