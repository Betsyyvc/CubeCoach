#!/usr/bin/env python3
"""Quick dependency checker for CubeCoach.

Run this after activating your virtual environment to see which packages are missing
and get suggested next steps.
"""
import importlib
import sys

pkgs = {
    'cv2': 'OpenCV (cv2) - required for webcam and vision processing',
    'PySimpleGUI': 'PySimpleGUI - GUI (optional for CLI demo)',
    'kociemba': 'kociemba - solver (may require build tools on Windows)'
}

missing = []

for name, desc in pkgs.items():
    try:
        importlib.import_module(name)
        print(f"{name}: OK — {desc}")
    except Exception as e:
        print(f"{name}: MISSING — {desc} (Error: {e.__class__.__name__}: {e})")
        missing.append(name)

if missing:
    print("\nNext steps:")
    print(" - Activate your virtual environment and run: python -m pip install -r requirements.txt")
    print(" - If 'kociemba' failed to build with a message about Microsoft Visual C++, install Visual C++ Build Tools:")
    print("   https://visualstudio.microsoft.com/visual-cpp-build-tools/")
    print(" - Alternatively, use WSL (Ubuntu): 'sudo apt update && sudo apt install build-essential python3-dev' and then pip install -r requirements.txt")
    print(" - If you don't need the solver immediately, remove or comment out 'kociemba' from requirements.txt and install other deps now; add the solver later when you have build tools available.")
    sys.exit(1)
else:
    print("\nAll required packages are installed. You're ready to run the demo!")
