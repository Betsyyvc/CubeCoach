# CubeCoach

CubeCoach is a Python app that uses computer vision to help users scan their Rubik's Cube and get step-by-step solve instructions.

## Quickstart

1. Create a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Bash (Git Bash / WSL) — in VS Code set terminal to Git Bash or open WSL
# then run (POSIX or Git Bash):
source .venv/bin/activate  # WSL or Unix-like
# or (Git Bash on Windows):
source .venv/Scripts/activate

# Or use the helper script for Bash:
./scripts/setup_env.sh

# Install dependencies (if not using helper)
pip install -r requirements.txt
```

2. Run the camera demo:

```bash
python -m cubecoach.cli --demo
```

## Project layout

```
cubecoach/
├─ vision/        # camera and detection code
├─ solver/        # solver wrappers
├─ gui.py         # minimal GUI demo
├─ cli.py         # entry points and demos
├─ __init__.py
```

Run the GUI demo with:

```bash
python -m cubecoach.gui
```

# Running from VS Code Git Bash terminal

If you use the integrated **Git Bash** terminal in VS Code, here are simple ways to run the demos without having to `source` the venv activation script.

- Preferred (direct venv python executable — works even if `source` doesn't):

```bash
.venv/Scripts/python -m cubecoach.cli --demo
.venv/Scripts/python -m cubecoach.cli --calibrate
.venv/Scripts/python -m cubecoach.cli --scan
```

- Or activate and run (if `source` works in your shell):

```bash
source .venv/Scripts/activate   # Git Bash
# or (WSL): source .venv/bin/activate
python -m cubecoach.cli --demo
```

- Quick install helper (installs everything except `kociemba`):

```bash
.venv/Scripts/python -m pip install --upgrade pip wheel
.venv/Scripts/python -m pip install -r requirements-except-kociemba.txt
```

- You can also use the VS Code Tasks I added: open the Command Palette (Ctrl+Shift+P) → "Tasks: Run Task" → choose `Run CubeCoach demo` / `Run CubeCoach scan` / `Install deps (except kociemba)`.


## Next steps

- Implement robust sticker detection in `cubecoach.vision.detector`
- Add a GUI and scan workflow to capture all 6 faces
- Integrate `cubecoach.solver.kociemba_solver` to provide step-by-step instructions

### Demo notes

- The camera demo shows live video. Use `cubecoach.vision.detector.sample_colors` as a starting point to sample sticker colors from regions-of-interest.
- You can run tests with `pytest` (there is a placeholder test in `tests/test_camera_demo.py`).

## Troubleshooting

- If you see an error like "source: the term 'source' is not recognized ..." you are in PowerShell, not Bash. In PowerShell activate the venv with:

```powershell
.\.venv\Scripts\Activate.ps1
```

  - If you get an execution policy error when running the PowerShell activation, run PowerShell as an administrator once and run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

- If you want Bash (Git Bash or WSL), open a **Git Bash** or **WSL** terminal in VS Code (Terminal → New Terminal → choose *Git Bash*), then activate with:

```bash
source .venv/bin/activate   # WSL / Unix-like
# or (Git Bash on Windows)
source .venv/Scripts/activate
```

- You can also use the Bash helper script (run it in a Bash terminal):

```bash
./scripts/setup_env.sh
```

- Make sure dependencies are installed into the same Python interpreter you run commands with. Use:

```bash
python -m pip install -r requirements.txt
python -m pip show opencv-python PySimpleGUI
python -c "import sys; print(sys.executable)"
```

- If `ModuleNotFoundError: No module named 'cv2'` or `No module named 'PySimpleGUI'` appears, the fix is to activate the correct venv and install the requirements (see commands above).

## Calibration tool

A simple calibration tool helps map camera colors to cube faces. Run it with:

```bash
python -m cubecoach.vision.calibrate
# or via the CLI
python -m cubecoach.cli --calibrate
```

- On the calibration screen press one of the keys to record samples for a face: `u` (Up), `r` (Right), `f` (Front), `d` (Down), `l` (Left), `b` (Back).
- Press `s` to save averaged HSV centers to `cubecoach/vision/calibration.json`.
- Press `t` to toggle a live test mode which maps the center patch to a label using the current calibration.
- After saving, the color mapper will use the calibration for `map_bgr_to_label` in `cubecoach.vision.colors`.

## Scanning faces (guided scan)

After you have a calibration saved, you can scan all 6 faces with the guided scanner:

```bash
python -m cubecoach.cli --scan
```

Instructions while scanning:

- Point the camera at one cube face and press the key for that face: `u`, `r`, `f`, `d`, `l`, `b` corresponding to Up/Right/Front/Down/Left/Back.
- The scanner will detect the face, warp it to a top-down view, sample the 3×3 sticker colors and map them using your calibration.
- After pressing a face key, a **preview window** will appear showing the 3×3 *raw* color samples; press **`y`** to accept and store the raw samples for that face or **`n`** to reject and try again.
- After you have accepted all 6 faces the scanner will automatically compute a **per-camera calibration** from the collected faces (center sticker heuristic + median fallback), save it to `cubecoach/vision/calibration.json`, remap all stickers using perceptual LAB distances, and save the final face images and mappings under `cubecoach/vision/scans/`.
- Repeat scanning if any face looks incorrect — you can re-scan a face by pressing its key again before the 6 faces are complete.

Note: if you already have a calibration file, you can still use `--calibrate` to fine-tune it manually; the automatic calibration is a convenience when you don't have one yet.


### "Failed to build wheel for kociemba" (Windows)

- If you see an error during `pip install` like "Failed to build kociemba" with the message "Microsoft Visual C++ 14.0 or greater is required", that means a C extension needs MSVC to compile.

  - Recommended: Install the **Microsoft Visual C++ Build Tools** (choose "C++ build tools") from:

    https://visualstudio.microsoft.com/visual-cpp-build-tools/

    After installing, re-open your terminal, activate the venv, and re-run:

    ```bash
    python -m pip install -r requirements.txt
    ```

  - Alternative: Use **WSL (Ubuntu)** — open a WSL terminal and run:

    ```bash
    sudo apt update && sudo apt install build-essential python3-dev
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

  - Temporary workaround: If you don't need the solver immediately, remove or comment out `kociemba` from `requirements.txt`, install the other deps, and add the solver later once build tools are available.

- Quick diagnostic: run the included script to check dependencies and get guidance:

```bash
python scripts/check_dependencies.py
```

This will tell you which packages are missing and point you to the appropriate next steps.


---

---
