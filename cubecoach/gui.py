"""Minimal GUI for CubeCoach using PySimpleGUI (placeholder)."""
import PySimpleGUI as sg


def run_gui():
    layout = [
        [sg.Text("CubeCoach — demo GUI", font=(None, 16))],
        [sg.Button("Start Camera", key="-START-")],
        [sg.Button("Scan Face", key="-SCAN-"), sg.Button("Compute Solution", key="-SOLVE-")],
        [sg.Multiline(size=(60, 10), key="-LOG-")],
        [sg.Button("Exit")],
    ]

    window = sg.Window("CubeCoach", layout)

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, "Exit"):
            break
        if event == "-START-":
            window["-LOG-"].print("Camera started (demo). Use CLI demo to view webcam.")
        if event == "-SCAN-":
            window["-LOG-"].print("Scan face: TODO - open camera and collect sticker colors")
        if event == "-SOLVE-":
            window["-LOG-"].print("Compute solution: TODO - call solver with collected faces")

    window.close()


if __name__ == "__main__":
    run_gui()
