# SmartPos AI setup and demonstration

See the main [README](README.md) for the detector policy, troubleshooting, local webcam validation instructions, and current limitations.

## Windows PowerShell

Run from the folder containing `app.py`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:5000, allow camera access, choose the correct view, and apply an upright calibration before starting a session. The default webcam index is 0; set `$env:SMARTPOS_CAMERA="1"` before launching to select another.

Hybrid mode is the default. The binary image model is diagnostic only until independent live webcam sessions have been collected and reviewed. Rule-only and image-only testing modes are available in the dashboard. The legacy five-class posture model is not used by the app; its files and historical notes are preserved in `legacy/five_class_pose/`.

No live webcam accuracy or physical neck angle is claimed. Frontal 2-D landmarks cannot reliably measure forward/backward depth. Use side-profile view for measurable image-plane evidence and treat UNKNOWN as an appropriate result when landmarks are insufficient.
