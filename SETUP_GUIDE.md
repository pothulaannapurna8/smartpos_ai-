# smartpos ai — setup, demonstration and viva guide

## Requirements and installation

Use Windows, VS Code and Python 3.12. In PowerShell, change to the directory containing `app.py`, then run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

In VS Code, use **Python: Select Interpreter** and choose `.\.venv\Scripts\python.exe`.

The direct requirements are Flask `>=3.0,<4`, OpenCV contrib `>=4.8,<5`, MediaPipe `==0.10.14`, NumPy `>=1.26,<2`, scikit-learn `==1.8.0`, and joblib `>=1.4,<2`. MediaPipe has additional transitive dependencies. Python 3.12 is recommended because the pinned MediaPipe release provides a Windows CPython 3.12 wheel.

## Run and test

```powershell
.\.venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:5000` and allow camera access in Windows. The default camera index is 0; to select another camera, set `$env:SMARTPOS_CAMERA="1"` before starting the process.

Run the automated tests in a second PowerShell terminal:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The app binds only to localhost. The session is held in memory and resets when the process exits.

## Demonstration procedure

1. Position the camera near eye level; use even lighting and keep your head and both shoulders in view. Hips are not required.
2. Confirm the dashboard shows a live camera and landmark overlay.
3. Start/resume a session and check the posture label and measured angles.
4. Briefly change your head/shoulder position to observe rule-based classification and the delayed visual warning. Do not force an uncomfortable posture.
5. Pause and resume; stop and reset. Observe the timer and valid good/bad duration score.
6. Hold your upright pose for five seconds during calibration, then accept the suggested limits. Repeat and reject a suggestion to verify the existing settings remain.
7. Change a threshold and save. Move out of view to check missing-landmark handling.
8. Browser sound is optional and may require an interaction with the page or audio permissions.
9. Select **Experimental ML model** to display the five-class posture prediction and general ergonomic advice. The model requires nose, both ears and both shoulders; insufficient visibility or low model probability produces UNKNOWN. Select **Rule-based** to return to the geometric threshold detector.

The classifier's reported offline evaluation is 57.04% accuracy and 56.40% balanced accuracy on three held-out subjects. These are not live webcam results, and the model is experimental rather than medically validated. See [README_MODEL.md](README_MODEL.md) for its input and limitations.

In ML mode, `TUP` maps to GOOD/UPRIGHT; the four lean classes map to BAD. A BAD `LEAN_FORWARD` result displays: “Move your head back over your shoulders and sit comfortably upright.” Warnings appear only after BAD persists for the configured delay and reset when GOOD, UNKNOWN, or a stale camera frame occurs. Open **Live prediction diagnostics** to inspect raw class, probability, final status, rejection reason and landmark visibility. Spoken corrections are optional; the visual warning and browser beep remain separate.

## Retraining the experimental model

The training dataset is not included in this project. If you have the authorized MultiPosture `data.csv`, run this from the directory containing `app.py`:

```powershell
.\.venv\Scripts\python.exe train_model.py --data ..\data.csv
```

By default, this replaces `posture_model.joblib` and `metrics.json` in the project directory. Keep a backup if you want to retain the current artifacts. Confirm the dataset's terms before redistribution. The Flask app loads the model once at startup; restart Flask after retraining.

## System architecture and deep-learning explanation

OpenCV captures frames locally. MediaPipe Pose applies its pretrained deep-learning model to infer body landmarks. The app requires both visible shoulders plus a visible head point (the midpoint of both ears, the nose fallback, or one visible ear); it computes shoulder tilt and an image-space head-to-vertical alignment angle without requiring hip landmarks. Configurable threshold rules decide `good`, `bad` or `unknown`. Flask serves the dashboard, status/action JSON APIs and an MJPEG stream; browser JavaScript updates the display and optional sound alert.

MediaPipe's pretrained deep-learning pose estimator infers landmarks. The default posture detector then uses independent geometry and configurable threshold rules. The optional experimental classifier is a supplied scikit-learn `MLPClassifier`, retrained locally from the provided MultiPosture dataset when its original serialized artifact was incompatible with this environment. It uses five head/shoulder landmarks; its prediction and confidence are separate from geometric angle measurements. The dashboard's score uses the selected detector's tracked GOOD/BAD durations; unknown intervals and model probability are excluded from score arithmetic. This is an educational prototype, not a medical device or diagnosis.

## Common viva questions

**Where is deep learning used?** In MediaPipe's pretrained pose-estimation inference.

**What does the posture classifier do?** MediaPipe infers landmarks with pretrained deep-learning models. The default detector applies geometry and rules. An optional experimental MLP classifier was trained on the supplied MultiPosture dataset; it is separate from MediaPipe.

**What is the posture score?** Good posture duration divided by good+bad valid duration, times 100.

**What if the person leaves the frame?** Posture becomes unknown, and unknown time is excluded from the score.

**What is calibration?** A five-second sample of upright posture; medians yield suggested threshold settings.

**What is a limitation?** Front-camera geometry is sensitive to perspective and visibility.

**How is rule-based posture classified?** The neck angle and shoulder tilt are compared with adjustable thresholds. In experimental ML mode, normalized nose, ear and shoulder coordinates are passed to the MLP classifier.

**Is the classifier accurate or medically validated?** Its subject-separated offline test accuracy was 57.04% (balanced accuracy 56.40%). Live webcam accuracy is unverified and it is not clinically validated.

**What does the webcam stream do?** The local server returns annotated JPEG frames using an MJPEG response.

## Limitations and troubleshooting

- Measurement needs both shoulders and at least one sufficiently visible head landmark; hips are not used by the posture geometry.
- Perspective, camera placement, lighting and clothing affect the measurements; thresholds are configurable heuristics.
- Calibration needs enough visible samples. If it reports too few, improve visibility and repeat.
- Webcam access can fail if another application owns the device or Windows permissions deny access. Try another index with `SMARTPOS_CAMERA`.
- Browser audio is optional; the visual warning does not depend on sound.
- If the ML mode is unavailable, inspect the model-load error from `http://127.0.0.1:5000/api/status`; the default rule-based mode remains available.
- The model artifact must come from a trusted source because joblib deserialization uses pickle.
- No video is uploaded or recorded. This app has no login and should remain on the local machine.

## Publishing to GitHub

Create an empty GitHub repository called `smartpos-ai`. From this project directory, after reviewing the files and your institution's academic integrity and AI-use requirements:

```powershell
git init
git add .
git commit -m "Initial smartpos ai implementation"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/smartpos-ai.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username. The `.gitignore` excludes the virtual environment, secrets and common temporary files. Do not claim the pretrained MediaPipe model was developed or trained by you, and do not push until you have approved the repository contents.
