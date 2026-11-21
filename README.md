# smartpos ai

**Real-time posture monitoring with MediaPipe pose estimation and optional experimental posture classification.**

smartpos ai is an educational Flask application that processes a local webcam with OpenCV and MediaPipe Pose, measures upper-body geometry, and provides posture feedback through a responsive browser dashboard.

## Features

- Webcam MJPEG stream with MediaPipe landmark overlay
- Head-to-vertical neck alignment angle measured at the shoulder midpoint (no hip landmarks required)
- Shoulder tilt and normalized head-to-shoulder-center offset
- Configurable good/bad/unknown posture rules and delayed visual alerts
- Optional five-class scikit-learn pose mode and binary HOG image mode; rule-based detection remains the default
- ML posture category and general ergonomic suggestions, separate from geometric measurements
- Live prediction diagnostics for raw class, probability, final status, rejection reason and landmark visibility
- Delayed visual/beep warning with optional spoken correction
- Session start, pause, stop and reset
- Time-weighted posture score excluding missing-landmark intervals
- Five-second upright calibration using medians, with accept/reject
- Settings validation, responsive dark dashboard and optional browser beep

## Technologies and architecture

`Webcam → OpenCV capture → MediaPipe Pose inference → landmark visibility checks → geometric measurements → (rule-based detection or one explicitly selected experimental classifier) → Flask JSON/MJPEG endpoints → HTML/CSS/JavaScript dashboard`

The Flask process loads the optional models at startup and owns one background camera worker and a thread-safe session state. `/api/status` supplies geometric values, selected result, recent posture transitions and session warning count; `/api/mode` switches between rule-based, five-class pose, and binary image detection; `/api/session`, `/api/calibration` and `/api/settings` accept dashboard actions; `/video_feed` streams locally generated JPEG frames with posture overlays. Browser JavaScript polls status and plays an optional Web Audio beep when a warning first appears. The session score uses the selected detector's GOOD/BAD durations; model probability is displayed separately, is not calibrated, and is not the score.

**Model explanation:** MediaPipe Pose uses pretrained deep-learning models to infer body landmarks. The optional five-class posture classifier is a supplied scikit-learn `MLPClassifier` trained on MultiPosture data; the separately trained GOOD/BAD model uses HOG image features and logistic regression. The five-class artifact's participant-held-out offline accuracy is **57.04%** (balanced accuracy **56.40%**). The binary image model's 44-image, grouped same-session evaluation is reported in [IMAGE_MODEL_README.md](IMAGE_MODEL_README.md), including per-class metrics. These offline figures are not directly comparable because their data and split strategies differ; neither is verified real-world webcam accuracy, and the reported probabilities are not calibrated safety confidence. Pose-ML directional outputs are conservatively checked against independent geometry: an unsupported forward/backward guess is shown as UNKNOWN, and supported depth-oriented labels are reduced to generic head-alignment feedback because frontal 2-D geometry cannot verify depth. The rule-based mode remains the default. This project is an educational demonstration, **not medical advice**.

| Mode | Evidence and selection strategy |
|---|---|
| Rule-based (default) | Uses measured neck alignment, shoulder tilt and normalized head-center offset. Missing/low-visibility landmarks remain UNKNOWN. No trained-model accuracy is claimed. |
| Experimental pose ML | Uses the five-class landmark model (57.04% participant-held-out offline accuracy). Model labels are not fused blindly: upright must agree with GOOD geometry, lateral labels need a matching head offset, and forward/backward labels need a measurable alignment deviation but are reported only as generic head alignment. |
| Experimental image | Uses the binary GOOD/BAD HOG model (86.36% on its small grouped same-session split; per-class results in [IMAGE_MODEL_README.md](IMAGE_MODEL_README.md)). Requires a visible head and both shoulders; it cannot identify a specific posture issue. |

The two offline model scores have different datasets and split designs and must not be used as a head-to-head comparison or webcam accuracy claim.

## Run on Windows (PowerShell)

Use **Python 3.12**. From the project directory:

```powershell
cd smartpos-ai
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:5000 in your browser and allow Windows camera access. The default camera is index 0. To select the environment in VS Code, run **Python: Select Interpreter** and choose `.\.venv\Scripts\python.exe`.

To use another webcam in PowerShell:

```powershell
$env:SMARTPOS_CAMERA="1"
.\.venv\Scripts\python.exe app.py
```

On Linux/macOS use `python3.12 -m venv .venv`, `source .venv/bin/activate`, then `pip install -r requirements.txt` and `python app.py`.

## How to demonstrate

1. Face the webcam with your **head and both shoulders visible**. Hips are not required; MediaPipe uses visible ears (or nose fallback) plus both shoulders.
2. Click **Start / Resume** and hold your head approximately over your shoulders to see the good classification.
3. Safely move your head forward/sideways or tilt your shoulders to see the bad classification and warning after the configured delay.
4. Pause, resume, stop and reset the session; verify elapsed, good/bad tracked time and score.
5. Hold an upright pose and click **Calibrate · 5 sec**. Review the suggested thresholds and try both **Apply thresholds** and **Reject**.
6. Adjust a valid threshold and click **Save settings**. Move out of frame to observe the unknown/missing-landmark state.
7. Optionally select **Experimental ML model** to see a model label, five-class category and general ergonomic suggestion. Switch back to **Rule-based** for geometric classification. If required model landmarks are absent or probability is below threshold, ML mode reports UNKNOWN.
8. Use **Posture history** and the warning count while the session is running; warning sounds trigger once per sustained warning episode, not once per frame.

## Default thresholds

- Neck minimum: **150°** (lower is classified as bad)
- Shoulder tilt maximum: **10°** (higher is classified as bad)
- Head-center offset maximum: **0.18 shoulder widths** from the calibrated neutral center
- Alert delay: **3 seconds**

The neck value is the angle between the head-to-shoulder line and a vertical-down reference at the shoulder midpoint; upright alignment is near 180°. It and the shoulder/head-center measurements are image-space approximations, not medical measurements. The frontal-view geometry can identify visible shoulder tilt and head offset, but cannot reliably infer forward/backward depth displacement or diagnose neck angles. Decisions require eight stable frames; the rule-based detector also uses narrow exit hysteresis to reduce threshold flicker. Calibration uses median neck angle minus 10°, median shoulder tilt plus 5°, and the observed neutral head-center offset. These are heuristics, not clinical thresholds.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover geometry, head/shoulder measurements with partial-body visibility, missing landmarks, sideways and shoulder-tilt rules, model label/probability mapping, image-model loading, rule/ML/image mode separation, score arithmetic, calibration acceptance/rejection and expiry, session start/stop/reset, warning counting, model-loading fallback, overlays and Flask routes/input validation. The reported offline metrics are not a live accuracy guarantee.

## Project layout

```text
smartpos-ai/
├── app.py                 Flask app and JSON/video endpoints
├── smartpos/
│   ├── camera.py          Webcam + MediaPipe inference worker
│   ├── geometry.py        Independent geometry/score helpers
│   └── state.py           Session, alerts, calibration state
├── templates/index.html  Dashboard markup
├── static/styles.css     Dashboard appearance
├── static/app.js         Live polling and controls
├── tests/test_geometry.py
├── posture_model.joblib  Experimental trained classifier
├── posture_inference.py  Feature extraction and prediction adapter
├── image_posture_model.joblib  Experimental GOOD/BAD image classifier
├── image_posture_inference.py  Image feature prediction adapter
├── train_model.py        Reproducible, subject-separated training
├── train_image_posture.py  Image model training/evaluation
├── metrics.json          Held-out-subject evaluation metrics
├── README_MODEL.md       Model details and limitations
├── requirements.txt
├── README.md
└── SETUP_GUIDE.md
```

## Troubleshooting

- **Camera unavailable:** close Zoom/Teams, check camera permissions, try `SMARTPOS_CAMERA=1`.
- **UNKNOWN or missing angles:** keep your head and both shoulders visible; hips are not needed. Use even lighting and face the camera.
- **ML unavailable:** check `/api/status` for `model_error`, keep the model file beside `app.py`, and use the pinned scikit-learn requirement. Rule-based mode remains available if model loading fails.
- **Pose ML shows UNKNOWN:** it requires visible nose, both ears and both shoulders with sufficient visibility and a probability of at least 0.55.
- **Image model shows UNKNOWN:** its maximum binary class probability is below 0.65, the model failed, or no usable camera frame is available.
- **MediaPipe import fails:** use Python 3.12 and reinstall requirements in the project virtual environment. `mediapipe==0.10.14` includes a Windows CPython 3.12 wheel.
- **Dependency download/hash error:** retry on a stable network and use a fresh virtual environment. Do not disable package integrity checks.
- **Score stays at zero:** click Start and keep the required landmarks visible.
- **Alerts are too sensitive:** use calibration or change thresholds. Browser sound requires browser audio support and may require user interaction; the visual warning remains available.

## Limitations and privacy

Single-user local-only demo. Measurements and model classifications depend on camera angle, clothing, landmark visibility and lighting. Front-facing 2-D shoulder/head geometry does not reliably measure depth-oriented leaning. The five-class model has modest held-out accuracy; the image model has only a small same-session evaluation, and neither has verified live-camera accuracy. The supplied 123-image source dataset is not included, so near-duplicate images cannot be independently audited from this workspace; the training script groups timestamped adjacent frames, but this does not establish independent-person validation. The session data is in memory and is lost when the process exits. No recordings, cloud uploads, account system or persistent database are included. The server binds to `127.0.0.1` and has no authentication; do not expose it to an untrusted network.

## Reference and development note

Functionality was informed by the public [Posture-Detection-ComputerVision reference project](https://github.com/MuhammadIshaq-AI/Posture-Detection-ComputerVision). This codebase was written separately rather than copied from that repository. The posture classifier was trained from the supplied MultiPosture data; consult the dataset terms before redistribution. MediaPipe, OpenCV, Flask, NumPy, scikit-learn, joblib and their transitive packages are external dependencies governed by their respective licenses. AI assistance was used during development; review and follow your institution's AI-use and attribution requirements.

## Publishing to GitHub

The project directory can be initialized as a new repository after review. Create an empty repository named `smartpos-ai` on GitHub, then follow the commands in [SETUP_GUIDE.md](SETUP_GUIDE.md). Do not push credentials or the `.venv` directory.
