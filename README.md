# SmartPos AI

SmartPos AI is a local Flask application for webcam-based posture feedback. It uses OpenCV capture, MediaPipe Pose landmarks, an explicitly calibrated geometry detector, and the project's existing binary GOOD/BAD HOG image classifier.

## What changed in the active detector

- **Hybrid is the default.** It uses calibrated geometry for final decisions. A reliable geometric BAD remains BAD regardless of AI output; a confident AI BAD that conflicts with geometric GOOD becomes UNKNOWN, never BAD. The binary image model's raw label/confidence remain visible in diagnostics, but the model is advisory until it has been validated on separate live webcam sessions.
- **Rule-only** and **image-only** modes are available for testing. Image-only low-confidence predictions remain UNKNOWN. The legacy five-class posture classifier is no longer imported, loaded, selectable, or used in decisions.
- The old five-class model, inference/training scripts, metrics, and source dataset are preserved under [`legacy/five_class_pose/`](legacy/five_class_pose/). Its historical attribution and model notes remain in that folder.
- Frontal geometry measures head displacement relative to the shoulder midpoint, normalized horizontal head shift, head/shoulder image-plane alignment, shoulder tilt, and optional torso lean when hips are visible. Side-profile mode uses the visible profile shoulder/head and, when available, hip landmarks to measure image-plane alignment and torso lean.
- Apply the five-second upright calibration before relying on rule/hybrid status. The baseline is view-specific; changing frontal/side view clears it and requires recalibration. Missing/low-visibility required landmarks immediately produce UNKNOWN. GOOD/BAD transitions are smoothed, with wider recovery hysteresis for BAD results.
- Frontal 2-D landmarks cannot establish forward/backward depth. Use **Side profile** for visible screen-plane evidence; the app does not infer a physical neck angle or use MediaPipe's uncertain frontal depth coordinate as a certainty.
- The dashboard shows rule status/reason, binary AI label/confidence/reason, final prediction/reason, landmark visibility, head shift, image-plane alignment, shoulder tilt, optional torso lean, calibration, alerts and session score.

## Root causes identified in the previous source

1. The previous geometry used only 2-D head/shoulder positions. A frontal camera cannot see a forward lean that moves mainly along the camera's depth axis, so the posture can remain GOOD without any measurable 2-D change.
2. Sideways posture depended on one normalized horizontal head-offset threshold and a zero baseline unless calibration had been manually accepted. The old offset denominator also included the shoulder line's vertical component, which could reduce the reported horizontal offset when shoulders tilted. The previous pipeline did not require a valid calibrated baseline before returning GOOD.
3. The old five-class model was loaded automatically at startup even though geometry was the default; its labels and APIs were still present in dashboard mode selection and inference branches.
4. The saved binary image model's preprocessing and class mapping were internally consistent: BGR input, grayscale conversion, 64×64 full-frame resize, histogram equalization, 1,764 HOG features, class `0 = BAD`, class `1 = GOOD`. Live inference used the original unmirrored BGR camera frame just as training used `cv2.imread`; no crop is applied in either path. This verifies preprocessing parity, **not** webcam accuracy. If live predictions differ, domain shift remains possible and must be measured with the validation utility rather than assumed.

## Setup and start (Windows PowerShell)

Use Python 3.12. Run these commands from the directory containing `app.py`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) and allow camera access. The default webcam index is `0`. For another camera, stop the app, then run:

```powershell
$env:SMARTPOS_CAMERA="1"
.\.venv\Scripts\python.exe app.py
```

For Linux/macOS, create a Python 3.12 virtual environment, install `requirements.txt`, and run `python app.py`. The server binds to localhost only.

## First use and manual webcam checks

1. Choose **Frontal** view when facing the camera, or **Side profile** when the camera sees your profile.
2. Keep the required head and shoulder landmarks visible. Click **Calibrate · 5 sec** while sitting comfortably upright; apply the proposed baseline. If calibration reports too few samples, improve lighting/framing and try again.
3. Start a session and check that upright posture becomes GOOD after the stable-frame delay. The score counts only final GOOD/BAD time; UNKNOWN intervals are excluded.
4. In frontal view, gently shift your head left/right or raise one shoulder to test measurable geometry. Do not expect frontal 2-D geometry to identify a pure forward/backward depth movement.
5. To test forward movement, select **Side profile**, recalibrate in that view, then make a small comfortable forward head/torso movement. If landmarks do not support the measurement, UNKNOWN is intentional.
6. Compare the independent AI and rule predictions in **Live detector diagnostics**. Use **Image-only (testing)** to observe the binary model. Its probability is uncalibrated and its offline score is not live webcam accuracy.
7. Test pause/resume/stop/reset, score, warning delay, visual/beep warning, and optional spoken correction. Adjust limits under **Sensitivity settings** if necessary.

To try a different camera index, close the app first and set `SMARTPOS_CAMERA` as shown above. If the webcam cannot be opened, the connection banner and `/api/status` report the problem.

## Consent-based webcam validation

The local utility stores user-approved images under `webcam-validation/`, which is git-ignored. Nothing is uploaded. Ask permission from anyone visible, run it interactively, and type `YES` at the consent prompt:

```powershell
.\.venv\Scripts\python.exe webcam_validation.py collect --view frontal
```

In the capture window, hold one comfortable pose and press:

| Key | Label |
|---|---|
| `g` | `GOOD_UPRIGHT` |
| `f` | `BAD_FORWARD` |
| `l` | `BAD_LEFT` |
| `r` | `BAD_RIGHT` |
| `t` | `BAD_SHOULDER_TILT` |
| `q` | Quit |

Each label key saves a short 12-frame burst to a new session. Run collection again to create a different session. Do not deliberately hold an uncomfortable posture.

For a rule/hybrid evaluation, first collect a **calibration-only session** containing only `GOOD_UPRIGHT`, note its printed session ID, and then collect separate evaluation sessions. Evaluate them with:

```powershell
.\.venv\Scripts\python.exe webcam_validation.py evaluate --mode hybrid --calibration-session SESSION_ID
```

Use `--mode rules` or `--mode image` to evaluate either detector alone. Calibration sessions are excluded from evaluation; each burst is summarized as one prediction after stable smoothing. The JSON report includes a GOOD/BAD-by-GOOD/BAD/UNKNOWN confusion matrix, GOOD/BAD precision and recall, balanced accuracy, per-posture BAD recall, UNKNOWN rate, and the evaluation/calibration session IDs. Missing classes have `null` recall rather than an invented value. The utility never trains or retrains a model. Keep evaluation sessions separate from any future training data; correlated frames in a burst are not independent samples. Review the report and collect additional sessions across people, cameras, and lighting before making any accuracy claim.

Do not claim 80%+ live accuracy without sufficient, independently labeled measurement. No webcam validation was performed as part of this source change.

## Binary image model verification and limits

The active artifact is [`image_posture_model.joblib`](image_posture_model.joblib). The checked bundle has feature kind `hog_image_binary_v1`, classes `[0, 1]`, and 1,764 input features; inference validates these facts at load. Training and inference share the same preprocessing function in [`smartpos/image_features.py`](smartpos/image_features.py). Inference keeps both the raw class and confidence when the confidence threshold rejects a result. The model requires a visible head and both shoulders.

The existing offline grouped evaluation used 44 images from the same-session dataset split: 86.36% accuracy/balanced accuracy, BAD recall 1.00, and GOOD recall 0.727. The model was then refit on 123 images. This small, grouped same-session evaluation does **not** establish independent-person, webcam, or medical accuracy. Its probabilities are not calibrated. Do not retrain this model until baseline webcam sessions have been collected and the failure cause is understood.

The image model labels only GOOD/BAD; it cannot explain a lean direction. Hybrid mode does not let its unvalidated BAD/GOOD predictions override geometry. A confident image prediction is diagnostic evidence only until new live-session validation has been reviewed; a confident AI BAD conflicting with geometric GOOD produces UNKNOWN.

The source dataset's existing attribution and license notes remain in [`IMAGE_MODEL_README.md`](IMAGE_MODEL_README.md). Preserve them if redistributing the model or training code.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover frontal/side geometry, calibrated baseline decisions, hysteresis, immediate UNKNOWN on missing landmarks, hybrid policy, model schema/labels/preprocessing, API mode and settings validation, overlays, and webcam-validation metrics.

## Troubleshooting

- **Missing Python dependencies:** activate/use the project's Python 3.12 environment and reinstall `requirements.txt`. Startup reports a missing Flask/OpenCV/MediaPipe dependency with installation guidance when possible.
- **Camera unavailable or wrong index:** check OS camera permissions, close applications using the webcam, and retry with `$env:SMARTPOS_CAMERA="1"` (or another non-negative index). A bad `SMARTPOS_CAMERA` value is reported rather than silently selected.
- **UNKNOWN / calibration failed:** confirm the selected frontal/side view, improve lighting, keep the head and required shoulder landmarks visible, and collect at least 10 reliable calibration samples. UNKNOWN is preferable to a fabricated measurement.
- **Forward lean remains GOOD in frontal view:** pure depth movement is not measurable from frontal 2-D landmarks. Change to **Side profile**, recalibrate, and test again; if the head/torso shift is not visible, status remains UNKNOWN.
- **Binary model missing or corrupt:** `/api/status` reports `image_model_error`; hybrid/rule mode remains available, but image-only mode is disabled.
- **Image model appears wrong:** verify the diagnostic confidence and compare it with rules. Preprocessing matches training; poor live behavior can be domain shift. Collect independent labeled webcam sessions and inspect the report rather than changing the model or claiming accuracy.
- **Score stays at zero:** click Start/Resume and wait for a stable GOOD/BAD result. UNKNOWN time, paused time, and stale camera frames are excluded.
- **Alerts feel too sensitive:** apply a clean upright baseline, adjust the geometric thresholds, and change alert delay. Image confidence is not a safety confidence.
- **MediaPipe import or inference error:** check the reported camera error and reinstall the pinned project requirements in Python 3.12. OpenCV/MediaPipe must use the same environment as Flask.
- **Browser audio unavailable:** allow browser audio or interact with the page; visual warnings remain available.

Camera frames are processed locally and are not recorded by the dashboard. The validation utility records only after explicit local consent and stores its files in the ignored `webcam-validation/` folder. This is an educational project, not medical advice or a diagnosis.
