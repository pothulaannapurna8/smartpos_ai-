# SmartPos AI — independently trained posture classifier

## What is inside
- `posture_model.joblib`: trained StandardScaler + scikit-learn MLPClassifier (48, 24 hidden units).
- `posture_inference.py`: landmark extraction and inference adapter.
- `train_model.py`: reproducible training script.
- `metrics.json`: subject-separated evaluation details.
- `data.csv`: user-supplied MultiPosture dataset, included for reproducibility; check original dataset terms before redistribution.

## Actual evaluation
4,794 labeled rows. 3,714 training rows from 10 participants, 1,080 test rows from three entirely held-out participants (IDs 1, 10, 12). Five-class test accuracy **57.04%**, balanced accuracy **56.40%**. This is a prototype, not a clinical tool. The dataset was not gathered from your own camera.

## Important compatibility
This model uses only **nose, left/right ears, left/right shoulders**, x and y coordinates, centered on the shoulder midpoint and scaled by shoulder width. No hips required. The training CSV contains hip-centered landmarks; translation is removed in preprocessing. Webcam image normalization and the dataset camera conditions can differ substantially. Results on the live camera are **not verified**.

## Install on Windows (inside project folder)
```powershell
.\.venv\Scripts\python.exe -m pip install scikit-learn==1.8.0 joblib
```
**Compatibility warning:** The model was serialized with scikit-learn 1.8.0 in this environment. For reliable deserialization, use scikit-learn **1.8.0** with a compatible Python version, or **retrain locally** using `train_model.py` under your installed scikit-learn version. If your environment cannot install scikit-learn 1.8.0, install a compatible scikit-learn version and run `train_model.py` to create a compatible file. Never load untrusted joblib/pickle files.

## How to integrate
Copy `posture_inference.py` and `posture_model.joblib` into a `models` folder, or adjust model path. Example (after you obtain MediaPipe's `results.pose_landmarks`):
```python
from posture_inference import PosturePredictor
predictor = PosturePredictor('models/posture_model.joblib')  # once at startup
result = predictor.predict(results.pose_landmarks) if results.pose_landmarks else {'status': 'UNKNOWN', 'issue': None, 'confidence': None}
```
`result` contains `status` (GOOD/BAD/UNKNOWN), `issue` (UPRIGHT, LEAN_FORWARD, LEAN_BACKWARD, LEAN_LEFT, LEAN_RIGHT) and `confidence` (a model probability, not clinical confidence). The Flask dashboard and API must be explicitly wired to this output. Keep existing geometry measurements, session score, calibration, and rule-based fallback; do not automatically equate model confidence with posture score.

## Limitations
- Subject-held-out test accuracy is low; use only as an experimental option alongside rule-based posture detection.
- Five posture classes are dataset labels, not clinical diagnoses.
- Generalization to different camera placements and subjects is unknown.
- The original MultiPosture dataset should be credited when presenting this project.
