# Experimental posture classifier

## What is included

- `posture_model.joblib`: a `StandardScaler` + scikit-learn `MLPClassifier` (48, 24 hidden units).
- `posture_inference.py`: MediaPipe feature extraction and prediction adapter.
- `train_model.py`: reproducible subject-separated training script.
- `metrics.json`: evaluation metrics and runtime versions for the current model artifact.

The Flask app loads the classifier once during app creation. Rule-based geometry remains the default. ML mode is opt-in from the dashboard, and model inference receives real MediaPipe landmarks. Geometric neck/shoulder measurements remain independent. Session GOOD/BAD score accounting uses the active mode's labels, not model probabilities. The dashboard's live diagnostics expose the raw class/category, probability, final status, rejection reason and observed landmark visibility.

## Current artifact and reported evaluation

The supplied MultiPosture `data.csv` contained 4,794 labeled rows. Training used 3,714 rows from 10 participants; evaluation used 1,080 rows from three held-out participants (IDs 1, 10, 12). The current artifact was retrained locally with Python 3.12.10, scikit-learn 1.8.0 and NumPy 1.26.4 after the originally supplied artifact failed to deserialize in this environment (`MT19937` BitGenerator compatibility error).

The regenerated held-out-subject results are **57.04% accuracy** and **56.40% balanced accuracy**. These values describe that offline split only. Live webcam accuracy has not been established, and the model is not medically validated.

## Features, labels, and missing landmarks

The classifier uses x/y coordinates, in this exact order: nose, left ear, right ear, left shoulder, right shoulder (x then y for each). Training and inference share the same shoulder-midpoint translation and Euclidean shoulder-width normalization, followed by the serialized `StandardScaler` and MLP pipeline. It does not require hips. All five model landmarks must have visibility of at least 0.45; predicted probabilities below 0.55 yield UNKNOWN. `TUP` maps to GOOD/UPRIGHT; `TLF`, `TLB`, `TLL` and `TLR` map to BAD with LEAN_FORWARD, LEAN_BACKWARD, LEAN_LEFT and LEAN_RIGHT respectively. A probability is not a calibrated measure of safety or clinical confidence.

The model uses only normalized 2-D image coordinates and does not receive MediaPipe depth (`z`). A frontal webcam view can therefore provide weak or ambiguous evidence of forward depth lean, and subject-held-out accuracy is modest. It must not force a BAD result based on an expected pose: low visibility or low probability remains UNKNOWN. Five-class categories are dataset labels, not diagnoses.

The separate GOOD/BAD HOG image classifier is opt-in and does not replace this landmark model. Its training/inference preprocessing, class mapping and same-session evaluation are documented in [IMAGE_MODEL_README.md](IMAGE_MODEL_README.md). Rule-based detection is the default and the user selects one mode rather than combining multiple ML models. In pose-ML mode, the classifier's output is an experimental candidate: UPRIGHT must agree with geometric GOOD; sideways labels need a matching normalized head-center offset; and forward/backward labels need a measured alignment deviation, are presented only as generic `NECK_ALIGNMENT`, and explain that 2-D cannot determine depth direction. Unsupported labels become UNKNOWN. The image model uses its own visible-head/shoulders guard and does not infer a specific issue.

## Retraining

The dataset is not included in this project folder. Use an authorized copy of the MultiPosture `data.csv`, and check its terms before redistribution. From PowerShell in the folder containing `app.py`:

```powershell
.\.venv\Scripts\python.exe train_model.py --data ..\data.csv
```

By default this overwrites `posture_model.joblib` and `metrics.json`. The Flask process must be restarted to load a newly trained artifact. Never load an untrusted joblib file: deserialization uses pickle.

## Limitations

- The modest subject-held-out accuracy is experimental; generalization to new people and camera placements is unknown.
- Live predictions have not been independently evaluated.
- MediaPipe's pretrained deep-learning pose estimator finds landmarks; this MLP classifier is a separate posture model.
- The tool is an educational prototype, not a medical or ergonomic assessment.

Credit the MultiPosture dataset and follow its terms when presenting or distributing a model trained with it. MediaPipe, scikit-learn, joblib and other dependencies are external software under their respective licenses.
