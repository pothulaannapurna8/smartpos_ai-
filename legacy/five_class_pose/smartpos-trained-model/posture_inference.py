"""Inference adapter for MultiPosture upper-body classifier.
Requires scikit-learn, joblib, numpy and MediaPipe Pose landmarks.
"""
from pathlib import Path
import numpy as np
import joblib

JOINTS = ('nose', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder')
LABELS = {'TUP': 'UPRIGHT', 'TLF': 'LEAN_FORWARD', 'TLB': 'LEAN_BACKWARD',
          'TLL': 'LEAN_LEFT', 'TLR': 'LEAN_RIGHT'}


def features_from_points(points):
    """points: dict of joint names to (x,y) values in consistent coordinates.
    Dataset uses hip-centered, normalized image coordinates; translation is removed
    and all points are divided by shoulder width.
    """
    a = np.asarray([points[k][:2] for k in JOINTS], dtype=float)
    if not np.isfinite(a).all():
        raise ValueError('Missing or invalid landmark coordinates')
    midpoint = (a[3] + a[4]) / 2
    width = np.linalg.norm(a[3] - a[4])
    if width < 1e-4:
        raise ValueError('Shoulders too close together for reliable prediction')
    return ((a - midpoint) / width).reshape(-1)


def features_from_mediapipe(landmarks, min_visibility=0.45):
    """Accept MediaPipe NormalizedLandmarkList.landmark or a list of landmarks.
    Requires nose, ears, shoulders, even if the rule-based app needs fewer joints.
    """
    indices = {'nose': 0, 'left_ear': 7, 'right_ear': 8,
               'left_shoulder': 11, 'right_shoulder': 12}
    if hasattr(landmarks, 'landmark'):
        landmarks = landmarks.landmark
    points = {}
    for name, idx in indices.items():
        p = landmarks[idx]
        if getattr(p, 'visibility', 1.0) < min_visibility:
            raise ValueError(f'{name} not reliably visible')
        points[name] = (p.x, p.y)
    return features_from_points(points)


class PosturePredictor:
    def __init__(self, model_path=None):
        path = Path(model_path) if model_path else Path(__file__).parent / 'posture_model.joblib'
        # Only load model files from a trusted source; joblib uses pickle.
        self.bundle = joblib.load(path)
        self.pipeline = self.bundle['pipeline']

    def predict(self, landmarks, threshold=0.55):
        try:
            x = features_from_mediapipe(landmarks)
        except (ValueError, IndexError, TypeError, KeyError):
            return {'status': 'UNKNOWN', 'issue': None, 'confidence': None}
        probs = self.pipeline.predict_proba([x])[0]
        i = int(np.argmax(probs))
        label = str(self.bundle['labels'][int(self.pipeline.classes_[i])])
        conf = float(probs[i])
        if conf < threshold:
            return {'status': 'UNKNOWN', 'issue': None, 'confidence': conf}
        return {'status': 'GOOD' if label == 'TUP' else 'BAD',
                'issue': LABELS[label], 'confidence': conf}
