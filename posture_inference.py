"""Inference adapter for the experimental MultiPosture classifier."""
from pathlib import Path

import joblib
import numpy as np

JOINTS = ('nose', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder')
LANDMARK_INDICES = {
    'nose': 0,
    'left_ear': 7,
    'right_ear': 8,
    'left_shoulder': 11,
    'right_shoulder': 12,
}
FEATURE_NAMES_BY_DIMENSIONS = {
    dimensions: tuple(
        f'{joint}_{axis}'
        for joint in JOINTS
        for axis in (('x', 'y') if dimensions == 2 else ('x', 'y', 'z'))
    )
    for dimensions in (2, 3)
}
FEATURE_NAMES = FEATURE_NAMES_BY_DIMENSIONS[2]
FEATURE_NAMES_3D = FEATURE_NAMES_BY_DIMENSIONS[3]
LABELS = {'TUP': 'UPRIGHT', 'TLF': 'LEAN_FORWARD', 'TLB': 'LEAN_BACKWARD',
          'TLL': 'LEAN_LEFT', 'TLR': 'LEAN_RIGHT'}
CLASS_LABELS = tuple(sorted(LABELS))
STATUS_BY_LABEL = {
    'TUP': 'GOOD',
    'TLF': 'BAD',
    'TLB': 'BAD',
    'TLL': 'BAD',
    'TLR': 'BAD',
}


def landmark_visibility(landmarks):
    """Return observed visibility for model landmarks without inventing values."""
    visibility = {name: None for name in JOINTS}
    if landmarks is None:
        return visibility
    if hasattr(landmarks, 'landmark'):
        landmarks = landmarks.landmark
    try:
        for name, index in LANDMARK_INDICES.items():
            if index >= len(landmarks):
                continue
            value = getattr(landmarks[index], 'visibility', None)
            if value is not None:
                value = float(value)
                if np.isfinite(value) and 0 <= value <= 1:
                    visibility[name] = value
    except (TypeError, ValueError, IndexError, OverflowError):
        return {name: None for name in JOINTS}
    return visibility


def features_from_points(points, dimensions=2):
    """Normalize points around the shoulder midpoint and by 2-D shoulder width."""
    if dimensions not in FEATURE_NAMES_BY_DIMENSIONS:
        raise ValueError('Feature dimensions must be 2 or 3')
    axes = dimensions
    a = np.asarray([points[joint][:axes] for joint in JOINTS], dtype=float)
    if not np.isfinite(a).all():
        raise ValueError('Missing or invalid landmark coordinates')
    midpoint = (a[3] + a[4]) / 2
    width = np.linalg.norm(a[3, :2] - a[4, :2])
    if width < 1e-4:
        raise ValueError('Shoulders too close together for reliable prediction')
    return ((a - midpoint) / width).reshape(-1)


def features_from_mediapipe(landmarks, min_visibility=0.45, dimensions=2):
    """Extract the same 2-D or 3-D feature schema used by training."""
    if hasattr(landmarks, 'landmark'):
        landmarks = landmarks.landmark
    visibility = landmark_visibility(landmarks)
    points = {}
    for name, index in LANDMARK_INDICES.items():
        value = visibility[name]
        if value is None or value < min_visibility:
            measured = 'unavailable' if value is None else f'{value:.2f}'
            raise ValueError(
                f'{name} visibility {measured}; required >= {min_visibility:.2f}'
            )
        landmark = landmarks[index]
        coordinates = (landmark.x, landmark.y)
        if dimensions == 3:
            z = getattr(landmark, 'z', None)
            if z is None:
                raise ValueError(f'{name} depth coordinate is unavailable')
            coordinates += (z,)
        points[name] = coordinates
    return features_from_points(points, dimensions=dimensions)


class PosturePredictor:
    def __init__(self, model_path=None):
        path = Path(model_path) if model_path else Path(__file__).parent / 'posture_model.joblib'
        # Only load model files from a trusted source; joblib uses pickle.
        self.bundle = joblib.load(path)
        self.pipeline = self.bundle['pipeline']
        features = self.bundle.get('features')
        dimensions = next(
            (
                dimensions
                for dimensions, expected in FEATURE_NAMES_BY_DIMENSIONS.items()
                if features == list(expected)
            ),
            None,
        )
        if dimensions is None:
            raise ValueError('Model feature order does not match inference preprocessing')
        self.dimensions = dimensions
        if self.bundle.get('labels') != list(CLASS_LABELS):
            raise ValueError('Model label order does not match training classes')
        if list(self.pipeline.classes_) != list(range(len(self.bundle['labels']))):
            raise ValueError('Model class indices do not match its label mapping')

    @staticmethod
    def _unknown(reason, visibility):
        return {
            'status': 'UNKNOWN',
            'issue': None,
            'confidence': None,
            'raw_label': None,
            'raw_category': None,
            'rejection_reason': reason,
            'landmark_visibility': visibility,
        }

    def predict(self, landmarks, threshold=0.55):
        if not 0 <= threshold <= 1:
            raise ValueError('Prediction threshold must be between 0 and 1')
        visibility = landmark_visibility(landmarks)
        try:
            x = features_from_mediapipe(
                landmarks, dimensions=self.dimensions
            )
        except (ValueError, IndexError, TypeError, KeyError) as exc:
            return self._unknown(str(exc), visibility)
        probs = np.asarray(self.pipeline.predict_proba([x])[0], dtype=float)
        if (probs.shape != (len(CLASS_LABELS),)
                or not np.isfinite(probs).all()
                or np.any((probs < 0) | (probs > 1))):
            return self._unknown('Model returned invalid probabilities', visibility)
        index = int(np.argmax(probs))
        label = str(self.bundle['labels'][int(self.pipeline.classes_[index])])
        category = LABELS.get(label)
        probability = float(probs[index])
        base = {
            'status': 'UNKNOWN',
            'issue': None,
            'confidence': probability,
            'raw_label': label,
            'raw_category': category,
            'rejection_reason': None,
            'landmark_visibility': visibility,
        }
        if category is None or label not in STATUS_BY_LABEL:
            base['rejection_reason'] = f'Unsupported model label: {label}'
            return base
        if probability < threshold:
            base['rejection_reason'] = (
                f'Model probability {probability:.3f} is below threshold '
                f'{threshold:.2f}'
            )
            return base
        base['status'] = STATUS_BY_LABEL[label]
        base['issue'] = category
        return base
