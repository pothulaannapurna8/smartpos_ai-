"""Experimental binary image classifier; not validated for new webcam users."""
from pathlib import Path
import joblib
import numpy as np
from posture_inference import landmark_visibility
from train_image_posture import IMAGE_FEATURE_COUNT, image_features


class ImagePosturePredictor:
    def __init__(self, path=None):
        path = Path(path) if path else Path(__file__).with_name('image_posture_model.joblib')
        bundle = joblib.load(path)  # Only load trusted model files.
        if bundle.get('kind') != 'hog_image_binary_v1' or bundle.get('labels') != ['BAD', 'GOOD']:
            raise ValueError('Unsupported image posture model')
        self.pipeline = bundle['pipeline']
        classes = getattr(self.pipeline, 'classes_', None)
        if (classes is None or list(classes) != [0, 1]
                or not callable(getattr(self.pipeline, 'predict_proba', None))
                or getattr(self.pipeline, 'n_features_in_', None)
                != IMAGE_FEATURE_COUNT):
            raise ValueError(
                'Image model class indices, feature schema, or probability API are invalid'
            )

    def predict(self, frame, landmarks=None, threshold=.65):
        if not 0 <= threshold <= 1:
            raise ValueError('Prediction threshold must be between 0 and 1')
        visibility = landmark_visibility(landmarks)
        shoulders_visible = all(
            visibility[name] is not None and visibility[name] >= 0.45
            for name in ('left_shoulder', 'right_shoulder')
        )
        head_visible = any(
            visibility[name] is not None and visibility[name] >= 0.45
            for name in ('nose', 'left_ear', 'right_ear')
        )
        if not shoulders_visible or not head_visible:
            return {
                'status': 'UNKNOWN',
                'issue': None,
                'confidence': None,
                'raw_label': None,
                'raw_category': None,
                'rejection_reason': (
                    'Image prediction requires a visible head and both shoulders.'
                ),
                'landmark_visibility': visibility,
            }
        probs = np.asarray(
            self.pipeline.predict_proba([image_features(frame)])[0],
            dtype=float,
        )
        classes = list(self.pipeline.classes_)
        if (probs.shape != (2,) or not np.isfinite(probs).all()
                or np.any((probs < 0) | (probs > 1))
                or not np.isclose(probs.sum(), 1.0, atol=1e-6)
                or classes != [0, 1]):
            raise ValueError('Image model returned invalid class probabilities')
        best = int(np.argmax(probs))
        label = ('BAD', 'GOOD')[classes[best]]
        confidence = float(probs[best])
        accepted = confidence >= threshold
        return {'status': label if accepted else 'UNKNOWN',
                'issue': ('UPRIGHT' if label == 'GOOD' else 'IMAGE_POSTURE') if accepted else None,
                'confidence': confidence,
                'raw_label': label, 'raw_category': label,
                'rejection_reason': None if accepted else 'Low image-model confidence',
                'landmark_visibility': visibility}
