"""Shared MediaPipe landmark metadata helpers."""
import math


JOINTS = (
    'nose', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
)
LANDMARK_INDICES = {
    'nose': 0,
    'left_ear': 7,
    'right_ear': 8,
    'left_shoulder': 11,
    'right_shoulder': 12,
}


def landmark_visibility(landmarks):
    """Return observed landmark visibility without inventing missing values."""
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
                if math.isfinite(value) and 0 <= value <= 1:
                    visibility[name] = value
    except (TypeError, ValueError, IndexError, OverflowError):
        return {name: None for name in JOINTS}
    return visibility


def posture_points(landmarks):
    """Extract image-plane points used by geometry from a Pose result."""
    if landmarks is None:
        return {}
    if hasattr(landmarks, 'landmark'):
        landmarks = landmarks.landmark
    indices = {
        **LANDMARK_INDICES,
        'left_hip': 23,
        'right_hip': 24,
    }
    points = {}
    for name, index in indices.items():
        try:
            landmark = landmarks[index]
            points[name] = (
                float(landmark.x),
                float(landmark.y),
                float(landmark.visibility),
            )
        except (AttributeError, TypeError, ValueError, IndexError):
            continue
    return points
