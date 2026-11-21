"""Shared BGR-to-HOG preprocessing for training and live image inference."""
import cv2
import numpy as np

IMAGE_SIZE = (64, 64)
IMAGE_FEATURE_COUNT = 1764


def image_features(image):
    """Extract the original model's full-frame, equalized 64x64 HOG vector."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError('Invalid BGR image')
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected a three-channel BGR image')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.equalizeHist(gray)
    hog = cv2.HOGDescriptor(IMAGE_SIZE, (16, 16), (8, 8), (8, 8), 9)
    features = hog.compute(gray).reshape(-1)
    if features.size != IMAGE_FEATURE_COUNT:
        raise RuntimeError(
            f'Unexpected HOG feature count: {features.size}; '
            f'expected {IMAGE_FEATURE_COUNT}'
        )
    return features
