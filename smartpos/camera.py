"""Background webcam capture with pretrained MediaPipe Pose inference."""
import logging
import threading
import time
import cv2
from .geometry import head_offset_ratio, posture_measurements


def _wrap_overlay_text(text, max_width, font_scale, thickness):
    words = text.split()
    lines = []
    current = ''
    for word in words:
        candidate = f'{current} {word}'.strip()
        width = cv2.getTextSize(
            candidate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )[0][0]
        if current and width > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _face_rectangle(landmarks, pose_landmark_ids, width, height):
    if landmarks is None or pose_landmark_ids is None:
        return None
    points = []
    for name in ('NOSE', 'LEFT_EAR', 'RIGHT_EAR'):
        index = getattr(pose_landmark_ids, name, None)
        if index is None or index >= len(landmarks):
            continue
        landmark = landmarks[index]
        visibility = getattr(landmark, 'visibility', 0.0)
        if visibility >= 0.35:
            points.append((landmark.x * width, landmark.y * height))
    if not points:
        return None
    return (
        max(0, int(min(point[0] for point in points) - 35)),
        max(0, int(min(point[1] for point in points) - 45)),
        min(width, int(max(point[0] for point in points) + 35)),
        min(height, int(max(point[1] for point in points) + 40)),
    )


def _rectangles_overlap(first, second):
    return not (
        first[2] <= second[0] or second[2] <= first[0]
        or first[3] <= second[1] or second[3] <= first[1]
    )


def _highlight_posture_landmarks(frame, landmarks, pose_landmark_ids, issue):
    if landmarks is None or pose_landmark_ids is None:
        return
    names = (
        ('LEFT_SHOULDER', 'RIGHT_SHOULDER')
        if issue == 'UNEVEN_SHOULDERS'
        else ('NOSE', 'LEFT_EAR', 'RIGHT_EAR', 'LEFT_SHOULDER', 'RIGHT_SHOULDER')
    )
    height, width = frame.shape[:2]
    for name in names:
        index = getattr(pose_landmark_ids, name, None)
        if index is None or index >= len(landmarks):
            continue
        landmark = landmarks[index]
        if getattr(landmark, 'visibility', 0.0) < 0.35:
            continue
        point = (
            max(0, min(width - 1, int(landmark.x * width))),
            max(0, min(height - 1, int(landmark.y * height))),
        )
        cv2.circle(frame, point, 8, (35, 40, 240), 3, cv2.LINE_AA)


def draw_posture_overlay(frame, status, landmarks=None, pose_landmark_ids=None):
    """Draw the same stable status and correction exposed by the status API."""
    height, width = frame.shape[:2]
    label = status.get('label', 'unknown')
    issue = status.get('issue')
    correction = status.get('correction', '')
    if label == 'bad':
        color = (42, 42, 220)
        lines = [
            'BAD POSTURE',
            f'Issue: {issue or "POSTURE"}',
            f"Neck angle: {status['neck']:.1f} deg"
            if status.get('neck') is not None else 'Neck angle: unavailable',
            f"Shoulder tilt: {status['tilt']:.1f} deg"
            if status.get('tilt') is not None else 'Shoulder tilt: unavailable',
        ]
        if status.get('head_offset') is not None:
            lines.append(f"Head offset: {status['head_offset']:+.2f} shoulder widths")
        if status.get('probability') is not None:
            lines.append(f"Model probability: {status['probability']:.3f}")
        if status.get('alert'):
            lines.append('Warning active')
        elif status.get('warning_remaining') is not None:
            lines.append(f"Warning in {status['warning_remaining']:.1f}s")
        lines.extend(_wrap_overlay_text(correction, 400, 0.48, 1))
    elif label == 'good':
        color = (46, 160, 92)
        lines = ['GOOD POSTURE']
        if status.get('neck') is not None:
            lines.append(f"Neck angle: {status['neck']:.1f} deg")
        if status.get('tilt') is not None:
            lines.append(f"Shoulder tilt: {status['tilt']:.1f} deg")
        if status.get('head_offset') is not None:
            lines.append(f"Head offset: {status['head_offset']:+.2f} shoulder widths")
        if status.get('probability') is not None:
            lines.append(f"Model probability: {status['probability']:.3f}")
    else:
        color = (108, 116, 128)
        lines = ['Posture unavailable', correction]

    max_panel_width = min(440, width - 20)
    font_scale = 0.48 if max_panel_width >= 370 else 0.40
    thickness = 1 if font_scale < 0.45 else 2
    max_text_width = max_panel_width - 24
    wrapped = []
    for line in lines:
        wrapped.extend(
            _wrap_overlay_text(line, max_text_width, font_scale, thickness)
        )
    line_height = max(20, int(font_scale * 42))
    panel_width = max_panel_width
    if label == 'unknown':
        panel_width = min(max_panel_width, 320)
    elif label == 'good':
        panel_width = min(max_panel_width, 340)
    else:
        panel_width = max_panel_width
    panel_height = 14 + len(wrapped) * line_height
    margin = 10
    candidates = [
        (margin, margin),
        (width - panel_width - margin, margin),
        (margin, height - panel_height - margin),
        (width - panel_width - margin, height - panel_height - margin),
    ]
    face = _face_rectangle(landmarks, pose_landmark_ids, width, height)
    origin = candidates[0]
    if face is not None:
        for candidate in candidates:
            box = (
                candidate[0], candidate[1],
                candidate[0] + panel_width, candidate[1] + panel_height,
            )
            if not _rectangles_overlap(box, face):
                origin = candidate
                break

    x, y = origin
    x = max(0, min(x, width - panel_width))
    y = max(0, min(y, height - panel_height))
    if label == 'bad':
        _highlight_posture_landmarks(
            frame, landmarks, pose_landmark_ids, issue
        )
    region = frame[y:y + panel_height, x:x + panel_width]
    overlay = region.copy()
    cv2.rectangle(
        overlay, (0, 0), (panel_width - 1, panel_height - 1), color, -1
    )
    cv2.addWeighted(overlay, 0.88, region, 0.12, 0, region)
    cv2.rectangle(
        frame, (x, y), (x + panel_width - 1, y + panel_height - 1),
        color, 2,
    )
    text_color = (248, 250, 255)
    baseline = y + 23
    for index, line in enumerate(wrapped):
        cv2.putText(
            frame,
            line,
            (x + 12, baseline + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )


class CameraWorker:
    def __init__(self, state, camera_index=0, predictor=None, model_error='', image_predictor=None, image_model_error=''):
        self.state = state
        self.camera_index = camera_index
        self.predictor = predictor
        self.image_predictor = image_predictor
        self.image_model_available = image_predictor is not None
        self.image_model_error = image_model_error
        self.model_available = predictor is not None
        self.model_error = model_error
        self.lock = threading.Lock()
        self.latest_jpeg = None
        self.running = False
        self.thread = None
        self.error = ''

    def start(self):
        if self.running: return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread: self.thread.join(timeout=3)

    def frame(self):
        with self.lock:
            return self.latest_jpeg

    def _predict_ml(self, landmarks, visibility):
        if not self.model_available or self.predictor is None:
            return {
                'status': 'UNKNOWN',
                'issue': None,
                'confidence': None,
                'raw_label': None,
                'raw_category': None,
                'rejection_reason': self.model_error or 'Posture model unavailable.',
                'landmark_visibility': visibility,
            }
        try:
            return self.predictor.predict(landmarks)
        except Exception as exc:
            self.model_error = f'{type(exc).__name__}: {exc}'
            self.model_available = False
            logging.getLogger(__name__).exception(
                'ML posture prediction failed; switch to rules mode'
            )
            return {
                'status': 'UNKNOWN',
                'issue': None,
                'confidence': None,
                'raw_label': None,
                'raw_category': None,
                'rejection_reason': f'Model inference failed: {self.model_error}',
                'landmark_visibility': visibility,
            }

    def _predict_image(self, frame, landmarks):
        if not self.image_model_available or self.image_predictor is None:
            return {
                'status': 'UNKNOWN',
                'issue': None,
                'confidence': None,
                'raw_label': None,
                'raw_category': None,
                'rejection_reason': (
                    self.image_model_error or 'Image posture model unavailable.'
                ),
                'landmark_visibility': {},
            }
        try:
            return self.image_predictor.predict(frame, landmarks)
        except Exception as exc:
            self.image_model_error = f'{type(exc).__name__}: {exc}'
            self.image_model_available = False
            logging.getLogger(__name__).exception(
                'Image posture prediction failed; switch to another mode'
            )
            return {
                'status': 'UNKNOWN',
                'issue': None,
                'confidence': None,
                'raw_label': None,
                'raw_category': None,
                'rejection_reason': (
                    f'Image model inference failed: {self.image_model_error}'
                ),
                'landmark_visibility': {},
            }

    def _run(self):
        cap = None
        try:
            import mediapipe as mp
            pose_api = mp.solutions.pose
            drawing = mp.solutions.drawing_utils
            cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                self.error = 'Cannot open webcam. Check permissions and camera index.'
                self.running = False
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
            with pose_api.Pose(static_image_mode=False, model_complexity=1,
                               min_detection_confidence=0.6,
                               min_tracking_confidence=0.6) as model:
                while self.running:
                    ok, frame = cap.read()
                    if not ok:
                        self.state.update_measurement(None, None)
                        time.sleep(.1)
                        continue
                    source_frame = frame
                    frame = cv2.flip(frame, 1)
                    height, width = frame.shape[:2]
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    result = model.process(rgb)
                    neck = tilt = head_offset = None
                    prediction = None
                    pose_landmarks = (
                        result.pose_landmarks.landmark
                        if result.pose_landmarks else None
                    )
                    visibility = {
                        'nose': None,
                        'left_ear': None,
                        'right_ear': None,
                        'left_shoulder': None,
                        'right_shoulder': None,
                    }
                    if pose_landmarks is not None:
                        landmarks = pose_landmarks
                        ids = pose_api.PoseLandmark
                        indices = {
                            'nose': ids.NOSE,
                            'left_ear': ids.LEFT_EAR,
                            'right_ear': ids.RIGHT_EAR,
                            'left_shoulder': ids.LEFT_SHOULDER,
                            'right_shoulder': ids.RIGHT_SHOULDER,
                        }
                        visibility = {
                            name: float(landmarks[index].visibility)
                            for name, index in indices.items()
                        }
                        pose_points = {
                            name: (
                                landmarks[index].x,
                                landmarks[index].y,
                                landmarks[index].visibility,
                            )
                            for name, index in indices.items()
                        }
                        neck, tilt = posture_measurements(
                            pose_points, width, height
                        )
                        head_offset = head_offset_ratio(pose_points)
                        drawing.draw_landmarks(frame,result.pose_landmarks,pose_api.POSE_CONNECTIONS)
                    mode = self.state.current_mode()
                    if mode == 'ml':
                        prediction = self._predict_ml(pose_landmarks, visibility)
                    elif mode == 'image':
                        prediction = self._predict_image(
                            source_frame, pose_landmarks
                        )
                    else:
                        prediction = {
                            'status': 'UNKNOWN',
                            'issue': None,
                            'confidence': None,
                            'raw_label': None,
                            'raw_category': None,
                            'rejection_reason': (
                                'ML inference is disabled in rule-based mode.'
                            ),
                            'landmark_visibility': visibility,
                        }
                    self.state.update_measurement(
                        neck, tilt, prediction, head_offset
                    )
                    status = self.state.snapshot()
                    draw_posture_overlay(
                        frame,
                        status,
                        pose_landmarks,
                        pose_api.PoseLandmark if pose_landmarks is not None else None,
                    )
                    success, encoded = cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,78])
                    if success:
                        with self.lock: self.latest_jpeg = encoded.tobytes()
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
            logging.getLogger(__name__).exception('Camera worker failed')
        finally:
            if cap is not None:
                cap.release()
            self.running = False
