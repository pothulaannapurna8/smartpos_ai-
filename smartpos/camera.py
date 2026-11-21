"""Background webcam capture with pretrained MediaPipe Pose inference."""
import logging
import threading
import time
try:
    import cv2
except ImportError:
    cv2 = None
from .geometry import measure_posture
from .landmarks import landmark_visibility, posture_points


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
    metrics = status.get('metrics', {})
    baseline = status.get('baseline', {})
    debug = status.get('debug', {})
    if label == 'bad':
        color = (42, 42, 220)
        lines = [
            'BAD POSTURE',
            f'Issue: {issue or "POSTURE"}',
        ]
    elif label == 'good':
        color = (46, 160, 92)
        lines = ['GOOD POSTURE']
    else:
        color = (108, 116, 128)
        lines = ['Posture unavailable', correction]
    alignment = metrics.get('head_alignment_deg')
    tilt = metrics.get('shoulder_tilt_deg')
    tilt_signed = metrics.get('shoulder_tilt_signed_deg')
    tilt_baseline = baseline.get('shoulder_tilt_signed_deg')
    if alignment is not None:
        lines.append(
            f'Head/shoulder image-plane deviation: {abs(alignment):.1f} deg'
        )
    else:
        lines.append('Head/shoulder alignment: unavailable')
    vertical = metrics.get('head_vertical_displacement')
    if vertical is not None:
        lines.append(
            f"Vertical head shift: {vertical:.2f} "
            f"{metrics.get('head_offset_unit', 'normalized units')}"
        )
    if tilt is not None:
        delta = (
            tilt_signed - tilt_baseline
            if tilt_signed is not None and tilt_baseline is not None
            else None
        )
        tilt_line = f'Shoulder tilt: {tilt:.1f} deg'
        if delta is not None:
            tilt_line += f' (baseline delta {delta:+.1f} deg)'
        lines.append(tilt_line)
    if metrics.get('head_offset') is not None:
        unit = metrics.get('head_offset_unit', 'shoulder widths')
        offset = metrics['head_offset']
        offset_baseline = baseline.get('head_offset')
        offset_line = f'Head shift: {offset:+.2f} {unit}'
        if offset_baseline is not None:
            offset_line += f' (baseline {offset_baseline:+.2f})'
        lines.append(offset_line)
    if metrics.get('torso_lean_deg') is not None:
        lines.append(f"Torso lean: {metrics['torso_lean_deg']:+.1f} deg")
    image = status.get('image_prediction', {})
    image_label = image.get('raw_label') or image.get('status', 'UNKNOWN')
    image_confidence = image.get('confidence')
    image_line = f'Image AI: {image_label}'
    if image_confidence is not None:
        image_line += f' ({image_confidence:.2f})'
    lines.append(image_line)
    lines.append(f"Rule: {debug.get('rule_prediction', 'UNKNOWN')}")
    if label == 'bad':
        if status.get('alert'):
            lines.append('Warning active')
        elif status.get('warning_remaining') is not None:
            lines.append(f"Warning in {status['warning_remaining']:.1f}s")
    lines.extend(_wrap_overlay_text(correction, 400, 0.48, 1))

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
    def __init__(self, state, camera_index=0, image_predictor=None,
                 image_model_error=''):
        self.state = state
        self.camera_index = camera_index
        self.image_predictor = image_predictor
        self.image_model_available = image_predictor is not None
        self.image_model_error = image_model_error
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

    def _predict_image(self, frame, landmarks):
        if not self.image_model_available or self.image_predictor is None:
            return {
                'status': 'UNKNOWN',
                'issue': None,
                'confidence': None,
                'raw_label': None,
                'rejection_reason': (
                    self.image_model_error or 'Image posture model unavailable.'
                ),
                'landmark_visibility': landmark_visibility(landmarks),
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
                'rejection_reason': (
                    f'Image model inference failed: {self.image_model_error}'
                ),
                'landmark_visibility': landmark_visibility(landmarks),
            }

    def _run(self):
        cap = None
        try:
            if cv2 is None:
                self.error = (
                    'OpenCV is unavailable. Install the project requirements '
                    'in its Python 3.12 environment.'
                )
                return
            try:
                import mediapipe as mp
            except ImportError as exc:
                self.error = (
                    'MediaPipe is unavailable. Install the project requirements '
                    f'in its Python 3.12 environment. Details: {exc}'
                )
                return
            pose_api = mp.solutions.pose
            drawing = mp.solutions.drawing_utils
            cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                self.error = (
                    f'Cannot open webcam index {self.camera_index}. Check camera '
                    'permissions, close other camera apps, and try SMARTPOS_CAMERA=1.'
                )
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
                        self.error = (
                            f'Webcam index {self.camera_index} stopped delivering '
                            'frames. Check the connection and permissions.'
                        )
                        self.state.mark_camera_failure(
                            self.error
                        )
                        time.sleep(.1)
                        continue
                    self.error = ''
                    source_frame = frame
                    frame = cv2.flip(frame, 1)
                    height, width = frame.shape[:2]
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    result = model.process(rgb)
                    pose_landmarks = (
                        result.pose_landmarks.landmark
                        if result.pose_landmarks else None
                    )
                    settings = self.state.detector_settings()
                    pose_points = posture_points(pose_landmarks)
                    if pose_landmarks is not None:
                        drawing.draw_landmarks(frame,result.pose_landmarks,pose_api.POSE_CONNECTIONS)
                    measurements = measure_posture(
                        pose_points, width, height, settings['view_mode']
                    )
                    image_prediction = self._predict_image(
                        source_frame, pose_landmarks
                    )
                    self.state.update_frame(measurements, image_prediction)
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
