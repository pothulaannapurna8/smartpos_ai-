"""Thread-safe posture decisions, user calibration, alerts, and session state."""
import math
import statistics
import threading
import time

from .detection import combine_predictions, empty_prediction, rule_prediction
from .geometry import score

_CORRECTION_BY_CATEGORY = {
    'UPRIGHT': 'Your visible alignment is near its calibrated baseline.',
    'LEAN_LEFT': 'Gently bring your head toward the center above your shoulders.',
    'LEAN_RIGHT': 'Gently bring your head toward the center above your shoulders.',
    'HEAD_ALIGNMENT': (
        'Gently realign your head over your shoulders; in side view, check '
        'that your head is comfortably over your upper body.'
    ),
    'TORSO_LEAN': 'Return to a comfortable, supported upright position.',
    'UNEVEN_SHOULDERS': 'Relax your shoulders and keep them level.',
    'IMAGE_POSTURE': (
        'The image model indicates a posture change; adjust comfortably and '
        'check the geometric diagnostics.'
    ),
}


class MonitorState:
    def __init__(self):
        self.lock = threading.RLock()
        self.settings = {
            'mode': 'hybrid',
            'view_mode': 'frontal',
            'head_offset_max': 0.18,
            'alignment_max': 16.0,
            'tilt_max': 10.0,
            'torso_lean_max': 16.0,
            'alert_delay': 3.0,
        }
        self.baseline = {
            'calibrated': False,
            'head_offset': 0.0,
            'head_alignment_deg': 0.0,
            'shoulder_tilt_signed_deg': 0.0,
            'torso_lean_deg': 0.0,
        }
        self.label = 'unknown'
        self.issue = None
        self.active_probability = None
        self.correction = (
            'Position not detected. Keep the required head and shoulder '
            'landmarks visible.'
        )
        self.metrics = {
            'valid': False,
            'view_mode': self.settings['view_mode'],
            'head_offset': None,
            'head_offset_unit': 'shoulder widths',
            'head_vertical_displacement': None,
            'head_alignment_deg': None,
            'shoulder_tilt_deg': None,
            'shoulder_tilt_signed_deg': None,
            'torso_lean_deg': None,
            'missing_reason': 'Waiting for a camera frame.',
        }
        self.rule_prediction = empty_prediction('Waiting for geometry.')
        self.image_prediction = {
            **empty_prediction('Image prediction has not run yet.'),
            'rejection_reason': 'Image prediction has not run yet.',
            'landmark_visibility': {},
        }
        self.final_reason = 'Waiting for a camera frame.'
        self._pending_decision = None
        self._pending_frames = 0
        self._smoothing_frames = 8
        self.last_frame_at = 0.0
        self.bad_since = None
        self.warning_count = 0
        self.warning_counted = False
        self.history = []
        self.session_mode = 'idle'
        self.elapsed = 0.0
        self.good = 0.0
        self.bad = 0.0
        self.last_tick = None
        self.calibration = None
        self.calibration_result = None

    def _tick(self, now):
        if self.session_mode == 'running' and self.last_tick is not None:
            dt = max(0.0, min(now - self.last_tick, 1.0))
            self.elapsed += dt
            if now - self.last_frame_at < 1.5:
                if self.label == 'good':
                    self.good += dt
                elif self.label == 'bad':
                    self.bad += dt
        self.last_tick = now

    def session_action(self, action):
        with self.lock:
            now = time.monotonic()
            self._tick(now)
            if action == 'start':
                if self.session_mode in ('idle', 'stopped'):
                    self.elapsed = self.good = self.bad = 0.0
                    self.warning_count = 0
                    self.warning_counted = False
                    self.history = []
                    if (self.label in ('good', 'bad')
                            and now - self.last_frame_at < 1.5):
                        self.history.append({
                            'timestamp': time.time(),
                            'label': self.label.upper(),
                            'issue': self.issue,
                        })
                self.session_mode = 'running'
            elif action == 'pause' and self.session_mode == 'running':
                self.session_mode = 'paused'
            elif action == 'stop':
                self.session_mode = 'stopped'
            elif action == 'reset':
                self.elapsed = self.good = self.bad = 0.0
                self.session_mode = 'idle'
                self.warning_count = 0
                self.warning_counted = False
                self.history = []
            elif action not in ('start', 'pause', 'stop', 'reset'):
                raise ValueError('Unknown session action')
            self.last_tick = now

    def set_mode(self, mode):
        if mode not in ('hybrid', 'rules', 'image'):
            raise ValueError('Mode must be hybrid, rules, or image')
        with self.lock:
            if mode == self.settings.get('mode'):
                return
            self.settings['mode'] = mode
            self.label = 'unknown'
            self.issue = None
            self.active_probability = None
            self.bad_since = None
            self.warning_counted = False
            self._pending_decision = None
            self._pending_frames = 0
            self.rule_prediction = empty_prediction(
                'Awaiting the next frame in the selected mode.'
            )
            self.image_prediction = {
                **empty_prediction('Awaiting the next image prediction.'),
                'rejection_reason': 'Awaiting the next image prediction.',
                'landmark_visibility': {},
            }
            self.final_reason = 'Awaiting the next camera frame.'

    def current_mode(self):
        with self.lock:
            return self.settings.get('mode', 'hybrid')

    def detector_settings(self):
        with self.lock:
            return dict(self.settings)

    def update_settings(self, updates):
        with self.lock:
            if 'view_mode' in updates:
                view_mode = updates['view_mode']
                if view_mode not in ('frontal', 'side'):
                    raise ValueError('view_mode must be frontal or side')
                if view_mode != self.settings['view_mode']:
                    self.baseline = {
                        'calibrated': False,
                        'head_offset': 0.0,
                        'head_alignment_deg': 0.0,
                        'shoulder_tilt_signed_deg': 0.0,
                        'torso_lean_deg': 0.0,
                    }
                    self.calibration = None
                    self.calibration_result = None
                    self.label = 'unknown'
                    self.issue = None
                    self.active_probability = None
                    self.bad_since = None
                    self.warning_counted = False
                    self.metrics = {
                        'valid': False,
                        'view_mode': view_mode,
                        'head_offset': None,
                        'head_offset_unit': (
                            'shoulder widths' if view_mode == 'frontal'
                            else 'torso lengths'
                        ),
                        'head_vertical_displacement': None,
                        'head_alignment_deg': None,
                        'shoulder_tilt_deg': None,
                        'shoulder_tilt_signed_deg': None,
                        'torso_lean_deg': None,
                        'missing_reason': (
                            'Camera view changed; waiting for new measurements.'
                        ),
                    }
                    self.image_prediction = {
                        **empty_prediction(
                            'Camera view changed; waiting for a new frame.'
                        ),
                        'rejection_reason': (
                            'Camera view changed; waiting for a new frame.'
                        ),
                        'landmark_visibility': {},
                    }
                    self.final_reason = (
                        'Camera view changed; calibrate upright before monitoring.'
                    )
                    self.rule_prediction = empty_prediction(
                        'Camera view changed; recalibration is required.'
                    )
            for key, value in updates.items():
                if key != 'mode':
                    self.settings[key] = value
            self._pending_decision = None
            self._pending_frames = 0

    def _apply_decision(self, decision, now):
        label, issue, probability = decision
        previous_label = self.label
        previous_issue = self.issue
        if label == 'bad' and previous_label != 'bad':
            self.bad_since = now
        elif label != 'bad':
            self.bad_since = None
            self.warning_counted = False
        self.label = label
        self.issue = issue
        self.active_probability = probability
        self.correction = _CORRECTION_BY_CATEGORY.get(
            issue,
            'Position not detected. Follow the landmark guidance and try '
            'calibration again.' if label == 'unknown'
            else 'Adjust gently to a comfortable, supported posture.',
        )
        if (label, issue) != (previous_label, previous_issue):
            self.history.append({
                'timestamp': time.time(),
                'label': label.upper(),
                'issue': issue,
            })
            self.history = self.history[-12:]

    def _smooth_decision(self, decision, now):
        if decision[:2] == (self.label, self.issue):
            self._pending_decision = None
            self._pending_frames = 0
            if decision[0] in ('good', 'bad'):
                self.active_probability = decision[2]
            return
        pending = (
            self._pending_decision[:2]
            if self._pending_decision is not None else None
        )
        if decision[:2] == pending:
            self._pending_frames += 1
            self._pending_decision = decision
        else:
            self._pending_decision = decision
            self._pending_frames = 1
        if self._pending_frames >= self._smoothing_frames:
            self._apply_decision(self._pending_decision, now)
            self._pending_decision = None
            self._pending_frames = 0

    @staticmethod
    def _validated_image_prediction(prediction):
        prediction = prediction if isinstance(prediction, dict) else {}
        status = prediction.get('status', 'UNKNOWN')
        reason = prediction.get('rejection_reason')
        if status not in ('GOOD', 'BAD', 'UNKNOWN'):
            status = 'UNKNOWN'
            reason = 'Image model returned an unsupported status.'
        raw_label = prediction.get('raw_label')
        confidence = prediction.get('confidence')
        if (not isinstance(confidence, (int, float))
                or not math.isfinite(confidence)
                or not 0 <= confidence <= 1):
            confidence = None
        if status != 'UNKNOWN' and (
                raw_label != status or confidence is None):
            status = 'UNKNOWN'
            reason = (
                'Image status, binary label, and confidence are inconsistent.'
            )
        if status == 'UNKNOWN':
            issue = None
        else:
            issue = 'UPRIGHT' if status == 'GOOD' else 'IMAGE_POSTURE'
        visibility = prediction.get('landmark_visibility')
        if not isinstance(visibility, dict):
            visibility = {}
        return {
            'status': status,
            'issue': issue,
            'confidence': confidence,
            'raw_label': raw_label,
            'rejection_reason': reason,
            'landmark_visibility': dict(visibility),
        }

    def update_frame(self, measurements, image_prediction=None):
        with self.lock:
            now = time.monotonic()
            self._tick(now)
            self.last_frame_at = now
            if not isinstance(measurements, dict):
                measurements = {
                    'valid': False,
                    'missing_reason': 'No usable pose landmarks were detected.',
                }
            self.metrics = dict(measurements)
            settings = dict(self.settings)
            mode = settings['mode']
            self.rule_prediction = rule_prediction(
                self.metrics,
                settings,
                self.baseline,
                self.rule_prediction.get('status', 'UNKNOWN'),
            )
            self.image_prediction = self._validated_image_prediction(
                image_prediction
            )
            decision = combine_predictions(
                mode, self.rule_prediction, self.image_prediction
            )
            self.final_reason = decision['reason']
            confidence = (
                decision['confidence'] if mode == 'image' else None
            )
            decision_tuple = (
                decision['status'].lower(),
                decision.get('issue'),
                confidence,
            )
            if decision_tuple[0] == 'unknown':
                self._apply_decision(decision_tuple, now)
                self._pending_decision = None
                self._pending_frames = 0
            else:
                self._smooth_decision(decision_tuple, now)
            self._collect_calibration_sample(now)

    def mark_camera_failure(self, reason):
        with self.lock:
            now = time.monotonic()
            self._tick(now)
            self.last_frame_at = 0.0
            self.label = 'unknown'
            self.issue = None
            self.active_probability = None
            self.bad_since = None
            self.warning_counted = False
            self._pending_decision = None
            self._pending_frames = 0
            self.final_reason = reason

    def _collect_calibration_sample(self, now):
        if self.calibration is None or now >= self.calibration['ends_at']:
            return
        metrics = self.metrics
        if not metrics.get('valid'):
            return
        sample = {}
        for key in (
            'head_offset', 'head_alignment_deg',
            'shoulder_tilt_signed_deg', 'torso_lean_deg',
        ):
            value = metrics.get(key)
            if (isinstance(value, (int, float)) and math.isfinite(value)):
                sample[key] = float(value)
        if 'head_offset' in sample and 'head_alignment_deg' in sample:
            self.calibration['samples'].append(sample)

    def _finish_calibration(self, now):
        if self.calibration is None or now < self.calibration['ends_at']:
            return
        samples = self.calibration['samples']
        if len(samples) >= 10:
            baselines = {}
            for key in (
                'head_offset', 'head_alignment_deg',
                'shoulder_tilt_signed_deg', 'torso_lean_deg',
            ):
                values = [sample[key] for sample in samples if key in sample]
                if values:
                    baselines[key] = statistics.median(values)
            self.calibration_result = {
                'baselines': baselines,
                'samples': len(samples),
                'view_mode': self.settings['view_mode'],
            }
        else:
            self.calibration_result = {
                'error': (
                    f'Only {len(samples)} reliable samples were collected; '
                    'at least 10 are required. Check lighting, framing, and view mode.'
                )
            }
        self.calibration = None

    def calibration_action(self, action):
        with self.lock:
            if action == 'start':
                self.calibration = {
                    'ends_at': time.monotonic() + 5,
                    'samples': [],
                }
                self.calibration_result = None
            elif action == 'cancel':
                self.calibration = None
                self.calibration_result = None
            elif action == 'accept':
                if (not self.calibration_result
                        or 'error' in self.calibration_result):
                    raise ValueError('No calibration baseline is available')
                self.baseline.update(
                    self.calibration_result['baselines']
                )
                self.baseline['calibrated'] = True
                self.calibration_result = None
                self.rule_prediction = empty_prediction(
                    'Awaiting measurements against the new baseline.'
                )
                self._pending_decision = None
                self._pending_frames = 0
            elif action == 'reject':
                self.calibration_result = None
            else:
                raise ValueError('Unknown calibration action')

    def snapshot(self):
        with self.lock:
            now = time.monotonic()
            self._tick(now)
            self._finish_calibration(now)
            live = now - self.last_frame_at < 1.5
            label = self.label if live else 'unknown'
            if not live:
                self.label = 'unknown'
                self.issue = None
                self.active_probability = None
                self.bad_since = None
                self.warning_counted = False
                self._pending_decision = None
                self._pending_frames = 0
            warning_remaining = max(
                0.0,
                self.settings['alert_delay'] - (
                    max(0.0, now - self.bad_since)
                    if live and label == 'bad' and self.bad_since is not None
                    else 0.0
                ),
            )
            alert = bool(
                live and label == 'bad' and warning_remaining <= 0
            )
            if (alert and self.session_mode == 'running'
                    and not self.warning_counted):
                self.warning_count += 1
                self.warning_counted = True
            stale_reason = (
                self.final_reason if live else
                'No fresh camera frame; final status is UNKNOWN.'
            )
            image = dict(self.image_prediction)
            rule = dict(self.rule_prediction)
            if not live:
                image.update({
                    'status': 'UNKNOWN',
                    'issue': None,
                    'confidence': None,
                    'rejection_reason': 'No fresh camera frame.',
                })
                rule = empty_prediction('No fresh camera frame.')
            metrics = dict(self.metrics) if live else {
                **self.metrics,
                'valid': False,
                'missing_reason': 'No fresh camera frame.',
            }
            active_probability = (
                self.active_probability
                if live and self.settings.get('mode') == 'image'
                else None
            )
            return {
                'label': label,
                'issue': self.issue if live else None,
                'correction': self.correction if live else (
                    'Position not detected. Keep the required head and '
                    'shoulder landmarks visible.'
                ),
                'probability': active_probability,
                'metrics': metrics,
                'head_offset_baseline': self.baseline['head_offset'],
                'baseline': dict(self.baseline),
                'alert': alert,
                'warning_remaining': round(warning_remaining, 1),
                'mode': self.settings.get('mode', 'hybrid'),
                'view_mode': self.settings['view_mode'],
                'rule_prediction': rule,
                'image_prediction': image,
                'debug': {
                    'active_mode': self.settings.get('mode', 'hybrid'),
                    'view_mode': self.settings['view_mode'],
                    'rule_prediction': rule['status'],
                    'rule_reason': rule['reason'],
                    'image_prediction': (
                        image.get('raw_label') or image['status']
                    ),
                    'image_confidence': image.get('confidence'),
                    'image_reason': image.get('rejection_reason'),
                    'final_status': label.upper(),
                    'final_reason': stale_reason,
                    'landmark_visibility': dict(
                        image.get('landmark_visibility', {})
                    ),
                },
                'camera_live': live,
                'settings': {
                    **self.settings,
                    'mode': self.settings.get('mode', 'hybrid'),
                },
                'session': {
                    'mode': self.session_mode,
                    'elapsed': round(self.elapsed, 1),
                    'good': round(self.good, 1),
                    'bad': round(self.bad, 1),
                    'score': score(self.good, self.bad),
                    'warnings': self.warning_count,
                },
                'history': list(reversed(self.history)),
                'calibration': {
                    'active': self.calibration is not None,
                    'result': self.calibration_result,
                },
            }
