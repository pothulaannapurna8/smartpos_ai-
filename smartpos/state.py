"""Thread-safe session and baseline calibration state."""
import statistics
import threading
import time
from .geometry import posture_label, score

_STATUS_BY_CATEGORY = {
    'UPRIGHT': 'GOOD',
    'NECK_ALIGNMENT': 'BAD',
    'LEAN_FORWARD': 'BAD',
    'LEAN_BACKWARD': 'BAD',
    'LEAN_LEFT': 'BAD',
    'LEAN_RIGHT': 'BAD',
    'IMAGE_POSTURE': 'BAD',
}
_CORRECTION_BY_CATEGORY = {
    'UPRIGHT': 'Good posture. Keep it up.',
    'LEAN_FORWARD': 'Move your head back gently and align it above your shoulders.',
    'LEAN_BACKWARD': 'Move to a comfortable, supported upright position.',
    'LEAN_LEFT': 'Gently bring your head toward the center above your shoulders.',
    'LEAN_RIGHT': 'Gently bring your head toward the center above your shoulders.',
    'UNEVEN_SHOULDERS': 'Relax your shoulders and keep them level.',
    'IMAGE_POSTURE': 'Adjust your sitting posture; align your head and shoulders comfortably.',
    'NECK_ALIGNMENT': (
        'Gently align your head above your shoulders and sit comfortably upright.'
    ),
}


class MonitorState:
    def __init__(self):
        self.lock = threading.RLock()
        self.settings = {
            'neck_min': 150, 'tilt_max': 10, 'head_offset_max': 0.18,
            'alert_delay': 3,
        }
        self.label = 'unknown'
        self.issue = None
        self.active_probability = None
        self.head_offset_baseline = 0.0
        self.head_offset = None
        self.correction = 'Position not detected. Keep your head and both shoulders visible.'
        self._pending_decision = None
        self._pending_frames = 0
        self._smoothing_frames = 8
        self.rule_label = 'unknown'
        self.mode = 'rules'
        self.ml_prediction = {
            'status': 'UNKNOWN',
            'issue': None,
            'confidence': None,
            'raw_label': None,
            'raw_category': None,
            'rejection_reason': 'ML inference has not run yet.',
            'landmark_visibility': {},
        }
        self.neck = None
        self.tilt = None
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
            dt = max(0.0, min(now-self.last_tick, 1.0))
            self.elapsed += dt
            # Do not credit stale measurements, e.g. disconnected webcam.
            if now-self.last_frame_at < 1.5:
                if self.label == 'good': self.good += dt
                elif self.label == 'bad': self.bad += dt
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
            elif action not in ('start','pause','stop','reset'):
                raise ValueError('Unknown session action')
            self.last_tick = now

    def set_mode(self, mode):
        if mode not in ('rules', 'ml', 'image'):
            raise ValueError('Unknown posture detection mode')
        with self.lock:
            now = time.monotonic()
            self._tick(now)
            if mode == self.mode:
                return
            self.mode = mode
            self.label = 'unknown'
            self.issue = None
            self.active_probability = None
            self.correction = (
                'Position not detected. Keep your head and both shoulders visible.'
            )
            self._pending_decision = None
            self._pending_frames = 0
            self.bad_since = None
            self.warning_counted = False
            self.ml_prediction = {
                'status': 'UNKNOWN',
                'issue': None,
                'confidence': None,
                'raw_label': None,
                'raw_category': None,
                'rejection_reason': (
                    'Awaiting the next frame.' if mode in ('ml', 'image')
                    else 'ML inference is disabled in rule-based mode.'
                ),
                'landmark_visibility': {},
            }
    def current_mode(self):
        with self.lock:
            return self.mode

    def _active_decision(self):
        if self.mode in ('ml', 'image'):
            prediction = self.ml_prediction
            return (
                prediction['status'].lower(),
                prediction['issue'],
                prediction['confidence'],
            )
        if self.rule_label == 'good':
            issue = 'UPRIGHT'
        elif self.rule_label == 'bad':
            head_delta = (
                None if self.head_offset is None
                else abs(self.head_offset - self.head_offset_baseline)
            )
            if (self.issue == 'UNEVEN_SHOULDERS' and self.tilt is not None
                    and self.tilt > self.settings['tilt_max'] - 2):
                issue = 'UNEVEN_SHOULDERS'
            elif (self.issue in ('LEAN_LEFT', 'LEAN_RIGHT')
                  and head_delta is not None
                  and head_delta > self.settings['head_offset_max'] - 0.05):
                issue = (
                    'LEAN_LEFT'
                    if self.head_offset < self.head_offset_baseline
                    else 'LEAN_RIGHT'
                )
            elif (self.issue == 'NECK_ALIGNMENT' and self.neck is not None
                  and self.neck < self.settings['neck_min'] + 5):
                issue = 'NECK_ALIGNMENT'
            elif self.tilt is not None and self.tilt > self.settings['tilt_max']:
                issue = 'UNEVEN_SHOULDERS'
            elif (head_delta is not None
                  and head_delta > self.settings['head_offset_max']):
                issue = (
                    'LEAN_LEFT'
                    if self.head_offset < self.head_offset_baseline
                    else 'LEAN_RIGHT'
                )
            else:
                issue = 'NECK_ALIGNMENT'
        else:
            issue = None
        return self.rule_label, issue, None

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
            'Position not detected. Keep your head and both shoulders visible.'
            if label == 'unknown'
            else 'Gently align your head and shoulders in a comfortable position.',
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
        if decision[:2] == (
            self._pending_decision[:2]
            if self._pending_decision is not None
            else None
        ):
            self._pending_frames += 1
            self._pending_decision = decision
        else:
            self._pending_decision = decision
            self._pending_frames = 1
        if self._pending_frames >= self._smoothing_frames:
            self._apply_decision(self._pending_decision, now)
            self._pending_decision = None
            self._pending_frames = 0

    def update_measurement(self, neck, tilt, ml_prediction=None, head_offset=None):
        with self.lock:
            now = time.monotonic()
            self._tick(now)
            if now - self.last_frame_at >= 1.5:
                self.label = 'unknown'
                self.bad_since = None
            self.neck, self.tilt = neck, tilt
            self.head_offset = head_offset
            self.last_frame_at = now
            self.rule_label = posture_label(
                neck, tilt,
                self.settings['neck_min'],
                self.settings['tilt_max'],
            )
            if (self.rule_label == 'good' and head_offset is not None
                    and abs(head_offset - self.head_offset_baseline)
                    > self.settings['head_offset_max']):
                self.rule_label = 'bad'
            if self.mode == 'rules' and self.label == 'bad' and self.rule_label == 'good':
                if (self.issue == 'UNEVEN_SHOULDERS' and tilt is not None
                        and tilt > self.settings['tilt_max'] - 2):
                    self.rule_label = 'bad'
                elif (self.issue in ('LEAN_LEFT', 'LEAN_RIGHT')
                      and head_offset is not None
                      and abs(head_offset - self.head_offset_baseline)
                      > self.settings['head_offset_max'] - 0.05):
                    self.rule_label = 'bad'
                elif (self.issue == 'NECK_ALIGNMENT' and neck is not None
                      and neck < self.settings['neck_min'] + 5):
                    self.rule_label = 'bad'
            prediction = ml_prediction or {}
            status = prediction.get('status', 'UNKNOWN')
            issue = prediction.get('issue')
            rejection_reason = prediction.get('rejection_reason')
            if status not in ('GOOD', 'BAD', 'UNKNOWN'):
                status = 'UNKNOWN'
                rejection_reason = 'Model returned an unsupported status.'
            expected_status = _STATUS_BY_CATEGORY.get(issue)
            if status == 'UNKNOWN':
                issue = None
            elif expected_status != status:
                status = 'UNKNOWN'
                issue = None
                rejection_reason = 'Model status and posture category do not match.'
            if self.mode == 'ml' and status != 'UNKNOWN':
                if issue == 'UPRIGHT' and self.rule_label != 'good':
                    status = 'UNKNOWN'
                    issue = None
                    rejection_reason = (
                        'The UPRIGHT model result is not corroborated by the '
                        'visible geometric measurements.'
                    )
                elif issue in ('LEAN_FORWARD', 'LEAN_BACKWARD'):
                    lateral_offset = (
                        head_offset is not None
                        and abs(head_offset - self.head_offset_baseline)
                        > self.settings['head_offset_max']
                    )
                    supported_alignment = (
                        neck is not None
                        and neck < self.settings['neck_min']
                        and tilt is not None
                        and tilt <= self.settings['tilt_max']
                        and not lateral_offset
                    )
                    if supported_alignment:
                        issue = 'NECK_ALIGNMENT'
                        rejection_reason = (
                            'Head alignment is outside the configured range, '
                            'but a frontal 2-D view cannot reliably determine '
                            'forward versus backward depth.'
                        )
                    else:
                        status = 'UNKNOWN'
                        issue = None
                        rejection_reason = (
                            'The model-only forward/backward result is not '
                            'supported by the visible head and shoulder measurements.'
                        )
                elif issue in ('LEAN_LEFT', 'LEAN_RIGHT'):
                    lateral_offset = (
                        None if head_offset is None
                        else head_offset - self.head_offset_baseline
                    )
                    supported_side = (
                        lateral_offset is not None
                        and abs(lateral_offset) > self.settings['head_offset_max']
                        and (
                            (issue == 'LEAN_LEFT' and lateral_offset < 0)
                            or (issue == 'LEAN_RIGHT' and lateral_offset > 0)
                        )
                    )
                    if not supported_side:
                        status = 'UNKNOWN'
                        issue = None
                        rejection_reason = (
                            'The model-only sideways result is not corroborated '
                            'by the normalized head-center offset.'
                        )
            confidence = prediction.get('confidence')
            if (not isinstance(confidence, (int, float))
                    or not 0 <= confidence <= 1):
                confidence = None
            visibility = prediction.get('landmark_visibility')
            if not isinstance(visibility, dict):
                visibility = {}
            self.ml_prediction = {
                'status': status,
                'issue': issue,
                'confidence': confidence,
                'raw_label': prediction.get('raw_label'),
                'raw_category': prediction.get('raw_category'),
                'rejection_reason': rejection_reason,
                'landmark_visibility': dict(visibility),
            }
            self._smooth_decision(self._active_decision(), now)
            if self.calibration is not None:
                cal = self.calibration
                if (neck is not None and tilt is not None
                        and head_offset is not None):
                    cal['samples'].append((neck, tilt, head_offset))
                self._finish_calibration(now)

    def _finish_calibration(self, now):
        if self.calibration is None or now < self.calibration['ends_at']:
            return
        samples = self.calibration['samples']
        if len(samples) >= 10:
            neck = statistics.median(sample[0] for sample in samples)
            tilt = statistics.median(sample[1] for sample in samples)
            head_offset = statistics.median(sample[2] for sample in samples)
            self.calibration_result = {
                'neck_min': round(max(90, min(179, neck - 10))),
                'tilt_max': round(max(1, min(45, tilt + 5))),
                'head_offset_baseline': head_offset,
                'samples': len(samples),
            }
        else:
            self.calibration_result = {
                'error': 'Not enough visible landmark samples. Try brighter lighting.'
            }
        self.calibration = None

    def calibration_action(self, action):
        with self.lock:
            if action == 'start':
                self.calibration = {'ends_at': time.monotonic()+5, 'samples': []}
                self.calibration_result = None
            elif action == 'cancel':
                self.calibration = None
                self.calibration_result = None
            elif action == 'accept':
                if not self.calibration_result or 'error' in self.calibration_result:
                    raise ValueError('No calibration suggestion available')
                self.settings.update({k:self.calibration_result[k] for k in ('neck_min','tilt_max')})
                self.head_offset_baseline = self.calibration_result.get(
                    'head_offset_baseline', self.head_offset_baseline
                )
                self.calibration_result = None
            elif action == 'reject':
                self.calibration_result = None
            else:
                raise ValueError('Unknown calibration action')

    def snapshot(self):
        with self.lock:
            now = time.monotonic()
            self._tick(now)
            self._finish_calibration(now)
            live = now-self.last_frame_at < 1.5
            if not live:
                self.label = 'unknown'
                self.issue = None
                self.active_probability = None
                self.correction = (
                    'Position not detected. Keep your head and both shoulders visible.'
                )
                self._pending_decision = None
                self._pending_frames = 0
                self.bad_since = None
                self.warning_counted = False
                if self.mode in ('ml', 'image'):
                    self.ml_prediction = {
                        **self.ml_prediction,
                        'status': 'UNKNOWN',
                        'issue': None,
                        'confidence': None,
                        'rejection_reason': (
                            'No fresh camera frame; the last prediction is stale.'
                        ),
                    }
            label = self.label if live else 'unknown'
            bad_duration = (
                max(0.0, now - self.bad_since)
                if live and label == 'bad' and self.bad_since is not None
                else 0.0
            )
            warning_remaining = max(
                0.0, self.settings['alert_delay'] - bad_duration
            )
            alert = bool(live and label == 'bad' and warning_remaining <= 0)
            if (alert and self.session_mode == 'running'
                    and not self.warning_counted):
                self.warning_count += 1
                self.warning_counted = True
            return {'label': label, 'issue': self.issue if live else None,
                    'correction': self.correction if live else (
                        'Position not detected. Keep your head and both shoulders visible.'
                    ),
                    'probability': self.active_probability if live else None,
                    'neck': self.neck if live else None,
                    'tilt': self.tilt if live else None,
                    'head_offset': self.head_offset if live else None,
                    'head_offset_baseline': self.head_offset_baseline,
                    'alert': alert,
                    'warning_remaining': round(warning_remaining, 1),
                    'mode': self.mode,
                    'rule_label': self.rule_label if live else 'unknown',
                    'ml_prediction': dict(self.ml_prediction),
                    'debug': {
                        'active_mode': self.mode,
                        'raw_ml_prediction': self.ml_prediction['raw_label'],
                        'raw_posture_category': self.ml_prediction['raw_category'],
                        'model_probability': self.ml_prediction['confidence'],
                        'final_status': label.upper(),
                        'rejection_reason': self.ml_prediction['rejection_reason'],
                        'landmark_visibility': dict(
                            self.ml_prediction['landmark_visibility']
                        ),
                    },
                    'camera_live': live, 'settings': dict(self.settings),
                    'session': {'mode':self.session_mode,'elapsed':round(self.elapsed,1),
                                'good':round(self.good,1),'bad':round(self.bad,1),
                                'score':score(self.good,self.bad),
                                'warnings':self.warning_count},
                    'history': list(reversed(self.history)),
                    'calibration': {'active':self.calibration is not None,
                                    'result':self.calibration_result}}
