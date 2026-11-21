"""Consent-based local webcam sample collection and session-held-out evaluation."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import uuid

try:
    import cv2
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        'OpenCV and NumPy are required. Install project requirements.txt '
        f'in the active Python environment. Details: {exc}'
    ) from exc

try:
    from image_posture_inference import ImagePosturePredictor
except ImportError as exc:
    ImagePosturePredictor = None
    IMAGE_IMPORT_ERROR = f'{type(exc).__name__}: {exc}'
else:
    IMAGE_IMPORT_ERROR = ''
from smartpos.geometry import measure_posture
from smartpos.landmarks import posture_points
from smartpos.state import MonitorState

POSTURE_LABELS = (
    'GOOD_UPRIGHT',
    'BAD_FORWARD',
    'BAD_LEFT',
    'BAD_RIGHT',
    'BAD_SHOULDER_TILT',
)
LABEL_KEYS = {
    ord('g'): 'GOOD_UPRIGHT',
    ord('f'): 'BAD_FORWARD',
    ord('l'): 'BAD_LEFT',
    ord('r'): 'BAD_RIGHT',
    ord('t'): 'BAD_SHOULDER_TILT',
}
BURST_FRAMES = 12
PREDICTION_LABELS = ('GOOD', 'BAD', 'UNKNOWN')


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def confusion_report(rows):
    """Return a 2-by-3 confusion table; UNKNOWN remains an explicit outcome."""
    actual_classes = ('GOOD', 'BAD')
    predicted_classes = PREDICTION_LABELS
    counts = {
        actual: Counter(
            row['prediction'] for row in rows
            if row['actual'] == actual
        )
        for actual in actual_classes
    }
    matrix = [
        [counts[actual][predicted] for predicted in predicted_classes]
        for actual in actual_classes
    ]
    class_metrics = {}
    recalls = []
    for actual in actual_classes:
        tp = counts[actual][actual]
        support = sum(counts[actual].values())
        predicted_support = sum(
            1 for row in rows if row['prediction'] == actual
        )
        precision = _ratio(tp, predicted_support)
        recall = _ratio(tp, support)
        class_metrics[actual] = {
            'precision': precision,
            'recall': recall,
            'support': support,
        }
        if recall is not None:
            recalls.append(recall)
    posture_recall = {}
    for label in POSTURE_LABELS[1:]:
        matching = [row for row in rows if row['actual_label'] == label]
        posture_recall[label] = {
            'recall': _ratio(
                sum(row['prediction'] == 'BAD' for row in matching),
                len(matching),
            ),
            'support': len(matching),
        }
    total = len(rows)
    return {
        'labels': {
            'actual': list(actual_classes),
            'predicted': list(predicted_classes),
        },
        'confusion_matrix': matrix,
        'class_metrics': class_metrics,
        'balanced_accuracy': (
            sum(recalls) / len(recalls) if len(recalls) == 2 else None
        ),
        'posture_specific_recall': posture_recall,
        'unknown_rate': _ratio(
            sum(row['prediction'] == 'UNKNOWN' for row in rows), total
        ),
        'samples': total,
    }


def _read_samples(root):
    sessions = {}
    session_root = root / 'sessions'
    if not session_root.exists():
        raise ValueError(f'No collected session directory at {session_root}')
    for manifest in sorted(session_root.glob('*/manifest.jsonl')):
        session_id = manifest.parent.name
        records = []
        with manifest.open(encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f'Invalid JSON in {manifest}:{line_number}: {exc}'
                    ) from exc
                if (
                    not isinstance(record, dict)
                    or record.get('label') not in POSTURE_LABELS
                    or record.get('session_id') != session_id
                    or not isinstance(record.get('frame'), str)
                    or not isinstance(record.get('burst_id'), str)
                    or not isinstance(record.get('sequence'), int)
                    or record.get('view_mode', 'frontal') not in ('frontal', 'side')
                ):
                    raise ValueError(
                        f'Invalid sample record in {manifest}:{line_number}'
                    )
                frame_path = (manifest.parent / record['frame']).resolve()
                if not frame_path.is_relative_to(manifest.parent.resolve()):
                    raise ValueError(f'Unsafe frame path in {manifest}:{line_number}')
                if not frame_path.is_file():
                    raise ValueError(f'Missing collected frame: {frame_path}')
                records.append(record)
        if records:
            sessions[session_id] = (manifest.parent, records)
    if not sessions:
        raise ValueError('No labeled webcam frames were found.')
    return sessions


def _predict_session(records, session_dir, view_mode, mode, baseline,
                     image_predictor, pose_module):
    state = MonitorState()
    state.settings['mode'] = mode
    state.settings['view_mode'] = view_mode
    state.baseline.update(baseline)
    grouped = defaultdict(list)
    ordered = sorted(records, key=lambda row: row['sequence'])
    with pose_module.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.6,
    ) as pose:
        for record in ordered:
            frame = cv2.imread(str(session_dir / record['frame']))
            if frame is None:
                raise ValueError(
                    f"Could not decode collected frame {record['frame']}"
                )
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            landmarks = (
                result.pose_landmarks.landmark
                if result.pose_landmarks else None
            )
            measurements = measure_posture(
                posture_points(landmarks), width, height, view_mode
            )
            if image_predictor is None:
                image_prediction = {
                    'status': 'UNKNOWN',
                    'rejection_reason': 'Image model unavailable.',
                    'landmark_visibility': {},
                }
            else:
                image_prediction = image_predictor.predict(frame, landmarks)
            state.update_frame(measurements, image_prediction)
            grouped[record['burst_id']].append({
                'label': record['label'],
                'prediction': state.snapshot()['label'].upper(),
                'image_prediction': image_prediction.get('raw_label'),
                'rule_prediction': state.rule_prediction['status'],
            })
    results = []
    for burst_id, predictions in grouped.items():
        actual_labels = {item['label'] for item in predictions}
        if len(actual_labels) != 1:
            raise ValueError(
                f'Burst {burst_id} contains more than one ground-truth label.'
            )
        actual_label = actual_labels.pop()
        final = (
            predictions[-1]['prediction']
            if len(predictions) >= state._smoothing_frames
            else 'UNKNOWN'
        )
        results.append({
            'session_id': records[0]['session_id'],
            'burst_id': burst_id,
            'actual_label': actual_label,
            'actual': 'GOOD' if actual_label == 'GOOD_UPRIGHT' else 'BAD',
            'prediction': final,
            'image_prediction': predictions[-1]['image_prediction'],
            'rule_prediction': predictions[-1]['rule_prediction'],
            'frames': len(predictions),
        })
    return results


def evaluate_sessions(data_dir, mode='hybrid', calibration_session=None,
                      image_model_path=None):
    """Evaluate only recorded sessions; this function never fits a model."""
    if mode not in ('hybrid', 'rules', 'image'):
        raise ValueError('mode must be hybrid, rules, or image')
    root = Path(data_dir).resolve()
    sessions = _read_samples(root)
    if calibration_session is not None and calibration_session not in sessions:
        raise ValueError('The selected calibration session does not exist.')
    if len(sessions) - (1 if calibration_session else 0) < 1:
        raise ValueError('At least one evaluation session is required.')
    calibration_records = []
    if calibration_session:
        calibration_records = sessions[calibration_session][1]
        if any(row['label'] != 'GOOD_UPRIGHT' for row in calibration_records):
            raise ValueError(
                'The calibration session must contain only GOOD_UPRIGHT samples.'
            )
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            'MediaPipe is unavailable. Install requirements.txt in the project '
            f'environment. Details: {exc}'
        ) from exc
    try:
        if ImagePosturePredictor is None:
            raise ImportError(IMAGE_IMPORT_ERROR)
        image_predictor = ImagePosturePredictor(image_model_path)
    except Exception as exc:
        if mode == 'image':
            raise RuntimeError(
                f'Image-only evaluation cannot load its model: {exc}'
            ) from exc
        image_predictor = None

    baseline = {}
    view_modes = {
        row.get('view_mode', 'frontal') for row in calibration_records
    }
    if calibration_session and len(view_modes) != 1:
        raise ValueError('Calibration samples must use a single camera view.')
    if calibration_records:
        calibration_view = next(iter(view_modes))
        calibration_dir, _ = sessions[calibration_session]
        calibration_metrics = []
        with mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.6,
        ) as pose:
            for row in sorted(
                    calibration_records, key=lambda item: item['sequence']):
                frame = cv2.imread(str(calibration_dir / row['frame']))
                if frame is None:
                    raise ValueError(
                        f"Could not decode collected frame {row['frame']}"
                    )
                result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                landmarks = (
                    result.pose_landmarks.landmark
                    if result.pose_landmarks else None
                )
                measurement = measure_posture(
                    posture_points(landmarks),
                    frame.shape[1],
                    frame.shape[0],
                    calibration_view,
                )
                if measurement['valid']:
                    calibration_metrics.append(measurement)
        for key in (
            'head_offset', 'head_alignment_deg',
            'shoulder_tilt_signed_deg', 'torso_lean_deg',
        ):
            values = [
                row[key] for row in calibration_metrics
                if row.get(key) is not None
            ]
            if values:
                baseline[key] = float(np.median(values))
        if not baseline:
            raise ValueError(
                'The selected calibration session had no reliable pose measurements.'
            )
        baseline['calibrated'] = True
    elif mode in ('hybrid', 'rules'):
        raise ValueError(
            'Hybrid and rule-only evaluation require a separate upright '
            'calibration session via --calibration-session.'
        )

    evaluation_rows = []
    evaluation_sessions = []
    for session_id, (session_dir, records) in sessions.items():
        if session_id == calibration_session:
            continue
        session_views = {
            row.get('view_mode', 'frontal') for row in records
        }
        if len(session_views) != 1:
            raise ValueError(
                f'Session {session_id} contains inconsistent camera views.'
            )
        view_mode = next(iter(session_views))
        if calibration_records and view_mode != calibration_view:
            raise ValueError(
                'Calibration and evaluation sessions must use the same camera view.'
            )
        rows = _predict_session(
            records,
            session_dir,
            view_mode,
            mode,
            baseline,
            image_predictor,
            mp.solutions.pose,
        )
        evaluation_rows.extend(rows)
        evaluation_sessions.append(session_id)
    if not evaluation_rows:
        raise ValueError('No webcam bursts are available for evaluation.')
    metric_rows = [
        {**row, 'actual': row['actual'], 'prediction': row['prediction']}
        for row in evaluation_rows
    ]
    metrics = confusion_report(metric_rows)
    metrics['evaluation_sessions'] = evaluation_sessions
    metrics['calibration_session'] = calibration_session
    metrics['baseline'] = baseline
    metrics['mode'] = mode
    metrics['per_session_metrics'] = [
        {
            'session_id': session_id,
            **confusion_report([
                row for row in metric_rows
                if row['session_id'] == session_id
            ]),
        }
        for session_id in evaluation_sessions
    ]
    metrics['note'] = (
        'No training or model fitting was performed. These user-collected '
        'sessions are evaluation-only; frame bursts are correlated and are '
        'summarized as one prediction per labeled burst. This report does not '
        'establish performance for other users, cameras, or rooms.'
    )
    return metrics


def collect_samples(camera_index, output_dir, view_mode='frontal'):
    if view_mode not in ('frontal', 'side'):
        raise ValueError('view_mode must be frontal or side')
    if not sys.stdin.isatty():
        raise RuntimeError(
            'Sample collection requires an interactive consent prompt and webcam window.'
        )
    print(
        'Frames are saved only in the selected local output folder and are not '
        'uploaded. Obtain consent from anyone visible before recording.'
    )
    if input('Type YES to begin local webcam capture: ').strip() != 'YES':
        print('Capture cancelled; no samples were saved.')
        return None

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f'Cannot open webcam index {camera_index}. Check permissions and '
            'try a different --camera value.'
        )
    session_id = uuid.uuid4().hex
    session_dir = Path(output_dir) / 'sessions' / session_id
    session_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = session_dir / 'manifest.jsonl'
    sequence = 0
    try:
        with manifest_path.open('w', encoding='utf-8') as manifest:
            while True:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(
                        'Webcam stopped delivering frames during sample capture.'
                    )
                preview = cv2.flip(frame, 1)
                cv2.putText(
                    preview,
                    'g GOOD  f FORWARD  l LEFT  r RIGHT  t SHOULDER  q QUIT',
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (40, 235, 220),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    preview,
                    f'Session {session_id[:8]} · {view_mode} view · saved {sequence}',
                    (12, 54),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (245, 245, 245),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow('SmartPos local validation capture', preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                label = LABEL_KEYS.get(key)
                if label is None:
                    continue
                burst_id = uuid.uuid4().hex
                for _ in range(BURST_FRAMES):
                    ok, sample = capture.read()
                    if not ok:
                        raise RuntimeError(
                            'Webcam stopped delivering frames during a sample burst.'
                        )
                    filename = f'{sequence:06d}.jpg'
                    if not cv2.imwrite(str(session_dir / filename), sample):
                        raise OSError(
                            f'Could not write local validation frame {filename}.'
                        )
                    record = {
                        'session_id': session_id,
                        'sequence': sequence,
                        'burst_id': burst_id,
                        'frame': filename,
                        'label': label,
                        'view_mode': view_mode,
                        'captured_at': datetime.now(
                            timezone.utc
                        ).isoformat(),
                    }
                    manifest.write(json.dumps(record) + '\n')
                    sequence += 1
                manifest.flush()
                print(f'Saved {BURST_FRAMES} {label} frames to session {session_id}.')
    finally:
        capture.release()
        cv2.destroyAllWindows()
    print(f'Validation session ID: {session_id}')
    print(f'Local samples: {session_dir}')
    return session_id


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Collect consented local webcam samples and evaluate held-out sessions.'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)
    collect = subparsers.add_parser('collect', help='collect user-labeled webcam bursts')
    collect.add_argument('--camera', type=int, default=None)
    collect.add_argument('--output-dir', type=Path, default=Path('webcam-validation'))
    collect.add_argument('--view', choices=('frontal', 'side'), default='frontal')
    evaluate = subparsers.add_parser('evaluate', help='evaluate saved, unused webcam sessions')
    evaluate.add_argument('--data-dir', type=Path, default=Path('webcam-validation'))
    evaluate.add_argument('--mode', choices=('hybrid', 'rules', 'image'), default='hybrid')
    evaluate.add_argument('--calibration-session', default=None)
    evaluate.add_argument('--image-model', type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        if args.command == 'collect':
            camera_index = args.camera
            if camera_index is None:
                try:
                    camera_index = int(os.environ.get('SMARTPOS_CAMERA', '0'))
                except ValueError as exc:
                    raise ValueError(
                        'SMARTPOS_CAMERA must be a non-negative integer.'
                    ) from exc
            if camera_index < 0:
                raise ValueError('--camera must be a non-negative integer.')
            collect_samples(camera_index, args.output_dir, args.view)
            return 0

        report = evaluate_sessions(
            args.data_dir,
            args.mode,
            args.calibration_session,
            args.image_model,
        )
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        report_path = args.data_dir / f'evaluation-report-{stamp}.json'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
        print(f'Report saved locally to {report_path}')
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
