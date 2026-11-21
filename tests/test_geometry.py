import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from app import create_app
from image_posture_inference import ImagePosturePredictor
from smartpos.camera import CameraWorker, draw_posture_overlay
from smartpos.detection import combine_predictions, rule_prediction
from smartpos.geometry import measure_posture, score
from smartpos.state import MonitorState
from train_image_posture import image_features as training_image_features
from webcam_validation import confusion_report


def frontal_pose(head_x=0.5, shoulder_tilt=0.0, hips=True, hip_shift=0.0):
    pose = {
        'nose': (head_x, 0.25, 0.95),
        'left_ear': (head_x - 0.03, 0.25, 0.9),
        'right_ear': (head_x + 0.03, 0.25, 0.9),
        'left_shoulder': (0.35, 0.5, 0.95),
        'right_shoulder': (0.65, 0.5 + shoulder_tilt, 0.95),
    }
    if hips:
        pose.update({
            'left_hip': (0.42 + hip_shift, 0.8, 0.9),
            'right_hip': (0.58 + hip_shift, 0.8, 0.9),
        })
    return pose


def side_pose(head_x=0.51, torso_shift=0.0):
    return {
        'nose': (head_x, 0.2, 0.95),
        'left_ear': (head_x - 0.02, 0.22, 0.9),
        'left_shoulder': (0.5 + torso_shift, 0.45, 0.9),
        'right_shoulder': (0.51 + torso_shift, 0.45, 0.2),
        'left_hip': (0.5, 0.8, 0.9),
        'right_hip': (0.51, 0.8, 0.2),
    }


def good_measurement():
    return measure_posture(frontal_pose(), 640, 480, 'frontal')


class GeometryTests(unittest.TestCase):
    def test_frontal_measurements_are_normalized_and_hips_are_optional(self):
        metrics = measure_posture(frontal_pose(hips=False), 640, 480, 'frontal')
        self.assertTrue(metrics['valid'])
        self.assertAlmostEqual(metrics['head_offset'], 0.0)
        self.assertAlmostEqual(metrics['head_alignment_deg'], 0.0)
        self.assertAlmostEqual(metrics['head_vertical_displacement'], 0.625)
        self.assertAlmostEqual(metrics['shoulder_tilt_deg'], 0.0)
        self.assertIsNone(metrics['torso_lean_deg'])
        self.assertEqual(metrics['head_offset_unit'], 'shoulder widths')

    def test_head_offset_uses_pixel_normalized_shoulder_span(self):
        metrics = measure_posture(frontal_pose(head_x=0.59), 640, 480)
        self.assertAlmostEqual(metrics['head_offset'], 0.30, places=2)
        self.assertGreater(metrics['head_alignment_deg'], 0)

    def test_shoulder_tilt_and_optional_torso_lean_are_measured(self):
        tilted = measure_posture(
            frontal_pose(shoulder_tilt=0.08), 640, 480, 'frontal'
        )
        self.assertGreater(tilted['shoulder_tilt_deg'], 5)
        leaning = measure_posture(
            frontal_pose(hip_shift=0.08), 640, 480, 'frontal'
        )
        self.assertLess(leaning['torso_lean_deg'], -5)

    def test_side_profile_uses_single_visible_shoulder_and_hips(self):
        metrics = measure_posture(side_pose(), 640, 480, 'side')
        self.assertTrue(metrics['valid'])
        self.assertIsNotNone(metrics['head_offset'])
        self.assertEqual(metrics['head_offset_unit'], 'torso lengths')
        self.assertIsNotNone(metrics['torso_lean_deg'])

    def test_missing_or_invalid_landmarks_are_unknown(self):
        metrics = measure_posture(
            {'left_shoulder': (0.3, 0.5, 0.9)}, 640, 480, 'frontal'
        )
        self.assertFalse(metrics['valid'])
        result = rule_prediction(metrics, MonitorState().settings, {})
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertIn('both visible shoulders', result['reason'])

    def test_rule_detection_uses_calibrated_head_baseline_and_hysteresis(self):
        settings = MonitorState().settings
        upright = good_measurement()
        uncalibrated = rule_prediction(upright, settings, {})
        self.assertEqual(uncalibrated['status'], 'UNKNOWN')
        self.assertIn('calibration', uncalibrated['reason'])
        result = rule_prediction(
            upright,
            settings,
            {
                'calibrated': True,
                'head_offset': 0.1,
                'head_alignment_deg': 0.0,
            },
        )
        self.assertEqual(result['status'], 'GOOD')
        shifted = measure_posture(frontal_pose(head_x=0.6), 640, 480)
        result = rule_prediction(
            shifted,
            settings,
            {
                'calibrated': True,
                'head_offset': 0.0,
                'head_alignment_deg': 0.0,
            },
        )
        self.assertEqual(result['status'], 'BAD')
        self.assertEqual(result['issue'], 'LEAN_RIGHT')

    def test_significant_shoulder_tilt_is_bad_relative_to_calibrated_baseline(self):
        settings = MonitorState().settings
        settings['view_mode'] = 'frontal'
        metrics = measure_posture(
            frontal_pose(shoulder_tilt=0.1), 640, 480, 'frontal'
        )
        result = rule_prediction(
            metrics,
            settings,
            {
                'calibrated': True,
                'head_offset': metrics['head_offset'],
                'head_alignment_deg': metrics['head_alignment_deg'],
                'shoulder_tilt_signed_deg': 0.0,
            },
        )
        self.assertEqual(result['status'], 'BAD')
        self.assertEqual(result['issue'], 'UNEVEN_SHOULDERS')

        recovered = measure_posture(frontal_pose(head_x=0.55), 640, 480)
        result = rule_prediction(
            recovered,
            settings,
            {
                'calibrated': True,
                'head_offset': 0.0,
                'head_alignment_deg': 0.0,
            },
            previous_status='BAD',
        )
        self.assertEqual(result['status'], 'BAD')

    def test_side_profile_forward_displacement_triggers_alignment_warning(self):
        settings = MonitorState().settings
        settings['view_mode'] = 'side'
        baseline = measure_posture(side_pose(), 640, 480, 'side')
        forward = measure_posture(
            side_pose(head_x=0.62), 640, 480, 'side'
        )
        result = rule_prediction(
            forward,
            settings,
            {
                'calibrated': True,
                'head_offset': baseline['head_offset'],
                'head_alignment_deg': baseline['head_alignment_deg'],
                'torso_lean_deg': baseline['torso_lean_deg'],
            },
        )
        self.assertEqual(result['status'], 'BAD')
        self.assertEqual(result['issue'], 'HEAD_ALIGNMENT')

    def test_hybrid_never_lets_unvalidated_image_good_override_bad_geometry(self):
        image_good = {'status': 'GOOD', 'raw_label': 'GOOD', 'confidence': 0.99}
        geometry_bad = {
            'status': 'BAD', 'issue': 'LEAN_LEFT', 'reason': 'offset over limit',
        }
        result = combine_predictions('hybrid', geometry_bad, image_good)
        self.assertEqual(result['status'], 'BAD')
        geometry_unknown = {
            'status': 'UNKNOWN', 'issue': None, 'reason': 'landmarks missing',
        }
        self.assertEqual(
            combine_predictions('hybrid', geometry_unknown, image_good)['status'],
            'UNKNOWN',
        )
        geometry_good = {
            'status': 'GOOD', 'issue': 'UPRIGHT', 'reason': 'within limits',
        }
        image_bad = {
            'status': 'BAD', 'raw_label': 'BAD', 'confidence': 0.95,
        }
        conflict = combine_predictions('hybrid', geometry_good, image_bad)
        self.assertEqual(conflict['status'], 'UNKNOWN')
        self.assertIn('conflict', conflict['reason'])

    def test_state_smooths_final_decision_and_preserves_rule_and_ai_diagnostics(self):
        state = MonitorState()
        state.baseline['calibrated'] = True
        image = {
            'status': 'GOOD', 'issue': 'UPRIGHT', 'raw_label': 'GOOD',
            'confidence': 0.81,
        }
        for _ in range(state._smoothing_frames - 1):
            state.update_frame(good_measurement(), image)
        self.assertEqual(state.snapshot()['label'], 'unknown')
        state.update_frame(good_measurement(), image)
        snapshot = state.snapshot()
        self.assertEqual(snapshot['label'], 'good')
        self.assertEqual(snapshot['rule_prediction']['status'], 'GOOD')
        self.assertEqual(snapshot['image_prediction']['raw_label'], 'GOOD')
        self.assertIn('advisory only', snapshot['debug']['final_reason'])

    def test_missing_landmarks_force_unknown_without_smoothing_delay(self):
        state = MonitorState()
        state.baseline['calibrated'] = True
        state._smoothing_frames = 1
        for _ in range(1):
            state.update_frame(good_measurement(), {'status': 'UNKNOWN'})
        self.assertEqual(state.snapshot()['label'], 'good')
        state.update_frame(
            {'valid': False, 'missing_reason': 'shoulders not visible'},
            {'status': 'UNKNOWN'},
        )
        self.assertEqual(state.snapshot()['label'], 'unknown')

    def test_calibration_collects_and_applies_upright_baseline(self):
        state = MonitorState()
        state.calibration = {'ends_at': 20.0, 'samples': []}
        with patch('smartpos.state.time.monotonic', return_value=19.0):
            for _ in range(10):
                state.update_frame(good_measurement(), {})
        state._finish_calibration(21.0)
        result = state.calibration_result
        self.assertEqual(result['samples'], 10)
        self.assertIn('head_offset', result['baselines'])
        state.calibration_action('accept')
        self.assertAlmostEqual(state.baseline['head_offset'], 0.0)

    def test_score_excludes_unknown_time(self):
        self.assertEqual(score(3.0, 1.0), 75.0)
        self.assertEqual(score(0.0, 0.0), 0.0)

    def test_running_session_counts_fresh_stable_good_time(self):
        state = MonitorState()
        state.baseline['calibrated'] = True
        state._smoothing_frames = 1
        with patch(
                'smartpos.state.time.monotonic',
                side_effect=[100.0, 100.1, 100.6, 101.1]):
            state.update_frame(good_measurement(), {'status': 'UNKNOWN'})
            state.session_action('start')
            state.update_frame(good_measurement(), {'status': 'UNKNOWN'})
            snapshot = state.snapshot()
        self.assertEqual(snapshot['session']['mode'], 'running')
        self.assertEqual(snapshot['session']['good'], 1.0)
        self.assertEqual(snapshot['session']['score'], 100.0)


class ImageModelTests(unittest.TestCase):
    def test_saved_binary_model_loads_with_expected_feature_schema(self):
        predictor = ImagePosturePredictor()
        self.assertEqual(predictor.pipeline.classes_.tolist(), [0, 1])
        self.assertEqual(predictor.pipeline.n_features_in_, 1764)
        self.assertEqual(
            len(training_image_features(np.zeros((120, 160, 3), dtype=np.uint8))),
            1764,
        )

    def test_saved_model_runs_inference_on_full_bgr_frame(self):
        predictor = ImagePosturePredictor()
        landmarks = [SimpleNamespace(visibility=0.9) for _ in range(13)]
        result = predictor.predict(
            np.zeros((480, 640, 3), dtype=np.uint8), landmarks
        )
        self.assertIn(result['raw_label'], ('BAD', 'GOOD'))
        self.assertIn(result['status'], ('BAD', 'GOOD', 'UNKNOWN'))
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)

    def test_training_and_inference_use_the_same_preprocessing_function(self):
        import smartpos.image_features as shared_features
        self.assertIs(training_image_features, shared_features.image_features)

    def test_hog_feature_extraction_matches_original_bgr_preprocessing(self):
        import cv2
        from smartpos.image_features import image_features
        image = np.zeros((90, 130, 3), dtype=np.uint8)
        image[:, :65] = (255, 0, 0)
        image[:, 65:] = (0, 0, 255)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        equalized = cv2.equalizeHist(resized)
        descriptor = cv2.HOGDescriptor(
            (64, 64), (16, 16), (8, 8), (8, 8), 9
        ).compute(equalized).reshape(-1)
        np.testing.assert_array_equal(image_features(image), descriptor)

    def test_binary_class_mapping_and_low_confidence_are_explicit(self):
        predictor = ImagePosturePredictor.__new__(ImagePosturePredictor)
        predictor.pipeline = Mock()
        predictor.pipeline.classes_ = np.array([0, 1])
        predictor.pipeline.predict_proba.return_value = np.array([[0.15, 0.85]])
        landmarks = [SimpleNamespace(visibility=0.9) for _ in range(13)]
        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        with patch('image_posture_inference.image_features', return_value=[0]):
            result = predictor.predict(frame, landmarks)
        self.assertEqual(result['status'], 'GOOD')
        self.assertEqual(result['raw_label'], 'GOOD')
        self.assertAlmostEqual(result['confidence'], 0.85)

        predictor.pipeline.predict_proba.return_value = np.array([[0.5, 0.5]])
        with patch('image_posture_inference.image_features', return_value=[0]):
            result = predictor.predict(frame, landmarks)
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertEqual(result['raw_label'], 'BAD')
        self.assertIn('0.500', result['rejection_reason'])

    def test_model_requires_bgr_three_channel_full_frame_input(self):
        from smartpos.image_features import image_features
        with self.assertRaisesRegex(ValueError, 'three-channel BGR'):
            image_features(np.zeros((64, 64), dtype=np.uint8))


class CameraWorkerTests(unittest.TestCase):
    def test_image_predictor_receives_original_bgr_frame(self):
        frame = np.zeros((12, 16, 3), dtype=np.uint8)
        landmarks = [SimpleNamespace(visibility=0.9) for _ in range(13)]
        predictor = Mock()
        predictor.predict.return_value = {'status': 'GOOD'}
        worker = CameraWorker(MonitorState(), image_predictor=predictor)
        worker._predict_image(frame, landmarks)
        predictor.predict.assert_called_once_with(frame, landmarks)

    def test_image_inference_failure_is_reported_without_crashing(self):
        predictor = Mock()
        predictor.predict.side_effect = ValueError('invalid image')
        worker = CameraWorker(MonitorState(), image_predictor=predictor)
        result = worker._predict_image(np.zeros((8, 8, 3), dtype=np.uint8), None)
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertIn('invalid image', result['rejection_reason'])
        self.assertFalse(worker.image_model_available)

    def test_camera_worker_processes_bgr_frame_updates_state_and_encodes_jpeg(self):
        state = MonitorState()
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        predictor = Mock()
        predictor.predict.return_value = {
            'status': 'UNKNOWN',
            'rejection_reason': 'No usable landmarks.',
            'landmark_visibility': {},
        }
        worker = CameraWorker(state, image_predictor=predictor)

        class FakePose:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def process(self, _rgb):
                worker.running = False
                return SimpleNamespace(pose_landmarks=None)

        class FakeCapture:
            def isOpened(self):
                return True

            def set(self, *_args):
                return True

            def read(self):
                return True, frame

            def release(self):
                return None

        fake_pose_api = SimpleNamespace(
            Pose=lambda **_kwargs: FakePose(),
            POSE_CONNECTIONS=(),
            PoseLandmark=SimpleNamespace(),
        )
        fake_mediapipe = SimpleNamespace(
            solutions=SimpleNamespace(
                pose=fake_pose_api,
                drawing_utils=SimpleNamespace(draw_landmarks=Mock()),
            )
        )
        with patch.dict('sys.modules', {'mediapipe': fake_mediapipe}):
            with patch('smartpos.camera.cv2.VideoCapture', return_value=FakeCapture()):
                worker.running = True
                worker._run()
        predictor.predict.assert_called_once_with(frame, None)
        self.assertIsNotNone(worker.frame())
        self.assertEqual(state.snapshot()['mode'], 'hybrid')
        self.assertTrue(state.snapshot()['camera_live'])

    def test_overlay_uses_posture_label_and_never_calls_alignment_a_neck_angle(self):
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        status = {
            'label': 'bad',
            'issue': 'HEAD_ALIGNMENT',
            'metrics': {
                'head_alignment_deg': 22.5,
                'shoulder_tilt_deg': 7.2,
                'shoulder_tilt_signed_deg': 7.2,
                'head_offset': 0.2,
                'head_offset_unit': 'shoulder widths',
            },
            'baseline': {
                'head_offset': 0.0,
                'shoulder_tilt_signed_deg': 0.0,
            },
            'debug': {'rule_prediction': 'BAD'},
            'image_prediction': {
                'status': 'UNKNOWN', 'raw_label': 'BAD', 'confidence': 0.6,
            },
            'correction': 'Realign gently.',
            'warning_remaining': 2.5,
        }
        with patch('smartpos.camera.cv2.putText') as put_text:
            draw_posture_overlay(frame, status)
        rendered = ' '.join(call.args[1] for call in put_text.call_args_list)
        self.assertIn('BAD POSTURE', rendered)
        self.assertIn('image-plane deviation', rendered)
        self.assertIn('Image AI: BAD', rendered)
        self.assertIn('Rule: BAD', rendered)
        self.assertNotIn('Neck angle', rendered)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app(
            start_camera=False,
            image_predictor=Mock(),
        )
        self.client = self.app.test_client()

    def test_status_defaults_to_hybrid_and_exposes_independent_diagnostics(self):
        status = self.client.get('/api/status').json
        self.assertEqual(status['mode'], 'hybrid')
        self.assertIn('rule_prediction', status)
        self.assertIn('image_prediction', status)
        self.assertIn('final_reason', status['debug'])
        self.assertTrue(status['image_model_available'])

    def test_modes_support_hybrid_rules_image_but_reject_legacy_ml(self):
        for mode in ('hybrid', 'rules', 'image'):
            response = self.client.post('/api/mode', json={'mode': mode})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json['mode'], mode)
        response = self.client.post('/api/mode', json={'mode': 'ml'})
        self.assertEqual(response.status_code, 400)
        self.assertNotIn(b'five-class', self.client.get('/').data.lower())

    def test_settings_validate_view_and_configurable_thresholds(self):
        state = self.app.extensions['monitor_state']
        state.baseline['calibrated'] = True
        response = self.client.post('/api/settings', json={
            'view_mode': 'side',
            'head_offset_max': 0.22,
            'alignment_max': 18,
            'torso_lean_max': 20,
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['settings']['view_mode'], 'side')
        self.assertEqual(response.json['settings']['head_offset_max'], 0.22)
        self.assertFalse(state.baseline['calibrated'])
        self.assertEqual(
            self.client.post(
                '/api/settings', json={'view_mode': 'diagonal'}
            ).status_code,
            400,
        )
        self.assertEqual(
            self.client.post(
                '/api/settings', json={'alignment_max': float('nan')}
            ).status_code,
            400,
        )

    def test_model_file_error_is_reported_without_blocking_geometry(self):
        client = create_app(
            start_camera=False,
            image_model_path='missing-image-model.joblib',
        ).test_client()
        status = client.get('/api/status').json
        self.assertFalse(status['image_model_available'])
        self.assertIn('missing-image-model', status['image_model_error'])
        self.assertEqual(status['mode'], 'hybrid')
        self.assertEqual(
            client.post('/api/mode', json={'mode': 'image'}).status_code,
            503,
        )
        self.assertEqual(
            client.post('/api/mode', json={'mode': 'rules'}).status_code,
            200,
        )

    def test_invalid_camera_index_is_reported_instead_of_silently_used(self):
        import os
        with patch.dict(os.environ, {'SMARTPOS_CAMERA': 'not-a-number'}):
            client = create_app(start_camera=True).test_client()
        status = client.get('/api/status').json
        self.assertIn('SMARTPOS_CAMERA must', status['camera_error'])
        self.assertFalse(status['camera_live'])

    def test_dashboard_contains_model_and_rule_diagnostics_and_new_metrics(self):
        html = self.client.get('/').data
        self.assertIn(b'id="debug-rule"', html)
        self.assertIn(b'id="debug-image"', html)
        self.assertIn(b'id="alignment"', html)
        self.assertIn(b'value="hybrid"', html)


class WebcamValidationTests(unittest.TestCase):
    def test_confusion_report_keeps_unknown_and_posture_recall(self):
        report = confusion_report([
            {
                'actual': 'GOOD', 'actual_label': 'GOOD_UPRIGHT',
                'prediction': 'GOOD',
            },
            {
                'actual': 'BAD', 'actual_label': 'BAD_FORWARD',
                'prediction': 'UNKNOWN',
            },
            {
                'actual': 'BAD', 'actual_label': 'BAD_LEFT',
                'prediction': 'BAD',
            },
        ])
        self.assertEqual(
            report['confusion_matrix'], [[1, 0, 0], [0, 1, 1]]
        )
        self.assertAlmostEqual(report['balanced_accuracy'], 0.75)
        self.assertAlmostEqual(report['unknown_rate'], 1 / 3)
        self.assertEqual(
            report['posture_specific_recall']['BAD_FORWARD']['recall'], 0.0
        )
        self.assertIsNone(
            report['posture_specific_recall']['BAD_RIGHT']['recall']
        )

    def test_saved_manifest_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = root / 'sessions' / 'abc'
            session.mkdir(parents=True)
            (session / 'manifest.jsonl').write_text(
                json.dumps({
                    'session_id': 'abc',
                    'sequence': 0,
                    'burst_id': 'burst',
                    'frame': '..\\outside.jpg',
                    'label': 'GOOD_UPRIGHT',
                }) + '\n',
                encoding='utf-8',
            )
            from webcam_validation import _read_samples
            with self.assertRaisesRegex(ValueError, 'Unsafe frame path'):
                _read_samples(root)


if __name__ == '__main__':
    unittest.main()
