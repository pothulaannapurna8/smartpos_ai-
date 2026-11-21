import unittest
import time
import numpy as np
from types import SimpleNamespace
from unittest.mock import Mock, patch
from smartpos.geometry import (
    angle_at, head_offset_ratio, posture_measurements, shoulder_tilt,
    posture_label, score,
)
from smartpos.state import MonitorState
from smartpos.camera import CameraWorker, draw_posture_overlay
from app import create_app
from posture_inference import (
    CLASS_LABELS, FEATURE_NAMES, FEATURE_NAMES_3D, LABELS, STATUS_BY_LABEL,
    PosturePredictor, features_from_mediapipe,
)
from image_posture_inference import ImagePosturePredictor
from train_image_posture import image_features

class GeometryTests(unittest.TestCase):
    def test_straight_angle(self): self.assertAlmostEqual(angle_at((0,-1),(0,0),(0,1)),180)
    def test_tilt(self): self.assertAlmostEqual(shoulder_tilt((0,0),(2,0)),0)
    def test_missing(self): self.assertEqual(posture_label(None,4,150,10),'unknown')
    def test_normalized_head_offset_and_missing_landmarks(self):
        centered = {
            'nose': (0.5, 0.2, 0.9),
            'left_shoulder': (0.3, 0.5, 0.9),
            'right_shoulder': (0.7, 0.5, 0.9),
        }
        self.assertAlmostEqual(head_offset_ratio(centered), 0.0)
        centered['nose'] = (0.4, 0.2, 0.9)
        self.assertAlmostEqual(head_offset_ratio(centered), -0.25)
        del centered['nose']
        self.assertIsNone(head_offset_ratio(centered))
    def test_posture_classification(self):
        self.assertEqual(posture_label(160,5,150,10),'good')
        self.assertEqual(posture_label(140,5,150,10),'bad')
        self.assertEqual(posture_label(160,12,150,10),'bad')
    def test_partial_body_measurements_do_not_require_hips(self):
        neck, tilt = posture_measurements({
            'nose': (0.5, 0.2, 0.9),
            'left_shoulder': (0.3, 0.5, 0.9),
            'right_shoulder': (0.7, 0.5, 0.9),
        }, 640, 480)
        self.assertAlmostEqual(neck, 180)
        self.assertAlmostEqual(tilt, 0)
        self.assertEqual(posture_label(neck, tilt, 150, 10), 'good')

    def test_head_lean_changes_alignment_measurement(self):
        neck, tilt = posture_measurements({
            'nose': (0.65, 0.2, 0.9),
            'left_shoulder': (0.3, 0.5, 0.9),
            'right_shoulder': (0.7, 0.5, 0.9),
        }, 640, 480)
        self.assertLess(neck, 150)
        self.assertEqual(posture_label(neck, tilt, 150, 10), 'bad')

    def test_missing_head_or_shoulder_keeps_measurement_unknown(self):
        missing_head = {
            'left_shoulder': (0.3, 0.5, 0.9),
            'right_shoulder': (0.7, 0.5, 0.9),
        }
        missing_shoulder = {
            'nose': (0.5, 0.2, 0.9),
            'left_shoulder': (0.3, 0.5, 0.9),
        }
        self.assertEqual(posture_measurements(missing_head, 640, 480), (None, None))
        self.assertEqual(posture_measurements(missing_shoulder, 640, 480), (None, None))
        self.assertEqual(posture_label(None, None, 150, 10), 'unknown')

    def test_low_visibility_ear_uses_visible_nose(self):
        neck, tilt = posture_measurements({
            'left_ear': (0.35, 0.2, 0.2),
            'right_ear': (0.65, 0.2, 0.2),
            'nose': (0.5, 0.2, 0.9),
            'left_shoulder': (0.3, 0.5, 0.9),
            'right_shoulder': (0.7, 0.5, 0.9),
        }, 640, 480)
        self.assertAlmostEqual(neck, 180)
        self.assertAlmostEqual(tilt, 0)
    def test_score(self):
        self.assertEqual(score(0,0),0)
        self.assertEqual(score(3,1),75)
    def test_calibration_accept_reject(self):
        s=MonitorState()
        s.calibration_result={'neck_min':155,'tilt_max':7,'samples':20}
        s.calibration_action('reject')
        self.assertEqual(s.settings['neck_min'],150)
        s.calibration_result={'neck_min':155,'tilt_max':7,'samples':20}
        s.calibration_result['head_offset_baseline'] = 0.12
        s.calibration_action('accept')
        self.assertEqual(s.settings['neck_min'],155)
        self.assertAlmostEqual(s.head_offset_baseline, 0.12)
    def test_session_reset(self):
        s=MonitorState()
        s.session_action('start')
        s.session_action('pause')
        s.session_action('reset')
        self.assertEqual(s.snapshot()['session']['mode'],'idle')
        s.session_action('start')
        s.session_action('stop')
        self.assertEqual(s.snapshot()['session']['mode'],'stopped')

    def test_rules_distinguish_lateral_offset_without_labeling_upright_forward(self):
        state = MonitorState()
        for _ in range(state._smoothing_frames):
            state.update_measurement(170, 2, head_offset=0.0)
        self.assertEqual(state.snapshot()['label'], 'good')
        self.assertNotEqual(state.snapshot()['issue'], 'LEAN_FORWARD')

        state = MonitorState()
        for _ in range(state._smoothing_frames):
            state.update_measurement(170, 2, head_offset=-0.3)
        self.assertEqual(state.snapshot()['label'], 'bad')
        self.assertEqual(state.snapshot()['issue'], 'LEAN_LEFT')
        self.assertIn('center', state.snapshot()['correction'])

    def test_rule_hysteresis_stabilizes_near_threshold_offsets(self):
        state = MonitorState()
        for _ in range(state._smoothing_frames):
            state.update_measurement(170, 2, head_offset=-0.3)
        self.assertEqual(state.snapshot()['label'], 'bad')
        for _ in range(state._smoothing_frames):
            state.update_measurement(170, 2, head_offset=-0.15)
        self.assertEqual(state.snapshot()['label'], 'bad')
        for _ in range(state._smoothing_frames):
            state.update_measurement(170, 2, head_offset=-0.10)
        self.assertEqual(state.snapshot()['label'], 'good')

    def test_warning_episode_is_counted_once(self):
        state = MonitorState()
        state.label = 'bad'
        state.bad_since = 100.0
        state.last_frame_at = 104.0
        state.session_mode = 'running'
        with patch('smartpos.state.time.monotonic', return_value=104.0):
            first = state.snapshot()
            second = state.snapshot()
        self.assertTrue(first['alert'])
        self.assertTrue(second['alert'])
        self.assertEqual(first['session']['warnings'], 1)
        self.assertEqual(second['session']['warnings'], 1)

    def test_calibration_expires_without_new_frames(self):
        s=MonitorState()
        s.calibration={'ends_at':time.monotonic()-1,'samples':[]}
        result=s.snapshot()['calibration']
        self.assertFalse(result['active'])
        self.assertIn('error',result['result'])

    def test_session_time_counts_valid_posture_and_pauses(self):
        s=MonitorState()
        with patch('smartpos.state.time.monotonic',
                   side_effect=[
                       100,100,100,100,100.1,100.5,100.6,100.7,
                       100.8,100.9,101.0,101.1,101.1,
                   ]):
            s._smoothing_frames = 3
            for _ in range(3):
                s.update_measurement(160,0)
            s.session_action('start')
            for _ in range(3):
                s.update_measurement(140,12)
            for _ in range(3):
                s.update_measurement(None,None)
            s.session_action('pause')
            session=s.snapshot()['session']
        self.assertEqual(session['mode'],'paused')
        self.assertEqual(session['elapsed'],1.0)
        self.assertEqual(session['good'],0.6)
        self.assertEqual(session['bad'],0.3)
        self.assertEqual(session['score'],66.7)

    def test_bad_posture_alert_uses_configured_delay(self):
        s=MonitorState()
        now = [100.0]
        with patch('smartpos.state.time.monotonic', side_effect=lambda: now[0]):
            for _ in range(s._smoothing_frames):
                s.update_measurement(140,0)
            for moment in (100.8, 101.6, 102.4, 102.9):
                now[0] = moment
                s.update_measurement(140,0)
            now[0] = 102.9
            self.assertFalse(s.snapshot()['alert'])
            now[0] = 103.1
            s.update_measurement(140,0)
            self.assertTrue(s.snapshot()['alert'])

    def test_ml_warning_delay_restarts_after_good_or_unknown(self):
        state = MonitorState()
        now = [100.0]
        with patch('smartpos.state.time.monotonic', side_effect=lambda: now[0]):
            state.set_mode('image')
            bad = {'status': 'BAD', 'issue': 'IMAGE_POSTURE'}
            good = {'status': 'GOOD', 'issue': 'UPRIGHT'}
            unknown = {'status': 'UNKNOWN', 'issue': None}
            for _ in range(state._smoothing_frames):
                state.update_measurement(None, None, bad)
            for moment in (100.8, 101.6, 102.4, 102.9):
                now[0] = moment
                state.update_measurement(None, None, bad)
            now[0] = 102.9
            self.assertFalse(state.snapshot()['alert'])
            now[0] = 103.2
            state.update_measurement(None, None, bad)
            self.assertTrue(state.snapshot()['alert'])
            for _ in range(state._smoothing_frames):
                state.update_measurement(None, None, good)
            self.assertIsNone(state.bad_since)
            for _ in range(state._smoothing_frames):
                state.update_measurement(None, None, bad)
            now[0] = 104.0
            state.update_measurement(None, None, bad)
            self.assertFalse(state.snapshot()['alert'])
            for _ in range(state._smoothing_frames):
                state.update_measurement(None, None, unknown)
            self.assertIsNone(state.bad_since)
            for _ in range(state._smoothing_frames):
                state.update_measurement(None, None, bad)
            for moment in (104.7, 105.4, 106.1, 106.9):
                now[0] = moment
                state.update_measurement(None, None, bad)
            now[0] = 106.9
            self.assertFalse(state.snapshot()['alert'])
            now[0] = 107.2
            state.update_measurement(None, None, bad)
            self.assertTrue(state.snapshot()['alert'])

    def test_ml_forward_label_is_withheld_when_geometry_is_upright(self):
        state = MonitorState()
        state.update_measurement(160, 2)
        state.set_mode('ml')
        prediction = {
            'status': 'BAD', 'issue': 'LEAN_FORWARD', 'confidence': 0.8,
            'raw_label': 'TLF', 'raw_category': 'LEAN_FORWARD',
        }
        for _ in range(state._smoothing_frames):
            state.update_measurement(160, 2, prediction, head_offset=0.0)
        snapshot = state.snapshot()
        self.assertEqual(snapshot['label'], 'unknown')
        self.assertEqual(snapshot['rule_label'], 'good')
        self.assertEqual(snapshot['neck'], 160)
        self.assertEqual(snapshot['tilt'], 2)
        self.assertEqual(snapshot['ml_prediction']['raw_label'], 'TLF')
        self.assertIsNone(snapshot['issue'])
        self.assertIn('not supported', snapshot['debug']['rejection_reason'])

    def test_ml_forward_label_becomes_generic_alignment_when_supported(self):
        state = MonitorState()
        state.set_mode('ml')
        prediction = {
            'status': 'BAD', 'issue': 'LEAN_FORWARD', 'confidence': 0.8,
            'raw_label': 'TLF', 'raw_category': 'LEAN_FORWARD',
        }
        for _ in range(state._smoothing_frames):
            state.update_measurement(140, 2, prediction, head_offset=0.0)
        snapshot = state.snapshot()
        self.assertEqual(snapshot['label'], 'bad')
        self.assertEqual(snapshot['issue'], 'NECK_ALIGNMENT')
        self.assertIn('cannot reliably determine', snapshot['debug']['rejection_reason'])
        self.assertIn('align your head', snapshot['correction'])

    def test_ml_missing_prediction_remains_unknown(self):
        state = MonitorState()
        state.set_mode('ml')
        state.update_measurement(None, None, {
            'status': 'UNKNOWN', 'issue': None, 'confidence': None,
            'rejection_reason': 'nose visibility 0.12; required >= 0.45',
            'landmark_visibility': {'nose': 0.12},
        })
        snapshot = state.snapshot()
        self.assertEqual(snapshot['label'], 'unknown')
        self.assertEqual(snapshot['neck'], None)
        self.assertEqual(snapshot['tilt'], None)
        self.assertIn('0.12', snapshot['debug']['rejection_reason'])
        self.assertEqual(snapshot['debug']['landmark_visibility']['nose'], 0.12)

    def test_ml_category_mapping_and_status_mismatch(self):
        self.assertEqual(STATUS_BY_LABEL, {
            'TUP': 'GOOD',
            'TLF': 'BAD',
            'TLB': 'BAD',
            'TLL': 'BAD',
            'TLR': 'BAD',
        })
        state = MonitorState()
        state.set_mode('ml')
        measurements = {
            'TUP': (160, 2, 0.0),
            'TLF': (140, 2, 0.0),
            'TLB': (140, 2, 0.0),
            'TLL': (160, 2, -0.3),
            'TLR': (160, 2, 0.3),
        }
        for label, category in LABELS.items():
            prediction = {
                'status': STATUS_BY_LABEL[label],
                'issue': category,
                'raw_label': label,
                'raw_category': category,
                'confidence': 0.9,
            }
            neck, tilt, head_offset = measurements[label]
            for _ in range(state._smoothing_frames):
                state.update_measurement(
                    neck, tilt, prediction, head_offset=head_offset
                )
            snapshot = state.snapshot()
            self.assertEqual(snapshot['label'], STATUS_BY_LABEL[label].lower())
            expected_issue = (
                'NECK_ALIGNMENT'
                if label in ('TLF', 'TLB')
                else category
            )
            self.assertEqual(snapshot['issue'], expected_issue)
        mismatch = {
            'status': 'GOOD', 'issue': 'LEAN_FORWARD',
        }
        for _ in range(state._smoothing_frames):
            state.update_measurement(None, None, mismatch)
        self.assertEqual(state.snapshot()['label'], 'unknown')
        self.assertIn('do not match', state.snapshot()['debug']['rejection_reason'])

    def test_decision_smoothing_prevents_single_frame_status_flicker(self):
        state = MonitorState()
        state.set_mode('image')
        bad = {'status': 'BAD', 'issue': 'IMAGE_POSTURE'}
        for _ in range(state._smoothing_frames - 1):
            state.update_measurement(None, None, bad)
        self.assertEqual(state.snapshot()['label'], 'unknown')
        state.update_measurement(None, None, bad)
        self.assertEqual(state.snapshot()['label'], 'bad')
        self.assertEqual(state.snapshot()['issue'], 'IMAGE_POSTURE')

    def test_rule_overlay_category_and_correction_are_geometry_derived(self):
        state = MonitorState()
        for _ in range(state._smoothing_frames):
            state.update_measurement(160, 12)
        snapshot = state.snapshot()
        self.assertEqual(snapshot['label'], 'bad')
        self.assertEqual(snapshot['issue'], 'UNEVEN_SHOULDERS')
        self.assertEqual(
            snapshot['correction'],
            'Relax your shoulders and keep them level.',
        )

class CameraWorkerTests(unittest.TestCase):
    def test_successful_ml_prediction_is_not_overwritten(self):
        prediction={
            'status':'BAD','issue':'LEAN_FORWARD','confidence':0.82,
            'raw_label':'TLF','raw_category':'LEAN_FORWARD',
        }
        predictor=Mock()
        predictor.predict.return_value=prediction
        state=MonitorState()
        state.set_mode('ml')
        worker=CameraWorker(state,predictor=predictor)
        result=worker._predict_ml([],{'nose':0.9})
        self.assertIs(result,prediction)
        self.assertTrue(worker.model_available)
        for _ in range(state._smoothing_frames):
            state.update_measurement(140,2,result,head_offset=0.0)
        self.assertEqual(state.snapshot()['label'],'bad')

    def test_ml_inference_failure_is_reported_as_unknown(self):
        predictor=Mock()
        predictor.predict.side_effect=ValueError('bad model input')
        worker=CameraWorker(MonitorState(),predictor=predictor)
        result=worker._predict_ml([],{'nose':0.2})
        self.assertEqual(result['status'],'UNKNOWN')
        self.assertIn('bad model input',result['rejection_reason'])
        self.assertEqual(result['landmark_visibility']['nose'],0.2)
        self.assertFalse(worker.model_available)

    def test_image_inference_failure_is_reported_without_crashing(self):
        predictor = Mock()
        predictor.predict.side_effect = ValueError('invalid image')
        worker = CameraWorker(MonitorState(), image_predictor=predictor)
        result = worker._predict_image(object(), None)
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertIn('invalid image', result['rejection_reason'])
        self.assertFalse(worker.image_model_available)

    def test_image_worker_passes_unmodified_camera_frame_to_predictor(self):
        frame = np.zeros((12, 16, 3), dtype='uint8')
        landmarks = [SimpleNamespace(visibility=0.9)]
        predictor = Mock()
        predictor.predict.return_value = {'status': 'GOOD'}
        worker = CameraWorker(MonitorState(), image_predictor=predictor)
        worker._predict_image(frame, landmarks)
        predictor.predict.assert_called_once_with(frame, landmarks)

    def test_unavailable_camera_is_released_and_reported(self):
        state=MonitorState()
        worker=CameraWorker(state)
        capture=Mock()
        capture.isOpened.return_value=False
        with patch('smartpos.camera.cv2.VideoCapture',return_value=capture):
            worker._run()
        capture.release.assert_called_once()
        self.assertIn('Cannot open webcam',worker.error)

class ApiTests(unittest.TestCase):
    def setUp(self): self.client=create_app(start_camera=False, predictor=Mock()).test_client()
    def test_home(self): self.assertEqual(self.client.get('/').status_code,200)
    def test_status(self):
        status = self.client.get('/api/status').json
        self.assertIn('session', status)
        self.assertIn('history', status)
        self.assertIn('warnings', status['session'])
    def test_bad_setting(self): self.assertEqual(self.client.post('/api/settings',json={'neck_min':999}).status_code,400)
    def test_head_offset_setting(self):
        response = self.client.post(
            '/api/settings', json={'head_offset_max': 0.22}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['settings']['head_offset_max'], 0.22)
    def test_non_object_payload(self):
        response=self.client.post('/api/settings',json=[])
        self.assertEqual(response.status_code,400)
        self.assertEqual(response.json['error'],'Expected a JSON object')
    def test_session(self): self.assertEqual(self.client.post('/api/session',json={'action':'start'}).status_code,200)
    def test_calibration(self): self.assertEqual(self.client.post('/api/calibration',json={'action':'start'}).status_code,200)
    def test_ml_mode_can_be_enabled(self):
        response=self.client.post('/api/mode',json={'mode':'ml'})
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json['mode'],'ml')
    def test_detection_mode_can_switch_to_image_and_back(self):
        response = self.client.post('/api/mode', json={'mode': 'image'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['mode'], 'image')
        response = self.client.post('/api/mode', json={'mode': 'rules'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['mode'], 'rules')
    def test_invalid_mode_is_rejected(self):
        response=self.client.post('/api/mode',json={'mode':'automatic'})
        self.assertEqual(response.status_code,400)
    def test_ml_mode_rejects_unavailable_model(self):
        client=create_app(
            start_camera=False,
            model_path='missing-posture-model.joblib',
        ).test_client()
        response=client.post('/api/mode',json={'mode':'ml'})
        self.assertEqual(response.status_code,503)
        self.assertEqual(client.get('/api/status').json['mode'],'rules')
    def test_model_is_loaded_once_per_app(self):
        with patch('posture_inference.PosturePredictor') as predictor:
            create_app(start_camera=False)
        predictor.assert_called_once()
    def test_api_and_dashboard_expose_corroborated_generic_forward_feedback(self):
        app=create_app(start_camera=False, predictor=Mock())
        state=app.extensions['monitor_state']
        state.set_mode('ml')
        prediction = {
            'status':'BAD',
            'issue':'LEAN_FORWARD',
            'raw_label':'TLF',
            'raw_category':'LEAN_FORWARD',
            'confidence':0.9,
            'landmark_visibility':{
                'nose':0.9,'left_ear':0.9,'right_ear':0.9,
                'left_shoulder':0.9,'right_shoulder':0.9,
            },
        }
        for _ in range(state._smoothing_frames):
            state.update_measurement(140,2,prediction)
        client=app.test_client()
        status=client.get('/api/status').json
        self.assertEqual(status['label'],'bad')
        self.assertEqual(status['issue'],'NECK_ALIGNMENT')
        self.assertEqual(
            status['correction'],
            'Gently align your head above your shoulders and sit comfortably upright.',
        )
        self.assertEqual(status['debug']['raw_ml_prediction'],'TLF')
        self.assertEqual(status['debug']['final_status'],'BAD')
        self.assertEqual(status['ml_prediction']['raw_category'],'LEAN_FORWARD')
        self.assertEqual(status['ml_prediction']['issue'],'NECK_ALIGNMENT')
        self.assertEqual(status['debug']['landmark_visibility']['nose'],0.9)
        html_response=client.get('/')
        script_response=client.get('/static/app.js')
        html=html_response.get_data()
        script=script_response.get_data()
        html_response.close()
        script_response.close()
        self.assertIn(b'id="debug-raw"',html)
        self.assertIn(b'data.correction',script)

class ModelTests(unittest.TestCase):
    def test_retrained_model_loads_and_missing_landmarks_are_unknown(self):
        predictor=PosturePredictor()
        self.assertEqual(predictor.pipeline.classes_.tolist(),[0,1,2,3,4])
        self.assertEqual(predictor.bundle['labels'], list(CLASS_LABELS))
        self.assertIn(
            predictor.bundle['features'],
            [list(FEATURE_NAMES), list(FEATURE_NAMES_3D)],
        )
        self.assertEqual(
            predictor.pipeline.n_features_in_, len(predictor.bundle['features'])
        )
        missing = predictor.predict([])
        self.assertEqual(missing['status'], 'UNKNOWN')
        self.assertIn('visibility unavailable', missing['rejection_reason'])
        self.assertIsNone(missing['raw_label'])

    def test_image_model_loads_and_exposes_binary_class_mapping(self):
        predictor = ImagePosturePredictor()
        self.assertEqual(predictor.pipeline.classes_.tolist(), [0, 1])
        self.assertEqual(
            predictor.pipeline.n_features_in_,
            len(image_features(np.zeros((64, 64, 3), dtype='uint8'))),
        )

    def test_image_probability_maps_classes_with_visible_landmarks(self):
        predictor = ImagePosturePredictor.__new__(ImagePosturePredictor)
        predictor.pipeline = Mock()
        predictor.pipeline.classes_ = [0, 1]
        predictor.pipeline.predict_proba.return_value = [[0.2, 0.8]]
        frame = object()
        landmarks = [SimpleNamespace(visibility=0.9) for _ in range(13)]
        with patch('image_posture_inference.image_features', return_value=[0]):
            result = predictor.predict(frame, landmarks=landmarks)
        self.assertEqual(result['status'], 'GOOD')
        self.assertEqual(result['raw_label'], 'GOOD')
        self.assertEqual(result['issue'], 'UPRIGHT')
        self.assertEqual(result['confidence'], 0.8)
        self.assertEqual(result['landmark_visibility']['nose'], 0.9)

        predictor.pipeline.predict_proba.return_value = [[0.8, 0.2]]
        with patch('image_posture_inference.image_features', return_value=[0]):
            result = predictor.predict(frame, landmarks=landmarks)
        self.assertEqual(result['status'], 'BAD')
        self.assertEqual(result['raw_label'], 'BAD')
        self.assertEqual(result['issue'], 'IMAGE_POSTURE')

    def test_image_model_rejects_empty_scene_as_unknown(self):
        predictor = ImagePosturePredictor()
        result = predictor.predict(np.zeros((480, 640, 3), dtype='uint8'))
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertIsNone(result['confidence'])
        self.assertIn('visible head', result['rejection_reason'])

    def test_forward_label_is_bad_and_low_probability_is_unknown(self):
        predictor=PosturePredictor()
        with patch(
            'posture_inference.features_from_mediapipe',
            return_value=[0]*len(predictor.bundle['features']),
        ):
            with patch.object(
                predictor.pipeline, 'predict_proba',
                return_value=[[0.05,0.8,0.05,0.05,0.05]],
            ):
                forward=predictor.predict(None)
            with patch.object(
                predictor.pipeline, 'predict_proba',
                return_value=[[0.1,0.4,0.1,0.1,0.3]],
            ):
                uncertain=predictor.predict(None)
        self.assertEqual(forward['raw_label'],'TLF')
        self.assertEqual(forward['issue'],'LEAN_FORWARD')
        self.assertEqual(forward['status'],'BAD')
        self.assertEqual(uncertain['raw_category'],'LEAN_FORWARD')
        self.assertEqual(uncertain['status'],'UNKNOWN')
        self.assertIn('below threshold',uncertain['rejection_reason'])

    def test_unreliable_model_landmark_returns_unknown_with_visibility(self):
        predictor=PosturePredictor()
        landmarks=[SimpleNamespace(visibility=0.9) for _ in range(13)]
        landmarks[0].visibility=0.2
        result=predictor.predict(landmarks)
        self.assertEqual(result['status'],'UNKNOWN')
        self.assertEqual(result['landmark_visibility']['nose'],0.2)
        self.assertIn('nose visibility 0.20',result['rejection_reason'])

    def test_three_dimensional_features_use_depth_and_shared_order(self):
        landmarks = [
            SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=0.9)
            for _ in range(13)
        ]
        landmarks[0] = SimpleNamespace(x=0.5, y=0.2, z=-0.3, visibility=0.9)
        landmarks[7] = SimpleNamespace(x=0.4, y=0.2, z=-0.2, visibility=0.9)
        landmarks[8] = SimpleNamespace(x=0.6, y=0.2, z=-0.2, visibility=0.9)
        landmarks[11] = SimpleNamespace(x=0.3, y=0.5, z=0.1, visibility=0.9)
        landmarks[12] = SimpleNamespace(x=0.7, y=0.5, z=0.1, visibility=0.9)
        values = features_from_mediapipe(landmarks, dimensions=3)
        self.assertEqual(len(values), len(FEATURE_NAMES_3D))
        self.assertAlmostEqual(values[2], -1.0)

class OverlayTests(unittest.TestCase):
    def test_bad_overlay_shows_status_category_real_angles_and_correction(self):
        frame = __import__('numpy').zeros((540, 960, 3), dtype='uint8')
        status = {
            'label': 'bad',
            'issue': 'LEAN_FORWARD',
            'neck': 142.5,
            'tilt': 7.2,
            'correction': 'Move your head back over your shoulders.',
            'warning_remaining': 3.0,
        }
        with patch('smartpos.camera.cv2.putText') as put_text:
            draw_posture_overlay(frame, status)
        drawn_text = [call.args[1] for call in put_text.call_args_list]
        rendered = ' '.join(drawn_text)
        self.assertIn('BAD POSTURE', rendered)
        self.assertIn('Issue: LEAN_FORWARD', rendered)
        self.assertIn('Neck angle: 142.5 deg', rendered)
        self.assertIn('Shoulder tilt: 7.2 deg', rendered)
        self.assertIn('Warning in 3.0s', rendered)
        self.assertIn('Move your head back over your shoulders.', rendered)

    def test_good_and_unknown_overlays_use_distinct_neutral_states(self):
        numpy = __import__('numpy')
        with patch('smartpos.camera.cv2.putText') as put_text:
            draw_posture_overlay(
                numpy.zeros((540, 960, 3), dtype='uint8'),
                {'label': 'good', 'issue': 'UPRIGHT'},
            )
            draw_posture_overlay(
                numpy.zeros((540, 960, 3), dtype='uint8'),
                {'label': 'unknown'},
            )
        rendered = [call.args[1] for call in put_text.call_args_list]
        self.assertIn('GOOD POSTURE', rendered)
        self.assertIn('Posture unavailable', rendered)

if __name__=='__main__': unittest.main()
