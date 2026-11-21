"""SmartPos AI: local-only posture monitoring dashboard."""
import atexit
import logging
import math
import os
import time
from pathlib import Path

try:
    from flask import Flask, Response, jsonify, render_template, request
except ImportError as exc:
    raise SystemExit(
        'Flask is missing. From the project folder run: '
        '.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt'
    ) from exc

try:
    from image_posture_inference import ImagePosturePredictor
except ImportError as exc:
    ImagePosturePredictor = None
    IMAGE_IMPORT_ERROR = f'{type(exc).__name__}: {exc}'
else:
    IMAGE_IMPORT_ERROR = ''
from smartpos.camera import CameraWorker
from smartpos.state import MonitorState


def create_app(start_camera=True, image_model_path=None,
               image_predictor=None, camera_index=None):
    app = Flask(__name__)
    state = MonitorState()
    image_model_error = ''
    if image_predictor is None:
        if ImagePosturePredictor is None:
            image_model_error = IMAGE_IMPORT_ERROR
        else:
            try:
                path = image_model_path or Path(__file__).with_name(
                    'image_posture_model.joblib'
                )
                image_predictor = ImagePosturePredictor(path)
            except Exception as exc:
                image_model_error = f'{type(exc).__name__}: {exc}'
                app.logger.exception(
                    'Could not load binary image model; hybrid geometry remains available'
                )

    camera_config_error = ''
    if camera_index is None:
        try:
            camera_index = int(os.environ.get('SMARTPOS_CAMERA', '0'))
        except ValueError:
            camera_index = 0
            camera_config_error = (
                'SMARTPOS_CAMERA must be a non-negative integer camera index.'
            )
    else:
        try:
            camera_index = int(camera_index)
        except (TypeError, ValueError):
            camera_index = 0
            camera_config_error = (
                'Camera index must be a non-negative integer.'
            )
    if camera_index < 0:
        camera_config_error = (
            'Camera index must be a non-negative integer.'
        )
        camera_index = 0

    worker = CameraWorker(
        state,
        camera_index,
        image_predictor=image_predictor,
        image_model_error=image_model_error,
    )
    worker.error = camera_config_error
    app.extensions['monitor_state'] = state
    app.extensions['camera_worker'] = worker
    app.extensions['image_posture_predictor'] = image_predictor
    if start_camera and not camera_config_error:
        worker.start()
        atexit.register(worker.stop)

    @app.get('/')
    def home():
        return render_template('index.html')

    @app.get('/api/status')
    def status():
        return jsonify({
            **state.snapshot(),
            'camera_error': worker.error,
            'image_model_available': worker.image_model_available,
            'image_model_error': worker.image_model_error,
        })

    @app.post('/api/mode')
    def detection_mode():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({'error': 'Expected a JSON object'}), 400
        mode = payload.get('mode')
        if mode not in ('hybrid', 'rules', 'image'):
            return jsonify({
                'error': 'Mode must be hybrid, rules, or image'
            }), 400
        if mode == 'image' and not worker.image_model_available:
            return jsonify({
                'error': worker.image_model_error or 'Image model unavailable'
            }), 503
        state.set_mode(mode)
        return jsonify({
            **state.snapshot(),
            'image_model_available': worker.image_model_available,
            'image_model_error': worker.image_model_error,
        })

    @app.post('/api/session')
    def session():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({'error': 'Expected a JSON object'}), 400
        try:
            state.session_action(payload.get('action'))
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400
        return jsonify(state.snapshot())

    @app.post('/api/calibration')
    def calibration():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({'error': 'Expected a JSON object'}), 400
        try:
            state.calibration_action(payload.get('action'))
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400
        return jsonify(state.snapshot())

    @app.post('/api/settings')
    def settings():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({'error': 'Expected a JSON object'}), 400
        limits = {
            'head_offset_max': (0.05, 0.5),
            'alignment_max': (3, 45),
            'tilt_max': (1, 45),
            'torso_lean_max': (3, 45),
            'alert_delay': (1, 20),
        }
        try:
            updates = {}
            for key, value in payload.items():
                if key == 'view_mode':
                    if value not in ('frontal', 'side'):
                        raise ValueError('view_mode must be frontal or side')
                    updates[key] = value
                    continue
                if key not in limits:
                    raise ValueError(f'Unknown setting: {key}')
                number = float(value)
                low, high = limits[key]
                if not math.isfinite(number) or not low <= number <= high:
                    raise ValueError(f'{key} must be between {low} and {high}')
                updates[key] = number
            state.update_settings(updates)
        except (ValueError, TypeError) as exc:
            return jsonify({'error': str(exc)}), 400
        return jsonify(state.snapshot())

    @app.get('/video_feed')
    def video_feed():
        def frames():
            while True:
                jpeg = worker.frame()
                if jpeg:
                    yield (
                        b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                        + jpeg + b'\r\n'
                    )
                time.sleep(.065)
        return Response(
            frames(),
            mimetype='multipart/x-mixed-replace; boundary=frame',
        )

    return app


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    create_app().run(
        host='127.0.0.1', port=5000, debug=False, threaded=True
    )
