"""Smartpos AI: local-only posture monitoring dashboard."""
import atexit
import os
import time
from pathlib import Path
from flask import Flask, Response, jsonify, render_template, request
from smartpos.camera import CameraWorker
from smartpos.state import MonitorState


def create_app(start_camera=True, model_path=None, predictor=None):
    app = Flask(__name__)
    state = MonitorState()
    model_error = ''
    if predictor is None:
        try:
            from posture_inference import PosturePredictor
            path = model_path or Path(__file__).with_name('posture_model.joblib')
            predictor = PosturePredictor(path)
        except Exception as exc:
            model_error = f'{type(exc).__name__}: {exc}'
            app.logger.exception(
                'Could not load posture model; rule-based detection remains available'
            )
    image_predictor = None
    image_model_error = ''
    try:
        from image_posture_inference import ImagePosturePredictor
        image_predictor = ImagePosturePredictor()
    except Exception as exc:
        image_model_error = f'{type(exc).__name__}: {exc}'
    worker = CameraWorker(
        state,
        int(os.environ.get('SMARTPOS_CAMERA', '0')),
        predictor=predictor,
        model_error=model_error,
        image_predictor=image_predictor,
        image_model_error=image_model_error,
    )
    app.extensions['monitor_state'] = state
    app.extensions['camera_worker'] = worker
    app.extensions['posture_predictor'] = predictor
    if start_camera:
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
            'model_available': worker.model_available,
            'image_model_available': worker.image_model_available,
            'image_model_error': worker.image_model_error,
            'model_error': worker.model_error,
        })

    @app.post('/api/mode')
    def detection_mode():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({'error': 'Expected a JSON object'}), 400
        mode = payload.get('mode')
        if mode not in ('rules', 'ml', 'image'):
            return jsonify({'error': 'Mode must be rules, ml, or image'}), 400
        if mode == 'ml' and not worker.model_available:
            message = worker.model_error or 'Posture model is unavailable'
            return jsonify({'error': message}), 503
        if mode == 'image' and not worker.image_model_available:
            return jsonify({'error': worker.image_model_error or 'Image model unavailable'}), 503
        try:
            state.set_mode(mode)
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400
        return jsonify({
            **state.snapshot(),
            'model_available': worker.model_available,
            'image_model_available': worker.image_model_available,
            'image_model_error': worker.image_model_error,
            'model_error': worker.model_error,
        })

    @app.post('/api/session')
    def session():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({'error': 'Expected a JSON object'}), 400
        action = payload.get('action')
        try: state.session_action(action)
        except ValueError as exc: return jsonify({'error':str(exc)}), 400
        return jsonify(state.snapshot())

    @app.post('/api/calibration')
    def calibration():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({'error': 'Expected a JSON object'}), 400
        action = payload.get('action')
        try: state.calibration_action(action)
        except ValueError as exc: return jsonify({'error':str(exc)}), 400
        return jsonify(state.snapshot())

    @app.post('/api/settings')
    def settings():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({'error': 'Expected a JSON object'}), 400
        limits = {
            'neck_min': (90, 179),
            'tilt_max': (1, 45),
            'head_offset_max': (0.05, 0.5),
            'alert_delay': (1, 20),
        }
        try:
            updates = {}
            for key, value in payload.items():
                if key not in limits: raise ValueError('Unknown setting: '+key)
                number = float(value)
                if not limits[key][0] <= number <= limits[key][1]:
                    raise ValueError(f'{key} must be between {limits[key][0]} and {limits[key][1]}')
                updates[key] = number
            with state.lock: state.settings.update(updates)
        except (ValueError,TypeError) as exc:
            return jsonify({'error':str(exc)}), 400
        return jsonify(state.snapshot())

    @app.get('/video_feed')
    def video_feed():
        def frames():
            while True:
                jpeg = worker.frame()
                if jpeg:
                    yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n'
                time.sleep(.065)
        return Response(frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

    return app


if __name__ == '__main__':
    create_app().run(host='127.0.0.1',port=5000,debug=False,threaded=True)
