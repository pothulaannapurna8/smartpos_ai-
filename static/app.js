'use strict';
const el = id => document.getElementById(id);
let lastSettings = '';
let lastCal = '';
let lastAlert = false;
let voiceUnavailableShown = false;
let audioContext;

const advice = {
  UPRIGHT: 'Your detected pose is near upright. Vary your posture and take regular movement breaks.',
  LEAN_FORWARD: 'Move your head back over your shoulders and sit comfortably upright.',
  LEAN_BACKWARD: 'Return gently to a supported, comfortable position; keep the screen near eye level.',
  LEAN_LEFT: 'Center your head and shoulders comfortably; adjust the screen or chair if needed.',
  LEAN_RIGHT: 'Center your head and shoulders comfortably; adjust the screen or chair if needed.'
};

function duration(seconds) {
  const n = Math.floor(seconds);
  return [Math.floor(n / 3600), Math.floor(n % 3600 / 60), n % 60]
    .map(value => String(value).padStart(2, '0')).join(':');
}

function toast(message) {
  const element = el('toast');
  element.textContent = message;
  element.style.display = 'block';
  setTimeout(() => {
    if (element.textContent === message) element.style.display = 'none';
  }, 3500);
}

async function post(path, body) {
  const response = await fetch(path, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  });
  const data = await response.json();
  if (!response.ok) throw Error(data.error || 'Request failed');
  return data;
}

function beep() {
  try {
    audioContext ||= new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gain = audioContext.createGain();
    oscillator.type = 'sine';
    oscillator.frequency.value = 680;
    gain.gain.value = .08;
    oscillator.connect(gain);
    gain.connect(audioContext.destination);
    oscillator.start();
    oscillator.stop(audioContext.currentTime + .17);
  } catch (error) {
    console.warn('Browser beep is unavailable.', error);
  }
}

function speakCorrection(text) {
  if (!('speechSynthesis' in window) || !('SpeechSynthesisUtterance' in window)) {
    if (!voiceUnavailableShown) {
      toast('Voice alerts are not supported by this browser.');
      voiceUnavailableShown = true;
    }
    return;
  }
  try {
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
  } catch (error) {
    console.error('Could not speak posture correction.', error);
    toast('Could not start the voice alert.');
  }
}

function renderPosture(data) {
  const prediction = data.ml_prediction || {};
  const category = data.issue;
  const statusText = {good: 'GOOD', bad: 'BAD', unknown: 'UNKNOWN'};
  el('posture').textContent = statusText[data.label] || 'UNKNOWN';
  el('posture').className = 'posture ' + (data.label || 'unknown');

  if (category) {
    const probability = ['ml', 'image'].includes(data.mode) && data.probability != null
      ? ` · uncalibrated model probability ${Number(data.probability).toFixed(3)}`
      : '';
    el('posture-category').textContent = `${category}${probability}`;
  } else {
    el('posture-category').textContent = 'No reliable posture category';
  }

  const correction = data.correction || (
    data.label === 'bad'
      ? advice[category] || 'Gently align your head over your shoulders and keep both shoulders comfortably level.'
      : data.label === 'good'
        ? advice.UPRIGHT
        : 'Move into view so the required landmarks can be evaluated.'
  );
  el('ergonomic-advice').textContent = correction;
  return correction;
}

function renderDiagnostics(data) {
  const debug = data.debug || {};
  const raw = debug.raw_ml_prediction
    ? `${debug.raw_ml_prediction}${debug.raw_posture_category ? ` · ${debug.raw_posture_category}` : ''}`
    : 'Not available';
  const visibility = debug.landmark_visibility || {};
  el('debug-mode').textContent = debug.active_mode || data.mode || '—';
  el('debug-raw').textContent = raw;
  el('debug-probability').textContent = debug.model_probability == null
    ? '—'
    : debug.model_probability.toPrecision(9);
  el('debug-status').textContent = debug.final_status || String(data.label || 'unknown').toUpperCase();
  el('debug-reason').textContent = debug.rejection_reason || '—';
  el('debug-visibility').textContent = Object.keys(visibility).length
    ? Object.entries(visibility).map(([name, value]) =>
      `${name}: ${value == null ? 'unavailable' : value.toPrecision(7)}`).join(' · ')
    : 'Unavailable';
}

async function refresh() {
  try {
    const response = await fetch('/api/status', {cache: 'no-store'});
    const data = await response.json();
    el('connection').textContent = data.camera_live
      ? 'Camera connected'
      : (data.camera_error || 'Waiting for camera');
    const correction = renderPosture(data);
    renderDiagnostics(data);

    if (document.activeElement !== el('detection-mode')) {
      el('detection-mode').value = data.mode;
    }
    el('ml-mode').disabled = !data.model_available;
    el('image-mode').disabled = !data.image_model_available;
    el('model-info').textContent = data.mode === 'image'
      ? 'Experimental HOG image model; a small grouped same-session evaluation is not independent-person or verified webcam accuracy.'
      : data.mode === 'ml'
        ? data.model_available
          ? 'Experimental pose model · labels are geometry-checked; unsupported directional guesses become UNKNOWN. Live webcam accuracy is not established.'
          : `Experimental pose model unavailable: ${data.model_error || 'model failed to load'}.`
        : 'Rule-based measurements are the default and remain available independently of the optional experimental classifiers.';

    const warning = Boolean(data.alert);
    el('alert').textContent = warning
      ? 'Posture warning'
      : data.label === 'bad'
        ? `Warning in ${Number(data.warning_remaining || 0).toFixed(1)}s`
        : 'No alert';
    el('alert').className = 'alert-chip' + (warning ? ' on' : '');
    if (warning && !lastAlert) {
      beep();
      if (el('voice-alert').checked) speakCorrection(correction);
    } else if (!warning && lastAlert && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    lastAlert = warning;

    el('neck').textContent = data.neck == null ? '—' : data.neck.toFixed(1) + '°';
    el('tilt').textContent = data.tilt == null ? '—' : data.tilt.toFixed(1) + '°';
    el('head-offset').textContent = data.head_offset == null
      ? '—'
      : `${data.head_offset >= 0 ? '+' : ''}${data.head_offset.toFixed(2)}`;
    el('score').textContent = data.session.score.toFixed(1) + '%';
    el('elapsed').textContent = duration(data.session.elapsed);
    el('good').textContent = Math.round(data.session.good) + 's';
    el('bad').textContent = Math.round(data.session.bad) + 's';
    el('session-mode').textContent = 'Session: ' + data.session.mode;
    el('warning-count').textContent = data.session.warnings;
    el('neck-target').textContent = data.settings.neck_min;
    el('tilt-target').textContent = data.settings.tilt_max;
    const historyList = el('history-list');
    historyList.replaceChildren();
    if (!data.history.length) {
      const empty = document.createElement('li');
      empty.className = 'helper';
      empty.textContent = 'No posture transitions yet.';
      historyList.append(empty);
    } else {
      for (const entry of data.history.slice(0, 8)) {
        const item = document.createElement('li');
        const timeLabel = new Date(entry.timestamp * 1000).toLocaleTimeString();
        item.textContent = `${timeLabel} · ${entry.label}${entry.issue ? ` — ${entry.issue}` : ''}`;
        historyList.append(item);
      }
    }

    const settings = JSON.stringify(data.settings);
    if (settings !== lastSettings) {
      for (const [key, value] of Object.entries(data.settings)) {
        const field = document.querySelector(`[name="${key}"]`);
        if (field && document.activeElement !== field) field.value = value;
      }
      lastSettings = settings;
    }

    const calibration = data.calibration;
    el('cal-message').textContent = calibration.active
      ? 'Collecting upright posture samples…'
      : calibration.result?.error ||
        (calibration.result ? 'Calibration ready — review suggested values' : 'Calibration not running');
    el('cal-offer').hidden = !(calibration.result && !calibration.result.error);
    if (calibration.result && !calibration.result.error) {
      el('cal-values').textContent =
        `Neck ≥ ${calibration.result.neck_min}°, shoulder ≤ ${calibration.result.tilt_max}°; head baseline ${Number(calibration.result.head_offset_baseline).toFixed(2)} shoulder widths (${calibration.result.samples} samples)`;
    }
    const calibrationKey = JSON.stringify(calibration.result);
    if (calibrationKey !== lastCal && calibration.result?.error) {
      toast(calibration.result.error);
    }
    lastCal = calibrationKey;
  } catch (error) {
    console.error('Could not refresh posture dashboard.', error);
    el('connection').textContent = 'Server unavailable';
  }
}

for (const button of document.querySelectorAll('[data-session]')) {
  button.addEventListener('click', async () => {
    try {
      await post('/api/session', {action: button.dataset.session});
      refresh();
    } catch (error) {
      toast(error.message);
    }
  });
}

for (const [id, action] of [
  ['calibrate', 'start'], ['cancel-cal', 'cancel'],
  ['accept-cal', 'accept'], ['reject-cal', 'reject']
]) {
  el(id).addEventListener('click', async () => {
    try {
      await post('/api/calibration', {action});
      refresh();
    } catch (error) {
      toast(error.message);
    }
  });
}

el('settings').addEventListener('submit', async event => {
  event.preventDefault();
  const values = Object.fromEntries(new FormData(event.target).entries());
  try {
    await post('/api/settings', values);
    toast('Settings saved');
    refresh();
  } catch (error) {
    toast(error.message);
  }
});

el('detection-mode').addEventListener('change', async event => {
  try {
    await post('/api/mode', {mode: event.target.value});
    refresh();
  } catch (error) {
    toast(error.message);
    refresh();
  }
});

el('voice-alert').addEventListener('change', event => {
  if (!event.target.checked && 'speechSynthesis' in window) {
    window.speechSynthesis.cancel();
  }
});

refresh();
setInterval(refresh, 500);
