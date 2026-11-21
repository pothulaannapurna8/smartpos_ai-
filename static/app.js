'use strict';
const el = id => document.getElementById(id);
let lastSettings = '';
let lastCalibration = '';
let lastAlert = false;
let voiceUnavailableShown = false;
let audioContext;

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
  const category = data.issue;
  const statusText = {good: 'GOOD', bad: 'BAD', unknown: 'UNKNOWN'};
  el('posture').textContent = statusText[data.label] || 'UNKNOWN';
  el('posture').className = 'posture ' + (data.label || 'unknown');
  el('posture-category').textContent = category
    ? `${category} · ${data.mode} mode`
    : 'No reliable posture category';
  el('ergonomic-advice').textContent = data.correction ||
    'Move into view so the required landmarks can be evaluated.';
  return data.correction || '';
}

function renderDiagnostics(data) {
  const debug = data.debug || {};
  const visibility = debug.landmark_visibility || {};
  const confidence = debug.image_confidence;
  el('debug-mode').textContent =
    `${debug.active_mode || data.mode || '—'} · ${debug.view_mode || data.view_mode || '—'}`;
  el('debug-rule').textContent = debug.rule_prediction || 'UNKNOWN';
  el('debug-rule-reason').textContent = debug.rule_reason || '—';
  el('debug-image').textContent = debug.image_prediction || 'UNKNOWN';
  el('debug-confidence').textContent = confidence == null
    ? '—'
    : Number(confidence).toFixed(3);
  el('debug-image-reason').textContent = debug.image_reason || '—';
  const vertical = data.metrics?.head_vertical_displacement;
  const verticalUnit = data.metrics?.head_offset_unit || 'normalized units';
  el('debug-head-vertical').textContent = vertical == null
    ? '—'
    : `${Number(vertical).toFixed(2)} ${verticalUnit}`;
  el('debug-status').textContent =
    debug.final_status || String(data.label || 'unknown').toUpperCase();
  el('debug-reason').textContent = debug.final_reason || '—';
  el('debug-visibility').textContent = Object.keys(visibility).length
    ? Object.entries(visibility).map(([name, value]) =>
      `${name}: ${value == null ? 'unavailable' : Number(value).toFixed(2)}`).join(' · ')
    : 'Unavailable';
}

function renderMeasurements(data) {
  const metrics = data.metrics || {};
  const baseline = data.baseline || {};
  const number = value => value == null || !Number.isFinite(Number(value))
    ? null
    : Number(value);
  const alignment = number(metrics.head_alignment_deg);
  const alignmentBaseline = number(baseline.head_alignment_deg);
  el('alignment').textContent = alignment == null || alignmentBaseline == null
    ? '—'
    : `Δ ${(alignment - alignmentBaseline >= 0 ? '+' : '')}${(alignment - alignmentBaseline).toFixed(1)}°`;

  const tilt = number(metrics.shoulder_tilt_deg);
  const signedTilt = number(metrics.shoulder_tilt_signed_deg);
  const tiltBaseline = number(baseline.shoulder_tilt_signed_deg);
  el('tilt').textContent = tilt == null
    ? '—'
    : `${tilt.toFixed(1)}°${signedTilt != null && tiltBaseline != null
      ? ` · Δ ${(signedTilt - tiltBaseline >= 0 ? '+' : '')}${(signedTilt - tiltBaseline).toFixed(1)}°`
      : ''}`;

  const offset = number(metrics.head_offset);
  const offsetBaseline = number(baseline.head_offset);
  el('head-offset').textContent = offset == null
    ? '—'
    : `${offset >= 0 ? '+' : ''}${offset.toFixed(2)}${offsetBaseline == null
      ? ''
      : ` · Δ ${(offset - offsetBaseline >= 0 ? '+' : '')}${(offset - offsetBaseline).toFixed(2)}`}`;
  el('head-offset-unit').textContent =
    `Normalized by ${metrics.head_offset_unit || 'shoulder width'}`;

  const torso = number(metrics.torso_lean_deg);
  const torsoBaseline = number(baseline.torso_lean_deg);
  el('torso-lean').textContent = torso == null
    ? '—'
    : `${torso >= 0 ? '+' : ''}${torso.toFixed(1)}°${torsoBaseline == null
      ? ''
      : ` · Δ ${(torso - torsoBaseline >= 0 ? '+' : '')}${(torso - torsoBaseline).toFixed(1)}°`}`;
}

function renderCalibration(data) {
  const calibration = data.calibration || {};
  el('cal-message').textContent = calibration.active
    ? 'Collecting reliable upright samples…'
    : calibration.result?.error ||
      (calibration.result
        ? 'Baseline ready — review the values before applying'
        : 'Calibration not started');
  el('cal-offer').hidden = !(calibration.result && !calibration.result.error);
  if (calibration.result && !calibration.result.error) {
    const values = calibration.result.baselines || {};
    const shown = [];
    if (values.head_offset != null) {
      shown.push(`head shift ${Number(values.head_offset).toFixed(2)}`);
    }
    if (values.head_alignment_deg != null) {
      shown.push(`image-plane alignment ${Number(values.head_alignment_deg).toFixed(1)}°`);
    }
    if (values.shoulder_tilt_signed_deg != null) {
      shown.push(`shoulder tilt ${Number(values.shoulder_tilt_signed_deg).toFixed(1)}°`);
    }
    if (values.torso_lean_deg != null) {
      shown.push(`torso lean ${Number(values.torso_lean_deg).toFixed(1)}°`);
    }
    el('cal-values').textContent =
      `${calibration.result.samples} samples · ${calibration.result.view_mode} view · ${shown.join(' · ')}`;
  }
  const key = JSON.stringify(calibration.result);
  if (key !== lastCalibration && calibration.result?.error) {
    toast(calibration.result.error);
  }
  lastCalibration = key;
}

async function refresh() {
  try {
    const response = await fetch('/api/status', {cache: 'no-store'});
    const data = await response.json();
    el('connection').textContent = data.camera_live
      ? 'Camera connected'
      : (data.camera_error || data.debug?.final_reason || 'Waiting for camera');
    const correction = renderPosture(data);
    renderDiagnostics(data);
    renderMeasurements(data);

    if (document.activeElement !== el('detection-mode')) {
      el('detection-mode').value = data.mode;
    }
    el('image-mode').disabled = !data.image_model_available;
    el('model-info').textContent = data.mode === 'image'
      ? 'Image-only test mode. This HOG classifier has not been validated on live webcam sessions; probabilities are uncalibrated.'
      : data.mode === 'rules'
        ? 'Rule-only mode. Geometry is measured in the selected image plane; frontal view cannot resolve forward/backward depth.'
        : data.image_model_available
          ? 'Hybrid mode uses calibrated geometry for the final status. Image AI is diagnostic only until independent live-session validation.'
          : `Hybrid geometry is available; image model could not load: ${data.image_model_error || 'model unavailable'}.`;

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

    el('score').textContent = `${Number(data.session.score).toFixed(1)}%`;
    el('elapsed').textContent = duration(data.session.elapsed);
    el('good').textContent = `${Math.round(data.session.good)}s`;
    el('bad').textContent = `${Math.round(data.session.bad)}s`;
    el('session-mode').textContent = `Session: ${data.session.mode}`;
    el('warning-count').textContent = data.session.warnings;
    el('history-list').replaceChildren();
    if (!data.history.length) {
      const empty = document.createElement('li');
      empty.className = 'helper';
      empty.textContent = 'No posture transitions yet.';
      el('history-list').append(empty);
    } else {
      for (const entry of data.history.slice(0, 8)) {
        const item = document.createElement('li');
        const timeLabel = new Date(entry.timestamp * 1000).toLocaleTimeString();
        item.textContent = `${timeLabel} · ${entry.label}${entry.issue ? ` — ${entry.issue}` : ''}`;
        el('history-list').append(item);
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
    renderCalibration(data);
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
