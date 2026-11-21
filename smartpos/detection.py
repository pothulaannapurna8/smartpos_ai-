"""Rule-first hybrid policy for the independent posture evidence sources."""


def empty_prediction(reason):
    return {
        'status': 'UNKNOWN',
        'issue': None,
        'confidence': None,
        'raw_label': None,
        'reason': reason,
    }


def rule_prediction(measurements, settings, baseline, previous_status='UNKNOWN'):
    if not measurements.get('valid'):
        return empty_prediction(
            measurements.get('missing_reason')
            or 'Required pose landmarks are not reliable.'
        )
    if not baseline.get('calibrated', False):
        return empty_prediction(
            'No accepted upright baseline. Complete and apply calibration.'
        )

    view_mode = settings['view_mode']
    head_offset = measurements.get('head_offset')
    alignment = measurements.get('head_alignment_deg')
    head_offset_baseline = baseline.get('head_offset', 0.0)
    alignment_baseline = baseline.get('head_alignment_deg', 0.0)
    head_delta = (
        head_offset - head_offset_baseline
        if head_offset is not None and head_offset_baseline is not None
        else None
    )
    alignment_delta = (
        alignment - alignment_baseline
        if alignment is not None and alignment_baseline is not None
        else None
    )
    recovery = previous_status == 'BAD'
    multiplier = 0.8 if recovery else 1.0
    checks = []

    tilt = measurements.get('shoulder_tilt_signed_deg')
    tilt_baseline = baseline.get('shoulder_tilt_signed_deg')
    if tilt is not None and tilt_baseline is not None:
        tilt_delta = abs(tilt - tilt_baseline)
        tilt_limit = max(
            1.0,
            settings['tilt_max'] - 2.0 if recovery else settings['tilt_max'],
        )
        if tilt_delta > tilt_limit:
            checks.append((
                'UNEVEN_SHOULDERS',
                f'shoulder tilt changed {tilt_delta:.1f}° from calibrated baseline '
                f'(limit {tilt_limit:.1f}°)',
            ))

    torso = measurements.get('torso_lean_deg')
    torso_baseline = baseline.get('torso_lean_deg')
    if torso is not None and torso_baseline is not None:
        torso_delta = abs(torso - torso_baseline)
        torso_limit = settings['torso_lean_max'] * multiplier
        if torso_delta > torso_limit:
            checks.append((
                'TORSO_LEAN',
                f'torso lean changed {torso_delta:.1f}° from calibrated baseline '
                f'(limit {torso_limit:.1f}°)',
            ))

    if head_delta is not None:
        offset_limit = settings['head_offset_max'] * multiplier
        if abs(head_delta) > offset_limit:
            if view_mode == 'frontal':
                issue = 'LEAN_LEFT' if head_delta < 0 else 'LEAN_RIGHT'
            else:
                issue = 'HEAD_ALIGNMENT'
            checks.append((
                issue,
                f'head shifted {abs(head_delta):.2f} {measurements["head_offset_unit"]} '
                f'from calibrated baseline (limit {offset_limit:.2f})',
            ))

    if alignment_delta is not None:
        alignment_limit = settings['alignment_max'] * multiplier
        if abs(alignment_delta) > alignment_limit:
            issue = (
                ('LEAN_LEFT' if alignment_delta < 0 else 'LEAN_RIGHT')
                if view_mode == 'frontal'
                else 'HEAD_ALIGNMENT'
            )
            if not any(item[0] == issue for item in checks):
                checks.append((
                    issue,
                    f'head/shoulder image-plane alignment changed '
                    f'{abs(alignment_delta):.1f}° from calibrated baseline '
                    f'(limit {alignment_limit:.1f}°)',
                ))

    if checks:
        issue, reason = checks[0]
        if len(checks) > 1:
            reason += '; also: ' + '; '.join(item[1] for item in checks[1:])
        return {'status': 'BAD', 'issue': issue, 'confidence': None,
                'raw_label': None, 'reason': reason}
    return {
        'status': 'GOOD',
        'issue': 'UPRIGHT',
        'confidence': None,
        'raw_label': None,
        'reason': (
            'Visible image-plane measurements are within the configured '
            'calibrated limits.'
        ),
    }


def combine_predictions(mode, rule, image):
    """Keep the unvalidated image classifier advisory in hybrid mode."""
    if mode == 'rules':
        return {
            **rule,
            'reason': f'Rule-only mode: {rule["reason"]}',
        }
    if mode == 'image':
        status = image.get('status', 'UNKNOWN')
        return {
            'status': status,
            'issue': image.get('issue'),
            'confidence': image.get('confidence'),
            'raw_label': image.get('raw_label'),
            'reason': (
                f'Image-only mode: {image.get("rejection_reason") or "binary image result"}'
            ),
        }

    status = rule['status']
    if status == 'BAD':
        reason = f'Reliable geometry indicates BAD: {rule["reason"]}'
    elif status == 'GOOD':
        ai_status = image.get('status', 'UNKNOWN')
        ai_label = image.get('raw_label') or ai_status
        if ai_status == 'BAD':
            status = 'UNKNOWN'
            reason = (
                'Geometry supports GOOD but the image model reports BAD. '
                'This conflict is UNKNOWN because the image model is not '
                'validated on live webcam sessions.'
            )
        else:
            reason = (
                'Calibrated geometry supports GOOD. The binary image model is '
                f'advisory only until independently validated on live webcam sessions '
                f'(current diagnostic: {ai_label}).'
            )
    else:
        reason = (
            f'Geometry is UNKNOWN: {rule["reason"]} The binary image classifier '
            'is diagnostic only until independently validated on live webcam sessions.'
        )
    return {
        'status': status,
        'issue': rule.get('issue') if status != 'UNKNOWN' else None,
        'confidence': None,
        'raw_label': None,
        'reason': reason,
    }
