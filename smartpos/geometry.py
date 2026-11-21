"""Image-plane geometry for frontal and side-profile posture monitoring."""
import math


def angle_at(a, vertex, b):
    """Angle in degrees at vertex between two 2-D points."""
    u = (a[0] - vertex[0], a[1] - vertex[1])
    v = (b[0] - vertex[0], b[1] - vertex[1])
    nu, nv = math.hypot(*u), math.hypot(*v)
    if nu < 1e-8 or nv < 1e-8:
        return None
    cos_value = (u[0] * v[0] + u[1] * v[1]) / (nu * nv)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_value))))


def shoulder_tilt(left, right):
    """Absolute deviation of the shoulder line from horizontal, in degrees."""
    dx, dy = right[0] - left[0], right[1] - left[1]
    if math.hypot(dx, dy) < 1e-8:
        return None
    return math.degrees(math.atan2(abs(dy), abs(dx)))


def signed_shoulder_tilt(left, right):
    """Signed image-plane shoulder-line angle; positive means right is lower."""
    dx, dy = right[0] - left[0], right[1] - left[1]
    if math.hypot(dx, dy) < 1e-8:
        return None
    if dx < 0:
        dx, dy = -dx, -dy
    return math.degrees(math.atan2(dy, dx))


def midpoint(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def _visible_point(landmarks, name, width, height, min_visibility):
    landmark = landmarks.get(name)
    if landmark is None:
        return None
    try:
        x, y, visibility = landmark[:3]
        x, y, visibility = float(x), float(y), float(visibility)
    except (TypeError, ValueError, IndexError):
        return None
    if (not all(math.isfinite(value) for value in (x, y, visibility))
            or visibility < min_visibility):
        return None
    return (x * width, y * height, visibility)


def measure_posture(landmarks, width, height, view_mode='frontal',
                    min_visibility=0.55):
    """Return measured image-plane features; no 3-D depth inference is used."""
    result = {
        'valid': False,
        'view_mode': view_mode,
        'head_offset': None,
        'head_offset_unit': 'shoulder widths',
        'head_vertical_displacement': None,
        'head_alignment_deg': None,
        'shoulder_tilt_deg': None,
        'shoulder_tilt_signed_deg': None,
        'torso_lean_deg': None,
        'missing_reason': None,
    }
    if width <= 0 or height <= 0:
        result['missing_reason'] = 'Invalid camera frame dimensions.'
        return result
    if view_mode not in ('frontal', 'side'):
        result['missing_reason'] = 'Unknown camera view mode.'
        return result

    point = lambda name: _visible_point(
        landmarks, name, width, height, min_visibility
    )
    left_shoulder = point('left_shoulder')
    right_shoulder = point('right_shoulder')
    left_hip = point('left_hip')
    right_hip = point('right_hip')

    if view_mode == 'frontal':
        if left_shoulder is None or right_shoulder is None:
            result['missing_reason'] = (
                'Frontal view requires both visible shoulders.'
            )
            return result
        shoulder_x, shoulder_y = midpoint(left_shoulder, right_shoulder)
        head_left, head_right = point('left_ear'), point('right_ear')
        if head_left is not None and head_right is not None:
            head_x, head_y = midpoint(head_left, head_right)
        else:
            nose = point('nose')
            if nose is None:
                result['missing_reason'] = (
                    'Frontal view requires a visible nose or both ears.'
                )
                return result
            head_x, head_y = nose[:2]
        shoulder_width = math.hypot(
            right_shoulder[0] - left_shoulder[0],
            right_shoulder[1] - left_shoulder[1],
        )
        if shoulder_width < 8:
            result['missing_reason'] = (
                'Shoulder landmarks are too close for a frontal measurement.'
            )
            return result
        scale = shoulder_width
        torso_pairs = (
            (left_shoulder, left_hip),
            (right_shoulder, right_hip),
        )
        if left_shoulder is not None and right_shoulder is not None:
            tilt_signed = signed_shoulder_tilt(
                left_shoulder[:2], right_shoulder[:2]
            )
            result['shoulder_tilt_signed_deg'] = tilt_signed
            result['shoulder_tilt_deg'] = (
                abs(tilt_signed) if tilt_signed is not None else None
            )
        if left_hip is not None and right_hip is not None:
            hip_x, hip_y = midpoint(left_hip, right_hip)
            torso_pairs = ((midpoint(left_shoulder, right_shoulder),
                            midpoint(left_hip, right_hip)),)
        else:
            torso_pairs = tuple(
                (shoulder[:2], hip[:2])
                for shoulder, hip in torso_pairs
                if shoulder is not None and hip is not None
            )
    else:
        available_shoulders = [
            ('left', left_shoulder), ('right', right_shoulder)
        ]
        available_shoulders = [
            item for item in available_shoulders if item[1] is not None
        ]
        if not available_shoulders:
            result['missing_reason'] = (
                'Side view requires at least one visible shoulder.'
            )
            return result
        side, shoulder = max(available_shoulders, key=lambda item: item[1][2])
        nose = point('nose')
        ears = [point('left_ear'), point('right_ear')]
        visible_ears = [ear for ear in ears if ear is not None]
        head = nose or (max(visible_ears, key=lambda item: item[2])
                        if visible_ears else None)
        if head is None:
            result['missing_reason'] = (
                'Side view requires a visible nose or ear.'
            )
            return result
        shoulder_x, shoulder_y = shoulder[:2]
        head_x, head_y = head[:2]
        matching_hip = left_hip if side == 'left' else right_hip
        if matching_hip is not None:
            scale = math.hypot(
                shoulder_x - matching_hip[0],
                shoulder_y - matching_hip[1],
            )
            result['head_offset_unit'] = 'torso lengths'
            torso_pairs = ((shoulder[:2], matching_hip[:2]),)
        else:
            scale = abs(shoulder_y - head_y)
            result['head_offset_unit'] = 'head-to-shoulder lengths'
            torso_pairs = ()
        if scale < 8:
            result['missing_reason'] = (
                'Side-view landmarks are too close for a stable measurement.'
            )
            return result

    vertical = shoulder_y - head_y
    if vertical <= 3:
        result['missing_reason'] = (
            'Head and shoulder vertical separation is insufficient.'
        )
        return result

    horizontal = head_x - shoulder_x
    result['head_offset'] = horizontal / scale
    result['head_vertical_displacement'] = vertical / scale
    result['head_alignment_deg'] = math.degrees(
        math.atan2(horizontal, vertical)
    )
    if torso_pairs:
        shoulder_point, hip_point = torso_pairs[0]
        torso_vertical = hip_point[1] - shoulder_point[1]
        if torso_vertical > 3:
            result['torso_lean_deg'] = math.degrees(
                math.atan2(
                    shoulder_point[0] - hip_point[0],
                    torso_vertical,
                )
            )
    result['valid'] = True
    return result


def score(good_seconds, bad_seconds):
    total = good_seconds + bad_seconds
    return round(100 * good_seconds / total, 1) if total > 0 else 0.0
