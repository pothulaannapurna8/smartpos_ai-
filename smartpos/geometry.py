"""Small, testable geometric helpers for a frontal-camera posture demo."""
import math


def angle_at(a, vertex, b):
    """Angle in degrees at vertex between two 2-D points."""
    u = (a[0] - vertex[0], a[1] - vertex[1])
    v = (b[0] - vertex[0], b[1] - vertex[1])
    nu, nv = math.hypot(*u), math.hypot(*v)
    if nu < 1e-8 or nv < 1e-8:
        return None
    cos_value = (u[0]*v[0] + u[1]*v[1]) / (nu*nv)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_value))))


def shoulder_tilt(left, right):
    """Absolute deviation from horizontal, 0..90 degrees."""
    dx, dy = right[0]-left[0], right[1]-left[1]
    if math.hypot(dx, dy) < 1e-8:
        return None
    return math.degrees(math.atan2(abs(dy), abs(dx)))


def midpoint(a, b):
    return ((a[0]+b[0])/2, (a[1]+b[1])/2)


def posture_measurements(landmarks, width, height, min_visibility=0.55):
    """Measure upper-body posture from head and shoulder landmarks only."""
    if width <= 0 or height <= 0:
        return None, None

    def visible_point(name):
        landmark = landmarks.get(name)
        if landmark is None:
            return None
        x, y, visibility = landmark
        if (not all(math.isfinite(value) for value in (x, y, visibility))
                or visibility < min_visibility):
            return None
        return (x * width, y * height)

    left_shoulder = visible_point('left_shoulder')
    right_shoulder = visible_point('right_shoulder')
    if left_shoulder is None or right_shoulder is None:
        return None, None

    ears = visible_point('left_ear'), visible_point('right_ear')
    if ears[0] is not None and ears[1] is not None:
        head = midpoint(*ears)
    else:
        head = visible_point('nose')
        if head is None:
            head = ears[0] if ears[0] is not None else ears[1]
    if head is None:
        return None, None

    shoulders = midpoint(left_shoulder, right_shoulder)
    vertical_down = (shoulders[0], shoulders[1] + height)
    neck_angle = angle_at(head, shoulders, vertical_down)
    tilt = shoulder_tilt(left_shoulder, right_shoulder)
    return neck_angle, tilt


def head_offset_ratio(landmarks, min_visibility=0.55):
    """Return horizontal head displacement from shoulder center, in shoulder widths."""
    def visible_point(name):
        landmark = landmarks.get(name)
        if landmark is None:
            return None
        x, y, visibility = landmark
        if (not all(math.isfinite(value) for value in (x, y, visibility))
                or visibility < min_visibility):
            return None
        return (x, y)

    left = visible_point('left_shoulder')
    right = visible_point('right_shoulder')
    if left is None or right is None:
        return None
    width = math.hypot(right[0] - left[0], right[1] - left[1])
    if width < 1e-8:
        return None

    left_ear = visible_point('left_ear')
    right_ear = visible_point('right_ear')
    if left_ear is not None and right_ear is not None:
        head_x = (left_ear[0] + right_ear[0]) / 2
    else:
        nose = visible_point('nose')
        if nose is None:
            return None
        head_x = nose[0]

    shoulder_center_x = (left[0] + right[0]) / 2
    return (head_x - shoulder_center_x) / width


def posture_label(neck, tilt, neck_min, tilt_max):
    if neck is None or tilt is None:
        return 'unknown'
    return 'bad' if neck < neck_min or tilt > tilt_max else 'good'


def score(good_seconds, bad_seconds):
    total = good_seconds + bad_seconds
    return round(100 * good_seconds / total, 1) if total > 0 else 0.0
