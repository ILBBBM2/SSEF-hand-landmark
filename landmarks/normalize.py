import numpy as np

NUM_LANDMARKS = 21
LANDMARK_DIM = 3
HAND_FEATURE_DIM = NUM_LANDMARKS * LANDMARK_DIM
NUM_HAND_SLOTS = 2
PRESENCE_FEATURE_DIM = NUM_HAND_SLOTS
RELATIVE_WRIST_DIM = LANDMARK_DIM
FEATURE_DIM = HAND_FEATURE_DIM * NUM_HAND_SLOTS + PRESENCE_FEATURE_DIM + RELATIVE_WRIST_DIM


def landmarks_to_vector(landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32).flatten()


def normalize_landmarks(landmarks_flat):
    pts = np.asarray(landmarks_flat, dtype=np.float32).reshape(NUM_LANDMARKS, LANDMARK_DIM).copy()
    pts -= pts[0]

    scale = np.linalg.norm(pts[9])
    if scale < 1e-6:
        scale = 1.0
    pts /= scale

    return pts.flatten().astype(np.float32)


def hand_scale(landmarks_flat):
    """Return a stable palm scale for comparing the positions of two hands."""
    pts = np.asarray(landmarks_flat, dtype=np.float32).reshape(NUM_LANDMARKS, LANDMARK_DIM)
    return max(float(np.linalg.norm(pts[9] - pts[0])), 1e-6)


def two_hand_feature(left_landmarks=None, right_landmarks=None):
    """Encode zero, one, or two raw MediaPipe hands into one fixed-size vector.

    The first two sections are individually wrist-normalized hand shapes.  The
    final values indicate which slots are present and, when both are present,
    the right-wrist position relative to the left wrist in palm-size units.
    """
    left_raw = landmarks_to_vector(left_landmarks) if left_landmarks is not None else None
    right_raw = landmarks_to_vector(right_landmarks) if right_landmarks is not None else None
    left_present = left_raw is not None
    right_present = right_raw is not None

    left = normalize_landmarks(left_raw) if left_present else np.zeros(HAND_FEATURE_DIM, dtype=np.float32)
    right = normalize_landmarks(right_raw) if right_present else np.zeros(HAND_FEATURE_DIM, dtype=np.float32)
    relative_wrist = np.zeros(RELATIVE_WRIST_DIM, dtype=np.float32)
    if left_present and right_present:
        left_pts = left_raw.reshape(NUM_LANDMARKS, LANDMARK_DIM)
        right_pts = right_raw.reshape(NUM_LANDMARKS, LANDMARK_DIM)
        scale = (hand_scale(left_raw) + hand_scale(right_raw)) / 2.0
        relative_wrist = (right_pts[0] - left_pts[0]) / scale

    return np.concatenate(
        [
            left,
            right,
            np.asarray([left_present, right_present], dtype=np.float32),
            relative_wrist.astype(np.float32),
        ]
    ).astype(np.float32)


def augment_landmarks(landmarks_flat, noise_std=0.02, rotate_deg=15.0):
    features = np.asarray(landmarks_flat, dtype=np.float32).copy()
    if features.size == HAND_FEATURE_DIM:
        return _augment_hand(features, noise_std, rotate_deg)
    if features.size != FEATURE_DIM:
        raise ValueError(f"Expected {HAND_FEATURE_DIM} or {FEATURE_DIM} landmark features, got {features.size}.")

    for start, present_index in ((0, HAND_FEATURE_DIM * 2), (HAND_FEATURE_DIM, HAND_FEATURE_DIM * 2 + 1)):
        if features[present_index] >= 0.5:
            features[start : start + HAND_FEATURE_DIM] = _augment_hand(
                features[start : start + HAND_FEATURE_DIM], noise_std, rotate_deg
            )

    # Rotate the wrist-to-wrist displacement by the same amount as the hands.
    angle = np.random.uniform(-rotate_deg, rotate_deg) * np.pi / 180.0
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    features[-RELATIVE_WRIST_DIM:-1] = features[-RELATIVE_WRIST_DIM:-1] @ rotation.T
    return features


def _augment_hand(landmarks_flat, noise_std, rotate_deg):
    pts = landmarks_flat.reshape(NUM_LANDMARKS, LANDMARK_DIM).copy()

    angle = np.random.uniform(-rotate_deg, rotate_deg) * np.pi / 180.0
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    pts[:, :2] = pts[:, :2] @ rotation.T

    pts += np.random.normal(0.0, noise_std, pts.shape).astype(np.float32)
    return pts.flatten()
