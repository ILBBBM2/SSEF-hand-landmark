import numpy as np

NUM_LANDMARKS = 21
LANDMARK_DIM = 3
FEATURE_DIM = NUM_LANDMARKS * LANDMARK_DIM


def landmarks_to_vector(landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32).flatten()


def normalize_landmarks(landmarks_flat):
    pts = landmarks_flat.reshape(NUM_LANDMARKS, LANDMARK_DIM).copy()
    pts -= pts[0]

    scale = np.linalg.norm(pts[9])
    if scale < 1e-6:
        scale = 1.0
    pts /= scale

    return pts.flatten().astype(np.float32)


def augment_landmarks(landmarks_flat, noise_std=0.02, rotate_deg=15.0):
    pts = landmarks_flat.reshape(NUM_LANDMARKS, LANDMARK_DIM).copy()

    angle = np.random.uniform(-rotate_deg, rotate_deg) * np.pi / 180.0
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    pts[:, :2] = pts[:, :2] @ rotation.T

    pts += np.random.normal(0.0, noise_std, pts.shape).astype(np.float32)
    return pts.flatten()
