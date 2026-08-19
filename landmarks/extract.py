from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from landmarks.normalize import landmarks_to_vector, normalize_landmarks


def draw_hand_landmarks(frame, result):
    """Draw a HandLandmarkerResult in MediaPipe's official visual style.

    This is the same Tasks drawing utility and hand topology used by Google's
    Hand Landmarker sample. In addition to the 21-point skeleton, it labels
    each detected hand with MediaPipe's left/right handedness classification.
    """
    if result is None or not result.hand_landmarks:
        return frame

    # MediaPipe 0.10.35 moved drawing helpers under Tasks rather than exposing
    # the older ``mp.solutions`` module. Import it here so image-only workflows
    # still give a useful initialization error when MediaPipe is unavailable.
    from mediapipe.tasks.python.vision import drawing_utils
    from mediapipe.tasks.python.vision import hand_landmarker

    landmark_style = drawing_utils.DrawingSpec(
        color=(0, 255, 0), thickness=2, circle_radius=2
    )
    connection_style = drawing_utils.DrawingSpec(color=(0, 165, 255), thickness=2)
    height, width = frame.shape[:2]

    for index, landmarks in enumerate(result.hand_landmarks):
        drawing_utils.draw_landmarks(
            frame,
            landmarks,
            hand_landmarker.HandLandmarksConnections.HAND_CONNECTIONS,
            landmark_drawing_spec=landmark_style,
            connection_drawing_spec=connection_style,
        )

        if index < len(result.handedness) and result.handedness[index]:
            handedness = result.handedness[index][0].category_name
            text_x = int(min(landmark.x for landmark in landmarks) * width)
            text_y = max(25, int(min(landmark.y for landmark in landmarks) * height) - 10)
            cv2.putText(
                frame,
                handedness,
                (text_x, text_y),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (88, 205, 54),
                1,
                cv2.LINE_AA,
            )
    return frame


class HandLandmarkExtractor:
    """Extract normalized landmarks with MediaPipe's current Tasks API."""

    def __init__(
        self,
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self._landmarker = None
        self._init_error = None
        self.last_result = None
        self.last_landmarks = None
        self.last_handedness = None
        self.last_world_landmarks = None
        self._last_timestamp_ms = -1

        try:
            import mediapipe as mp
            model_path = Path(__file__).with_name("hand_landmarker.task")
            if not model_path.is_file():
                raise FileNotFoundError(f"Hand Landmarker model not found: {model_path}")

            self._running_mode = (
                mp.tasks.vision.RunningMode.IMAGE
                if static_image_mode
                else mp.tasks.vision.RunningMode.VIDEO
            )
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=self._running_mode,
                num_hands=max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._mp = mp
            self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        except Exception as exc:  # pragma: no cover - environment dependent
            self._init_error = exc

    def extract_from_rgb(self, rgb_image, timestamp_ms=None):
        if self._landmarker is None:
            raise RuntimeError(
                "MediaPipe Hand Landmarker could not be initialized: "
                f"{self._init_error}"
            )

        if rgb_image.dtype != np.uint8:
            rgb_image = rgb_image.astype(np.uint8)
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise ValueError("Expected an RGB image with shape (height, width, 3).")

        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_image)
        if self._running_mode == self._mp.tasks.vision.RunningMode.VIDEO:
            if timestamp_ms is None:
                timestamp_ms = int(time.monotonic() * 1000)
            timestamp_ms = max(int(timestamp_ms), self._last_timestamp_ms + 1)
            self._last_timestamp_ms = timestamp_ms
            result = self._landmarker.detect_for_video(image, timestamp_ms)
        else:
            result = self._landmarker.detect(image)
        self.last_result = result
        if not result.hand_landmarks:
            self.last_landmarks = None
            self.last_handedness = None
            self.last_world_landmarks = None
            return None

        self.last_landmarks = result.hand_landmarks[0]
        self.last_handedness = result.handedness[0] if result.handedness else None
        self.last_world_landmarks = (
            result.hand_world_landmarks[0] if result.hand_world_landmarks else None
        )
        raw = landmarks_to_vector(self.last_landmarks)
        return normalize_landmarks(raw)

    def extract_from_path(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return self.extract_from_rgb(np.array(image))

    def extract_from_bgr(self, bgr_image, timestamp_ms=None):
        return self.extract_from_rgb(
            cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB), timestamp_ms=timestamp_ms
        )

    def close(self):
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
