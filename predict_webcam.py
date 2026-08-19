import argparse
import json
import os
from collections import Counter, deque

import cv2
import torch

from landmarks.extract import HandLandmarkExtractor, draw_hand_landmarks
from models import build_model
from utils import (
    CHECKPOINT_DIR,
    DEVICE,
    checkpoint_path_for,
    get_model_state_dict,
    class_names_path_for,
)


def load_class_names(model_name):
    classes_path = class_names_path_for(model_name)
    if not os.path.exists(classes_path):
        classes_path = os.path.join(CHECKPOINT_DIR, "class_names.json")

    with open(classes_path) as f:
        return json.load(f)


def load_model(model_name, num_classes):
    model, _ = build_model(model_name, num_classes)
    model.load_state_dict(get_model_state_dict(checkpoint_path_for(model_name)))
    model = model.to(DEVICE)
    model.eval()
    return model


def predict_from_landmarks(model, features, class_names):
    tensor = torch.from_numpy(features).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, index = probabilities.max(1)

    label = class_names[index.item()]
    return label, confidence.item()


def run_webcam(model_name, camera_index=0, confidence_threshold=0.70):
    class_names = load_class_names(model_name)
    model = load_model(model_name, len(class_names))

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    print("Press SPACE to add a stable sign, BACKSPACE to delete, C to clear, Q to quit.")
    recent_predictions = deque(maxlen=8)
    translation = []
    stable_label = None

    with HandLandmarkExtractor(static_image_mode=False) as extractor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            features = extractor.extract_from_bgr(frame, timestamp_ms=int(cap.get(cv2.CAP_PROP_POS_MSEC)))
            display = frame.copy()

            if features is not None:
                label, confidence = predict_from_landmarks(model, features, class_names)
                if confidence >= confidence_threshold:
                    recent_predictions.append(label)
                if recent_predictions:
                    stable_label, votes = Counter(recent_predictions).most_common(1)[0]
                    if votes < max(3, len(recent_predictions) // 2):
                        stable_label = None
                draw_hand_landmarks(display, extractor.last_result)
                text = f"{label} ({confidence * 100:.1f}%)"
                cv2.putText(
                    display,
                    text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    display,
                    "No hand detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            stable_text = f"Stable: {stable_label or '-'}"
            cv2.putText(display, stable_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
            cv2.putText(
                display,
                "Translation: " + " ".join(translation),
                (20, display.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            # Display with fallback if OpenCV GUI is not available
            try:
                cv2.imshow("Sign Language (MediaPipe)", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" ") and stable_label:
                    translation.append(stable_label)
                elif key in (8, 127) and translation:
                    translation.pop()
                elif key == ord("c"):
                    translation.clear()
            except cv2.error:
                # GUI not available; periodically save annotated frames and print status
                if not hasattr(run_webcam, "_fallback_count"):
                    run_webcam._fallback_count = 0
                run_webcam._fallback_count += 1
                if run_webcam._fallback_count % 30 == 0:
                    out_path = os.path.join(os.getcwd(), f"annotated_frame_{run_webcam._fallback_count}.jpg")
                    try:
                        cv2.imwrite(out_path, display)
                        latest_label = locals().get("label", "-")
                        latest_conf = locals().get("confidence", 0.0)
                        print(f"GUI unavailable. Saved annotated frame to {out_path}. Latest: {latest_label} ({latest_conf*100:.1f}%)")
                    except Exception as e:
                        print("Failed to save annotated frame:", e)
                # allow quitting with Ctrl+C
                continue

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Realtime sign language prediction via webcam.")
    parser.add_argument("--model", default="landmark_mlp")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.70)
    args = parser.parse_args()

    run_webcam(args.model, camera_index=args.camera, confidence_threshold=args.threshold)


if __name__ == "__main__":
    main()
