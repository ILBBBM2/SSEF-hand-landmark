from models.mobilenetv2 import build_mobilenetv2
from models.mobilenetv3 import build_mobilenetv3
from models.customcnn import build_custom_cnn
from models.landmark_mlp import build_landmark_mlp

MODEL_REGISTRY = {
    "landmark_mlp": build_landmark_mlp,
    "mobilenetv2": build_mobilenetv2,
    "mobilenetv3": build_mobilenetv3,
    "customcnn": build_custom_cnn,
}

LANDMARK_MODELS = {"landmark_mlp"}
IMAGE_MODELS = {"mobilenetv2", "mobilenetv3", "customcnn"}


def build_model(name, num_classes):
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY)
        raise ValueError(f"Unknown model '{name}'. Choose from: {available}")

    model, trainable_params = MODEL_REGISTRY[name](num_classes)
    return model, trainable_params
