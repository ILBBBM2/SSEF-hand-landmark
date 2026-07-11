import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def build_mobilenetv3(num_classes, freeze_features=True):
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    trainable_params = model.classifier.parameters()
    return model, trainable_params
