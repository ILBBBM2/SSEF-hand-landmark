import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def build_mobilenetv2(num_classes, freeze_features=True):
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes,
    )

    trainable_params = model.classifier.parameters()
    return model, trainable_params
