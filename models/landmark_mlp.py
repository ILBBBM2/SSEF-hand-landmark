import torch
import torch.nn as nn

from landmarks.normalize import FEATURE_DIM


def build_landmark_mlp(num_classes, input_dim=FEATURE_DIM):
    model = nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes),
    )
    return model, model.parameters()
