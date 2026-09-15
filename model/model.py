import torch
import torch.nn as nn
from torchvision import models


NUM_CLASSES = 2


def create_model():
    """
    Create the SignalScope ResNet-18 architecture.

    Architecture:
    - ResNet-18
    - layer3 trainable during V2 training
    - layer4 trainable during V2 training
    - 2-class classifier
    """

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    return model


def load_model(model_path, device=None):
    """
    Load the trained SignalScope model.
    """

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model = create_model()

    state_dict = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(state_dict)

    model = model.to(device)

    model.eval()

    return model