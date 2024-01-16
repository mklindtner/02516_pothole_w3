import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def get_resnet_model(network=resnet50, weights=ResNet50_Weights.IMAGENET1K_V1):
    model = network(weights=weights)

    # Replace the final layer
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, 1)

    return model
