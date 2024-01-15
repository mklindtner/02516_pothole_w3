from torchvision.models import resnet50, ResNet50_Weights


def get_resnet_model(model=resnet50, weights=ResNet50_Weights.IMAGENET1K_V1):
    return model(weights=weights)
