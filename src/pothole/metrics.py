import torch.nn.functional as F


def bce_loss(y_pred, y_target):
    return F.binary_cross_entropy(F.sigmoid(y_pred), y_target)
