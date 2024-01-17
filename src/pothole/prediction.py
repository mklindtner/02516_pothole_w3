import torch
import torch.nn.functional as F


from pothole.datasets import crop_bounding_box


def classify_proposals(
    image,
    proposals,
    model,
    threshold=0.5,
    prepare_image=crop_bounding_box,
):
    results = []

    model.eval()

    for prop in proposals:
        with torch.no_grad():
            confidence = F.sigmoid(model(prepare_image(image, prop))).item()

        if confidence > threshold:
            results.append((confidence, prop))

    return results
