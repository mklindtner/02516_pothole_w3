def xyxy_to_xywh(bb):
    """Convert bounding box from [xmin, ymin, xmax, ymax] to [x, y, w, h]."""

    return bb[0], bb[1], bb[2]-bb[0], bb[3]-bb[1]


def box_iou(a, b):
    """Calculate the intersection over union metric for two bounding boxes.

    The boxes are expected to be specified as to vectors in xywh format, i.e.:

    [xmin, ymin, width, height]

    Arguments:
        a: The first bounding box.
        b: The second bounding box.
    """

    # Intersection corners.
    xmin = max(a[0], b[0])
    ymin = max(a[1], b[1])
    xmax = min(a[0]+a[2], b[0]+b[2])
    ymax = min(a[1]+a[3], b[1]+b[3])

    if xmin > xmax or ymin > ymax:
        intersection = 0
    else:
        intersection = (xmax-xmin) * (ymax-ymin)

    union = a[2]*a[3] + b[2]*b[3] - intersection

    return intersection / union


def filter_proposals(proposals, target, metric=box_iou, k1=0.3, k2=0.7):
    pos = []
    background = []

    for prop in proposals:
        scores = [metric(prop, t) for t in target]

        if max(scores) >= k2:
            pos.append(prop)
        elif max(scores) < k1:
            background.append(prop)

    return pos, background
