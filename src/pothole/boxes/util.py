import csv


def xyxy_to_xywh(bb):
    """Convert bounding box from [xmin, ymin, xmax, ymax] to [x, y, w, h]."""

    return bb[0], bb[1], bb[2]-bb[0], bb[3]-bb[1]


def xywh_to_xyxy(bb):
    """Convert bounding box from [x, y, w, h] to [xmin, ymin, xmax, ymax]."""

    return bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[3]


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


def save_proposals(file, proposals):
    writer = csv.writer(file)

    for xmlfile, proposals in proposals.items():
        for bb in proposals['background']:
            line = [xmlfile, 0]
            line.extend(bb)
            writer.writerow(line)

        for bb in proposals['pothole']:
            line = [xmlfile, 1]
            line.extend(bb)
            writer.writerow(line)


def load_proposals(file):
    proposals = {}

    reader = csv.reader(file)

    for row in reader:
        # Extract fields.
        xmlfile = row[0]
        bg_or_ph = 'pothole' if row[1] == 1 else 'background'
        bb = list(map(int, row[2:6]))

        if xmlfile not in proposals:
            proposals[xmlfile] = {
                'background': [],
                'pothole': [],
            }

        proposals[xmlfile][bg_or_ph].append(bb)

    return proposals
