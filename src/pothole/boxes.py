def xyxy_to_xywh(bb):
    """Convert bounding box from [xmin, ymin, xmax, ymax] to [x, y, w, h]."""

    return bb[0], bb[1], bb[2]-bb[0], bb[3]-bb[1]
