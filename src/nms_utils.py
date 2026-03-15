"""Utilities for IoU and Non-Maximum Suppression (NMS)."""


def compute_iou(box1, box2, eps=1e-6):
    """
    Compute IoU between two bounding boxes.
    box1, box2: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = box1
    x1_b, y1_b, x2_b, y2_b = box2

    inter_x1 = max(x1, x1_b)
    inter_y1 = max(y1, y1_b)
    inter_x2 = min(x2, x2_b)
    inter_y2 = min(y2, y2_b)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_b - x1_b) * (y2_b - y1_b)

    iou = inter_area / (box1_area + box2_area - inter_area + eps)
    return iou


def apply_nms(results, iou_threshold=0.5):
    """
    Apply NMS to detection results.
    results: [(class_id, x_center, y_center, width, height, confidence), ...]
    """
    if len(results) == 0:
        return []

    boxes = []
    confidences = []
    class_ids = []

    for result in results:
        class_id, x_center, y_center, width, height, confidence = result
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        boxes.append([x1, y1, x2, y2])
        confidences.append(confidence)
        class_ids.append(class_id)

    indices = list(range(len(boxes)))
    indices.sort(key=lambda i: confidences[i], reverse=True)

    nms_indices = []
    while indices:
        current = indices.pop(0)
        nms_indices.append(current)
        indices = [i for i in indices if compute_iou(boxes[current], boxes[i]) < iou_threshold]

    nms_results = [
        (
            class_ids[i],
            (boxes[i][0] + boxes[i][2]) / 2,
            (boxes[i][1] + boxes[i][3]) / 2,
            boxes[i][2] - boxes[i][0],
            boxes[i][3] - boxes[i][1],
            confidences[i],
        )
        for i in nms_indices
    ]
    return nms_results
