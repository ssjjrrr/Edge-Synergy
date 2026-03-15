import cv2
VIS_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_bbox_cv(image, bbox, color=(0, 255, 0), label: str = ""):
    """Draw one bounding box with an optional label."""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(image, label, (x1, max(y1-5,0)),
                    VIS_FONT, 0.5, color, 1, cv2.LINE_AA)
