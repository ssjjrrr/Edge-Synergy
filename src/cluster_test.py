import numpy as np
import cv2
from sklearn.cluster import MeanShift
import time


detections = []
with open('/home/edge/work/ultralytics/runs/detect/predict27/label_nms/IMG_01_28.txt', 'r') as f:
    for line in f:
        parts = line.strip().split(' ')
        if len(parts) == 6:
            cls, x, y, w, h, conf = parts
            detections.append({
                'class': int(cls),
                'x': float(x),
                'y': float(y),
                'w': float(w),
                'h': float(h),
                'confidence': float(conf)
            })

class0_detections = [det for det in detections if det['class'] == 0]

positions = []
for det in class0_detections:
    center_x = det['x'] + det['w'] / 2
    center_y = det['y'] + det['h'] / 2
    positions.append([center_x, center_y])
positions = np.array(positions)

if len(positions) > 0:
    ms = MeanShift(bandwidth=0.15, bin_seeding=True) 
    ms.fit(positions)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
else:
    labels = np.array([])
    cluster_centers = np.array([])

image = cv2.imread('/home/edge/work/datasets/PANDA_dataset/images/val/IMG_01_28.jpg')
if image is None:
    print("Failed to read image")
    exit()
height, width = image.shape[:2]
for det in detections:
    det['x'] *= width
    det['y'] *= height
    det['w'] *= width
    det['h'] *= height

num_clusters = len(np.unique(labels))
colors = []
np.random.seed(42)
for i in range(num_clusters):
    color = [int(c) for c in np.random.randint(0, 255, 3)]
    colors.append(color)

for idx, det in enumerate(class0_detections):
    center_x, center_y = det['x'], det['y']
    w, h = det['w'], det['h']
    x1 = int(center_x - w / 2)
    y1 = int(center_y - h / 2)
    x2 = int(center_x + w / 2)
    y2 = int(center_y + h / 2)
    if len(labels) > 0:
        cluster_label = labels[idx]
        color = colors[cluster_label]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f'Cluster {cluster_label}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    else:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, 'No Cluster', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imwrite('clustered_image.jpg', image)

