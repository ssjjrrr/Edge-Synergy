import os
import glob
import numpy as np
import cv2
from config import *

def load_yolo_annotations(annotation_dir, image_dir):
    annotations = {}
    for file_path in glob.glob(os.path.join(annotation_dir, '*.txt')):
        image_id = os.path.basename(file_path).split('.')[0]
        image_file = os.path.join(image_dir, image_id + '.jpg')
        image = cv2.imread(image_file)
        height, width, _ = image.shape
        
        annotations[image_id] = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                class_id, x_center, y_center, w, h = map(float, line.strip().split())
                xmin = (x_center - w / 2) * width
                ymin = (y_center - h / 2) * height
                xmax = (x_center + w / 2) * width
                ymax = (y_center + h / 2) * height
                area = w * width * h * height
                annotations[image_id].append([xmin, ymin, xmax, ymax, 0, int(class_id), area])
    return annotations

def load_yolo_results(result_dir, image_dir):
    results = {}
    for file_path in glob.glob(os.path.join(result_dir, '*.txt')):
        image_id = os.path.basename(file_path).split('.')[0]
        image_file = os.path.join(image_dir, image_id + '.jpg')
        image = cv2.imread(image_file)
        height, width, _ = image.shape
        
        results[image_id] = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                class_id, x_center, y_center, w, h, confidence = map(float, line.strip().split())
                xmin = (x_center - w / 2) * width
                ymin = (y_center - h / 2) * height
                xmax = (x_center + w / 2) * width
                ymax = (y_center + h / 2) * height
                area = w * width * h * height
                results[image_id].append([xmin, ymin, xmax, ymax, confidence, int(class_id), area])
    return results

def ious(a, b):
    aleft, atop, aright, abottom = [a[i] for i in range(4)]
    bleft, btop, bright, bbottom = [b[i] for i in range(4)]
    
    cross_left = np.maximum(aleft, bleft)        # M x N
    cross_top = np.maximum(atop, btop)           # M x N
    cross_right = np.minimum(aright, bright)     # M x N
    cross_bottom = np.minimum(abottom, bbottom)  # M x N
    
    cross_area = (cross_right - cross_left + 1).clip(0) * (cross_bottom - cross_top + 1).clip(0)
    union_area = (aright - aleft + 1) * (abottom - atop + 1) + (bright - bleft + 1) * (bbottom - btop + 1) - cross_area
    return cross_area / union_area

def build_matched_table(classes_index, groundtruths, detections, size_category, maxDets=1000):
    matched_table = []
    sum_groundtruths = 0
    for image_id in groundtruths:
        select_detections = np.array(list(filter(lambda x: x[5] == classes_index and size_category(x[6]), detections[image_id])))    
        select_groundtruths = np.array(list(filter(lambda x: x[5] == classes_index and size_category(x[6]), groundtruths[image_id])))
        num_detections = len(select_detections)
        num_groundtruths = len(select_groundtruths)
        num_use_detections = min(num_detections, maxDets)
        sum_groundtruths += num_groundtruths

        if num_detections == 0:
            continue

        if len(select_groundtruths) == 0:
            for detection_index in range(num_use_detections):
                confidence = select_detections[detection_index, 4]
                matched_table.append([confidence, 0, -1, image_id])
            continue

        sgt = select_groundtruths.T.reshape(7, -1, 1)
        sdt = select_detections.T.reshape(7, 1, -1)
        groundtruth_detection_ious = ious(sgt, sdt)

        for detection_index in range(num_use_detections):
            confidence = select_detections[detection_index, 4]
            matched_groundtruth_index = groundtruth_detection_ious[:, detection_index].argmax()
            matched_iou = groundtruth_detection_ious[matched_groundtruth_index, detection_index]
            matched_table.append([confidence, matched_iou, matched_groundtruth_index, image_id])

    matched_table = sorted(matched_table, key=lambda x: x[0], reverse=True)
    return matched_table, sum_groundtruths

def compute_AP(matched_table, iou_threshold, sum_groundtruths):
    if sum_groundtruths == 0:
        return 0.0
    num_detections = len(matched_table)
    if num_detections == 0:
        return 0.0
    
    true_positive = np.zeros((num_detections,))
    groundtruth_seen_map = {item[3]: set() for item in matched_table}
    for index in range(num_detections):
        confidence, matched_iou, matched_groundtruth_index, image_id = matched_table[index]
        image_seen_map = groundtruth_seen_map[image_id]
        if matched_iou > iou_threshold and matched_groundtruth_index not in image_seen_map:
            true_positive[index] = 1
            image_seen_map.add(matched_groundtruth_index)
                
    TP_count = np.cumsum(true_positive)
    detection_count = np.arange(1, num_detections + 1)
    precision = TP_count / detection_count
    recall = TP_count / sum_groundtruths
    
    if len(recall) == 0 or len(precision) == 0:
        return 0.0
    
    mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
    mpre = np.concatenate(([0.], precision, [0.]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    
    AP = np.mean(np.interp(np.linspace(0, 1, 101), mrec, mpre))
    return AP

def compute_mAP(groundtruths, detections, classes, size_categories, maxDets=1000):
    APs = {category: [] for category in size_categories}
    for classes_index in range(len(classes)):
        for category, size_category in size_categories.items():
            matched_table, sum_groundtruths = build_matched_table(classes_index, groundtruths, detections, size_category, maxDets)
            AP50 = compute_AP(matched_table, 0.5, sum_groundtruths)
            AP = np.mean([compute_AP(matched_table, iou_threshold, sum_groundtruths) for iou_threshold in np.arange(0.5, 1.0, 0.05)])
            APs[category].append([AP, AP50])
    return APs

if __name__ == "__main__":
    classes = ["person", "fakeperson"]

    size_categories = {
        '0-8': lambda area: area < 8 * 8,
        '8-16': lambda area: 8 * 8 <= area < 16 * 16,
        '16-24': lambda area: 16 * 16 <= area < 24 * 24,
        '24-32': lambda area: 24 * 24 <= area < 32 * 32,
        '32-48': lambda area: 32 * 32 <= area < 48 * 48,
        '48-64': lambda area: 48 * 48 <= area < 64 * 64,
        '64-80': lambda area: 64 * 64 <= area < 80 * 80,
        '80-96': lambda area: 80 * 80 <= area < 96 * 96,
        '96-128': lambda area: 96 * 96 <= area < 128 * 128,
        '128-160': lambda area: 128 * 128 <= area < 160 * 160,
        '160-196': lambda area: 160 * 160 <= area < 196 * 196,
        '196-': lambda area: area >= 196 * 196
    }

    annotations = load_yolo_annotations(annotation_dir=os.path.join(label_dir, 'val'), image_dir=os.path.join(image_dir, 'val'))
    results = load_yolo_results(result_dir, image_dir=os.path.join(image_dir, 'val'))
    mAPs = compute_mAP(annotations, results, classes, size_categories)
    
    for category, aps in mAPs.items():
        print(f"mAP for pixel size category {category} objects:")
        for i, (ap, ap50) in enumerate(aps):
            if classes[i] == "person":
                print(f"Class {classes[i]} - mAP50:95 : {ap}")
