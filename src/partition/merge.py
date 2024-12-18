import os
import numpy as np

# Updated slice configuration for 3x3 grid
image_width = 3840
image_height = 2160
slice_row = 3
slice_col = 3
base_height = image_height // slice_row
base_width = image_width // slice_col
overlap = 400
iou_threshold = 0.3

def convert_coordinates(slice_index, x, y, w, h, base_width, base_height, overlap, image_width, image_height, slice_row, slice_col):
    row = slice_index // slice_col  # Corrected to use slice_col for row calculation
    col = slice_index % slice_col

    x_offset = col * base_width
    y_offset = row * base_height

    # Adjust overlap based on position in the grid
    slice_width = base_width + (overlap if col < slice_col - 1 else 0)
    slice_height = base_height + (overlap if row < slice_row - 1 else 0)

    abs_x_center = (x * slice_width + x_offset) / image_width
    abs_y_center = (y * slice_height + y_offset) / image_height
    abs_w = w * slice_width / image_width
    abs_h = h * slice_height / image_height

    return abs_x_center, abs_y_center, abs_w, abs_h

def load_yolo_results(file_path):
    results = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            score = float(parts[5])
            results.append((class_id, x_center, y_center, width, height, score))
    return results

def save_yolo_results(file_path, results):
    with open(file_path, 'w') as file:
        for result in results:
            class_id, x_center, y_center, width, height, score = result
            file.write(f"{class_id} {x_center} {y_center} {width} {height} {score}\n")

def compute_iou(box1, box2, eps=1e-6):
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

    nms_results = [(class_ids[i], (boxes[i][0] + boxes[i][2]) / 2, (boxes[i][1] + boxes[i][3]) / 2,
                    boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1], confidences[i]) for i in nms_indices]

    return nms_results

def merge_yolo_results_for_image(image_name, slice_dir, output_dir, base_width, base_height, overlap, image_width, image_height, slice_row, slice_col, iou_threshold):
    output_file = os.path.join(output_dir, f"{image_name}.txt")
    all_results = []
    total_slices = slice_row * slice_col

    for slice_index in range(total_slices):
        slice_file = os.path.join(slice_dir, f"{image_name}_{slice_index + 1}.txt")
        if not os.path.exists(slice_file):
            continue
        with open(slice_file, 'r') as infile:
            for line in infile:
                parts = line.strip().split()
                class_id = int(parts[0])
                x, y, w, h, score = map(float, parts[1:])
                abs_x, abs_y, abs_w, abs_h = convert_coordinates(
                    slice_index, x, y, w, h, base_width, base_height, overlap, image_width, image_height, slice_row, slice_col
                )
                all_results.append((class_id, abs_x, abs_y, abs_w, abs_h, score))
    # Apply Non-Maximum Suppression
    nms_results = apply_nms(all_results, iou_threshold)
    # Save the merged results
    save_yolo_results(output_file, nms_results)

def process_all_images(image_dir, slice_dir, output_dir, base_width, base_height, overlap, image_width, image_height, slice_row, slice_col, iou_threshold=0.5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_file in os.listdir(image_dir):
        if image_file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_name, _ = os.path.splitext(image_file)
            merge_yolo_results_for_image(
                image_name, slice_dir, output_dir, base_width, base_height, overlap, image_width, image_height, slice_row, slice_col, iou_threshold
            )

if __name__ == "__main__":
    image_dir = "data/PANDA/images/val"
    slice_dir = "runs/detect/3x3_v8l/labels"
    output_dir = slice_dir.rstrip('s') + "_merged_results"  # Adjusted to correctly replace 'labels' with 'label_merged_results'

    process_all_images(
        image_dir,
        slice_dir,
        output_dir,
        base_width,
        base_height,
        overlap,
        image_width,
        image_height,
        slice_row,
        slice_col,
        iou_threshold
    )
