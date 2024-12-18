import os
import time
import cv2
import numpy as np
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor

partition_plan = {
    'partitions': [{'coords': (0, 0, 960.0, 540.0)}, {'coords': (960.0, 0, 1920.0, 540.0)}, {'coords': (0, 540.0, 960.0, 1080.0)}, {'coords': (960.0, 540.0, 1920.0, 1080.0)}, {'coords': (1920.0, 0, 2880.0, 540.0)}, {'coords': (2880.0, 0, 3840, 540.0)}, {'coords': (1920.0, 540.0, 2880.0, 1080.0)}, {'coords': (2880.0, 540.0, 3840, 1080.0)}, {'coords': (0, 1080.0, 960.0, 1620.0)}, {'coords': (960.0, 1080.0, 1920.0, 1620.0)}, {'coords': (0, 1620.0, 960.0, 2160)}, {'coords': (960.0, 1620.0, 1920.0, 2160)}, {'coords': (1920.0, 1080.0, 2880.0, 1620.0)}, {'coords': (2880.0, 1080.0, 3840, 1620.0)}, {'coords': (1920.0, 1620.0, 2880.0, 2160)}, {'coords': (2880.0, 1620.0, 3840, 2160)}], 'network_assignments': ['YOLOv8n', 'YOLOv8n', 'YOLOv8l', 'YOLOv8l', 'YOLOv8n', 'YOLOv8n', 'YOLOv8l', 'YOLOv8l', 'YOLOv8l', 'YOLOv8l', 'YOLOv8l', 'YOLOv8l', 'YOLOv8l', 'YOLOv8l', 'YOLOv8l', 'YOLOv8l'], 'eap': 49.905199999999994, 'latency': 1996.8000000000002}

image_width = 3840
image_height = 2160
overlap_pixels = 200  # 重叠像素数
iou_threshold = 0.3    # NMS的IOU阈值
CONF = 0.25

model_mapping = {
    'YOLOv8n': '/home/edge/work/Edge-Synergy/checkpoints/yolov8n_640_ep300.pt',
    'YOLOv8s': '/home/edge/work/Edge-Synergy/checkpoints/yolov8s_768_ep300.pt',
    'YOLOv8m': '/home/edge/work/Edge-Synergy/checkpoints/yolov8m_896_ep300.pt',
    'YOLOv8l': '/home/edge/work/Edge-Synergy/checkpoints/yolov8l_1024_ep300.pt',
    'YOLOv8x': '/home/edge/work/Edge-Synergy/checkpoints/yolov8x_1280_ep300.pt'
}

loaded_models = {}
for model_name in set(partition_plan['network_assignments']):
    model_path = model_mapping.get(model_name)
    if model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件 {model_path} 未找到。请确保模型文件存在。")
        print(f"加载模型: {model_name} from {model_path}")
        loaded_models[model_name] = YOLO(model_path)
    else:
        raise ValueError(f"未找到模型路径: {model_name}")

def adjust_partition_coords(partition, overlap, image_width, image_height):
    """
    调整每个分区的坐标，向右和向下各增加overlap像素，确保不超出图像边界。
    """
    x1, y1, x2, y2 = partition['coords']
    x2_new = min(x2 + overlap, image_width)
    y2_new = min(y2 + overlap, image_height)
    partition['coords'] = (x1, y1, x2_new, y2_new)

def convert_coordinates(coords, x, y, w, h, image_width, image_height):
    """
    将切片中的相对坐标转换为全图中的绝对坐标，并进行归一化。
    coords: (x1, y1, x2, y2) 切片在全图中的坐标
    x, y, w, h: 切片内的相对坐标 (0到1)
    返回归一化后的全图坐标 (x_center, y_center, width, height)
    """
    x1, y1, x2, y2 = coords
    slice_width = x2 - x1
    slice_height = y2 - y1

    abs_x_center = (x * slice_width + x1) / image_width
    abs_y_center = (y * slice_height + y1) / image_height
    abs_w = w * slice_width / image_width
    abs_h = h * slice_height / image_height

    return abs_x_center, abs_y_center, abs_w, abs_h

def compute_iou(box1, box2, eps=1e-6):
    """
    计算两个边界框的交并比（IoU）。
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
    对检测结果应用非极大值抑制（NMS）。
    results: [(class_id, x_center, y_center, width, height, confidence), ...]
    返回经过NMS处理后的结果列表。
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
    # 按置信度从高到低排序
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
            confidences[i]
        )
        for i in nms_indices
    ]

    return nms_results

def save_yolo_results(file_path, results):
    """
    将检测结果保存为YOLO格式的文本文件。
    """
    with open(file_path, 'w') as file:
        for result in results:
            class_id, x_center, y_center, width, height, score = result
            file.write(f"{class_id} {x_center} {y_center} {width} {height} {score}\n")

def load_yolo_results(file_path):
    """
    从YOLO格式的文本文件中加载检测结果。
    """
    results = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 6:
                continue  # 跳过格式不正确的行
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            score = float(parts[5])
            results.append((class_id, x_center, y_center, width, height, score))
    return results

def process_partition(image, partition, model):
    """
    处理单个分区，运行YOLO模型，返回检测结果。
    """
    coords = partition['coords']
    x1, y1, x2, y2 = map(int, coords)
    slice_image = image[y1:y2, x1:x2]

    # 运行YOLO模型
    results = model(slice_image, conf=CONF)

    slice_results = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            score = float(box.conf[0])
            x, y, w, h = box.xywh[0].tolist()
            # 转换为全图坐标
            abs_x, abs_y, abs_w, abs_h = convert_coordinates(
                coords, x / slice_image.shape[1], y / slice_image.shape[0],
                w / slice_image.shape[1], h / slice_image.shape[0],
                image_width, image_height
            )
            slice_results.append((cls, abs_x, abs_y, abs_w, abs_h, score))

    return slice_results

def merge_yolo_results(all_results, final_output_file, iou_threshold=0.3):
    """
    合并所有分区的检测结果，应用NMS，并保存最终结果。
    """
    # 应用NMS
    nms_results = apply_nms(all_results, iou_threshold)

    # 保存最终合并的结果
    save_yolo_results(final_output_file, nms_results)

def draw_detections(image_path, detection_file, output_image_path):
    """
    将检测结果绘制到原始图像上，并保存可视化结果。
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return

    with open(detection_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 6:
                continue  # 跳过格式不正确的行
            class_id = int(parts[0])
            x_center = float(parts[1]) * image.shape[1]
            y_center = float(parts[2]) * image.shape[0]
            width = float(parts[3]) * image.shape[1]
            height = float(parts[4]) * image.shape[0]
            score = float(parts[5])

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 标注类别和置信度
            cv2.putText(image, f"{class_id} {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite(output_image_path, image)

def main():
    image_dir = "data/PANDA/images/scene/2"  # 输入图像目录
    slice_dir = "runs/detect/coarse_detection/labels"  # 原始切片结果目录
    output_dir = "runs/detect/merged_results"  # 合并结果输出目录
    visualization_dir = os.path.join(output_dir, "visualizations")  # 可视化结果目录

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(visualization_dir):
        os.makedirs(visualization_dir)

    # 调整每个分区的坐标，增加重叠
    for partition in partition_plan['partitions']:
        adjust_partition_coords(partition, overlap_pixels, image_width, image_height)

    # 处理每张图像
    for image_file in os.listdir(image_dir):
        start_time = time.time()
        if image_file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(image_dir, image_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图像: {image_path}")
                continue

            image_name, _ = os.path.splitext(image_file)
            print(f"处理图像: {image_name}")

            all_results = []

            # 按照划分计划逐个处理分区
            for idx, (partition, model_name) in enumerate(zip(partition_plan['partitions'], partition_plan['network_assignments'])):
                model = loaded_models.get(model_name)
                if model is None:
                    print(f"模型 {model_name} 未加载，跳过分区 {idx + 1}")
                    continue

                print(f"  处理分区 {idx + 1}/{len(partition_plan['partitions'])} 使用模型 {model_name}")
                slice_results = process_partition(image, partition, model)
                all_results.extend(slice_results)

            # 合并所有分区的检测结果
            final_output_file = os.path.join(output_dir, f"{image_name}.txt")
            merge_yolo_results(all_results, final_output_file, iou_threshold)
            print(f"  合并结果保存到: {final_output_file}")
            print(f"  处理图像 {image_name} 完成，耗时: {time.time() - start_time:.2f} 秒")
            # 可选：可视化结果
            visualize = False  # 设置为True以生成可视化图像
            if visualize:
                detection_file = final_output_file
                output_image_path = os.path.join(visualization_dir, f"{image_name}_visualized.jpg")
                draw_detections(image_path, detection_file, output_image_path)
                print(f"  可视化结果保存到: {output_image_path}")

if __name__ == "__main__":
    main()