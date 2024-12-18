import os
import time
import cv2
import math
import numpy as np
from ultralytics import YOLO

# 配置参数
image_width = 3840
image_height = 2160
block_width = 2120
block_height = 1280
overlap_pixels = 200  # 重叠像素数
iou_threshold = 0.001    # NMS的IOU阈值

# 定义YOLO模型路径
YOLO_MODEL_PATH = '/home/edge/work/Edge-Synergy/checkpoints/yolov8x_1280_ep300.pt'

# 输入输出目录
image_dir = "data/PANDA/images/scene/6"  # 输入图像目录
output_dir = "runs/detect/scene6_x"  # 合并结果输出目录
visualization_dir = os.path.join(output_dir, "visualizations")  # 可视化结果目录

# 加载统一的YOLO模型
if not os.path.exists(YOLO_MODEL_PATH):
    raise FileNotFoundError(f"YOLO模型文件 {YOLO_MODEL_PATH} 未找到。请确保模型文件存在。")
print(f"加载YOLO模型: {YOLO_MODEL_PATH}")
model = YOLO(YOLO_MODEL_PATH)

def generate_uniform_partitions(image_width, image_height, block_width, block_height, overlap):
    """
    生成均匀分区计划，包含向右和向下的重叠。
    返回一个包含分区坐标的列表，每个坐标为 (x1, y1, x2, y2)。
    """
    partitions = []
    step_x = block_width - overlap
    step_y = block_height - overlap

    num_blocks_x = math.ceil((image_width - overlap) / step_x)
    num_blocks_y = math.ceil((image_height - overlap) / step_y)

    for y in range(num_blocks_y):
        for x in range(num_blocks_x):
            x1 = x * step_x
            y1 = y * step_y
            x2 = x1 + block_width
            y2 = y1 + block_height

            # 确保分区不超出图像边界
            if x2 > image_width:
                x2 = image_width
                x1 = max(x2 - block_width, 0)
            if y2 > image_height:
                y2 = image_height
                y1 = max(y2 - block_height, 0)

            partitions.append({'coords': (x1, y1, x2, y2)})

    return partitions

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
    results = model(slice_image, conf=0.001)  # 设置较低的置信度阈值以获取更多候选框

    slice_results = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            score = float(box.conf[0])
            x, y, w, h = box.xywh[0].tolist()
            # 转换为全图坐标
            abs_x, abs_y, abs_w, abs_h = convert_coordinates(
                coords,
                x / slice_image.shape[1],
                y / slice_image.shape[0],
                w / slice_image.shape[1],
                h / slice_image.shape[0],
                image_width,
                image_height
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

def process_image(image_file, image_dir, output_dir, visualization_dir, partitions, model):
    """
    处理单张图像：分区检测，合并结果，保存检测结果和可视化。
    """
    image_path = os.path.join(image_dir, image_file)
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return

    image_name, _ = os.path.splitext(image_file)
    print(f"处理图像: {image_name}")

    all_results = []

    # 按照分区计划逐个处理分区
    for idx, partition in enumerate(partitions):
        print(f"  处理分区 {idx + 1}/{len(partitions)}: {partition['coords']}")
        slice_results = process_partition(
            image,
            partition,
            model
        )
        all_results.extend(slice_results)

    print(f"  共检测到 {len(all_results)} 个目标。")

    # 合并检测结果并应用NMS
    final_results = apply_nms(all_results, iou_threshold)
    print(f"  应用NMS后，剩余 {len(final_results)} 个目标。")

    # 保存合并后的检测结果
    final_output_file = os.path.join(output_dir, f"{image_name}.txt")
    merge_yolo_results(all_results, final_output_file, iou_threshold)
    print(f"  合并结果保存到: {final_output_file}")

    # 可选：可视化检测结果
    visualize = False  # 设置为True以生成可视化图像
    if visualize:
        output_image_path = os.path.join(visualization_dir, f"{image_name}_visualized.jpg")
        draw_detections(image_path, final_output_file, output_image_path)
        print(f"  可视化结果保存到: {output_image_path}")

def main():
    # 检查并创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(visualization_dir):
        os.makedirs(visualization_dir)

    # 生成均匀分区计划
    partitions = generate_uniform_partitions(
        image_width,
        image_height,
        block_width,
        block_height,
        overlap_pixels
    )

    print(f"生成了 {len(partitions)} 个分区。")

    # 处理每张图像
    for image_file in os.listdir(image_dir):
        if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue  # 跳过非图像文件
        start_time = time.time()
        process_image(
            image_file,
            image_dir,
            output_dir,
            visualization_dir,
            partitions,
            model
        )
        print(f"处理图像 {image_file} 完成，耗时: {time.time() - start_time:.2f} 秒")

if __name__ == "__main__":
    main()
