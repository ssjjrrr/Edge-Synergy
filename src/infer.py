import os
import json
import time
import cv2
import math
from ultralytics import YOLO

# 配置参数
IMAGE_DIR = "data/PANDA/images/scene/6"  # 输入图像目录
CLUSTER_DIR = "/home/edge/work/Edge-Synergy/cluster_output"      # 聚类JSON文件目录
RESULT_JSON_PATH = "/home/edge/work/Edge-Synergy/cluster_output/results_1200.json"     # result1000.json文件路径
OUTPUT_DIR = "runs/detect/repose_results"  # 合并结果输出目录
VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, "visualizations")  # 可视化结果目录
CONF = 0.15

# 定义YOLO模型路径
YOLO_MODELS = {
    's': '/home/edge/work/Edge-Synergy/checkpoints/yolov8s_768_ep300.pt',
    'm': '/home/edge/work/Edge-Synergy/checkpoints/yolov8m_896_ep300.pt',
    'l': '/home/edge/work/Edge-Synergy/checkpoints/yolov8l_1024_ep300.pt',
}

# NMS的IoU阈值
IOU_THRESHOLD = 0.3

# 定义YOLO模型加载
def load_models(model_paths):
    """
    加载指定路径的YOLO模型。
    model_paths: dict, 键为模型标识（'s', 'm', 'l'），值为模型路径。
    返回一个字典，键为模型标识，值为加载的模型实例。
    """
    models = {}
    for key, path in model_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"YOLO模型文件 {path} 未找到。请确保模型文件存在。")
        print(f"加载YOLO模型 '{key}': {path}")
        models[key] = YOLO(path)
    return models

# 计算IoU
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

# 应用非极大值抑制（NMS）
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

# 保存YOLO格式的检测结果
def save_yolo_results(file_path, results):
    """
    将检测结果保存为YOLO格式的文本文件。
    """
    with open(file_path, 'w') as file:
        for result in results:
            class_id, x_center, y_center, width, height, score = result
            file.write(f"{class_id} {x_center} {y_center} {width} {height} {score}\n")

# 绘制检测结果到图像上（可选）
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

# 处理单个图像
def process_image(image_json, offloading_plan, models, image_dir, cluster_dir, output_dir, visualization_dir):
    """
    处理单张图像：对每个聚类区域使用指定模型进行推理，合并结果并保存。
    """
    image_name = os.path.splitext(image_json)[0]
    cluster_json_path = os.path.join(cluster_dir, image_json)
    
    # 读取聚类信息
    with open(cluster_json_path, 'r') as f:
        clusters = json.load(f)
    
    # 假设图像文件与JSON文件同名，但扩展名为.jpg/.png等
    # 尝试多种常见扩展名
    possible_exts = ['.jpg', '.jpeg', '.png']
    image_path = None
    for ext in possible_exts:
        temp_path = os.path.join(image_dir, image_name + ext)
        if os.path.exists(temp_path):
            image_path = temp_path
            break
    if image_path is None:
        print(f"未找到图像文件对应于 {image_json}")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return

    print(f"\n处理图像: {image_name}")

    all_results = []

    # 读取offloading_plan for this image
    if image_json not in offloading_plan:
        print(f"图像 {image_json} 不在offloading_plan中，跳过。")
        return
    image_plan = offloading_plan[image_json]["offloading_plan"]

    # 逐个处理每个聚类
    for cluster in clusters:
        cluster_id = str(cluster["cluster_id"])
        bounding_box = cluster["bounding_box"]
        x1, y1, x2, y2 = bounding_box["x1"], bounding_box["y1"], bounding_box["x2"], bounding_box["y2"]

        if cluster_id not in image_plan:
            print(f"聚类ID {cluster_id} 在图像 {image_json} 的offloading_plan中未找到，跳过。")
            continue

        model_key = image_plan[cluster_id]
        if model_key not in models:
            print(f"模型 '{model_key}' 未加载，跳过聚类ID {cluster_id}。")
            continue

        model = models[model_key]

        # 裁剪图像到聚类区域
        slice_image = image[y1:y2, x1:x2]

        # 运行YOLO模型
        results = model(slice_image, conf=CONF)  # 设置较低的置信度阈值以获取更多候选框

        # 处理检测结果
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                score = float(box.conf[0])
                x, y, w, h = box.xywh[0].tolist()
                # 将相对坐标转换为全图绝对坐标并归一化
                abs_x_center = (x + x1) / image.shape[1]
                abs_y_center = (y + y1) / image.shape[0]
                abs_w = w / image.shape[1]
                abs_h = h / image.shape[0]
                all_results.append((cls, abs_x_center, abs_y_center, abs_w, abs_h, score))

    print(f"  共检测到 {len(all_results)} 个目标。")

    # 合并检测结果并应用NMS
    final_results = apply_nms(all_results, IOU_THRESHOLD)
    print(f"  应用NMS后，剩余 {len(final_results)} 个目标。")

    # 保存合并后的检测结果
    final_output_file = os.path.join(output_dir, f"{image_name}.txt")
    save_yolo_results(final_output_file, final_results)
    print(f"  合并结果保存到: {final_output_file}")

    # 可选：可视化检测结果
    visualize = True  # 设置为True以生成可视化图像
    if visualize:
        output_image_path = os.path.join(visualization_dir, f"{image_name}_visualized.jpg")
        draw_detections(image_path, final_output_file, output_image_path)
        print(f"  可视化结果保存到: {output_image_path}")

# 主函数
def main():
    # 检查并创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(VISUALIZATION_DIR):
        os.makedirs(VISUALIZATION_DIR)

    # 加载YOLO模型
    models = load_models(YOLO_MODELS)

    # 加载result1000.json
    if not os.path.exists(RESULT_JSON_PATH):
        raise FileNotFoundError(f"result.json 文件未找到: {RESULT_JSON_PATH}")
    with open(RESULT_JSON_PATH, 'r') as f:
        result_data = json.load(f)

    # 处理每个图像
    for image_json in result_data:
        start_time = time.time()
        process_image(
            image_json,
            result_data,
            models,
            IMAGE_DIR,
            CLUSTER_DIR,
            OUTPUT_DIR,
            VISUALIZATION_DIR
        )
        elapsed_time = time.time() - start_time
        print(f"处理图像 {image_json} 完成，耗时: {elapsed_time:.2f} 秒")

if __name__ == "__main__":
    main()
