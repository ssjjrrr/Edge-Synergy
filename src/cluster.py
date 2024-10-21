import numpy as np
import cv2
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score
import time
from pathlib import Path
import os
import logging

def setup_logging():
    """
    设置日志记录。
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("processing.log", mode='w')
        ]
    )

def process_image(image_path, annotation_path, output_path, class_filter=0, bandwidth=0.1, alpha=0.5, k=1.0):
    """
    处理单个图像和其对应的注释文件，进行聚类并绘制结果图像。
    对 y 坐标进行非线性变换，并根据变换后的 y 坐标调整 x 坐标，以适应不同深度的目标。

    Args:
        image_path (Path): 图像文件路径。
        annotation_path (Path): 注释文件路径。
        output_path (Path): 结果图像保存路径。
        class_filter (int, optional): 要过滤的目标类别。默认为 0。
        bandwidth (float, optional): MeanShift 聚类的带宽参数。默认为 0.1。
        alpha (float, optional): y 坐标的幂函数变换参数。默认为 0.5。
        k (float, optional): 根据 y 坐标调整 x 坐标的缩放因子。默认为 1.0。
    """
    start_time = time.time()
    detections = []
    try:
        with open(annotation_path, 'r') as f:
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
    except Exception as e:
        logging.error(f"读取注释文件 {annotation_path} 失败: {e}")
        return

    # 过滤指定类别的检测结果
    class_detections = [det for det in detections if det['class'] == class_filter]

    # 计算每个检测框的中心位置
    positions = []
    for det in class_detections:
        center_x = det['x'] + det['w'] / 2
        center_y = det['y'] + det['h'] / 2
        positions.append([center_x, center_y])
    positions = np.array(positions)

    # 对 y 坐标进行非线性变换，并根据 y 调整 x 坐标
    if len(positions) > 0:
        # 保证 y 坐标在 (0,1] 范围内，避免 y=0 时的数学问题
        epsilon = 1e-5
        y_original = positions[:, 1].copy()
        y_original = np.clip(y_original, epsilon, 1.0)  # 防止 y=0
        y_transformed = y_original ** alpha

        # 根据变换后的 y 坐标调整 x 坐标
        # x' = x * (1 + k * y')
        x_original = positions[:, 0].copy()
        x_transformed = x_original * (1 + k * y_transformed)

        # 更新 positions
        positions_transformed = np.column_stack((x_transformed, y_transformed))

        # 进行 MeanShift 聚类
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True) 
        ms.fit(positions_transformed)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_

        # 计算 Silhouette Score
        if len(np.unique(labels)) > 1:
            score = silhouette_score(positions_transformed, labels)
            logging.info(f"{image_path.name} - Silhouette Score: {score:.2f}")
        else:
            logging.info(f"{image_path.name} - Only one cluster detected.")
    else:
        labels = np.array([])
        cluster_centers = np.array([])

    end_time = time.time()
    logging.info(f"{image_path.name} - MeanShift done, cost {end_time - start_time:.2f} seconds")

    # 读取图像
    image = cv2.imread(str(image_path))
    if image is None:
        logging.error(f"无法读取图像 {image_path}")
        return
    height, width = image.shape[:2]

    # 将相对坐标转换为绝对坐标
    for det in detections:
        det['x'] *= width
        det['y'] *= height
        det['w'] *= width
        det['h'] *= height

    # 生成随机颜色用于不同的聚类
    num_clusters = len(np.unique(labels))
    colors = []
    np.random.seed(42)
    for i in range(num_clusters):
        color = [int(c) for c in np.random.randint(0, 255, 3)]
        colors.append(color)

    # 绘制检测框和聚类标签
    for idx, det in enumerate(class_detections):
        center_x, center_y = det['x'], det['y']
        w, h = det['w'], det['h']
        x1 = int(center_x - w / 2)
        y1 = int(center_y - h / 2)
        x2 = int(center_x + w / 2)
        y2 = int(center_y + h / 2)
        
        if len(labels) > 0 and idx < len(labels):
            cluster_label = labels[idx]
            color = colors[cluster_label]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f'Cluster {cluster_label}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, 'No Cluster', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    try:
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
        logging.info(f"已保存: {output_path}")
    except Exception as e:
        logging.error(f"保存图像 {output_path} 失败: {e}")

def main():
    setup_logging()
    # 定义目录路径
    images_dir = Path('/home/edge/work/Edge-Synergy/data/PANDA/images/val')  # 替换为您的图像文件夹路径
    annotations_dir = Path('/home/edge/work/Edge-Synergy/runs/detect/predict/labels')  # 替换为您的注释文件夹路径
    output_dir = Path('/home/edge/work/Edge-Synergy/clustered_results')  # 替换为您希望保存结果图像的文件夹路径
    
    # 支持的图像格式
    image_extensions = ['jpg', 'jpeg', 'png']
    
    # 获取所有图像文件
    image_files = [img for img in images_dir.iterdir() if img.suffix.lower().lstrip('.') in image_extensions]
    
    logging.info(f"找到 {len(image_files)} 张图像进行处理.")
    
    # 顺序处理每张图像
    for image_path in image_files:
        annotation_filename = image_path.stem + '.txt'
        annotation_path = annotations_dir / annotation_filename
        
        if not annotation_path.exists():
            logging.warning(f"注释文件 {annotation_path} 不存在，跳过 {image_path.name}")
            continue
        
        output_path = output_dir / image_path.name
        process_image(image_path, annotation_path, output_path)

if __name__ == "__main__":
    main()
