import json
import logging
import numpy as np
import cv2
from pathlib import Path
from agent import setup_logging, perform_meanshift, load_yolo_detections  # 移除RL环境和test_agent的导入

def perform_meanshift_no_transform(data, bandwidth=None):
    """
    Perform MeanShift clustering on the data without any coordinate transformations.
    """
    from sklearn.cluster import MeanShift, estimate_bandwidth

    data_no_transform = data.copy()

    if bandwidth is None:
        bandwidth = estimate_bandwidth(data_no_transform[:, :2], quantile=0.2)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(data_no_transform[:, :2])
    labels = ms.labels_
    return labels


def process_image_without_rl(image_path, annotation_path, output_path, class_filter=0, bandwidth=None):
    """
    Process a single image and its annotation, perform MeanShift clustering,
    and save the result as an image with bounding boxes and cluster labels.
    """

    image = cv2.imread(str(image_path))
    if image is None:
        logging.error(f"Cannot read image {image_path}")
        return
    height, width = image.shape[:2]

    detections = []
    with open(annotation_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) >= 5:
                cls, x, y, w, h = parts[:5]
                detections.append({
                    'class': int(cls),
                    'x': float(x),
                    'y': float(y),
                    'w': float(w),
                    'h': float(h)
                })

    class_detections = [det for det in detections if det['class'] == class_filter]

    positions_sizes = []
    for det in class_detections:
        center_x = det['x']
        center_y = det['y']
        w = det['w']
        h = det['h']
        positions_sizes.append([center_x, center_y, w, h])
    positions_sizes = np.array(positions_sizes)

    # Perform MeanShift clustering
    if len(positions_sizes) > 0:
        final_labels = perform_meanshift_no_transform(positions_sizes, bandwidth=bandwidth)
    else:
        # No detections, just return
        final_labels = np.array([])

    # Convert normalized coordinates back to absolute values
    for det in class_detections:
        det['x'] *= width
        det['y'] *= height
        det['w'] *= width
        det['h'] *= height

    # Generate random colors for clusters
    unique_final_labels = np.unique(final_labels)
    unique_final_labels = unique_final_labels[unique_final_labels != -1]
    num_final_clusters = len(unique_final_labels)
    colors = []
    np.random.seed(42)
    for _ in unique_final_labels:
        color = [int(c) for c in np.random.randint(0, 255, 3)]
        colors.append(color)

    # Draw detection boxes and cluster labels
    for idx, det in enumerate(class_detections):
        center_x, center_y = det['x'], det['y']
        w, h = det['w'], det['h']
        x1 = int(center_x - w / 2)
        y1 = int(center_y - h / 2)
        x2 = int(center_x + w / 2)
        y2 = int(center_y + h / 2)

        if len(final_labels) > idx and final_labels[idx] != -1:
            cluster_label = final_labels[idx]
            color = colors[cluster_label % len(colors)]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f'Cluster {cluster_label}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, 'No Cluster', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
        logging.info(f"Saved: {output_path}")
    except Exception as e:
        logging.error(f"Failed to save image {output_path}: {e}")

def main():
    setup_logging()
    # Define directory paths
    images_dir = Path('data/PANDA/images/val')  # Replace with your image folder path
    annotations_dir = Path('runs/detect/val_x_1280/labels')  # Replace with your annotation folder path
    output_dir = Path('ms_no_trans')  # Replace with your desired output folder path

    # Supported image extensions
    image_extensions = ['jpg', 'jpeg', 'png']

    # Get all image files
    image_files = [img for img in images_dir.iterdir() if img.suffix.lower().lstrip('.') in image_extensions]

    logging.info(f"Found {len(image_files)} images to process.")

    # Process each image only with MeanShift
    for idx, image_path in enumerate(image_files):
        annotation_filename = image_path.stem + '.txt'
        annotation_path = annotations_dir / annotation_filename

        if not annotation_path.exists():
            logging.warning(f"Annotation file {annotation_path} does not exist, skipping {image_path.name}")
            continue

        output_path = output_dir / image_path.name
        process_image_without_rl(
            image_path=image_path,
            annotation_path=annotation_path,
            output_path=output_path,
            class_filter=0,
            bandwidth=0.1
        )

if __name__ == "__main__":
    main()