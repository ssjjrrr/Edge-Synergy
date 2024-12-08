import json
import logging
import numpy as np
import cv2
from pathlib import Path
from stable_baselines3 import PPO
from agent import setup_logging, RLClusteringEnv, perform_meanshift, test_agent

def process_image_with_rl(image_path, annotation_path, output_path, model, class_filter=0,
                          bandwidth=None, num_clusters_min=10, num_clusters_max=15):
    """
    Process a single image and its annotation, perform clustering, and save the cluster data.
    Computes expanded bounding rectangles and records area of each detection box in each cluster.
    Ensures expanded rectangles do not exceed the image boundaries.
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

    if len(positions_sizes) > 0:
        initial_labels = perform_meanshift(positions_sizes, bandwidth=bandwidth, alpha=0.5)
    else:
        initial_labels = np.array([])

    env = RLClusteringEnv(
        image=image,
        detections=positions_sizes,
        initial_labels=initial_labels,
        num_clusters_min=num_clusters_min,
        num_clusters_max=num_clusters_max,
        alpha=50.0,
        beta=1,
        gamma=1000000,
        delta=5.0,
        max_steps_per_episode=30,
        y_transform_alpha=0.5
    )

    final_clusters = test_agent(env, model)

    final_labels = np.full(len(class_detections), -1)
    cluster_id = 0
    for cluster in final_clusters:
        if len(cluster) == 0:
            continue
        for idx in cluster:
            final_labels[idx] = cluster_id
        cluster_id += 1

    for det in class_detections:
        det['x'] *= width
        det['y'] *= height
        det['w'] *= width
        det['h'] *= height

    clusters_dict = {}
    for idx, det in enumerate(class_detections):
        cluster_label = final_labels[idx]
        if cluster_label == -1:
            continue
        cluster_label = int(cluster_label)  
        if cluster_label not in clusters_dict:
            clusters_dict[cluster_label] = {
                'detection_areas': [],
                'max_width': 0,
                'max_height': 0
            }

        area = det['w'] * det['h']

        clusters_dict[cluster_label]['detection_areas'].append(float(area))

        if det['w'] > clusters_dict[cluster_label]['max_width']:
            clusters_dict[cluster_label]['max_width'] = det['w']
        if det['h'] > clusters_dict[cluster_label]['max_height']:
            clusters_dict[cluster_label]['max_height'] = det['h']

    output_data = []
    for cluster_label, cluster_info in clusters_dict.items():
        detection_areas = cluster_info['detection_areas']
        max_width = cluster_info['max_width']
        max_height = cluster_info['max_height']

        x1_list = []
        y1_list = []
        x2_list = []
        y2_list = []
        for det in class_detections:
            if final_labels[class_detections.index(det)] != cluster_label:
                continue
            x = det['x']
            y = det['y']
            w = det['w']
            h = det['h']
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            x1_list.append(x1)
            y1_list.append(y1)
            x2_list.append(x2)
            y2_list.append(y2)

        x1_cluster = min(x1_list)
        y1_cluster = min(y1_list)
        x2_cluster = max(x2_list)
        y2_cluster = max(y2_list)

        x1_expanded = x1_cluster - 0.1 * max_width
        y1_expanded = y1_cluster - 0.1 * max_height
        x2_expanded = x2_cluster + 0.1 * max_width
        y2_expanded = y2_cluster + 0.1 * max_height

        x1_expanded = max(0, x1_expanded)
        y1_expanded = max(0, y1_expanded)
        x2_expanded = min(width, x2_expanded)
        y2_expanded = min(height, y2_expanded)

        output_data.append({
            'cluster_id': cluster_label,
            'bounding_box': {
                'x1': int(x1_expanded),
                'y1': int(y1_expanded),
                'x2': int(x2_expanded),
                'y2': int(y2_expanded)
            },
            'detection_areas': detection_areas
        })

    output_json_path = output_path.with_suffix('.json')

    try:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        logging.info(f"Saved cluster data: {output_json_path}")

    except Exception as e:
        logging.error(f"Failed to save cluster data {output_json_path}: {e}")


def main():
    """
    Main function to process images and apply the trained RL agent.
    """
    setup_logging()
    # Define directory paths
    images_dir = Path('data/PANDA/images/val')  # Replace with your image folder path
    annotations_dir = Path('runs/detect/val_x_1280/labels')  # Replace with your annotation folder path
    output_dir = Path('cluster_output')  # Replace with your desired output folder path

    # Supported image extensions
    image_extensions = ['jpg', 'jpeg', 'png']

    # Get all image files
    image_files = [img for img in images_dir.iterdir() if img.suffix.lower().lstrip('.') in image_extensions]

    logging.info(f"Found {len(image_files)} images to process.")

    # Load the trained RL model
    try:
        model = PPO.load("checkpoints/ppo_rl_clustering")
        logging.info("Loaded trained RL agent model ppo_rl_clustering.zip")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return

    # Process and save each image
    for idx, image_path in enumerate(image_files):
        annotation_filename = image_path.stem + '.txt'
        annotation_path = annotations_dir / annotation_filename

        if not annotation_path.exists():
            logging.warning(f"Annotation file {annotation_path} does not exist, skipping {image_path.name}")
            continue

        output_path = output_dir / image_path.name
        process_image_with_rl(
            image_path=image_path,
            annotation_path=annotation_path,
            output_path=output_path,
            model=model,
            class_filter=0,
            num_clusters_min=10,
            num_clusters_max=15
        )

if __name__ == "__main__":
    main()
