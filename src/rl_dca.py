import json
import logging
import pathlib
import os
import numpy as np
import cv2
import sklearn
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
    suffix = pathlib.Path(annotation_path).suffix.lower()

    if suffix == '.json':
        with open(annotation_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'detections' in data:
                det_iter = data['detections']
            else:
                det_iter = [d for cl in data for d in cl.get('detections_xywh', [])]

            for det in det_iter:
                cls = det.get('class_id', 0)
                x1, y1, x2, y2 = det['bbox']
                cx = (x1 + x2) / 2 / width
                cy = (y1 + y2) / 2 / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height
                detections.append({'class': cls, 'x': cx, 'y': cy, 'w': w, 'h': h})

    else:
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
                'max_height': 0,
                'detections_xywh': []
            }

        area = det['w'] * det['h']
        clusters_dict[cluster_label]['detection_areas'].append(float(area))
        clusters_dict[cluster_label]['detections_xywh'].append({
            'x': float(det['x']),
            'y': float(det['y']),
            'w': float(det['w']),
            'h': float(det['h'])
        })

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

        for det_item in cluster_info['detections_xywh']:
            x = det_item['x']
            y = det_item['y']
            w = det_item['w']
            h = det_item['h']
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
            'detection_areas': detection_areas,
            'detections_xywh': cluster_info['detections_xywh']
        })

    output_json_path = pathlib.Path(os.path.splitext(output_path)[0] + ".json")

    try:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        logging.info(f"Saved cluster data: {output_json_path}")

    except Exception as e:
        logging.error(f"Failed to save cluster data {output_json_path}: {e}")
    colors = []
    np.random.seed(42)
    unique_final_labels = np.unique(final_labels)
    unique_final_labels = unique_final_labels[unique_final_labels != -1]
    num_final_clusters = len(unique_final_labels)
    for _ in unique_final_labels:
        color = [int(c) for c in np.random.randint(0, 255, 3)]
        colors.append(color)
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

def cluster_from_mem(
        image: "np.ndarray",
        detections_dict: dict,
        model=None,
        class_filter: int = 0,
        num_clusters_min: int = 0,
        num_clusters_max: int = 5
):
    """
    Args
    ----
    image            : BGR ndarray (original image)
    detections_dict  : {'width':W,'height':H,'detections':[{'bbox':[x1,y1,x2,y2],'class_id':..},...]}
    model            : loaded PPO model or None
    Returns
    -------
    clusters         : list[dict] in the same schema as legacy cluster.json
    """
    import numpy as np

    H, W = image.shape[:2]

    class_dets = []
    for d in detections_dict.get("detections", []):
        if d.get("class_id", 0) != class_filter:
            continue
        x1, y1, x2, y2 = d["bbox"]
        cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        w, h   = (x2 - x1) / W, (y2 - y1) / H
        class_dets.append([cx, cy, w, h])
    det_arr = np.asarray(class_dets, dtype=np.float32)
    if det_arr.size:
        init_labels = perform_meanshift(det_arr, bandwidth=None, alpha=0.5)
    else:
        init_labels = np.array([])

    from agent import RLClusteringEnv, test_agent
    env = RLClusteringEnv(
        image=image,
        detections=det_arr,
        initial_labels=init_labels,
        num_clusters_min=num_clusters_min,
        num_clusters_max=num_clusters_max,
        alpha=50.0, beta=1, gamma=1e6, delta=5.0,
        max_steps_per_episode=30, y_transform_alpha=0.5
    )
    final_clusters = test_agent(env, model)

    clusters_out = []
    label_map = np.full(len(det_arr), -1, int)

    for cid, cl in enumerate(final_clusters):
        valid_inds = [idx for idx in cl if 0 <= idx < len(det_arr)]
        for idx in valid_inds:
            label_map[idx] = cid


    for cid in range(label_map.max() + 1):
        inds = np.where(label_map == cid)[0]
        if inds.size == 0:
            continue
        xs = (det_arr[inds, 0] * W)
        ys = (det_arr[inds, 1] * H)
        ws = (det_arr[inds, 2] * W)
        hs = (det_arr[inds, 3] * H)
        x1s = xs - ws / 2; y1s = ys - hs / 2
        x2s = xs + ws / 2; y2s = ys + hs / 2
        bb = [int(max(0,  x1s.min())),
              int(max(0,  y1s.min())),
              int(min(W, x2s.max())),
              int(min(H, y2s.max()))]

        clusters_out.append({
            "cluster_id": int(cid),
            "bounding_box": {"x1":bb[0],"y1":bb[1],"x2":bb[2],"y2":bb[3]},
            "detection_areas": (ws*hs).tolist(),
            "detections_xywh": [
                {"x": float(x), "y": float(y), "w": float(w), "h": float(h)}
                for x, y, w, h in zip(xs, ys, ws, hs)
            ]
        })
    return clusters_out
def main():
    """
    Main function to process images and apply the trained RL agent.
    """
    setup_logging()
    # Define directory paths
    images_dir = Path('data/PANDA/images/val')  # Replace with your image folder path
    annotations_dir = Path('/home/edge/work/Edge-Synergy/runs/detect/coarse_detection_2x2/label_merged_results')  # Replace with your annotation folder path
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

