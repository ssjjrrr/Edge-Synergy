import os
import cv2
import time
import logging
import torch
import json
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import gymnasium as gym
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from functools import partial
from stable_baselines3.common.callbacks import BaseCallback

def setup_logging():
    """
    Set up logging configuration.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("processing.log", mode='w')
        ]
    )

def load_yolo_detections(csv_path=None, json_path=None, npy_path=None, txt_path=None):
    """
    Load YOLO detection boxes from various file formats.

    Args:
        csv_path (str, optional): Path to CSV file.
        json_path (str, optional): Path to JSON file.
        npy_path (str, optional): Path to NumPy file.
        txt_path (str, optional): Path to TXT file.

    Returns:
        np.ndarray: Array of detection boxes with shape (N, 4).
    """
    if csv_path:
        df = pd.read_csv(csv_path)
        # Assuming CSV contains 'x_center', 'y_center', 'width', 'height'
        detections = df[['x_center', 'y_center', 'width', 'height']].values
    elif json_path:
        with open(json_path, 'r') as f:
            data = json.load(f)
        detections = np.array([[item['x_center'], item['y_center'], item['width'], item['height']] for item in data])
    elif npy_path:
        detections = np.load(npy_path)
    elif txt_path:
        # Read TXT file, assuming format: class x y w h score
        df = pd.read_csv(txt_path, sep=' ', header=None, names=['class', 'x_center', 'y_center', 'width', 'height', 'confidence'])
        detections = df[['x_center', 'y_center', 'width', 'height']].values
    else:
        raise ValueError("Please provide at least one file path (csv_path, json_path, npy_path, or txt_path).")
    return detections

def perform_meanshift(detections, bandwidth=None, quantile=0.2, n_samples=500):
    """
    Perform initial clustering using MeanShift.

    Args:
        detections (np.ndarray): Detection boxes with shape (N, 4).
        bandwidth (float, optional): Bandwidth parameter for MeanShift.
        quantile (float, optional): Quantile parameter for bandwidth estimation.
        n_samples (int, optional): Number of samples to use for bandwidth estimation.

    Returns:
        np.ndarray: Cluster labels for each detection box.
    """
    # Use only the positions (first two columns) for clustering
    positions = detections[:, :2]

    if bandwidth is None:
        bandwidth = estimate_bandwidth(positions, quantile=quantile, n_samples=min(n_samples, len(positions)))
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(positions)
    labels = ms.labels_
    return labels

class RLClusteringEnv(gym.Env):
    """
    Custom clustering environment for RL agent to adjust clusters by merging or splitting.
    """

    def __init__(self, detections, initial_labels, num_clusters_min=3, num_clusters_max=10,
                 alpha=1.0, beta=1.0, gamma=1.0, max_steps_per_episode=50):
        """
        Initialize the environment.

        Args:
            detections (np.ndarray): Detection boxes with shape (N, 4).
            initial_labels (np.ndarray): Initial cluster labels with shape (N,).
            num_clusters_min (int): Minimum number of clusters.
            num_clusters_max (int): Maximum number of clusters.
            alpha (float): Weight for tightness reward.
            beta (float): Weight for cluster count penalty.
            gamma (float): Weight for size variance penalty.
            max_steps_per_episode (int): Maximum steps per episode.
        """
        super(RLClusteringEnv, self).__init__()
        self.detections = detections
        self.num_boxes = detections.shape[0]
        self.num_clusters_min = num_clusters_min
        self.num_clusters_max = num_clusters_max
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_steps_per_episode = max_steps_per_episode
        self.current_step = 0

        # Initial clustering
        self.initial_labels = initial_labels
        self.current_clusters = self._initial_clusters()

        # Define maximum possible values for normalization
        self.max_centroid_x = 1.0  # Assuming normalized positions in [0,1]
        self.max_centroid_y = 1.0
        self.max_area = np.sum(self.detections[:, 2] * self.detections[:, 3])  # Sum of all box areas
        self.max_size = self.num_boxes  # Max number of boxes in a cluster
        self.max_size_variance = 1.0  # Assumed maximum variance

        # Define action space
        self.action_space = self._define_action_space()

        # Define observation space
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.num_clusters_max * 5,),
            dtype=np.float32
        )

    def _initial_clusters(self):
        """
        Build initial clusters based on initial labels.

        Returns:
            list: List of clusters, each containing indices of detection boxes.
        """
        clusters = []
        unique_labels = np.unique(self.initial_labels)
        for label in unique_labels:
            cluster = np.where(self.initial_labels == label)[0].tolist()
            clusters.append(cluster)
        # If initial number of clusters exceeds max_clusters, keep only the first num_clusters_max clusters
        if len(clusters) > self.num_clusters_max:
            clusters = clusters[:self.num_clusters_max]
        return clusters

    def _define_action_space(self):
        """
        Define the action space, including keep, merge, and split actions.

        Returns:
            spaces.Discrete: The action space.
        """
        # Calculate possible merge actions
        num_current_clusters = len(self.current_clusters)
        merge_actions = int(num_current_clusters * (num_current_clusters - 1) / 2)
        split_actions = num_current_clusters  # One split action per cluster

        total_actions = 1 + merge_actions + split_actions  # Keep + Merge + Split
        return spaces.Discrete(total_actions)

    def _get_merge_action_indices(self, action):
        """
        Decode the action into specific clusters to merge or split.

        Args:
            action (int): The action index.

        Returns:
            tuple or int: Indices of clusters to merge or split.
        """
        # Keep action
        if action == 0:
            return None

        num_current_clusters = len(self.current_clusters)
        merge_actions = int(num_current_clusters * (num_current_clusters - 1) / 2)
        # Merge actions
        if 1 <= action <= merge_actions:
            idx = action - 1
            for i in range(num_current_clusters):
                for j in range(i + 1, num_current_clusters):
                    if idx == 0:
                        return (i, j)
                    idx -= 1
        # Split actions
        split_action_start = 1 + merge_actions
        split_idx = action - split_action_start
        if 0 <= split_idx < num_current_clusters:
            return split_idx

        return None

    def _get_state(self):
        """
        Get the current state representation.

        Returns:
            np.ndarray: State vector with normalized features.
        """
        state = []
        for cluster in self.current_clusters:
            if len(cluster) == 0:
                # Pad empty clusters
                state.extend([0, 0, 0, 0, 0])
                continue
            # Compute cluster features
            cluster_boxes = self.detections[cluster]
            centroid = np.mean(cluster_boxes[:, :2], axis=0)  # x_center, y_center
            area = np.sum(cluster_boxes[:, 2] * cluster_boxes[:, 3])  # Total area
            size = len(cluster)
            # Compute size variance within the cluster
            var_w = np.var(cluster_boxes[:, 2])
            var_h = np.var(cluster_boxes[:, 3])
            size_variance = (var_w + var_h) / 2

            # Normalize features
            centroid_x_norm = np.clip(centroid[0] / self.max_centroid_x, 0.0, 1.0)
            centroid_y_norm = np.clip(centroid[1] / self.max_centroid_y, 0.0, 1.0)
            area_norm = np.clip(area / self.max_area, 0.0, 1.0)
            size_norm = np.clip(size / self.max_size, 0.0, 1.0)
            size_variance_norm = np.clip(size_variance / self.max_size_variance, 0.0, 1.0)

            state.extend([centroid_x_norm, centroid_y_norm, area_norm, size_norm, size_variance_norm])

        # Pad remaining clusters
        while len(state) < self.num_clusters_max * 5:
            state.extend([0, 0, 0, 0, 0])

        return np.array(state, dtype=np.float32)

    def step(self, action):
        """
        Execute an action.

        Args:
            action (int): The action index.

        Returns:
            Tuple[np.ndarray, float, bool, bool, dict]: observation, reward, terminated, truncated, info
        """
        action_indices = self._get_merge_action_indices(action)
        if action_indices is None:
            # Keep action
            pass
        elif isinstance(action_indices, tuple) and len(action_indices) == 2:
            # Merge action
            c1, c2 = action_indices
            if c1 < len(self.current_clusters) and c2 < len(self.current_clusters):
                # Merge c2 into c1
                self.current_clusters[c1].extend(self.current_clusters[c2])
                self.current_clusters[c2] = []
        elif isinstance(action_indices, int):
            # Split action
            c = action_indices
            if c < len(self.current_clusters) and len(self.current_clusters[c]) > 1:
                # Use KMeans to split into two sub-clusters
                cluster = self.current_clusters[c]
                cluster_boxes = self.detections[cluster]
                try:
                    kmeans = KMeans(n_clusters=2, random_state=0).fit(cluster_boxes)
                    labels = kmeans.labels_
                    cluster1 = [cluster[i] for i in range(len(cluster)) if labels[i] == 0]
                    cluster2 = [cluster[i] for i in range(len(cluster)) if labels[i] == 1]
                    self.current_clusters.append(cluster2)
                    self.current_clusters[c] = cluster1
                except Exception as e:
                    logging.error(f"KMeans split failed: {e}")
                    # Keep the original cluster if KMeans fails
                    pass

        # Compute reward
        reward = self._compute_reward()

        # Update step count
        self.current_step += 1

        # Determine if the episode is terminated
        if self.current_step >= self.max_steps_per_episode:
            terminated = True
        else:
            terminated = False
        truncated = False  # No truncation in this environment

        # Get observation
        observation = self._get_state()
        info = {}

        return observation, reward, terminated, truncated, info

    def _compute_reward(self):
        """
        Compute the reward based on current cluster configuration.

        Returns:
            float: The reward value.
        """
        tightness = 0
        variance = 0
        for cluster in self.current_clusters:
            if len(cluster) == 0:
                continue
            cluster_boxes = self.detections[cluster]
            # Compute the large bounding box
            x_min = np.min(cluster_boxes[:, 0] - cluster_boxes[:, 2] / 2)
            y_min = np.min(cluster_boxes[:, 1] - cluster_boxes[:, 3] / 2)
            x_max = np.max(cluster_boxes[:, 0] + cluster_boxes[:, 2] / 2)
            y_max = np.max(cluster_boxes[:, 1] + cluster_boxes[:, 3] / 2)
            large_w = x_max - x_min
            large_h = y_max - y_min
            large_area = large_w * large_h
            small_area = np.sum(cluster_boxes[:, 2] * cluster_boxes[:, 3])
            if large_area > 0:
                ratio = small_area / large_area
                tightness += ratio
            # Compute size variance
            var_w = np.var(cluster_boxes[:, 2])
            var_h = np.var(cluster_boxes[:, 3])
            variance += (var_w + var_h) / 2
        # Reward computation
        reward = self.alpha * tightness
        # Cluster count penalty
        num_clusters = len([c for c in self.current_clusters if len(c) > 0])
        if num_clusters < self.num_clusters_min:
            reward -= self.beta * (self.num_clusters_min - num_clusters)
        elif num_clusters > self.num_clusters_max:
            reward -= self.beta * (num_clusters - self.num_clusters_max)
        # Size variance penalty
        reward -= self.gamma * variance
        return reward

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment.

        Args:
            seed (int, optional): Random seed.
            options (dict, optional): Additional options.

        Returns:
            Tuple[np.ndarray, dict]: Initial observation and info.
        """
        super().reset(seed=seed)
        self.current_clusters = self._initial_clusters()
        self.current_step = 0
        return self._get_state(), {}

    def render(self, mode='human'):
        """
        Render the current cluster configuration (optional).
        """
        pass  # Implement visualization if needed

# def train_agent(env, model, total_timesteps=10000):
#     """
#     Train the RL agent using Stable Baselines3 PPO algorithm.

#     Args:
#         env (gym.Env): The clustering environment.
#         model (stable_baselines3.PPO): PPO model instance.
#         total_timesteps (int): Total training timesteps.
#     """
#     model.learn(total_timesteps=total_timesteps)
#     model.save("ppo_rl_clustering")
#     logging.info("RL agent trained and saved as ppo_rl_clustering.zip")

# def test_agent(env, model):
#     """
#     Test the trained RL agent to adjust clusters.

#     Args:
#         env (gym.Env): The clustering environment.
#         model (stable_baselines3.PPO): Trained PPO model.

#     Returns:
#         list: Final clustering result.
#     """
#     obs, info = env.reset()
#     done = False
#     while not done:
#         action, _states = model.predict(obs, deterministic=True)
#         obs, reward, done, truncated, info = env.step(action)
#     # Get final clusters
#     final_clusters = env.current_clusters
#     num_final_clusters = len([c for c in final_clusters if len(c) > 0])
#     logging.info(f"Final number of clusters after RL adjustment: {num_final_clusters}")
#     return final_clusters

def process_image_with_rl(image_path, annotation_path, output_path, model, class_filter=0,
                          bandwidth=None, num_clusters_min=3, num_clusters_max=10):
    """
    Process a single image and its annotation, perform clustering, and save the result image.
    Uses MeanShift for initial clustering and then adjusts clusters using the trained RL agent.

    Args:
        image_path (Path): Path to the image file.
        annotation_path (Path): Path to the annotation file.
        output_path (Path): Path to save the result image.
        model (stable_baselines3.PPO): Trained RL model.
        class_filter (int, optional): Target class to filter. Defaults to 0.
        bandwidth (float, optional): Bandwidth parameter for MeanShift. Defaults to 0.1.
        num_clusters_min (int, optional): Minimum number of clusters. Defaults to 3.
        num_clusters_max (int, optional): Maximum number of clusters. Defaults to 10.
    """
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
        logging.error(f"Failed to read annotation file {annotation_path}: {e}")
        return

    # Filter detections by class
    class_detections = [det for det in detections if det['class'] == class_filter]

    # Get positions and sizes
    positions_sizes = []
    for det in class_detections:
        center_x = det['x']
        center_y = det['y']
        w = det['w']
        h = det['h']
        positions_sizes.append([center_x, center_y, w, h])
    positions_sizes = np.array(positions_sizes)

    # Normalize positions and sizes to [0,1]
    positions_sizes[:, :4] = positions_sizes[:, :4] / np.max(positions_sizes[:, :4], axis=0)

    # Perform MeanShift clustering
    if len(positions_sizes) > 0:
        initial_labels = perform_meanshift(positions_sizes, bandwidth=bandwidth)

        # Compute Silhouette Score
        if len(np.unique(initial_labels)) > 1:
            score = silhouette_score(positions_sizes[:, :2], initial_labels)
            logging.info(f"{image_path.name} - MeanShift Silhouette Score: {score:.2f}")
        else:
            logging.info(f"{image_path.name} - Only one cluster detected by MeanShift.")
    else:
        initial_labels = np.array([])
        cluster_centers = np.array([])

    # Create RL environment
    env = RLClusteringEnv(
        detections=positions_sizes,
        initial_labels=initial_labels,
        num_clusters_min=num_clusters_min,
        num_clusters_max=num_clusters_max,
        alpha=1.0,
        beta=1.0,
        gamma=1.0
    )

    # Use the trained RL model to adjust clusters
    final_clusters = test_agent(env, model)

    # Compute final cluster labels
    final_labels = np.full(initial_labels.shape, -1)
    cluster_id = 0
    for cluster in final_clusters:
        if len(cluster) == 0:
            continue
        for idx in cluster:
            final_labels[idx] = cluster_id
        cluster_id += 1

    # Load the image
    image = cv2.imread(str(image_path))
    if image is None:
        logging.error(f"Cannot read image {image_path}")
        return
    height, width = image.shape[:2]

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

        if final_labels[idx] != -1:
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

def main_rl_training():
    """
    RL agent training process using data from multiple images.
    """
    setup_logging()
    # Define directory paths
    images_dir = Path('/home/edge/work/Edge-Synergy/data/PANDA/images/train')  # Replace with your image folder path
    annotations_dir = Path('/home/edge/work/Edge-Synergy/runs/detect/predict/labels')  # Replace with your annotation folder path

    # Supported image extensions
    image_extensions = ['jpg', 'jpeg', 'png']

    # Get all image files
    image_files = [img for img in images_dir.iterdir() if img.suffix.lower().lstrip('.') in image_extensions]

    logging.info(f"Found {len(image_files)} images to process.")

    # Initialize RL environment function list
    env_fns = []

    # Collect environment creation functions
    for image_path in image_files:
        annotation_filename = image_path.stem + '.txt'
        annotation_path = annotations_dir / annotation_filename

        if not annotation_path.exists():
            logging.warning(f"Annotation file {annotation_path} does not exist, skipping {image_path.name}")
            continue

        # Load detection data
        detections = load_yolo_detections(txt_path=str(annotation_path))
        if detections.size == 0:
            logging.warning(f"Annotation file {annotation_path} has no valid detections, skipping {image_path.name}")
            continue
        detections = detections / np.max(detections, axis=0)  # Normalize to [0,1]

        # Perform MeanShift clustering
        initial_labels = perform_meanshift(detections, bandwidth=None)

        # Define an environment creation function
        env_fn = partial(RLClusteringEnv,
                         detections=detections,
                         initial_labels=initial_labels,
                         num_clusters_min=3,
                         num_clusters_max=10,
                         alpha=1.0,
                         beta=1.0,
                         gamma=1.0,
                         max_steps_per_episode=50)
        env_fns.append(env_fn)

    logging.info(f"Created {len(env_fns)} RL clustering environment functions.")

    if len(env_fns) == 0:
        logging.error("No valid environments to train.")
        return

    # Use Vectorized Environment for batch training
    vec_env = DummyVecEnv(env_fns)
    n_envs = vec_env.num_envs
    logging.info(f"Number of environments: {n_envs}")

    # Check if the environment meets SB3 requirements
    try:
        check_env(env_fns[0]())
        logging.info("Environment passed the check.")
    except Exception as e:
        logging.error(f"Environment check failed: {e}")
        return

    # Define policy network parameters
    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )

    # Initialize PPO model
    model = PPO("MlpPolicy", vec_env, verbose=1, policy_kwargs=policy_kwargs)

    # Define total training timesteps
    steps_per_env = 1000  # Adjust as needed
    total_timesteps = steps_per_env * n_envs

    logging.info(f"Total training timesteps (all environments): {total_timesteps}")

    # Train the RL agent
    model.learn(total_timesteps=total_timesteps)

    # Save the model
    model.save("ppo_rl_clustering")
    logging.info("RL agent trained and saved as ppo_rl_clustering.zip")

def main():
    """
    Main function to process images and apply the trained RL agent.
    """
    setup_logging()
    # Define directory paths
    images_dir = Path('/home/edge/work/Edge-Synergy/data/PANDA/images/val')  # Replace with your image folder path
    annotations_dir = Path('/home/edge/work/Edge-Synergy/runs/detect/predict/labels')  # Replace with your annotation folder path
    output_dir = Path('/home/edge/work/Edge-Synergy/clustered_results')  # Replace with your desired output folder path

    # Supported image extensions
    image_extensions = ['jpg', 'jpeg', 'png']

    # Get all image files
    image_files = [img for img in images_dir.iterdir() if img.suffix.lower().lstrip('.') in image_extensions]

    logging.info(f"Found {len(image_files)} images to process.")

    # Load the trained RL model
    try:
        model = PPO.load("ppo_rl_clustering")
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
            num_clusters_min=3,
            num_clusters_max=10
        )

if __name__ == "__main__":
    # Choose training or application mode
    mode = input("Select mode: 1 - Train RL agent, 2 - Apply RL agent to adjust clusters (enter 1 or 2): ")
    if mode == '1':
        main_rl_training()
    elif mode == '2':
        main()
    else:
        print("Invalid choice. Please enter 1 or 2.")
