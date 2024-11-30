import os
import gymnasium as gym
import logging
import numpy as np
import cv2
from pathlib import Path
from functools import partial
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import BasePolicy
import torch.nn as nn
import torch
from gymnasium import spaces
import random

def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def load_yolo_detections(txt_path):
    """
    Load YOLO detection results from a text file.
    """
    detections = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) >= 5:
                cls, x, y, w, h = parts[:5]
                detections.append([float(x), float(y), float(w), float(h)])
    return np.array(detections)

def perform_meanshift(data, bandwidth=None, alpha=0.5):
    """
    Perform MeanShift clustering on the data.
    """
    from sklearn.cluster import MeanShift, estimate_bandwidth

    # Nonlinear transformation on y-coordinate
    data_transformed = data.copy()
    data_transformed[:, 1] = data_transformed[:, 1] ** alpha

    if bandwidth is None:
        bandwidth = estimate_bandwidth(data_transformed[:, :2], quantile=0.2)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(data_transformed[:, :2])
    labels = ms.labels_
    return labels

class RLClusteringEnv(gym.Env):
    """
    Custom RL environment for clustering adjustment.
    """
    metadata = {'render.modes': ['save_image', 'rgb_array']}

    def __init__(self, image, detections, initial_labels,
                 num_clusters_min=5, num_clusters_max=20,
                 alpha=1.0, beta=5.0, gamma=1.0,
                 max_steps_per_episode=100):
        super(RLClusteringEnv, self).__init__()
        self.image = image
        self.detections = detections  # Normalized detections [x_center, y_center, w, h]
        self.initial_labels = initial_labels
        self.num_clusters_min = num_clusters_min
        self.num_clusters_max = num_clusters_max
        self.alpha = alpha  # Tightness weight
        self.beta = beta    # Cluster count penalty weight
        self.gamma = gamma  # Size variance penalty weight
        self.max_steps_per_episode = max_steps_per_episode

        # Initialize clusters
        self.current_clusters = self._initial_clusters()
        self.current_step = 0

        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=((self.num_clusters_max + 10) * 5,),
            dtype=np.float32
        )

        # Initialize action space placeholders
        self.max_merge_actions = self.num_clusters_max  # Maximum possible merge actions
        self.max_split_actions = self.num_clusters_max  # Maximum possible split actions
        self.action_space = spaces.Discrete(1 + self.max_merge_actions + self.max_split_actions)

        # Initialize action lists
        self.merge_actions = []
        self.split_actions = []

    def _initial_clusters(self):
        clusters = []
        num_clusters = np.max(self.initial_labels) + 1
        for i in range(num_clusters):
            indices = np.where(self.initial_labels == i)[0].tolist()
            clusters.append(indices)
        return clusters

    def _define_action_space(self):
        """
        Define the action space based on current clusters.
        """
        # Keep action
        num_keep_actions = 1

        # Merge actions: Only nearest neighbor pairs
        nearest_pairs = self._get_nearest_neighbor_pairs()
        self.merge_actions = nearest_pairs  # Store for use in step()
        num_merge_actions = len(nearest_pairs)

        # Split actions: One split action per cluster
        num_current_clusters = len(self.current_clusters)
        self.split_actions = [idx for idx in range(num_current_clusters) if len(self.current_clusters[idx]) > 1]
        num_split_actions = len(self.split_actions)

        # Total actions
        total_actions = num_keep_actions + num_merge_actions + num_split_actions

        # Define fixed-size action space
        total_actions_fixed = 1 + self.max_merge_actions + self.max_split_actions
        self.action_space = spaces.Discrete(total_actions_fixed)

    def _get_nearest_neighbor_pairs(self):
        """
        Find the nearest neighbor pairs among clusters.
        Returns:
            List of tuples: Each tuple contains indices of clusters to be merged (c1, c2).
        """
        # Get centroids of clusters
        centroids = []
        valid_cluster_indices = []
        for idx, cluster in enumerate(self.current_clusters):
            if len(cluster) > 0:
                cluster_boxes = self.detections[cluster]
                centroid = np.mean(cluster_boxes[:, :2], axis=0)
                centroids.append(centroid)
                valid_cluster_indices.append(idx)
        centroids = np.array(centroids)

        if len(centroids) < 2:
            return []

        # Compute distance matrix between centroids
        distances = cdist(centroids, centroids, metric='euclidean')

        # Set self-distance to infinity
        np.fill_diagonal(distances, np.inf)

        # Find nearest neighbor for each cluster
        nearest_pairs = []
        for i in range(len(centroids)):
            j = np.argmin(distances[i])
            c1 = valid_cluster_indices[i]
            c2 = valid_cluster_indices[j]
            # Ensure each pair is added only once
            pair = tuple(sorted((c1, c2)))
            if pair not in nearest_pairs:
                nearest_pairs.append(pair)
                # Set distances to infinity to avoid duplicate pairs
                distances[i, j] = distances[j, i] = np.inf
        return nearest_pairs

    def _get_state(self):
        """
        Get the current state representation.
        """
        # Prepare the observation vector
        max_clusters = self.num_clusters_max + 10
        state = np.zeros((max_clusters * 5,), dtype=np.float32)
        for idx, cluster in enumerate(self.current_clusters):
            if idx >= max_clusters:
                break
            if len(cluster) == 0:
                continue
            cluster_boxes = self.detections[cluster]
            # Features: [mean_x, mean_y, mean_w, mean_h, cluster_size]
            mean_vals = np.mean(cluster_boxes, axis=0)
            cluster_size = len(cluster_boxes) / len(self.detections)
            state[idx * 5: (idx + 1) * 5] = np.concatenate([mean_vals, [cluster_size]])
        return state

    def _compute_reward(self):
        """
        Compute the reward based on the current cluster configuration.
        """
        tightness = 0  # Accumulated tightness value
        variance = 0   # Variance of cluster sizes
        num_valid_clusters = 0  # Number of valid clusters

        for cluster in self.current_clusters:
            if len(cluster) > 0:
                num_valid_clusters += 1
                # Get the detection boxes within the cluster
                cluster_boxes = self.detections[cluster]
                # Calculate the area of each detection box (w * h)
                areas = cluster_boxes[:, 2] * cluster_boxes[:, 3]
                sum_areas = np.sum(areas)

                # Calculate the minimal bounding rectangle that contains all detection boxes
                x_centers = cluster_boxes[:, 0]
                y_centers = cluster_boxes[:, 1]
                widths = cluster_boxes[:, 2]
                heights = cluster_boxes[:, 3]

                x_mins = x_centers - widths / 2
                x_maxs = x_centers + widths / 2
                y_mins = y_centers - heights / 2
                y_maxs = y_centers + heights / 2

                x_min = np.min(x_mins)
                x_max = np.max(x_maxs)
                y_min = np.min(y_mins)
                y_max = np.max(y_maxs)

                bounding_box_area = (x_max - x_min) * (y_max - y_min)

                # Compute tightness: sum of detection box areas divided by the area of the bounding rectangle
                if bounding_box_area > 0:
                    cluster_tightness = sum_areas / bounding_box_area
                else:
                    cluster_tightness = 0  # Avoid division by zero

                tightness += cluster_tightness

                # Compute cluster size (number of samples)
                cluster_size = len(cluster_boxes)
                variance += cluster_size

        if num_valid_clusters > 0:
            # Compute average tightness
            tightness /= num_valid_clusters
            # Compute variance of cluster sizes
            sizes = [len(cluster) for cluster in self.current_clusters if len(cluster) > 0]
            variance = np.var(sizes)
        else:
            tightness = 0
            variance = 0

        # Reward calculation (since higher tightness is better, we aim to maximize tightness)
        reward = self.alpha * tightness - self.gamma * variance

        # Penalty for cluster count
        num_clusters = num_valid_clusters
        if num_clusters < self.num_clusters_min:
            reward -= self.beta * (self.num_clusters_min - num_clusters)
        elif num_clusters > self.num_clusters_max:
            reward -= self.beta * (num_clusters - self.num_clusters_max)

        return reward

    def reset(self, seed=None):
        """
        Reset the environment.
        """
        super().reset(seed=seed)
        self.current_clusters = self._initial_clusters()
        self.current_step = 0
        self._define_action_space()
        return self._get_state(), {}

    def step(self, action):
        """
        Execute an action.
        Args:
            action (int): Action index.
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1

        # Update action lists
        self._define_action_space()

        num_keep_actions = 1
        num_merge_actions = len(self.merge_actions)
        num_split_actions = len(self.split_actions)
        total_actions = num_keep_actions + num_merge_actions + num_split_actions
        total_actions_fixed = self.action_space.n

        if action == 0:
            # Keep action
            pass
        elif 1 <= action <= self.max_merge_actions:
            merge_idx = action - 1
            if merge_idx < num_merge_actions:
                c1, c2 = self.merge_actions[merge_idx]
                self._merge_clusters(c1, c2)
            else:
                # Invalid merge action, optionally penalize
                pass
        elif (1 + self.max_merge_actions) <= action < total_actions_fixed:
            split_idx = action - (1 + self.max_merge_actions)
            if split_idx < num_split_actions:
                c = self.split_actions[split_idx]
                self._split_cluster(c)
            else:
                # Invalid split action, optionally penalize
                pass
        else:
            # Invalid action, optionally penalize
            pass

        # Compute reward
        reward = self._compute_reward()

        # Check termination condition
        terminated = self.current_step >= self.max_steps_per_episode
        truncated = False  # Not using truncated in this context

        # Get observation
        observation = self._get_state()
        info = {}

        return observation, reward, terminated, truncated, info

    def _merge_clusters(self, c1, c2):
        if c1 < len(self.current_clusters) and c2 < len(self.current_clusters):
            # Merge c2 into c1
            self.current_clusters[c1].extend(self.current_clusters[c2])
            self.current_clusters[c2] = []
        else:
            # Handle invalid indices
            pass

    def _split_cluster(self, c):
        if c < len(self.current_clusters) and len(self.current_clusters[c]) > 1:
            # Use KMeans to split the cluster into two sub-clusters
            cluster = self.current_clusters[c]
            cluster_boxes = self.detections[cluster]
            try:
                # Explicitly set n_init to suppress the warning
                kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(cluster_boxes)
                labels = kmeans.labels_
                cluster1 = [cluster[i] for i in range(len(cluster)) if labels[i] == 0]
                cluster2 = [cluster[i] for i in range(len(cluster)) if labels[i] == 1]
                self.current_clusters[c] = cluster1
                self.current_clusters.append(cluster2)
            except Exception as e:
                logging.error(f"KMeans split failed: {e}")
        else:
            # Cannot split clusters with only one element
            pass


    def render(self, mode='save_image'):
        """
        Render the current cluster configuration.
        """
        render_dir = "agent_render"
        os.makedirs(render_dir, exist_ok=True)

        vis_image = self.image.copy()
        image_height, image_width = vis_image.shape[:2]

        num_clusters = len(self.current_clusters)
        colors = []
        np.random.seed(42)
        for _ in range(num_clusters):
            color = [int(c) for c in np.random.randint(0, 255, 3)]
            colors.append(color)

        for cluster_idx, cluster in enumerate(self.current_clusters):
            if len(cluster) == 0:
                continue
            color = colors[cluster_idx % len(colors)]
            for idx in cluster:
                box = self.detections[idx]
                center_x, center_y, w, h = box

                x1 = int((center_x - w / 2) * image_width)
                y1 = int((center_y - h / 2) * image_height)
                x2 = int((center_x + w / 2) * image_width)
                y2 = int((center_y + h / 2) * image_height)
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis_image, f'C{cluster_idx}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if mode == 'save_image':
            save_path = os.path.join(render_dir, f"render_step_{self.current_step}.png")
            cv2.imwrite(save_path, vis_image)
            logging.info(f"Saved render image as {save_path}")
        elif mode == 'rgb_array':
            return vis_image

class RenderCallback(BaseCallback):
    def __init__(self, render_freq=100, verbose=0):
        super(RenderCallback, self).__init__(verbose)
        self.render_freq = render_freq

    def _on_step(self):
        if self.n_calls % self.render_freq == 0:
            env = self.model.get_env().envs[0]
            env.render()
        return True

def test_agent(env, model):
    """
    Test the trained RL agent to adjust clusters.
    Args:
        env (gym.Env): The clustering environment.
        model (stable_baselines3.PPO): Trained PPO model.
    Returns:
        list: Final clustering result.
    """
    obs, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        # env.render()
    # Get final clusters
    final_clusters = env.current_clusters
    num_final_clusters = len([c for c in final_clusters if len(c) > 0])
    logging.info(f"Final number of clusters after RL adjustment: {num_final_clusters}")
    return final_clusters

def process_image_with_rl(image_path, annotation_path, output_path, model, class_filter=0,
                          bandwidth=None, num_clusters_min=5, num_clusters_max=20):
    """
    Process a single image and its annotation, perform clustering, and save the result image.
    Uses MeanShift for initial clustering and then adjusts clusters using the trained RL agent.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        logging.error(f"Cannot read image {image_path}")
        return
    height, width = image.shape[:2]

    # Read annotations
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

    # Filter detections by class
    class_detections = [det for det in detections if det['class'] == class_filter]

    # Get positions and sizes (normalized)
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
        initial_labels = perform_meanshift(positions_sizes, bandwidth=bandwidth, alpha=0.5)

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
        image=image,
        detections=positions_sizes,
        initial_labels=initial_labels,
        num_clusters_min=num_clusters_min,
        num_clusters_max=num_clusters_max,
        alpha=0.5,
        beta=5.0,
        gamma=2.0,
        max_steps_per_episode=100
    )

    # Use the trained RL model to adjust clusters
    final_clusters = test_agent(env, model)

    # Compute final cluster labels
    final_labels = np.full(len(class_detections), -1)
    cluster_id = 0
    for cluster in final_clusters:
        if len(cluster) == 0:
            continue
        for idx in cluster:
            final_labels[idx] = cluster_id
        cluster_id += 1

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
    images_dir = Path('/home/edge/work/Edge-Synergy/data/PANDA/images/train')
    annotations_dir = Path('/home/edge/work/Edge-Synergy/runs/detect/train_x_1280/labels')

    # Supported image extensions
    image_extensions = ['jpg', 'jpeg', 'png']

    # Get all image files
    image_files = [img for img in images_dir.iterdir() if img.suffix.lower().lstrip('.') in image_extensions]

    logging.info(f"Found {len(image_files)} images to process.")

    # Initialize RL environment function list
    env_fns = []

    # Collect environment creation functions
    for image_path in image_files:
        image = cv2.imread(str(image_path))
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
        # Do NOT normalize again
        # detections = detections / np.max(detections, axis=0)  # Remove this line

        # Perform MeanShift clustering
        initial_labels = perform_meanshift(detections, bandwidth=None, alpha=0.5)

        # Define an environment creation function
        env_fn = partial(RLClusteringEnv,
                         image=image,
                         detections=detections,
                         initial_labels=initial_labels,
                         num_clusters_min=5,
                         num_clusters_max=20,
                         alpha=0.5,
                         beta=5.0,
                         gamma=2.0,
                         max_steps_per_episode=100)
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
    render_callback = RenderCallback(render_freq=15)
    # Train the RL agent
    model.learn(total_timesteps=total_timesteps, callback=None)

    # Save the model
    model.save("checkpoints/ppo_rl_clustering")
    logging.info("RL agent trained and saved as ppo_rl_clustering.zip")

def main():
    """
    Main function to process images and apply the trained RL agent.
    """
    setup_logging()
    # Define directory paths
    images_dir = Path('/home/edge/work/Edge-Synergy/data/PANDA/images/val')  # Replace with your image folder path
    annotations_dir = Path('/home/edge/work/Edge-Synergy/runs/detect/val_x_1280/labels')  # Replace with your annotation folder path
    output_dir = Path('/home/edge/work/Edge-Synergy/cluster_output')  # Replace with your desired output folder path

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
            num_clusters_min=5,
            num_clusters_max=20
        )

if __name__ == "__main__":
    mode = input("Select mode: 1 - Train RL agent, 2 - Apply RL agent to adjust clusters (enter 1 or 2): ")
    if mode == '1':
        main_rl_training()
    elif mode == '2':
        main()
    else:
        print("Invalid choice. Please enter 1 or 2.")
