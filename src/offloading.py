import json
import math
import os
import time


edge = [1.0/4, 1.0/3, 1.0/4]

models = {
    'n': {'size': 640, 'delay': 37.8},
    's': {'size': 768, 'delay': 150.3},
    'm': {'size': 896, 'delay': 210.0},
    'l': {'size': 1024, 'delay': 410},
}


precision_table = {
    'n': [
        ((0, 4096), 0.1),
        ((4096, 16384), 0.312),
        ((16384, 25600), 0.407),
        ((25600, 38416), 0.488),
        ((38416, math.inf), 0.575)
    ],
    's': [
        ((0, 4096), 0.2),
        ((4096, 16384), 0.401),
        ((16384, 25600), 0.458),
        ((25600, 38416), 0.493),
        ((38416, math.inf), 0.624)
    ],
    'm': [
        ((0, 4096), 0.3),
        ((4096, 16384), 0.453),
        ((16384, 25600), 0.523),
        ((25600, 38416), 0.527),
        ((38416, math.inf), 0.639)
    ],
    'l': [
        ((0, 4096), 0.4),
        ((4096, 16384), 0.507),
        ((16384, 25600), 0.564),
        ((25600, 38416), 0.590),
        ((38416, math.inf), 0.699)
    ]
}

config = {
    "normalization": {
        "area_enhancement_factor": 1.5,
        "height_normalization_factor": 1.5,
        "count_normalization_factor": 1
    },
    "allocation": {
        "m_model_ratio": 0.5,
        "s_model_ratio": 0.3,
        "n_model_ratio": 0.2
    },
    "migration": {
        "migration_threshold_factor": 0.3
    }
}



def get_precision_for_area(model, area):
    """Look up precision by detection area."""
    for (low, high), val in precision_table[model]:
        if low <= area < high:
            return val
    return 0.0


def compute_cluster_precision(cluster, models):
    """Compute per-model precision for one cluster."""
    bbox = cluster["bounding_box"]
    
    if "width" in bbox and "height" in bbox:
        W, H = bbox["width"], bbox["height"]
    elif all(k in bbox for k in ("x1", "y1", "x2", "y2")):
        W = bbox["x2"] - bbox["x1"]
        H = bbox["y2"] - bbox["y1"]
    else:
        raise ValueError(f"Unrecognized bounding_box format: {bbox}")

    areas = cluster["detection_areas"]
    M = len(areas)
    precision_dict = {}
    
    if M == 0:
        for m in models.keys():
            precision_dict[m] = 0.0
        return precision_dict

    for m in models.keys():
        S = models[m]['size']
        scale_factor = S / max(W, H) if max(W, H) > 0 else 1.0
        sum_p = 0.0
        for A in areas:
            A_scaled = A * (scale_factor ** 2)
            sum_p += get_precision_for_area(m, A_scaled)
        precision_dict[m] = sum_p / M
    
    return precision_dict



def calculate_cluster_metrics(cluster, image_height, image_width):
    """Compute raw cluster metrics: average area, position, and count."""
    areas = cluster["detection_areas"]
    avg_area = sum(areas) / len(areas) if areas else 0
    
    bbox = cluster["bounding_box"]
    
    if "x1" in bbox and "y1" in bbox and "x2" in bbox and "y2" in bbox:
        center_y = 0.5 * (bbox["y1"] + bbox["y2"])
        center_x = 0.5 * (bbox["x1"] + bbox["x2"])
    elif "center_y" in bbox and "center_x" in bbox:
        center_y = bbox["center_y"]
        center_x = bbox["center_x"]
    else:
        center_y = 0.5 * (bbox.get("y1", 0) + bbox.get("y2", 0))
        center_x = 0.5 * (bbox.get("x1", 0) + bbox.get("x2", 0))
    
    relative_width = min(abs(center_x - image_width / 4), 
                        abs(center_x - (image_width / 4) * 3)) / image_width
    relative_height = center_y / image_height if image_height > 0 else 0
    std_pos = math.sqrt(relative_width ** 2 + relative_height ** 2)
    
    count = len(areas)
    return avg_area, std_pos, count


def normalize_metrics(clusters, image_height, image_width):
    """Normalize cluster metrics."""
    raw_metrics = [calculate_cluster_metrics(c, image_height, image_width) for c in clusters]
    raw_areas = [m[0] for m in raw_metrics]
    raw_positions = [m[1] for m in raw_metrics]
    raw_counts = [m[2] for m in raw_metrics]

    min_area = min(raw_areas) if raw_areas else 0
    max_area = max(raw_areas) if raw_areas else 1
    area_range = max_area - min_area if max_area != min_area else 1
    normalized_areas = [(a - min_area) / area_range for a in raw_areas]
    enhanced_areas = [a / config["normalization"]["area_enhancement_factor"] for a in normalized_areas]
    
    normalized_positions = [p / config["normalization"]["height_normalization_factor"] for p in raw_positions]

    min_count = min(raw_counts) if raw_counts else 0
    max_count = max(raw_counts) if raw_counts else 0
    delta_x = max_count - min_count
    y = 1.0
    mapped_counts = []
    
    if max_count > 0:
        EPS = 1e-12
        if delta_x > 0:
            while y * math.log(max(y + max_count, EPS)) <= (delta_x / 0.1) and y < 10000:
                y *= 1.1
        denom = math.log(max(y + max_count, EPS))
        if denom <= 0:
            denom = EPS
        for count in raw_counts:
            safe_count = max(0, min(count, max_count))
            num = math.log(max(y + safe_count, EPS))
            score = 1.0 - (num / denom)
            mapped_counts.append(score)
    else:
        mapped_counts = [0.5] * len(raw_counts)

    normalized_metrics = list(zip(enhanced_areas, normalized_positions, mapped_counts))
    return normalized_metrics



def calculate_downgrade_cost(cluster_id, current_model, next_model, 
                            cluster_precisions, cluster_id_to_index, models):
    """
    Downgrade cost = precision loss rate / delay gain rate.
    Smaller values are better downgrade candidates.
    """
    cluster_index = cluster_id_to_index[cluster_id]
    current_precision = cluster_precisions[cluster_index][current_model]
    next_precision = cluster_precisions[cluster_index][next_model]
    
    if current_precision > 0:
        precision_loss_rate = (current_precision - next_precision) / current_precision
    else:
        precision_loss_rate = 0.0
    
    current_delay = models[current_model]['delay']
    next_delay = models[next_model]['delay']
    if current_delay > 0:
        delay_gain_rate = (current_delay - next_delay) / current_delay
    else:
        delay_gain_rate = 0.0
    
    if delay_gain_rate <= 0:
        return float('inf')
    
    return precision_loss_rate / delay_gain_rate


def downgrade_cluster(cluster_id, from_model, to_model, 
                     selected_models, model_clusters, cluster_id_to_list_index):
    """Downgrade one cluster from from_model to to_model."""
    idx = cluster_id_to_list_index[cluster_id]
    cid, _, at_bottom = selected_models[idx]
    selected_models[idx] = (cid, to_model, at_bottom)
    
    model_clusters[from_model].discard(cluster_id)
    model_clusters[to_model].add(cluster_id)


def enforce_delay_constraint(selected_models, model_clusters, cluster_precisions, 
                            cluster_id_to_index, cluster_id_to_list_index, 
                            D_max, models, max_iterations=1000):
    """
    Enforce D_max using greedy downgrades.
    Each step chooses the cluster with the minimum downgrade cost.
    """
    model_downgrade_map = {'l': 'm', 'm': 's', 's': 'n', 'n': None}
    iteration = 0
    
    while iteration < max_iterations:
        model_delays = {m: len(model_clusters[m]) * models[m]['delay'] 
                       for m in model_clusters}
        max_delay = max(model_delays.values()) if model_delays else 0.0
        
        if max_delay <= D_max:
            break
        
        bottleneck_model = max(model_delays, key=model_delays.get)
        
        next_model = model_downgrade_map[bottleneck_model]
        if next_model is None:
            print(f"Warning: D_max={D_max}ms cannot be satisfied (current max delay={max_delay:.2f}ms)")
            print("Already downgraded to model n; continue with current allocation")
            break
        
        candidates = []
        for cluster_id in list(model_clusters[bottleneck_model]):
            cost = calculate_downgrade_cost(cluster_id, bottleneck_model, next_model,
                                          cluster_precisions, cluster_id_to_index, models)
            candidates.append((cost, cluster_id))
        
        if not candidates:
            break
        
        candidates.sort(key=lambda x: x[0])
        _, best_cluster_id = candidates[0]
        
        downgrade_cluster(best_cluster_id, bottleneck_model, next_model,
                         selected_models, model_clusters, cluster_id_to_list_index)
        
        iteration += 1
    
    return selected_models, model_clusters



def solve_offloading(clusters, models, D_max):
    """
    Main solver for model offloading optimization.

    Returns:
        opt_val: total precision
        selected_models: [(cluster_id, model, is_at_bottom), ...]
        max_inference_time: max inference delay
        N: total number of clusters
        model_total_delay: {'n': delay, 's': delay, 'm': delay, 'l': delay}
    """
    N = len(clusters)
    image_height = 2160
    image_width = 3840

    
    cluster_id_to_is_at_bottom = {}
    for cluster in clusters:
        bbox = cluster["bounding_box"]
        x_center = (bbox["x1"] + bbox["x2"]) / 2
        cluster_area = abs(bbox["x1"] - bbox["x2"]) * abs(bbox["y1"] - bbox["y2"])
        
        if 'y2' in bbox:
            bottom_edge = bbox['y2']
        elif 'center_y' in bbox and 'height' in bbox:
            bottom_edge = bbox['center_y'] + 0.5 * bbox['height']
        else:
            bottom_edge = 0
        
        is_at_bottom = 0
        
        if bottom_edge >= ((1 - edge[0]) * image_height):
            is_at_bottom = 1
        elif bottom_edge >= (1 - edge[1]) * image_height:
            if x_center >= (1 - edge[2]) * image_width or x_center <= edge[2] * image_width:
                is_at_bottom = 1
        
        avg_detection_area = sum(cluster["detection_areas"]) / len(cluster["detection_areas"])
        if avg_detection_area >= (1.0/12) * cluster_area:
            is_at_bottom = 2
        
        cluster_id_to_is_at_bottom[cluster["cluster_id"]] = is_at_bottom
    
    cluster_precisions = [compute_cluster_precision(c, models) for c in clusters]
    cluster_id_to_index = {cluster["cluster_id"]: i for i, cluster in enumerate(clusters)}
    
    
    normalized_metrics = normalize_metrics(clusters, image_height, image_width)
    
    scores = []
    for enhanced_area, normalized_position, mapped_count in normalized_metrics:
        score = enhanced_area + normalized_position + mapped_count
        scores.append(score)
    
    cluster_score_pairs = list(zip(clusters, scores))
    cluster_score_pairs.sort(key=lambda x: x[1])
    
    
    l_count = math.floor(D_max / models['l']['delay'])
    l_count = min(l_count, N)
    
    remaining = N - l_count
    
    if remaining > 0:
        m_count = math.floor(remaining * config['allocation']['m_model_ratio'])
        s_count = math.floor(remaining * config['allocation']['s_model_ratio'])
        n_count = remaining - m_count - s_count
    else:
        m_count = 0
        s_count = 0
        n_count = 0
    
    selected_models = []
    cluster_id_to_list_index = {}
    model_clusters = {"l": set(), "m": set(), "s": set(), "n": set()}
    
    for i, (cluster, score) in enumerate(cluster_score_pairs):
        cluster_id = cluster["cluster_id"]
        is_at_bottom = cluster_id_to_is_at_bottom[cluster_id]
        
        if i < l_count:
            sel_model = 'l'
        elif i < l_count + m_count:
            sel_model = 'm'
        elif i < l_count + m_count + s_count:
            sel_model = 's'
        else:
            sel_model = 'n'
        
        selected_models.append((cluster_id, sel_model, is_at_bottom))
        cluster_id_to_list_index[cluster_id] = len(selected_models) - 1
        model_clusters[sel_model].add(cluster_id)
    
    
    selected_models, model_clusters = enforce_delay_constraint(
        selected_models, model_clusters, cluster_precisions,
        cluster_id_to_index, cluster_id_to_list_index, D_max, models
    )
    
    
    def compute_model_total_delay_from_sets():
        return {m: models[m]['delay'] * len(model_clusters[m]) for m in model_clusters}
    
    def calculate_upgrade_benefit(cluster_id, current_model, next_model):
        """
        Upgrade benefit = precision gain rate / delay increase rate.
        Larger values are better upgrade candidates.
        """
        cluster_index = cluster_id_to_index[cluster_id]
        current_precision = cluster_precisions[cluster_index][current_model]
        next_precision = cluster_precisions[cluster_index][next_model]
        
        if current_precision > 0:
            precision_gain_rate = (next_precision - current_precision) / current_precision
        else:
            precision_gain_rate = next_precision
        
        current_delay = models[current_model]['delay']
        next_delay = models[next_model]['delay']
        if current_delay > 0:
            delay_increase_rate = (next_delay - current_delay) / current_delay
        else:
            delay_increase_rate = 0.0
        
        if delay_increase_rate <= 0:
            return float('inf')
        
        return precision_gain_rate / delay_increase_rate
    
    max_upgrade_iterations = 1000
    upgrade_iteration = 0
    
    while upgrade_iteration < max_upgrade_iterations:
        upgrade_candidates = []
        
        model_upgrade_map = {'n': 's', 's': 'm', 'm': 'l', 'l': None}
        
        for current_model in ['n', 's', 'm']:
            next_model = model_upgrade_map[current_model]
            if next_model is None:
                continue
            
            for cluster_id in model_clusters[current_model]:
                cluster_index = cluster_id_to_index[cluster_id]
                current_precision = cluster_precisions[cluster_index][current_model]
                next_precision = cluster_precisions[cluster_index][next_model]
                
                if next_precision > current_precision:
                    next_model_new_delay = (len(model_clusters[next_model]) + 1) * models[next_model]['delay']
                    
                    if next_model_new_delay <= D_max:
                        benefit = calculate_upgrade_benefit(cluster_id, current_model, next_model)
                        upgrade_candidates.append((benefit, cluster_id, current_model, next_model, next_model_new_delay))
        
        if not upgrade_candidates:
            break
        
        upgrade_candidates.sort(key=lambda x: x[0], reverse=True)
        best_benefit, best_cluster_id, from_model, to_model, new_delay = upgrade_candidates[0]
        
        idx = cluster_id_to_list_index[best_cluster_id]
        cid, _, at_bottom = selected_models[idx]
        selected_models[idx] = (cid, to_model, at_bottom)
        model_clusters[from_model].discard(best_cluster_id)
        model_clusters[to_model].add(best_cluster_id)
        
        upgrade_iteration += 1
    
    
    opt_val = sum(cluster_precisions[cluster_id_to_index[cid]][model] 
                  for cid, model, _ in selected_models)
    model_total_delay = compute_model_total_delay_from_sets()
    max_inference_time = max(model_total_delay.values()) if model_total_delay else 0.0
    
    print(f"Final allocation: l={len(model_clusters['l'])}, m={len(model_clusters['m'])}, "
          f"s={len(model_clusters['s'])}, n={len(model_clusters['n'])}")
    print(f"Max delay: {max_inference_time:.2f}ms (constraint: {D_max}ms)")
    return opt_val, selected_models, max_inference_time, N, model_total_delay



if __name__ == "__main__":
    directory = "data_new"
    results = {}
    D_max = 400
    
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            t0 = time.time()
            filepath = os.path.join(directory, filename)
            
            with open(filepath, 'r') as f:
                clusters = json.load(f)
            
            opt_val, offloading_plan, max_delay, num_clusters, model_total_delay = \
                solve_offloading(clusters, models, D_max)
            
            plan_dict = {cid: {"model": m, "is_at_bottom": at_bottom} 
                        for cid, m, at_bottom in offloading_plan}
            model_delays_output = {m: delay for m, delay in model_total_delay.items()}
            
            results[filename] = {
                "num_clusters": num_clusters,
                "optimal_total_precision": opt_val,
                "total_delay_ms": max_delay,
                "offloading_plan": plan_dict,
                "model_assignments": model_delays_output
            }
            
            print(f"Finished {filename}, offloading time: {(time.time() - t0)*1000:.3f}ms\n")
    
    output_file = "./data22/results_xiezai3.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"All results saved to {output_file}")

