import os
import itertools
from PIL import Image


# Define size categories and their conditions
size_categories_list = [
    '0-8', '8-16', '16-24', '24-32', '32-48', '48-64',
    '64-80', '80-96', '96-128', '128-160', '160-196', '196-'
]

size_categories = {
    '0-8':    lambda area: area < 8*8,
    '8-16':   lambda area: 8*8 <= area < 16*16,
    '16-24':  lambda area: 16*16 <= area < 24*24,
    '24-32':  lambda area: 24*24 <= area < 32*32,
    '32-48':  lambda area: 32*32 <= area < 48*48,
    '48-64':  lambda area: 48*48 <= area < 64*64,
    '64-80':  lambda area: 64*64 <= area < 80*80,
    '80-96':  lambda area: 80*80 <= area < 96*96,
    '96-128': lambda area: 96*96 <= area < 128*128,
    '128-160':lambda area: 128*128 <= area < 160*160,
    '160-196':lambda area: 160*160 <= area < 196*196,
    '196-':   lambda area: area >= 196*196
}

# Define AP vectors for each network
ap_vectors = {
    'YOLOv8n': {
        '0-8':0.0,'8-16':0.000371,'16-24':0.00632,'24-32':0.0191,'32-48':0.0866,
        '48-64':0.165,'64-80':0.186,'80-96':0.236,'96-128':0.312,'128-160':0.407,
        '160-196':0.488,'196-':0.575
    },
    'YOLOv8s': {
        '0-8':0.0,'8-16':0.00429,'16-24':0.0161,'24-32':0.0545,'32-48':0.163,
        '48-64':0.277,'64-80':0.276,'80-96':0.294,'96-128':0.401,'128-160':0.458,
        '160-196':0.493,'196-':0.624
    },
    'YOLOv8m': {
        '0-8':0.0,'8-16':0.00483,'16-24':0.0423,'24-32':0.104,'32-48':0.239,
        '48-64':0.326,'64-80':0.326,'80-96':0.360,'96-128':0.453,'128-160':0.523,
        '160-196':0.527,'196-':0.639
    },
    'YOLOv8l': {
        '0-8':0.0,'8-16':0.00740,'16-24':0.0660,'24-32':0.142,'32-48':0.284,
        '48-64':0.393,'64-80':0.404,'80-96':0.422,'96-128':0.507,'128-160':0.564,
        '160-196':0.590,'196-':0.699
    }
}

# Define networks with input size, latency, and AP vector
networks = [
    {
        'name': 'YOLOv8n',
        'input_size': (640, 640),
        'latency': 37.8, # ms
        'ap_vector': ap_vectors['YOLOv8n']
    },
    {
        'name': 'YOLOv8s',
        'input_size': (768, 768),
        'latency': 45.3, # ms
        'ap_vector': ap_vectors['YOLOv8s']
    },
    {
        'name': 'YOLOv8m',
        'input_size': (896, 896),
        'latency': 89.0, # ms
        'ap_vector': ap_vectors['YOLOv8m']
    },
    {
        'name': 'YOLOv8l',
        'input_size': (1024, 1024),
        'latency': 153.8, # ms
        'ap_vector': ap_vectors['YOLOv8l']
    }
]

MAX_DEPTH = 2  # maximum depth for recursive partitioning

def compute_F_p(block_coords, objects, size_categories, network_input_size):
    """
    根据块在原图中的坐标，将块内的目标映射到网络输入大小的坐标系下，
    并计算该块在该网络输入大小下的目标分布 F_p。

    参数：
        block_coords (tuple): (xmin, ymin, xmax, ymax) 块的坐标（原图坐标）
        objects (list): [{'cx': float, 'cy': float, 'area': float}, ...] 的目标列表（原图坐标）
        size_categories (dict): {category_name: condition_func(area)}
        network_input_size (tuple): (N_w, N_h) 网络的输入大小，如(640,640)
    
    返回：
        F_p (dict): {category_name: proportion} 各尺寸类别目标比例
    """
    xmin, ymin, xmax, ymax = block_coords
    block_width = xmax - xmin
    block_height = ymax - ymin
    N_w, N_h = network_input_size

    # 分别计算x与y方向的缩放因子
    scale_x = N_w / block_width
    scale_y = N_h / block_height

    size_counts = {cat: 0 for cat in size_categories}
    total_objects = 0

    for obj in objects:
        cx, cy, area = obj['cx'], obj['cy'], obj['area']
        # 转换到块的局部坐标再缩放到网络输入坐标系
        scaled_x = (cx - xmin) * scale_x
        scaled_y = (cy - ymin) * scale_y
        scaled_area = area * scale_x * scale_y

        # 判断目标是否在网络输入坐标范围内
        if 0 <= scaled_x < N_w and 0 <= scaled_y < N_h:
            # 根据scaled_area分类尺寸类别
            for cat_name, condition in size_categories.items():
                if condition(scaled_area):
                    size_counts[cat_name] += 1
                    break
            total_objects += 1

    F_p = {}
    for cat_name in size_categories:
        if total_objects > 0:
            F_p[cat_name] = size_counts[cat_name]
        else:
            F_p[cat_name] = 0.0

    return F_p

def estimate_performance(network, F_p):
    """
    使用网络的 AP 向量和块内目标分布 F_p 计算估计检测精度和延迟。
    
    参数：
        network (dict): {
            'name': str,
            'input_size': (N_w, N_h),
            'latency': float,
            'ap_vector': {...}
        }
        F_p (dict): {category: probability} 块内目标分布
    
    返回：
        (eap, latency): (float, float)
        eap为估计检测精度
        latency为此网络推理的固定延迟(ms)
    """
    eap = 0.0
    for cat_name, p in F_p.items():
        ap_val = network['ap_vector'].get(cat_name, 0.0)
        eap += ap_val * p

    # 延迟恒定，因为输入分辨率固定
    latency = network['latency']
    return eap, latency

def generate_partition_plans(V, objects, networks, T, size_categories, depth=0, MAX_DEPTH=1):
    """
    增加剪枝逻辑：
    1. latency > T的计划不加入结果。
    2. 最终对plans进行延迟聚合，每个10ms延迟范围内只保留eap最高的计划。
    """
    xmin, ymin, xmax, ymax = V['coords']

    plans = []

    # 5个直接计划
    for network in networks:
        F_p = compute_F_p((xmin, ymin, xmax, ymax), objects, size_categories, network['input_size'])
        eap, latency = estimate_performance(network, F_p)
        if latency <= T:
            plan = {
                'partitions': [V],
                'network_assignments': [network['name']],
                'eap': eap,
                'latency': latency
            }
            plans.append(plan)

    # 若未达到最大深度则分块递归
    if depth < MAX_DEPTH:
        mid_x = (xmin + xmax) / 2
        mid_y = (ymin + ymax) / 2
        sub_blocks = [
            {'coords': (xmin, ymin, mid_x, mid_y)},
            {'coords': (mid_x, ymin, xmax, mid_y)},
            {'coords': (xmin, mid_y, mid_x, ymax)},
            {'coords': (mid_x, mid_y, xmax, ymax)}
        ]

        sub_plans_list = []
        for sub_V in sub_blocks:
            sxmin, symin, sxmax, symax = sub_V['coords']
            sub_objects = [obj for obj in objects if sxmin <= obj['cx'] < sxmax and symin <= obj['cy'] < symax]
            child_plans = generate_partition_plans(sub_V, sub_objects, networks, T, size_categories, depth+1, MAX_DEPTH)
            sub_plans_list.append(child_plans)

        # 笛卡尔积合成计划
        for combination in itertools.product(*sub_plans_list):
            total_eap = sum(cp['eap'] for cp in combination)
            total_latency = sum(cp['latency'] for cp in combination)
            if total_latency <= T:
                combined_partitions = []
                combined_network_assignments = []
                for cp in combination:
                    combined_partitions.extend(cp['partitions'])
                    combined_network_assignments.extend(cp['network_assignments'])

                combined_plan = {
                    'partitions': combined_partitions,
                    'network_assignments': combined_network_assignments,
                    'eap': total_eap,
                    'latency': total_latency
                }
                plans.append(combined_plan)

    # 对plans进行后处理：延迟聚合，只保留每个10ms范围内eap最高的计划
    # 步骤：
    # 1. 按 latency 升序排序
    # 2. 遍历plans，将延迟差<=10ms的计划分为一组，只保留eap最高的那一个
    plans.sort(key=lambda x: x['latency'])
    pruned_plans = []
    if plans:
        current_group_min_latency = plans[0]['latency']
        current_best_plan = plans[0]
        for p in plans[1:]:
            if p['latency'] - current_group_min_latency <= 10:
                # 同一延迟组内，只保留eap更高的那个
                if p['eap'] > current_best_plan['eap']:
                    current_best_plan = p
            else:
                # 切换到下一个延迟组
                pruned_plans.append(current_best_plan)
                current_group_min_latency = p['latency']
                current_best_plan = p
        # 别忘了最后一个组
        pruned_plans.append(current_best_plan)

    return pruned_plans


if __name__ == '__main__':
    # 定义图像和GT文件的目录
    image_dir = '/home/edge/work/Edge-Synergy/data/PANDA/images/train_all'       # 请替换为您的图像文件夹路径
    gt_boxes_dir = '/home/edge/work/Edge-Synergy/data/PANDA/labels/train_all'  # 请替换为您的GT文件夹路径
    scene = 'IMG_02_01'
    # breakpoint()
    scene_object_info = {}
    image_path = os.path.join(image_dir, f'{scene}.jpg')
    # GT 文件路径
    gt_boxes_path = os.path.join(gt_boxes_dir, f'{scene}.txt')

    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"图像文件不存在：{image_path}")
        # continue
    if not os.path.exists(gt_boxes_path):
        print(f"GT文件不存在：{gt_boxes_path}")
        # continue

    # 打开图像以获取宽度和高度
    with Image.open(image_path) as img:
        img_width, img_height = img.size

    # 读取 GT 检测框
    objects = []  # 存储每个目标的信息
    size_counts = {category: 0 for category in size_categories}
    total_objects = 0

    with open(gt_boxes_path, 'r') as f:
        for line in f:
            # 解析每一行，提取class, x, y, w, h
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(float(parts[0]))
                x_center_norm = float(parts[1])
                y_center_norm = float(parts[2])
                width_norm = float(parts[3])
                height_norm = float(parts[4])

                # 只处理class为0的目标
                if class_id != 0:
                    continue

                # 转换为像素坐标
                cx = x_center_norm * img_width
                cy = y_center_norm * img_height
                bbox_width = width_norm * img_width
                bbox_height = height_norm * img_height
                area = bbox_width * bbox_height

                # 存储目标信息
                objects.append({
                    'cx': cx,
                    'cy': cy,
                    'area': area
                })

                # 分类到对应的尺寸类别
                for category_name, condition in size_categories.items():
                    if condition(area):
                        size_counts[category_name] += 1
                        break

                total_objects += 1
    F_V = {}
    for category_name in size_categories:
        if total_objects > 0:
            F_V[category_name] = size_counts[category_name]
        else:
            F_V[category_name] = 0.0

    # 存储场景的目标信息和整体目标分布
    scene_object_info[scene] = {
        'image_size': (img_width, img_height),
        'objects': objects,
        'F_V': F_V
    }

    # 设定一个延迟预算 T
    T = 2000.0 # ms

    # 示例：对该场景进行自适应划分
    img_width, img_height = scene_object_info[scene]['image_size']
    objects = scene_object_info[scene]['objects']

    V = {
        'coords': (0, 0, img_width, img_height)
    }

    # 调用划分计划生成函数
    best_plans = generate_partition_plans(V, objects, networks, T, size_categories, 0, MAX_DEPTH)
    # breakpoint()
    if best_plans:
        best_plan = best_plans[-1]
        print(f"最佳划分计划：{best_plan}")
    else:
        print(f"环境 {scene} 没有符合延迟预算的划分计划。")

