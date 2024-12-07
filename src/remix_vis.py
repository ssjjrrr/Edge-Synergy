import os
from PIL import Image, ImageDraw

# Load the provided image
image_path = '/home/edge/work/Edge-Synergy/data/PANDA/images/train_all/IMG_01_01.jpg'
image = Image.open(image_path).convert("RGBA")

# Define network-to-color mapping
network_color_map = {
    'YOLOv8n': (255, 0, 0, 64),  # Red with transparency
    'YOLOv8s': (0, 255, 0, 64),  # Green with transparency
    'YOLOv8m': (0, 0, 255, 64),  # Blue with transparency
    'YOLOv8l': (255, 255, 0, 64),  # Yellow with transparency
    'YOLOv8x': (127, 255, 127, 64),  # Magenta with transparency
}

# The network assignments and partitions (example data from the screenshot)
network_assignments = [
    'YOLOv8n', 'YOLOv8n', 'YOLOv8n', 'YOLOv8s', 'YOLOv8l', 'YOLOv8s',
    'YOLOv8m', 'YOLOv8n', 'YOLOv8x', 'YOLOv8m', 'YOLOv8n', 'YOLOv8n',
    'YOLOv8n', 'YOLOv8n', 'YOLOv8n', 'YOLOv8m', 'YOLOv8n', 'YOLOv8s',
    'YOLOv8x', 'YOLOv8m', 'YOLOv8s', 'YOLOv8n', 'YOLOv8s', 'YOLOv8n', 'YOLOv8m'
]

partitions = [{'coords': (0, 0, 960.0, 540.0)},
                 {'coords': (960.0, 0, 1920.0, 540.0)},
                 {'coords': (0, 540.0, 480.0, 810.0)},
                 {'coords': (480.0, 540.0, 960.0, 810.0)},
                 {'coords': (0, 810.0, 480.0, 1080.0)},
                 {'coords': (480.0, 810.0, 960.0, 1080.0)},
                 {'coords': (960.0, 540.0, 1440.0, 810.0)},
                 {'coords': (1440.0, 540.0, 1920.0, 810.0)},
                 {'coords': (960.0, 810.0, 1440.0, 1080.0)},
                 {'coords': (1440.0, 810.0, 1920.0, 1080.0)},
                 {'coords': (1920.0, 0, 2880.0, 540.0)},
                 {'coords': (2880.0, 0, 3840, 540.0)},
                 {'coords': (1920.0, 540.0, 2400.0, 810.0)},
                 {'coords': (2400.0, 540.0, 2880.0, 810.0)},
                 {'coords': (1920.0, 810.0, 2400.0, 1080.0)},
                 {'coords': (2400.0, 810.0, 2880.0, 1080.0)},
                 {'coords': (2880.0, 540.0, 3360.0, 810.0)},
                 {'coords': (3360.0, 540.0, 3840, 810.0)},
                 {'coords': (2880.0, 810.0, 3360.0, 1080.0)},
                 {'coords': (3360.0, 810.0, 3840, 1080.0)},
                 {'coords': (0, 1080.0, 960.0, 1620.0)},
                 {'coords': (960.0, 1080.0, 1920.0, 1620.0)},
                 {'coords': (0, 1620.0, 960.0, 2160)},
                 {'coords': (960.0, 1620.0, 1920.0, 2160)},
                 {'coords': (1920.0, 1080.0, 3840, 2160)}]

# Create a transparent overlay
overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
draw = ImageDraw.Draw(overlay)

# Draw each partition with the corresponding color
for partition, network in zip(partitions, network_assignments):
    color = network_color_map.get(network, (255, 255, 255, 128))  # Default to white if network not in map
    draw.rectangle(partition['coords'], fill=color)

# Combine the original image with the overlay
result_image = Image.alpha_composite(image, overlay)

# Save and display the result
os.makedirs('remix', exist_ok=True)
result_path = 'remix/image_with_overlay.png'
result_image.save(result_path)
result_path
