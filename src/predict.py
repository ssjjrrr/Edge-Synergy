from ultralytics import YOLO
from config import *
import time


model_x = YOLO("checkpoints/yolov8l_1024_ep300.pt")
start_time = time.time()
results = model_x(source=f"{image_dir}/val", imgsz=1024, conf=0.2, iou=0.7, save_txt=True, save_conf=True, max_det=1000)
process_time = time.time() - start_time
process_time /= 78
print(f"Processing time per image: {process_time * 1000} mini seconds")
