from ultralytics import YOLO
from config import *


model_x = YOLO("checkpoints/yolov8x_1280_ep300.pt")
results = model_x(source=f"{image_dir}/val", imgsz=1280, conf=0.1, iou=0.7, save_txt=True, save_conf=True, max_det=1000)
