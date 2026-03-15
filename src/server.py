#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8 + ZeroMQ server (dual mode).

Mode 1:
  Request: <image-bytes>, b"1"
  Action: cache full image
  Reply: {"mode": 1, "detections": []}

Mode 2:
  Request: <json box [x1,y1,x2,y2]>, b"2", optional b"<opt>"
  Action: run detection on region from cached image
  Reply: {"mode": 2, "detections": [...], "time": <seconds>}
"""

import io
import json
from time import perf_counter
from typing import Tuple

import zmq
from PIL import Image, ImageOps
from ultralytics import YOLO

import config


cfg = config.Config()
model = YOLO(cfg.model[cfg.server_index])
model.to("cuda:0")
imgsz = cfg.imgsz[cfg.server_index]

ctx = zmq.Context()
sock = ctx.socket(zmq.REP)
sock.bind(cfg.servers[cfg.server_index])
print("Server started, waiting for requests...")

stored_image = None


def _parse_box(box) -> Tuple[int, int, int, int]:
    """Accept list or dict and normalize to (x1, y1, x2, y2)."""
    if isinstance(box, list) and len(box) == 4:
        return tuple(map(int, box))
    if isinstance(box, dict):
        return int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
    raise ValueError("Expect box [x1,y1,x2,y2] or {x1,y1,x2,y2}")


while True:
    send_t = perf_counter()
    try:
        parts = sock.recv_multipart()
        if len(parts) == 2:
            payload, mode_b = parts
            opt = 0
        elif len(parts) == 3:
            payload, mode_b, opt = parts
        else:
            raise ValueError(f"unexpected multipart length: {len(parts)}")

        mode = int(mode_b.decode().strip())

        if mode == 1:
            stored_image = Image.open(io.BytesIO(payload)).convert("RGB")
            sock.send_json({"mode": 1, "detections": []})
            continue

        if mode != 2:
            raise ValueError("Mode must be 1 or 2")

        if stored_image is None:
            raise RuntimeError("No image cached; please send mode 1 first")

        box_raw = json.loads(payload.decode())
        x1, y1, x2, y2 = _parse_box(box_raw)
        crop = stored_image.crop((x1, y1, x2, y2))

        shift_x = x1
        shift_y = y1

        if isinstance(opt, bytes):
            try:
                opt = int(opt.decode().strip())
            except Exception:
                opt = 0
        else:
            opt = int(opt)

        if opt == 1:
            w0 = x2 - x1
            h0 = y2 - y1
            pad_x = int(w0 * 0.3)
            pad_y = int(h0 * 0.3)
            crop = ImageOps.expand(crop, border=(pad_x, pad_y), fill=(255, 255, 255))
            shift_x = x1 - pad_x
            shift_y = y1 - pad_y
            results = model.predict(crop, conf=0.25, imgsz=imgsz, augment=True)
        elif opt == 2:
            target = 640
            w0, h0 = crop.size
            if w0 >= target or h0 >= target:
                left = top = right = bottom = 0
            else:
                left = (target - w0) // 2
                right = target - w0 - left
                top = (target - h0) // 2
                bottom = target - h0 - top
            crop = ImageOps.expand(crop, border=(left, top, right, bottom), fill=(255, 255, 255))
            shift_x = x1 - left
            shift_y = y1 - top
            results = model.predict(crop, conf=0.15, imgsz=target, augment=True)
        else:
            results = model.predict(crop, conf=0.25, imgsz=imgsz)

        dets = []
        for box in results[0].boxes:
            bx = box.xyxy[0].tolist()
            bbox_global = [bx[0] + shift_x, bx[1] + shift_y, bx[2] + shift_x, bx[3] + shift_y]
            dets.append(
                {
                    "bbox": [round(v, 2) for v in bbox_global],
                    "score": round(float(box.conf[0]), 3),
                    "class_id": int(box.cls[0]),
                    "class_name": model.names[int(box.cls[0])],
                }
            )

        sock.send_json({"mode": 2, "detections": dets, "time": perf_counter() - send_t})
    except Exception as exc:
        sock.send_json({"error": str(exc), "detections": [], "time": perf_counter() - send_t})
