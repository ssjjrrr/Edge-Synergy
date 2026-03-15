#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coarse detector based on 2x2 tiled inference.
"""

from pathlib import Path
from typing import Dict, List, Union

import cv2
import json

from client import MTYoloClient
from vis_utils import draw_bbox_cv


class CoarseDetector:
    _global_counter = 0

    def __init__(
        self,
        client: MTYoloClient,
        save_dir: Union[Path, str] = "coarse_vis",
        debug: bool = False,
        device_name: str = "coarse_client",
        model_name: str = "coarse_model",
    ):
        """
        debug=False returns in-memory results only.
        debug=True saves per-tile visualization and coarse JSON.
        """
        self.client = client
        self.save_dir = Path(save_dir)
        self.debug = debug
        self.device_name = device_name
        self.model_name = model_name
        if self.debug:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, img_in: Union[str, Path, "np.ndarray"]):
        """
        Args:
            img_in: image path or loaded ndarray.

        Returns:
            dict when debug=False, or saved JSON path when debug=True with file input.
        """
        if isinstance(img_in, (str, Path)):
            img_path = Path(img_in)
            img = cv2.imread(str(img_path))
            stem = img_path.stem
        else:
            img_path = None
            img = img_in
            stem = "in_mem"

        h, w = img.shape[:2]
        h2, w2 = h // 2, w // 2
        overlap = int(min(h, w) * 0.05)

        tiles = [
            (max(0, -overlap), max(0, -overlap), min(w, w2 + overlap), min(h, h2 + overlap)),
            (max(0, w2 - overlap), max(0, -overlap), min(w, w + overlap), min(h, h2 + overlap)),
            (max(0, -overlap), max(0, h2 - overlap), min(w, w2 + overlap), min(h, h + overlap)),
            (max(0, w2 - overlap), max(0, h2 - overlap), min(w, w + overlap), min(h, h + overlap)),
        ]

        futures = []
        for shard_id, xyxy in enumerate(tiles):
            payload = json.dumps(list(xyxy)).encode()
            fut = self.client.send_async(payload=payload, mode=2)
            futures.append((shard_id, xyxy, fut))

        detections: List[Dict] = []
        for shard_id, (x1, y1, x2, y2), fut in futures:
            _ = shard_id
            res = fut.result()
            vis_patch = img[y1:y2, x1:x2].copy()

            for det in res.get("detections", []):
                l, t, r, b = det["bbox"]
                draw_bbox_cv(
                    vis_patch,
                    (l - x1, t - y1, r - x1, b - y1),
                    color=(0, 255, 255),
                    label=f'{det["class_name"]} {det["score"]:.2f}',
                )
                detections.append(det)

            if self.debug:
                CoarseDetector._global_counter += 1
                seq = CoarseDetector._global_counter
                out_p = self.save_dir / f"{stem}_{seq}_{y1}_{x1}.jpg"
                cv2.imwrite(str(out_p), vis_patch)

        result = {
            "width": w,
            "height": h,
            "detections": detections,
        }

        if self.debug and img_path is not None:
            json_out = self.save_dir / f"{stem}_coarse.json"
            json_out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
            return json_out

        return result
