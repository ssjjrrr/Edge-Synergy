#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline:
  1) Broadcast cache (mode 1)
  2) 2x2 coarse detection
  3) RL clustering
  4) Cluster-box concurrent inference (mode 2)
  5) NMS + visualization + label export
"""

import json
import traceback
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, wait
from pathlib import Path

import cv2
from stable_baselines3 import PPO

import config
import offloading as offloading_improvement
from client import MTYoloClient
from coarse_detector import CoarseDetector
from nms_utils import apply_nms
from rl_dca import cluster_from_mem
from vis_utils import draw_bbox_cv


IOU_TH = 0.45
COARSE_SCORE_SCALE = 0.7
D_MAX = 1260
MIN_CLUSTER_AREA = 40000


def _extract_dets_from_coarse(coarse_res):
    """
    Normalize different coarse-result formats into:
    [{'class_id': int, 'class_name': str, 'bbox': [x1,y1,x2,y2], 'score': float}, ...]
    """
    out = []
    if coarse_res is None:
        return out

    if (
        isinstance(coarse_res, dict)
        and "detections" in coarse_res
        and isinstance(coarse_res["detections"], list)
    ):
        det_iter = coarse_res["detections"]
    elif isinstance(coarse_res, dict):
        det_iter = []
        for _, lst in coarse_res.items():
            if isinstance(lst, list):
                det_iter.extend(lst)
    elif isinstance(coarse_res, list):
        det_iter = coarse_res
    else:
        return out

    for det in det_iter:
        if not det:
            continue
        bbox = det.get("bbox") or det.get("box") or det.get("bounding_box")
        if isinstance(bbox, dict):
            x1, y1, x2, y2 = (
                bbox.get("x1", 0),
                bbox.get("y1", 0),
                bbox.get("x2", 0),
                bbox.get("y2", 0),
            )
        elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
        else:
            continue

        out.append(
            {
                "class_id": det.get("class_id", 0),
                "class_name": det.get("class_name", str(det.get("class_id", 0))),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(det.get("score", 1.0)),
            }
        )
    return out


def _filter_small_clusters(clusters):
    """Drop clusters whose bounding-box area is below MIN_CLUSTER_AREA."""
    valid_clusters = []
    for cluster in clusters:
        box = cluster["bounding_box"]
        area = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
        if area >= MIN_CLUSTER_AREA:
            valid_clusters.append(cluster)
        else:
            print(f"Skip small cluster {cluster.get('cluster_id')}, area={area:.2f}")
    return valid_clusters


def _iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def process_one(img_p: Path, out_dir: Path, clients, model) -> None:
    """Process one image and write stitched/nms/label outputs."""
    _, client, client_n, client_s, client_m, client_l = clients

    img = cv2.imread(str(img_p))
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_p}")
    h, w = img.shape[:2]

    # 1) Broadcast full image cache.
    _, buf = cv2.imencode(".png", img)
    client.broadcast_cache(buf.tobytes())

    # 2) Coarse detection (in-memory).
    coarse_res = CoarseDetector(
        client_s,
        save_dir=out_dir / "coarse_vis",
        debug=False,
        device_name="client_s",
        model_name="s",
    )(img)
    coarse_dets = _extract_dets_from_coarse(coarse_res)

    # 3) RL clustering.
    clusters = cluster_from_mem(
        image=img,
        detections_dict=coarse_res,
        model=model,
        num_clusters_min=0,
        num_clusters_max=5,
    )

    if clusters:
        clustered_img = img.copy()
        for cluster in clusters:
            box = cluster["bounding_box"]
            cid = cluster["cluster_id"]
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            cv2.rectangle(clustered_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                clustered_img,
                f"Cluster {cid}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        cluster_vis_dir = out_dir / "cluster_vis"
        cluster_vis_dir.mkdir(parents=True, exist_ok=True)
        vis_path = cluster_vis_dir / f"{img_p.stem}_clusters.jpg"
        cv2.imwrite(str(vis_path), clustered_img)
        print(f"Saved cluster visualization to: {vis_path}")

    clusters = _filter_small_clusters(clusters)

    # 4) Offloading and model selection.
    _, selected_models, _, _, _ = offloading_improvement.solve_offloading(
        clusters,
        offloading_improvement.models,
        D_max=D_MAX,
    )

    # Expand cluster boxes by 10% before sub-inference.
    enlarge_scale = 1.10
    for cluster in clusters:
        box = cluster["bounding_box"]
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = (x2 - x1) * enlarge_scale
        bh = (y2 - y1) * enlarge_scale
        cluster["bounding_box"] = {
            "x1": max(0, cx - bw / 2.0),
            "y1": max(0, cy - bh / 2.0),
            "x2": min(w - 1.0, cx + bw / 2.0),
            "y2": min(h - 1.0, cy + bh / 2.0),
        }

    selected_models_dict = defaultdict(list)
    selected_optimize_dict = defaultdict(list)
    for key, value, optimize_open in selected_models:
        selected_models_dict[key].append(value)
        selected_optimize_dict[key].append(optimize_open)

    # Optional visualization for offloading assignment.
    if clusters:
        offload_img = img.copy()
        for cluster in clusters:
            box = cluster["bounding_box"]
            cid = cluster["cluster_id"]
            sel = selected_models_dict[cid][0] if cid in selected_models_dict else "default"
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            color_map = {
                "s": (255, 0, 0),
                "m": (0, 255, 0),
                "l": (0, 0, 255),
                "n": (255, 255, 0),
                "default": (128, 128, 128),
            }
            color = color_map.get(sel, (128, 128, 128))
            overlay = offload_img.copy()
            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
            cv2.addWeighted(overlay, 0.3, offload_img, 0.7, 0, offload_img)
            cv2.rectangle(offload_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(
                offload_img,
                f"C{cid}:{sel}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        offload_vis_dir = out_dir / "offload_vis"
        offload_vis_dir.mkdir(parents=True, exist_ok=True)
        offload_path = offload_vis_dir / f"{img_p.stem}_offload.jpg"
        cv2.imwrite(str(offload_path), offload_img)
        print(f"Saved offloading visualization to: {offload_path}")

    # 5) Concurrent shard inference.
    pendings = {}
    for cluster in clusters:
        box = cluster["bounding_box"]
        cid = cluster["cluster_id"]
        payload = json.dumps(box).encode()
        sel = selected_models_dict[cid][0] if cid in selected_models_dict else None
        optimize_open = selected_optimize_dict[cid][0]

        if sel == "s":
            fut = client_s.send_async(payload=payload, mode=2, opt=optimize_open)
        elif sel == "m":
            fut = client_m.send_async(payload=payload, mode=2, opt=optimize_open)
        elif sel == "l":
            fut = client_l.send_async(payload=payload, mode=2, opt=optimize_open)
        elif sel == "n":
            fut = client_n.send_async(payload=payload, mode=2, opt=optimize_open)
        else:
            fut = client.send_async(payload=payload, mode=2, opt=optimize_open)

        pendings[fut] = {"cluster_id": cid}

    merged = []
    stitched = img.copy()
    while pendings:
        done, _ = wait(pendings.keys(), return_when=FIRST_COMPLETED)
        for fut in done:
            res = fut.result()
            _ = pendings.pop(fut)
            for det in res["detections"]:
                merged.append(det)
                draw_bbox_cv(
                    stitched,
                    det["bbox"],
                    color=(0, 0, 255),
                    label=f'{det["class_name"]} {det["score"]:.2f}',
                )

    # 6) Merge coarse + fine and run final NMS.
    fine_yolo = [
        (
            d["class_id"],
            ((d["bbox"][0] + d["bbox"][2]) / 2) / w,
            ((d["bbox"][1] + d["bbox"][3]) / 2) / h,
            (d["bbox"][2] - d["bbox"][0]) / w,
            (d["bbox"][3] - d["bbox"][1]) / h,
            d["score"],
        )
        for d in merged
    ]
    fine_nms = apply_nms(fine_yolo, IOU_TH)

    fine_boxes = []
    for _, x, y, ww, hh, _ in fine_nms:
        fine_boxes.append(
            [
                (x - ww / 2) * w,
                (y - hh / 2) * h,
                (x + ww / 2) * w,
                (y + hh / 2) * h,
            ]
        )

    filtered_coarse = []
    for det in coarse_dets:
        keep = True
        for fb in fine_boxes:
            if _iou_xyxy(det["bbox"], fb) >= 0.3:
                keep = False
                break
        if keep:
            filtered_coarse.append(det)

    all_dets = []
    all_dets.extend(merged)
    all_dets.extend(
        [
            {
                "class_id": det["class_id"],
                "class_name": det.get("class_name", str(det["class_id"])),
                "bbox": det["bbox"],
                "score": float(det.get("score", 1.0)) * COARSE_SCORE_SCALE,
                "source": "coarse",
            }
            for det in filtered_coarse
        ]
    )

    yolo_fmt = [
        (
            det["class_id"],
            ((det["bbox"][0] + det["bbox"][2]) / 2) / w,
            ((det["bbox"][1] + det["bbox"][3]) / 2) / h,
            (det["bbox"][2] - det["bbox"][0]) / w,
            (det["bbox"][3] - det["bbox"][1]) / h,
            det["score"],
        )
        for det in all_dets
    ]
    merged_nms = apply_nms(yolo_fmt, IOU_TH)

    # 7) Save outputs.
    (out_dir / "stitched").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "stitched" / f"{img_p.stem}.jpg"), stitched)

    nms_vis_dir = out_dir / "nms_vis"
    nms_vis_dir.mkdir(parents=True, exist_ok=True)
    nms_img = img.copy()
    for c, x, y, ww, hh, s in merged_nms:
        x1 = int((x - ww / 2) * w)
        y1 = int((y - hh / 2) * h)
        x2 = int((x + ww / 2) * w)
        y2 = int((y + hh / 2) * h)
        draw_bbox_cv(nms_img, [x1, y1, x2, y2], color=(0, 255, 0), label=f"{c} {s:.2f}")

    nms_path = nms_vis_dir / f"{img_p.stem}_nms.jpg"
    cv2.imwrite(str(nms_path), nms_img)
    print(f"Saved post-NMS visualization to: {nms_path}")

    (out_dir / "label").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "label" / f"{img_p.stem}.txt", "w", encoding="utf-8") as f:
        for c, x, y, ww, hh, s in merged_nms:
            f.write(f"{c} {x} {y} {ww} {hh} {s}\n")


def main(
    path_in: str,
    out_dir: str = "final_output",
    recursive: bool = False,
    rl_model: str = "../checkpoints/ppo_rl_clustering.zip",
):
    """Run pipeline for one image or all images in a directory."""
    cfg = config.Config()
    client = MTYoloClient(cfg.servers)
    client_n = MTYoloClient(cfg.servers_n)
    client_s = MTYoloClient(cfg.servers_s)
    client_m = MTYoloClient(cfg.servers_m)
    client_l = MTYoloClient(cfg.servers_l)
    clients = (cfg, client, client_n, client_s, client_m, client_l)

    model = None
    if rl_model:
        p = Path(rl_model)
        if p.exists():
            model = PPO.load(p)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    in_p = Path(path_in)
    if in_p.is_dir():
        pattern = "**/*" if recursive else "*"
        img_list = [
            p
            for p in in_p.glob(pattern)
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        ]
        img_list.sort()
    else:
        img_list = [in_p]

    if not img_list:
        print(f"No image files found in {in_p}.")
        return

    total = len(img_list)
    print(f"Start processing {total} image(s). Output directory: {out_dir}")

    ok = 0
    fail = 0
    for idx, img_p in enumerate(img_list, 1):
        try:
            process_one(img_p, out_dir, clients, model)
            ok += 1
            print(f"[{idx}/{total}] OK {img_p.name}")
        except Exception as exc:
            fail += 1
            print(f"[{idx}/{total}] FAIL {img_p}: {exc}")
            traceback.print_exc()

    print(f"Completed: success={ok}, failed={fail}")
    print("Outputs:")
    print(f"  - Visualization: {out_dir / 'stitched'}")
    print(f"  - Labels: {out_dir / 'label'}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Image path or directory path.")
    ap.add_argument("--out", default="final_output", help="Output directory.")
    ap.add_argument("--recursive", action="store_true", help="Recursively scan subdirectories.")
    args = ap.parse_args()
    main(args.path, args.out, args.recursive)
