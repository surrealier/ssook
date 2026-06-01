"""Batch video inference for 4 models on test set.
Saves per-frame detection results as JSON for each video×model combination.
"""
import os
import sys
import json
import time
import glob
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from core.model_loader import load_model
from core.inference import run_inference

# ── Config ──────────────────────────────────────────────
TEST_DIR = r"D:\bongkj\Projects\Embedder\source\video\testset"
OUTPUT_DIR = r"D:\bongkj\Projects\Visualizer\inference_results"
MODELS = [
    ("seq_anydetection_best_dynamic_m_960", "Models/seq_anydetection_best_dynamic_m_960.onnx", "seq_yolo"),
    ("SOREST_v1.12.5", "Models/SOREST_v1.12.5_100boxes_meta_opset17.onnx", "yolo"),
    ("seq_rfdetr_exp2_952", "Models/seq_rfdetr_exp2_952.onnx", "seq_rfdetr"),
    ("seq_dinov3_v7_960", "Models/seq_dinov3_v7_960.onnx", "seq_dinov3"),
]
CONF = 0.25
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mpv4", ".mkv"}


def find_videos(root):
    videos = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if Path(f).suffix.lower() in VIDEO_EXTS:
                videos.append(os.path.join(dirpath, f))
    return sorted(videos)


def run_video(model_info, video_path, conf):
    """Run inference on all frames, return list of per-frame results."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    results = []
    frame_idx = 0

    # Reset sequential buffers
    model_info._frame_buffer = []
    model_info._seq_tensor_buf = []
    model_info._seq_frame_counter = 0
    model_info._seq_last_result = None

    t_start = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        res = run_inference(model_info, frame, conf)
        detections = []
        for box, score, cid in zip(res.boxes, res.scores, res.class_ids):
            detections.append({
                "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                "score": float(score),
                "class_id": int(cid),
                "class_name": model_info.names.get(int(cid), str(int(cid))),
            })
        results.append({
            "frame_idx": frame_idx,
            "detections": detections,
            "infer_ms": round(res.infer_ms, 2),
        })
        frame_idx += 1

    cap.release()
    elapsed = time.perf_counter() - t_start
    avg_fps = frame_idx / elapsed if elapsed > 0 else 0
    print(f"    {frame_idx} frames, {avg_fps:.1f} FPS, {elapsed:.1f}s")
    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    videos = find_videos(TEST_DIR)
    print(f"Found {len(videos)} videos:")
    for v in videos:
        print(f"  {v}")
    print()

    for model_name, model_path, model_type in MODELS:
        print(f"═══ Loading: {model_name} ({model_type}) ═══")
        model_info = load_model(model_path, model_type=model_type)
        if model_info.session is None:
            print(f"  FAILED to load model")
            continue
        print(f"  input_size={model_info.input_size}, classes={len(model_info.names)}")

        for video_path in videos:
            video_name = Path(video_path).stem
            rel_dir = Path(video_path).parent.name  # True/False
            print(f"  ▶ {rel_dir}/{video_name}")

            results = run_video(model_info, video_path, CONF)

            # Save results
            out_dir = os.path.join(OUTPUT_DIR, model_name)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{rel_dir}_{video_name}.json")
            summary = {
                "model": model_name,
                "model_type": model_type,
                "video": video_path,
                "conf_threshold": CONF,
                "total_frames": len(results),
                "total_detections": sum(len(r["detections"]) for r in results),
                "frames": results,
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"    → Saved: {out_path}")
        print()

    print("Done!")


if __name__ == "__main__":
    main()
