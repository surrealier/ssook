"""Render detection boxes onto source videos using JSON inference results."""
import os
import json
import cv2
from pathlib import Path

RESULTS_DIR = r"D:\bongkj\Projects\Visualizer\inference_results"
OUTPUT_DIR = r"D:\bongkj\Projects\Visualizer\inference_videos"
COLORS = [(0, 255, 0), (0, 0, 255), (255, 165, 0), (255, 0, 255), (0, 255, 255)]


def draw_boxes(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cid = det["class_id"]
        color = COLORS[cid % len(COLORS)]
        label = f'{det["class_name"]} {det["score"]:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


def process_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    video_path = data["video"]
    if not os.path.exists(video_path):
        print(f"  SKIP (video not found): {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  SKIP (cannot open): {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model_name = data["model"]
    json_stem = Path(json_path).stem
    out_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{json_stem}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frames_data = {fr["frame_idx"]: fr["detections"] for fr in data["frames"]}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in frames_data:
            draw_boxes(frame, frames_data[frame_idx])
        # Add model name overlay
        cv2.putText(frame, model_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"  → {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_files = []
    for root, _, files in os.walk(RESULTS_DIR):
        for f in files:
            if f.endswith(".json"):
                json_files.append(os.path.join(root, f))

    print(f"Found {len(json_files)} JSON results")
    for jf in sorted(json_files):
        print(f"Processing: {Path(jf).name}")
        try:
            process_json(jf)
        except Exception as e:
            print(f"  ERROR: {e}")
    print("Done!")


if __name__ == "__main__":
    main()
