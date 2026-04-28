"""/api/viewer/* 라우터."""
import os
import time
import uuid

import cv2
import numpy as np
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.app_config import AppConfig
from core.inference import run_inference, run_classification, run_segmentation
from server.model_manager import ensure_model
from server.state import executor
from server.utils import imread, get_color, draw_label, generate_palette, overlay_segmentation

router = APIRouter()

# ── Video streaming (MJPEG) ─────────────────────────────
_video_sessions = {}  # session_id -> dict with state
_SESSION_TIMEOUT = 300  # 5분 무활동 시 세션 자동 정리
_SESSION_TIMEOUT_PLAYING = 600  # playing 세션도 10분 무응답 시 정리


def _cleanup_stale_sessions():
    """타임아웃된 비디오 세션 정리 (playing 세션 포함)"""
    now = time.time()
    stale = [sid for sid, s in _video_sessions.items()
             if (not s.get("playing") and now - s.get("last_access", 0) > _SESSION_TIMEOUT)
             or (s.get("playing") and now - s.get("last_access", 0) > _SESSION_TIMEOUT_PLAYING)]
    for sid in stale:
        sess = _video_sessions.pop(sid, None)
        if sess:
            sess["playing"] = False
            try:
                sess["cap"].release()
            except Exception:
                pass
            sess["last_frame"] = None
            sess["last_result"] = None
            print(f"[Session] Cleaned up stale session: {sid}")


class VideoStartRequest(BaseModel):
    model_path: str
    video_path: str
    conf: float = 0.25
    stream_max_height: int = 0  # 0=원본, 720/480 등 지정 시 다운스케일
    tracker_type: str = "none"  # none / bytetrack / sort


@router.post("/api/viewer/start")
async def viewer_start(req: VideoStartRequest):
    """Start a video inference session, returns session_id."""
    try:
        cfg = AppConfig()
        model = ensure_model(req.model_path, cfg.model_type, cfg)

        if model.task_type == "embedding":
            return {"error": "CLIP/Embedder 모델은 뷰어에서 사용할 수 없습니다."}

        cap = cv2.VideoCapture(req.video_path)
        if not cap.isOpened():
            return {"error": "Cannot open video"}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        sid = str(uuid.uuid4())[:8]
        tracker = None
        if req.tracker_type and req.tracker_type != "none":
            from core.tracking import create_tracker
            tracker = create_tracker(req.tracker_type)
        _video_sessions[sid] = {
            "cap": cap, "model": model, "conf": req.conf,
            "playing": True, "paused": False,
            "fps": fps, "speed": 1.0,
            "total": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_idx": 0, "last_detections": 0, "last_infer_ms": 0,
            "last_frame": None, "last_result": None,
            "seek_to": None, "step_request": None,
            "video_path": req.video_path,
            "last_access": time.time(),
            "stream_max_height": req.stream_max_height,
            "tracker": tracker,
        }
        return {"session_id": sid, "fps": fps,
                "total_frames": _video_sessions[sid]["total"]}
    except Exception as e:
        return {"error": str(e)}


@router.get("/api/viewer/stream/{session_id}")
async def viewer_stream(session_id: str):
    """MJPEG stream of inference results."""
    sess = _video_sessions.get(session_id)
    if not sess:
        return {"error": "Invalid session"}

    def generate():
        from core.inference import run_inference, run_classification, run_segmentation, run_embedding
        from core.app_config import AppConfig
        cap = sess["cap"]
        model = sess["model"]
        names = model.names or {}
        cfg = AppConfig()  # 루프 밖에서 한 번만 참조
        max_h = sess.get("stream_max_height", 0)

        try:
            while sess.get("playing", False):
                # Handle seek
                seek_to = sess.get("seek_to")
                if seek_to is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, seek_to)
                    sess["seek_to"] = None

                # Handle step
                step = sess.get("step_request")
                if step is not None:
                    cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cur + step))
                    sess["step_request"] = None

                # Handle pause
                if sess.get("paused", False):
                    time.sleep(0.05)
                    continue

                speed = sess.get("speed", 1.0)
                skip = max(0, int(speed) - 1)
                target_delay = 1.0 / sess["fps"]
                t0 = time.time()
                ret, frame = cap.read()
                if not ret:
                    sess["playing"] = False
                    break
                # frame skip
                for _ in range(skip):
                    r2, _ = cap.read()
                    if not r2:
                        break

                sess["frame_idx"] = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                if model.task_type == "classification":
                    result = run_classification(model, frame)
                    sess["last_detections"] = 0
                    sess["last_infer_ms"] = round(result.infer_ms, 2)
                    sess["last_frame"] = frame.copy()
                    sess["last_result"] = None
                    vis = frame
                    y = 30
                    for cid, conf in result.top_k[:5]:
                        label = f"{names.get(cid, str(cid))}: {conf:.3f}"
                        cv2.putText(vis, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        y += 30
                    _, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 65])
                elif model.task_type == "segmentation":
                    result = run_segmentation(model, frame)
                    sess["last_detections"] = 0
                    sess["last_infer_ms"] = round(result.infer_ms, 2)
                    sess["last_frame"] = frame.copy()
                    sess["last_result"] = None
                    vis = overlay_segmentation(frame, result.mask)
                    _, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 65])
                elif model.model_type.startswith("pose_"):
                    from core.inference import run_pose, COCO_SKELETON
                    pose_res = run_pose(model, frame, cfg.conf_threshold)
                    sess["last_detections"] = len(pose_res.boxes)
                    sess["last_infer_ms"] = round(pose_res.infer_ms, 2)
                    sess["last_frame"] = frame.copy()
                    sess["last_result"] = None
                    vis = frame
                    for i in range(len(pose_res.boxes)):
                        x1, y1, x2, y2 = pose_res.boxes[i].astype(int)
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        kpts = pose_res.keypoints[i]
                        for kx, ky, kc in kpts:
                            if kc > 0.5:
                                cv2.circle(vis, (int(kx), int(ky)), 3, (0, 0, 255), -1)
                        for a, b in COCO_SKELETON:
                            if kpts[a][2] > 0.5 and kpts[b][2] > 0.5:
                                cv2.line(vis, (int(kpts[a][0]), int(kpts[a][1])),
                                         (int(kpts[b][0]), int(kpts[b][1])), (255, 255, 0), 2)
                    _, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 65])
                elif model.model_type.startswith("instseg_"):
                    from core.inference import run_instance_seg
                    iseg_res = run_instance_seg(model, frame, cfg.conf_threshold)
                    sess["last_detections"] = len(iseg_res.boxes)
                    sess["last_infer_ms"] = round(iseg_res.infer_ms, 2)
                    sess["last_frame"] = frame.copy()
                    sess["last_result"] = None
                    vis = frame.copy()
                    palette = [(int(hash(str(i)*3)%200+55), int(hash(str(i)*7)%200+55), int(hash(str(i)*11)%200+55)) for i in range(max(len(iseg_res.masks), 1))]
                    for i, mask in enumerate(iseg_res.masks):
                        color = palette[i % len(palette)]
                        overlay = vis.copy()
                        overlay[mask > 0] = color
                        cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)
                        x1, y1, x2, y2 = iseg_res.boxes[i].astype(int)
                        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    _, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 65])
                else:
                    result = run_inference(model, frame, cfg.conf_threshold)
                    sess["last_detections"] = len(result.boxes)
                    sess["last_infer_ms"] = round(result.infer_ms, 2)
                    sess["last_frame"] = frame.copy()
                    sess["last_result"] = result

                    # Tracker integration
                    tracker = sess.get("tracker")
                    tracks = None
                    if tracker and len(result.boxes) > 0:
                        import numpy as _np
                        tracks = tracker.update(
                            _np.array(result.boxes), _np.array(result.scores), _np.array(result.class_ids)
                        )

                    thickness = cfg.box_thickness
                    label_size = cfg.label_size
                    total_cls = len(names)

                    if tracks:
                        for tr in tracks:
                            cid_int = int(tr.class_id)
                            style = cfg.get_class_style(cid_int)
                            if not style.enabled:
                                continue
                            x1, y1, x2, y2 = map(int, tr.box)
                            t_val = style.thickness or thickness
                            color = get_color(style, cid_int, total_cls)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, t_val)
                            parts = [f"ID:{tr.id}"]
                            if cfg.show_labels:
                                parts.append(names.get(cid_int, str(cid_int)))
                            if cfg.show_confidence:
                                parts.append(f"{tr.score:.2f}")
                            draw_label(frame, " ".join(parts), x1, y1, color, label_size, max(1, t_val - 1), cfg.show_label_bg)
                            # Draw trajectory
                            if len(tr.trajectory) > 1:
                                for j in range(1, len(tr.trajectory)):
                                    p1 = (int(tr.trajectory[j-1][0]), int(tr.trajectory[j-1][1]))
                                    p2 = (int(tr.trajectory[j][0]), int(tr.trajectory[j][1]))
                                    cv2.line(frame, p1, p2, color, max(1, t_val - 1))
                    else:
                        for box, score, cid in zip(result.boxes, result.scores, result.class_ids):
                            cid_int = int(cid)
                            style = cfg.get_class_style(cid_int)
                            if not style.enabled:
                                continue
                            x1, y1, x2, y2 = map(int, box)
                            t_val = style.thickness or thickness
                            color = get_color(style, cid_int, total_cls)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, t_val)
                            parts = []
                            if cfg.show_labels:
                                parts.append(names.get(cid_int, str(cid_int)))
                            if cfg.show_confidence:
                                parts.append(f"{score:.2f}")
                            if parts:
                                draw_label(frame, " ".join(parts), x1, y1, color, label_size, max(1, t_val - 1), cfg.show_label_bg)
                    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
                # 스트리밍 다운스케일 (노트북 최적화)
                if max_h > 0 and buf is not None:
                    out_frame = vis if model.task_type in ("classification", "segmentation") else frame
                    oh, ow = out_frame.shape[:2]
                    if oh > max_h:
                        scale = max_h / oh
                        small = cv2.resize(out_frame, (int(ow * scale), max_h), interpolation=cv2.INTER_AREA)
                        _, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 65])
                sess["last_display_jpeg"] = buf.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                       sess["last_display_jpeg"] + b'\r\n')

                elapsed = time.time() - t0
                remaining = target_delay - elapsed
                if remaining > 0:
                    time.sleep(remaining)
                elif remaining < -target_delay:
                    # 추론이 2프레임 이상 지연 시 프레임 스킵
                    extra_skip = int(-remaining / target_delay)
                    for _ in range(extra_skip):
                        r2, _ = cap.read()
                        if not r2:
                            break
        except Exception as exc:
            import traceback
            print(f"[MJPEG ERROR] {exc}")
            traceback.print_exc()
            sess["playing"] = False

    return StreamingResponse(generate(),
                             media_type='multipart/x-mixed-replace; boundary=frame')


@router.get("/api/viewer/status/{session_id}")
async def viewer_status(session_id: str):
    _cleanup_stale_sessions()
    sess = _video_sessions.get(session_id)
    if not sess:
        return {"error": "Invalid session"}
    sess["last_access"] = time.time()
    return {
        "playing": sess["playing"],
        "paused": sess.get("paused", False),
        "frame_idx": sess["frame_idx"],
        "total": sess["total"],
        "detections": sess["last_detections"],
        "infer_ms": sess["last_infer_ms"],
        "speed": sess.get("speed", 1.0),
    }


@router.post("/api/viewer/stop/{session_id}")
async def viewer_stop(session_id: str):
    sess = _video_sessions.pop(session_id, None)
    if sess:
        sess["playing"] = False
        sess["cap"].release()
    return {"ok": True}


@router.post("/api/viewer/pause/{session_id}")
async def viewer_pause(session_id: str):
    sess = _video_sessions.get(session_id)
    if sess:
        sess["paused"] = not sess.get("paused", False)
    return {"paused": sess.get("paused", False) if sess else False}


class SeekRequest(BaseModel):
    frame: int


@router.post("/api/viewer/seek/{session_id}")
async def viewer_seek(session_id: str, req: SeekRequest):
    sess = _video_sessions.get(session_id)
    if sess:
        sess["seek_to"] = req.frame
    return {"ok": True}


class SpeedRequest(BaseModel):
    speed: float


@router.post("/api/viewer/speed/{session_id}")
async def viewer_speed(session_id: str, req: SpeedRequest):
    sess = _video_sessions.get(session_id)
    if sess:
        sess["speed"] = req.speed
    return {"ok": True}


class StepRequest(BaseModel):
    delta: int = 1


@router.post("/api/viewer/step/{session_id}")
async def viewer_step(session_id: str, req: StepRequest):
    sess = _video_sessions.get(session_id)
    if sess:
        sess["step_request"] = req.delta
    return {"ok": True}


@router.post("/api/viewer/snapshot/{session_id}")
async def viewer_snapshot(session_id: str):
    sess = _video_sessions.get(session_id)
    if not sess:
        return {"error": "No session"}
    os.makedirs("snapshots", exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("snapshots", f"snapshot_{ts}.jpg")
    # 디스플레이된 화면 그대로 저장
    jpeg_data = sess.get("last_display_jpeg")
    if jpeg_data:
        with open(path, "wb") as f:
            f.write(jpeg_data)
    elif sess.get("last_frame") is not None:
        cv2.imwrite(path, sess["last_frame"])
    else:
        return {"error": "No frame available"}
    return {"ok": True, "path": path}


@router.post("/api/viewer/save-crops/{session_id}")
async def viewer_save_crops(session_id: str):
    """Save cropped detection boxes from the last frame."""
    sess = _video_sessions.get(session_id)
    if not sess:
        return {"error": "No session"}
    frame = sess.get("last_frame")
    result = sess.get("last_result")
    if frame is None or result is None or len(result.boxes) == 0:
        return {"error": "No detections available"}
    names = (sess.get("model") and sess["model"].names) or {}
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("snapshots", f"crops_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    h, w = frame.shape[:2]
    saved = 0
    for i, (box, score, cid) in enumerate(zip(result.boxes, result.scores, result.class_ids)):
        x1, y1, x2, y2 = max(0, int(box[0])), max(0, int(box[1])), min(w, int(box[2])), min(h, int(box[3]))
        if x2 <= x1 or y2 <= y1:
            continue
        cls_name = names.get(int(cid), str(int(cid)))
        crop = frame[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(out_dir, f"{i:03d}_{cls_name}_{score:.2f}.jpg"), crop)
        saved += 1
    return {"ok": True, "path": out_dir, "count": saved}

