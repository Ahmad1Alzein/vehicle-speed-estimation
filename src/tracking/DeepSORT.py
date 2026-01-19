# src/pipelines/deepsort.py
"""
Pipeline: YOLO + DeepSORT tracking + speed estimation.

Fixes included:
- ✅ VIDEO time: t_now = frame_idx / fps (correct even if processing is slow)
- ✅ DeepSORT detections TLWH [x, y, w, h] (prevents huge boxes)
- ✅ Crossing point = bottom-center with offset (stable for vehicles)
- ✅ Speed shown ONLY after both crossings (t1 and t2 exist)
- ✅ Overlay shows t1, t2, dt in seconds with 2 decimals (for validation)
- ✅ Debug status shows which lines were detected: [L1], [L1,L2], or [...]

Uses your existing modules:
- src.speed.geometry.Line
- src.speed.speed_estimator.SpeedEstimator
- src.speed.rules.get_max_speed_kmh, color_for_speed
"""

from __future__ import annotations

import os
from typing import Dict, List

import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

from src.speed.geometry import Line
from src.speed.speed_estimator import SpeedEstimator
from src.speed.rules import get_max_speed_kmh, color_for_speed


def process_video(
    video_path: str,
    output_video_path: str,
    model_path: str,
    line1: Line,
    line2: Line,
    distance_m: float,
    rules: Dict[str, float],
    conf: float = 0.3,
    infer_width: int = 640,
    device: str = "cpu",
    tracker_max_age: int = 30,
    tracker_n_init: int = 2,
    tracker_max_iou_distance: float = 0.6,
    crossing_point_bottom_offset: int = 10,
    draw_crossing_point: bool = True,
) -> List[Dict]:
    """
    Args:
        video_path: input video path
        output_video_path: output mp4 path
        model_path: YOLO weights path
        line1, line2: Line objects (geometry.py uses a/b)
        distance_m: real-world distance between the two lines (meters)
        rules: dict {class_name: max_speed_kmh}
        conf: YOLO confidence threshold
        infer_width: resize width for inference speed
        device: "cpu" or "0" for GPU
        tracker_*: DeepSORT parameters
        crossing_point_bottom_offset: use bottom center minus this offset (pixels)
        draw_crossing_point: draw a small dot at the point used for crossing detection

    Returns:
        rows: list of dicts for CSV export (one row per track once speed computed)
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load YOLO model
    yolo_model = YOLO(model_path)
    names = yolo_model.names  # class_id -> class_name

    # DeepSORT tracker
    tracker = DeepSort(
        max_age=tracker_max_age,
        n_init=tracker_n_init,
        max_iou_distance=tracker_max_iou_distance,
    )

    # Speed estimator (your module)
    estimator = SpeedEstimator(line1=line1, line2=line2, distance_m=distance_m)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, float(fps), (w0, h0))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open VideoWriter: {output_video_path}")

    rows: List[Dict] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # ✅ video time (seconds) independent of processing speed
        t_now = frame_idx / fps

        # Resize for inference speed
        scale = infer_width / w0
        infer_frame = cv2.resize(frame, (infer_width, int(h0 * scale)))

        # YOLO inference
        results = yolo_model.predict(
            source=infer_frame,
            conf=conf,
            imgsz=infer_width,
            device=device,
            verbose=False
        )[0]

        # Build detections for DeepSORT
        detections = []
        if results.boxes is not None and len(results.boxes) > 0:
            xyxy = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy().astype(int)

            inv = 1.0 / scale
            for (x1, y1, x2, y2), c, cls_id in zip(xyxy, confs, clss):
                # Scale back to original frame coords
                x1, y1, x2, y2 = x1 * inv, y1 * inv, x2 * inv, y2 * inv

                # ✅ DeepSORT expects TLWH: [x, y, w, h]
                w = x2 - x1
                h = y2 - y1
                if w <= 1 or h <= 1:
                    continue

                detections.append(([float(x1), float(y1), float(w), float(h)], float(c), int(cls_id)))

        # Update tracker
        tracks = tracker.update_tracks(detections, frame=frame)

        # Draw the two lines
        cv2.line(frame, line1.a, line1.b, (0, 255, 255), 3)
        cv2.line(frame, line2.a, line2.b, (255, 0, 255), 3)

        for tr in tracks:
            if not tr.is_confirmed():
                continue
            if tr.time_since_update > 1:
                continue

            track_id = int(tr.track_id)
            l, t, r, b = map(int, tr.to_ltrb())

            cls_id = getattr(tr, "det_class", None)
            cls_name = names.get(int(cls_id), str(cls_id)) if cls_id is not None else "obj"

            # ✅ point used for crossing detection (bottom-center - offset)
            cx = (l + r) // 2
            cy = max(0, b - crossing_point_bottom_offset)

            if draw_crossing_point:
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            # Update speed estimator with VIDEO time
            st_state = estimator.update(
                track_id=track_id,
                centroid=(cx, cy),
                t_now=t_now,
                label=cls_name,
            )

            max_kmh = get_max_speed_kmh(st_state.label, rules, default_max=60.0)
            col = color_for_speed(st_state.speed_kmh, max_kmh)

            # Draw bbox
            cv2.rectangle(frame, (l, t), (r, b), col, 2)

            # Debug status: which lines were detected?
            status = []
            if st_state.t1 is not None:
                status.append("L1")
            if st_state.t2 is not None:
                status.append("L2")
            status_txt = ",".join(status) if status else "..."

            # ✅ Overlay times with 2 decimals
            if st_state.t1 is not None and st_state.t2 is not None and st_state.speed_kmh is not None:
                dt = st_state.t2 - st_state.t1
                text = (
                    f"{st_state.label}:{track_id} "
                    f"{st_state.speed_kmh:.1f} km/h | "
                    f"t1={st_state.t1:.2f}s "
                    f"t2={st_state.t2:.2f}s "
                    f"dt={dt:.2f}s "
                    f"(max {max_kmh:.0f}) [{status_txt}]"
                )
            else:
                t1_txt = f"{st_state.t1:.2f}s" if st_state.t1 is not None else "--"
                t2_txt = f"{st_state.t2:.2f}s" if st_state.t2 is not None else "--"
                text = (
                    f"{st_state.label}:{track_id} | "
                    f"t1={t1_txt} t2={t2_txt} [{status_txt}]"
                )

            cv2.putText(
                frame,
                text,
                (l, max(20, t - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                col,
                2
            )

            # Log once when speed becomes available
            if st_state.speed_kmh is not None and not st_state.logged:
                rows.append({
                    "track_id": track_id,
                    "label": st_state.label,
                    "speed_kmh": float(st_state.speed_kmh),
                    "max_kmh": float(max_kmh),
                    "overspeed": bool(st_state.speed_kmh > max_kmh),
                    "t1": float(st_state.t1) if st_state.t1 is not None else None,
                    "t2": float(st_state.t2) if st_state.t2 is not None else None,
                    "dt_sec": float(st_state.t2 - st_state.t1) if (st_state.t1 is not None and st_state.t2 is not None) else None,
                    "distance_m": float(distance_m),
                    "fps": float(fps),
                })
                st_state.logged = True

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    return rows
