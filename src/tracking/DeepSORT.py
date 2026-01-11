import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

yolo_model = YOLO("./models/yolov8s/weights/best.pt")

tracker = DeepSort(max_age=15, n_init=2, max_iou_distance=0.6)
#max_age=25: How many frames a track is kept alive without a detection
# n_init=3: Number of consecutive detections before the track is confirmed
# max_iou_distance=0.6: Maximum IOU distance for association,I will match a detection to an existing track only if they overlap by at least 40%. (IoU distance = 1 − IoU) 
video_path = r"C:\Users\ahmad\Downloads\testingVideo1.mp4"
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

# Window setup (resizable + scalable)
win = "Tracking"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win, 1280, 720)  # initial size

names = yolo_model.names

# Speed knobs (CPU)
INFER_WIDTH = 640        # It is the width (in pixels) of the image that you give to YOLO for detection.

last_tracks = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    h0, w0 = frame.shape[:2]
    scale = INFER_WIDTH / w0
    infer_frame = cv2.resize(frame, (INFER_WIDTH, int(h0 * scale)))

    # YOLO inference (CPU)
    results = yolo_model.predict(
        source=infer_frame,
        conf=0.3,
        imgsz=INFER_WIDTH,
        device="cpu",
        verbose=False
    )[0]

    detections = []
    if results.boxes is not None and len(results.boxes) > 0:
        xyxy = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        clss = results.boxes.cls.cpu().numpy().astype(int)

        # Scale boxes back to original frame size
        inv = 1.0 / scale
        for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clss):
            x1, y1, x2, y2 = x1 * inv, y1 * inv, x2 * inv, y2 * inv
            detections.append(([float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                                float(conf),
                                int(cls)))

    last_tracks = tracker.update_tracks(detections, frame=frame)

    # Draw last known tracks (even on skipped frames)
    for track in last_tracks:
        if not track.is_confirmed():
            continue
        if track.time_since_update > 1:
            continue
        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())

        cls_id = getattr(track, "det_class", None)
        cls_name = names.get(int(cls_id), str(cls_id)) if cls_id is not None else "obj"

        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(frame, f"{cls_name}:{track_id}", (l, max(0, t - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # --- Fill the maximized window ---
    # Get current window size (works on most OpenCV builds)
    try:
        _, _, win_w, win_h = cv2.getWindowImageRect(win)
        disp = cv2.resize(frame, (win_w, win_h))
    except:
        # fallback (no dynamic scaling)
        disp = frame

    cv2.imshow(win, disp)

    if cv2.waitKey(1) & 0xFF == ord("q"): #means that if we press q the video will stop
        break

cap.release()
cv2.destroyAllWindows()
