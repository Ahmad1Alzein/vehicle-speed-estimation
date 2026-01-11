import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# --------------------
# Load models
# --------------------
yolo_model = YOLO("./models/yolov8s/weights/best.pt")

tracker = DeepSort(
    max_age=30,
    n_init=3,
    max_iou_distance=0.7
) # Initialize DeepSORT tracker with specified parameters : max_age (that means the maximum number of frames a track can be without being confirmed), n_init (number of frames to initialize a track), max_iou_distance (maximum IoU distance between tracks)

# --------------------
# Video input/output
# --------------------
video_path = "C:/Users/Hayssam/Downloads/COURSES/A1 M2 AI/Computer Vision/Project/sample_video.mp4"
cap = cv2.VideoCapture(video_path) #VideoCapture object to read video file into memory, cap is the variable name used to reference this object 

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) #to get frames per second of the video (the speed at which the video plays)

out = cv2.VideoWriter( #output video file 
    "tracked_output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

# --------------------
# Main loop
# --------------------
while cap.isOpened(): #while the video file is successfully opened
    ret, frame = cap.read() #reads the next frame from the video file and stores it in the variable frame
    if not ret:
        break

    # YOLO inference
    results = yolo_model(frame, conf=0.4)[0] #perform object detection on the current frame using the YOLO model with a confidence threshold of 0.4

    detections = []

    for box in results.boxes: #iterates over each detected bounding box in the results
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy() #extracts the coordinates of the bounding box, the coordinates of the top-left corner (x1, y1) and the bottom-right corner (x2, y2)
        conf = float(box.conf[0]) #extracts the confidence score associated with the bounding box
        cls = int(box.cls[0]) #extracts the class ID of the detected object

        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls)) #append the bounding box coordinates, confidence score, and class ID to the detections list

    # DeepSORT update
    tracks = tracker.update_tracks(detections, frame=frame) #update the DeepSORT tracker with the new detections for the current frame

    for track in tracks: #iterates over each track in the tracks list
        if not track.is_confirmed(): #checks if the track is confirmed
            continue

        track_id = track.track_id #retrieves the unique ID assigned to the track
        l, t, w_box, h_box = map(int, track.to_ltrb()) #retrieves the bounding box coordinates of the track in the format (left, top, right, bottom) and converts them to integers

        cv2.rectangle(frame, (l, t), (w_box, h_box), (0, 255, 0), 2) #draws a rectangle around the tracked object on the frame using the bounding box coordinates
        cv2.putText( 
            frame,
            f"ID {track_id}",
            (l, t - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    out.write(frame)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()


# python "C:/Users/Hayssam/Downloads/COURSES/A1 M2 AI/Computer Vision/Project/DeepSORT.py"