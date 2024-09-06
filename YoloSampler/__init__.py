__all__ = [
    'overlay_tracking_details',
    'open_webcam',
    'read_frame',
    'record_object_tracking',
    'release_capture',
    'track_frame',
    'track_video_stream',
    'track'
]

# System imports
import time
from collections import defaultdict

# Third-party imports
import cv2
import numpy as np
from ultralytics import YOLO

# Overlay last recorded bounding boxes on the current frame and the path of the object
def overlay_tracking_details(frame, prev_boxes, track_history, annotation_color = (0, 255, 0), path_color = (230, 230, 230)):
    for (x1, y1, x2, y2, label, confidence, track_id) in prev_boxes:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), annotation_color, 2)
        label_text = f"{label} {confidence:.2f} {track_id}"
        cv2.putText(frame, label_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, annotation_color, 2)
        if track_id is not None:
            points = [np.hstack(track_history[track_id]).astype(np.int32).reshape((-1, 1, 2))]  # for later: can we move np.int32 into the hstack call?
            cv2.polylines(frame, points, isClosed=False, color=path_color, thickness=7)

# Read tracked info out of rbox and store it in prev_boxes and track_history
def record_object_tracking(labels, prev_boxes, track_history, rbox, max_track_length = 30):
    box = rbox.xyxy[0]  # Bounding box in (x1, y1, x2, y2) format
    x, y, w, h = box # decompose the bounding box to get the center for plotting track history
    label = labels[int(rbox.cls)]  # Object label
    confidence = float(rbox.conf)  # Confidence score
    track_id = int(rbox.id) if hasattr(rbox, 'id') else None  # Unique tracking ID (if available)
    if track_id is not None:
        track = track_history[track_id]
        track.append((float(x/2 + w/2), float(y/2 + h/2)))  # x, y center point
        if len(track) > max_track_length:  # Keep only the last N points
            track.pop(0)
    prev_boxes.append((x, y, w, h, label, confidence, track_id))

# Open the webcam and set the frame width and height
def open_webcam(index = 0, width = 3840, height = 2160):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise Exception('Error: Could not open webcam.')
    return cap

# Release the capture and close windows
def release_capture(cap):
    cap.release()
    cv2.destroyAllWindows()

# Read a frame from the capture
def read_frame(cap):
    ret, frame = cap.read()
    if not ret:
        release_capture(cap)
        raise Exception('Error: Failed to grab frame.')
    return frame

# Track objects in the frame
def track_frame(tracking_model, frame, prev_boxes, track_history, max_track_length = 30):
    results = tracking_model.track(source=frame, persist=True, stream=True, verbose=False)
    for r in results:
        for rbox in r.boxes:
            record_object_tracking(tracking_model.names, prev_boxes, track_history, rbox, max_track_length)

# Track objects in the video stream, annotate the frame, and display the video
def track_video_stream(cap, tracking_model, detection_interval = 0.25, max_track_length = 30, frame_label = 'Object Tracking', annotation_color = (0, 255, 0), path_color = (230, 230, 230)):
    last_detection_time = 0  # Updated with the current time after each detection
    prev_boxes = []  # Format: [(x1, y1, x2, y2, label, confidence, track_id), ...]
    track_history = defaultdict(lambda: [])  # Format: {track_id: [(x1, y1), (x2, y2), ...]}
    while cap.isOpened():
        frame = read_frame(cap)  # Capture frame-by-frame
        # Run object tracking every N ms
        current_time = time.time()
        if current_time - last_detection_time >= detection_interval:
            prev_boxes = []
            track_frame(tracking_model, frame, prev_boxes, track_history, max_track_length)
            last_detection_time = current_time
        # Overlay previous bounding boxes on the current frame
        overlay_tracking_details(frame, prev_boxes, track_history, annotation_color, path_color)
        cv2.imshow(frame_label, frame)
        # Break the loop if the 'ESC' key is pressed
        if cv2.waitKey(1) == 27:  # 27 is the ASCII value for the 'ESC' key
            break
    release_capture(cap)

# Load YOLO model, open webcam and run object detection
def track(tracking_model_name = 'yolov8x.pt', device_index = 0, width = 3840, height = 2160, detection_interval = 0.25, max_track_length = 30, frame_label = 'Object Tracking', annotation_color = (0, 255, 0), path_color = (230, 230, 230)):
    tracking_model = YOLO(tracking_model_name)
    cap = open_webcam(device_index, width, height)
    track_video_stream(cap, tracking_model, detection_interval, max_track_length, frame_label, annotation_color, path_color)

# Main entry point
if __name__ == "__main__":
    track()
