# System imports
import time
from collections import defaultdict

# Third-party imports
import cv2
import numpy as np
from ultralytics import YOLO
import yaml

# Load configuration from config.yml
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)
    device_index = config['device_index']  # Capture device index
    width = config['width']  # Capture device frame width
    height = config['height']  # Capture device frame height
    detection_interval = config['detection_interval']  # Delay between detection attempts in seconds
    max_track_length = config['max_track_length']  # Maximum number of points to keep in the track history
    tracking_model_name = config['tracking_model_name']  # YOLO model name

def draw_bounding_boxes(frame, prev_boxes, track_history):
    for (x1, y1, x2, y2, label, confidence, track_id) in prev_boxes:
        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # Put label, confidence, and tracking ID
        label_text = f"{label} {confidence:.2f} {track_id}"
        cv2.putText(frame, label_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Draw the tracking lines
        track = track_history[track_id]
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=7)

def record_object_tracking(model, prev_boxes, track_history, max_track_length, rbox):
    box = rbox.xyxy[0]  # Bounding box in (x1, y1, x2, y2) format
    x, y, w, h = box # decompose the bounding box to get the center for plotting track history
    label = model.names[int(rbox.cls)]  # Object label
    confidence = float(rbox.conf)  # Confidence score
    track_id = int(rbox.id) if hasattr(rbox, 'id') else None  # Unique tracking ID (if available)
    if track_id is not None:
        track = track_history[track_id]
        track.append((float(x/2 + w/2), float(y/2 + h/2)))  # x, y center point
        if len(track) > max_track_length:  # Keep only the last N points
            track.pop(0)
    prev_boxes.append((x, y, w, h, label, confidence, track_id))

def open_webcam(index, width, height):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit(-1)
    return cap

def run_object_detection(cap, tracking_model, detection_interval, max_track_length):
    # Initialize the last detection time
    last_detection_time = 0  # Updated with the current time after each detection
    # Variable to store previous detections (bounding boxes and labels)
    prev_boxes = []  # Format: [(x1, y1, x2, y2, label, confidence, track_id), ...]
    # Store the track history so we can plot the trajectory of the object
    track_history = defaultdict(lambda: [])  # Format: {track_id: [(x1, y1), (x2, y2), ...]}

    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Get current time
        current_time = time.time()

        # Run detection every N ms
        if current_time - last_detection_time >= detection_interval:
            # Perform object tracking on the current frame
            results = tracking_model.track(source=frame, persist=True, stream=True, verbose=False)

            # Extract bounding boxes, labels, confidence scores, and tracking identifiers (if available)
            prev_boxes = []
            for r in results:
                for rbox in r.boxes:
                    record_object_tracking(tracking_model, prev_boxes, track_history, max_track_length, rbox)

            # Update the last detection time
            last_detection_time = current_time

        # Overlay previous bounding boxes on the current frame
        for (x1, y1, x2, y2, label, confidence, track_id) in prev_boxes:
            draw_bounding_boxes(frame, prev_boxes, track_history)

        # Display the current frame with the overlaid bounding boxes
        cv2.imshow('Object Detection', frame)

        # Break the loop if the 'ESC' key is pressed
        if cv2.waitKey(1) == 27:  # 27 is the ASCII value for the 'ESC' key
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

# Load the YOLO model
tracking_model = YOLO(tracking_model_name)

# Open the webcam
cap = open_webcam(device_index, width, height)

# Track detected objects and show the results
run_object_detection(cap, tracking_model, detection_interval, max_track_length)
