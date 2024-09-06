# System imports
import time
from collections import defaultdict

# Third-party imports
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model (use a pre-trained model, e.g., yolov8n, yolov8x, etc.)
model = YOLO('yolov8x.pt')

# Open the webcam 
index = 0 # capture device index (0 is the default camera, change if you have multiple cameras)
cap = cv2.VideoCapture(index)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# 4k resolution/high resolution (this will reduce to what the camera is capable of automatically)
width = 3840
height = 2160
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
# width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# _, frame = cap.read()
# height, width = frame.shape[:2]

# Variables to store the last detection time and the bounding boxes
last_detection_time = 0
detection_interval = 0.25  # 250 ms interval between detections

# Variable to store previous detections (bounding boxes and labels)
prev_boxes = []  # Format: [(x1, y1, x2, y2, label, confidence, track_id), ...]

# Store the track history so we can plot the trajectory of the object
track_history = defaultdict(lambda: [])
max_track_length = 90  # Maximum number of points to store in the track history

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Get current time
    current_time = time.time()

    # Run detection every 100ms
    if current_time - last_detection_time >= detection_interval:
        # Perform object tracking on the current frame
        results = model.track(source=frame, persist=True, stream=True, verbose=False)

        # Extract bounding boxes, labels, confidence scores, and tracking identifiers (if available)
        prev_boxes = []
        for r in results:
            for rbox in r.boxes:
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

        # Update the last detection time
        last_detection_time = current_time

    # Overlay previous bounding boxes on the current frame
    for (x1, y1, x2, y2, label, confidence, track_id) in prev_boxes:
        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # Put label, confidence, and tracking ID
        label_text = f"{label} {confidence:.2f} {track_id}"
        cv2.putText(frame, label_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Draw the tracking lines
        track = track_history[track_id]
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

    # Display the current frame with the overlaid bounding boxes
    cv2.imshow('YOLOv8 Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
