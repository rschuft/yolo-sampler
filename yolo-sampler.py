import cv2
import time
from ultralytics import YOLO

# Load the YOLOv8 model (use a pre-trained model, e.g., yolov8n, yolov8x, etc.)
model = YOLO('yolov8x.pt')

# Open the webcam (0 is the default camera, change if you have multiple cameras)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Variables to store the last detection time and the bounding boxes
last_detection_time = 0
detection_interval = 0.25  # 250 ms interval between detections

# Variable to store previous detections (bounding boxes and labels)
prev_boxes = []  # Format: [(x1, y1, x2, y2, label, confidence, track_id), ...]

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
        # Perform object detection
        # results = model(frame)

        # Perform object tracking instead of just detection
        results = model.track(source=frame, persist=True, stream=True, verbose=False)

        # Extract bounding boxes, labels, and confidence scores
        prev_boxes = []
        for r in results:
            for rbox in r.boxes:
                box = rbox.xyxy[0]  # Bounding box in (x1, y1, x2, y2) format
                label = model.names[int(rbox.cls)]  # Object label
                confidence = float(rbox.conf)  # Confidence score
                track_id = int(rbox.id) if hasattr(rbox, 'id') else None  # Unique tracking ID (if available)
                prev_boxes.append((box[0], box[1], box[2], box[3], label, confidence, track_id))

        # Update the last detection time
        last_detection_time = current_time

    # Overlay previous bounding boxes on the current frame
    for (x1, y1, x2, y2, label, confidence, track_id) in prev_boxes:
        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # Put label, confidence, and tracking ID
        label_text = f"{label} {confidence:.2f} {track_id}"
        cv2.putText(frame, label_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the current frame with the overlaid bounding boxes
    cv2.imshow('YOLOv8 Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
