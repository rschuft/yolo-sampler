import cv2
import time
from ultralytics import YOLO

# Load the YOLOv8 model (use a pre-trained model, e.g., yolov8n)
model = YOLO('yolov8x.pt')

# Open the webcam (0 is the default camera, change if you have multiple cameras)
cap = cv2.VideoCapture(0)

# Initialize variables
detection_time = 0.0
prev_annotated_frame = None  # To store the last annotated frame

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Start measuring time before detection
    start_time = time.time()

    # Perform detection if it's time for it
    if detection_time <= 0:
        # Run the YOLO model to detect objects in the frame
        results = model(frame)

        # Annotate detections on the frame
        prev_annotated_frame = results[0].plot()  # Save the annotated frame

        # Stop measuring time after detection
        detection_time = time.time() - start_time

        # Wait for 2x detection time before next detection
        throttle_delay = 2 * detection_time
    else:
        # Skip detection, use previous annotated frame
        annotated_frame = prev_annotated_frame

        # Display the resulting frame with previous annotations
        cv2.imshow('YOLOv8 Object Detection (Throttled)', annotated_frame)

        # Decrease detection_time counter
        detection_time -= 1 / 30  # Assuming ~30 FPS for the webcam

    # Display the current frame (annotated with previous detection)
    if prev_annotated_frame is not None:
        cv2.imshow('YOLOv8 Object Detection (Throttled)', prev_annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Sleep for a short time to simulate frame rate (can be adjusted)
    time.sleep(1 / 30)

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
