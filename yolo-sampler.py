import cv2
import numpy as np
import os
import urllib.request

# Download YOLO files if they don't exist
weights_url = "https://pjreddie.com/media/files/yolov3.weights"
weights_path = "yolov3.weights"
config_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
config_path = "yolov3.cfg"
coco_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/coco.data"
coco_path = "coco.data"

if not os.path.exists(weights_path):
    urllib.request.urlretrieve(weights_url, weights_path)

if not os.path.exists(config_path):
    urllib.request.urlretrieve(config_url, config_path)

if not os.path.exists(coco_path):
    urllib.request.urlretrieve(coco_url, coco_path)

classes = []
with open(coco_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load YOLO
net = cv2.dnn.readNet(weights_path, config_path)

# Set up webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

frame_count = 0
sample_rate = 50

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % sample_rate == 0:
        # Detect objects using YOLO
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(net.getUnconnectedOutLayersNames())

        # Get bounding box coordinates and draw labels
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    width = int(detection[2] * frame.shape[1])
                    height = int(detection[3] * frame.shape[0])
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Apply non-max suppression to remove overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Draw bounding boxes and labels on the sampled frame
        for i in indices:
            # i = i[0]
            box = boxes[i]
            left, top, width, height = box
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the annotated frame
    cv2.imshow("Video Feed", frame)

# Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()