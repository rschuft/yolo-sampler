# Local imports
from YoloSampler import config, track

# Attempt to load config from the YAML file with the same name as this script
config = config.load(__file__)
device_index = config['device_index'] or 0  # Capture device index
width = config['width'] or 3840  # Capture device frame width
height = config['height'] or 2160  # Capture device frame height
detection_interval = config['detection_interval'] or 0.25  # Delay between detection attempts in seconds
max_track_length = config['max_track_length'] or 30  # Maximum number of points to keep in the track history
tracking_model_name = config['tracking_model_name'] or 'yolov8x.pt' # YOLO model name
frame_label = config['frame_label'] or 'Object Tracking'  # Label for the window frame
annotation_color = config['annotation_color'] or (0, 255, 0)  # Bounding box and label color
path_color = config['path_color'] or (230, 230, 230)  # Tracking path color

# Main entry point
if __name__ == "__main__":
    track(tracking_model_name, device_index, width, height, detection_interval, max_track_length, frame_label, annotation_color, path_color)
