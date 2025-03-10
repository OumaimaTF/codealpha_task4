import cv2
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path='models/yolov8n.pt'):
        self.model = YOLO(model_path)  # Load YOLO model

    def detect_objects(self, frame):
        # Perform object detection
        results = self.model(frame)
        detections = []
        
        # Extract bounding boxes and labels
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                label = int(box.cls[0])  # Object class ID
                detections.append([x1, y1, x2, y2, conf])
        return detections
