import cv2


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from detector import YOLODetector
from tracker import ObjectTracker
from utils import draw_bboxes




def main(video_source=0):
    cap = cv2.VideoCapture(video_source)
    
    detector = YOLODetector()  # Initialize YOLO detector
    tracker = ObjectTracker()  # Initialize object tracker

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect_objects(frame)  # Perform detection
        track_ids = tracker.track_objects(detections)  # Track objects

        # Draw bounding boxes and track IDs
        frame = draw_bboxes(frame, track_ids, detector.model.names)
        
        cv2.imshow("Object Tracking", frame)  # Display frame
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
