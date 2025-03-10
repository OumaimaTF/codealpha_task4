
import numpy as np
from sort.sort import Sort

class ObjectTracker:
    def __init__(self):
        self.tracker = Sort()  # Initialize the SORT tracker

    def track_objects(self, detections):
        """
        Track objects based on detected bounding boxes.

        Parameters:
        - detections: List of tuples (x1, y1, x2, y2, confidence, class_id).

        Returns:
        - List of tracked objects (x1, y1, x2, y2, track_id).
        """
        if len(detections) == 0:
            detections_array = np.empty((0, 5))  # Empty array with 5 columns
        else:
            detections_array = np.array([det[:5] for det in detections])  # Ensure correct format

        # Debugging: Print the shape of the array
        print(f"[DEBUG] Detections Array Shape: {detections_array.shape}")

        # Ensure detections_array has the correct number of columns
        if detections_array.shape[1] != 5:
            raise ValueError(f"Incorrect detection format: Expected (N,5) but got {detections_array.shape}")

        track_ids = self.tracker.update(detections_array)

        # Convert to list format (bbox, track_id)
        tracked_objects = [(t[:4], int(t[4])) for t in track_ids]

        return tracked_objects
