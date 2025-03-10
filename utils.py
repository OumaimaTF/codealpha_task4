import cv2

def draw_bboxes(frame, tracked_objects, class_names=None):
    """
    Draw bounding boxes and track IDs on the frame.

    Parameters:
    - frame: The current frame from the video stream.
    - tracked_objects: List of tuples containing bounding boxes and tracking information.
    - class_names: List of class names (optional).

    Returns:
    - The frame with drawn bounding boxes.
    """
    for obj in tracked_objects:
        if len(obj) == 2:  # Format: (bbox, track_id)
            bbox, track_id = obj
            class_id = None  # No class ID available
        elif len(obj) == 3:  # Format: (bbox, track_id, class_id)
            bbox, track_id, class_id = obj
        else:
            print(f"Warning: Unexpected object format {obj}")
            continue  # Skip incorrectly formatted objects

        # Convert bbox values to integers
        x1, y1, x2, y2 = map(int, bbox)

        # Define label
        label = f"ID: {track_id}"
        if class_id is not None and class_names is not None and 0 <= class_id < len(class_names):
            label = f"{class_names[class_id]} {label}"

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame
