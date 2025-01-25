import cv2
import numpy as np

# Use OpenCV's ArUco submodule
# Make sure opencv-contrib-python is installed.
# Example: pip install opencv-contrib-python
from cv2 import aruco

class DetectedObject:
    def __init__(self, contour):
        self.contour = contour
        self.area = cv2.contourArea(contour)

def main():
    # Open the camera (using DirectShow on Windows for example)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # ArUco Dictionary (you can change to DICT_6X6_250, etc. if desired)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    aruco_params = aruco.DetectorParameters()

    # Adjust these thresholds for your object detection
    MIN_AREA = 500
    MAX_AREA = 100000

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- (1) Detect Normal Objects Using Contours ---

        # Convert to grayscale and blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge detection
        edged = cv2.Canny(blurred, 50, 150)

        # Morphological close to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

        # Find external contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area
        filtered_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if MIN_AREA < area < MAX_AREA:
                filtered_contours.append(cnt)

        # Create DetectedObject instances
        objects = [DetectedObject(c) for c in filtered_contours]

        # Draw bounding boxes for each detected object
        for obj in objects:
            x, y, w, h = cv2.boundingRect(obj.contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # --- (2) Detect ArUco Markers ---

        # Detect ArUco markers in the original (color) frame
        corners, ids, rejected = aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

        if ids is not None and len(ids) > 0:
            # Draw markers for visualization (optional)
            aruco.drawDetectedMarkers(frame, corners, ids)

            # For each detected marker, draw the ID text near its center
            for marker_corners, marker_id in zip(corners, ids):
                # Each marker_corners is [1, 4, 2]: 4 corner points (x, y)
                # We'll place text at the average of the 4 corners (marker center)
                corners_array = marker_corners[0]  # shape = (4, 2)
                center_x = int(np.mean(corners_array[:, 0]))
                center_y = int(np.mean(corners_array[:, 1]))

                cv2.putText(frame, f"ID: {marker_id[0]}", 
                            (center_x, center_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()