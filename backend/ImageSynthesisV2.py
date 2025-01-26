import os
import cv2
import numpy as np
from time import sleep

# -----------------------------------------------------------------------------------
# Global Projector Configuration
# -----------------------------------------------------------------------------------
PROJECTOR_WIDTH = 1920
PROJECTOR_HEIGHT = 1080


# -----------------------------------------------------------------------------------
# 1) generateCornerSquares
# -----------------------------------------------------------------------------------
def generateCornerSquares(square_size, save_path=None):
    """
    Creates a calibration image with 4 squares at each corner of the projector resolution.
    The squares are drawn on a black background.

    Parameters
    ----------
    square_size : int
        Size of each square in pixels (width and height).
    save_path : str, optional
        If provided, the path where the calibration image will be saved.

    Returns
    -------
    None
    """
    # Create a black canvas matching the projector's resolution
    canvas = np.zeros((PROJECTOR_HEIGHT, PROJECTOR_WIDTH, 3), dtype=np.uint8)

    # Define positions for the four corners (with a small offset from the edges)
    corners = [
        (30, 0),  # Top-left
        (PROJECTOR_WIDTH - square_size - 30, 0),                   # Top-right
        (30, PROJECTOR_HEIGHT - square_size),                      # Bottom-left
        (PROJECTOR_WIDTH - square_size - 30, PROJECTOR_HEIGHT - square_size)  # Bottom-right
    ]

    # Draw white squares at each corner
    for x, y in corners:
        cv2.rectangle(canvas,
                      (x, y),
                      (x + square_size, y + square_size),
                      (255, 255, 255),
                      -1)  # -1 = filled rectangle

    # Display the calibration image in full-screen mode
    window_name = "Calibration Corners"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    # Move the window to a specific location if needed (e.g., a second monitor).
    cv2.moveWindow(window_name, -1000, 0)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, canvas)

    # If you want to wait for a key press, uncomment these:
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Save the calibration image if a path is provided
    if save_path:
        cv2.imwrite(save_path, canvas)
        print(f"Calibration image saved to {save_path}")


# -----------------------------------------------------------------------------------
# 2) getImageOfCalibration
# -----------------------------------------------------------------------------------
def getImageOfCalibration():
    """
    Opens the camera and waits for the user to press 'c' to capture
    a calibration image of the currently projected content. Press 'q' to quit.
    The captured image is saved as 'captured_grid.jpg'.

    Returns
    -------
    None
    """
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Replace 0 with your camera index if needed
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    print("Press 'c' to capture the image or 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        cv2.imshow("Camera Feed", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            # Press 'c' to capture
            cv2.imwrite("captured_grid.jpg", frame)
            print("Calibration image captured and saved as 'captured_grid.jpg'.")
            break
        elif key == ord('q'):
            # Press 'q' to quit
            print("Exiting without capturing.")
            break

    cap.release()
    cv2.destroyAllWindows()


# -----------------------------------------------------------------------------------
# 3) detectCorners
# -----------------------------------------------------------------------------------
def detectCorners(frame, square_size):
    """
    Detects corner points (centers of squares) in the camera image.

    Parameters
    ----------
    frame : numpy.ndarray
        Image captured from the camera (BGR format).
    square_size : int
        Size of each square in pixels (used for reference, if needed).

    Returns
    -------
    numpy.ndarray
        Array of detected corner points (x, y) in camera space (float32).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    # Find external contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_points = []

    # For each contour, store the center of the bounding box
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        print(x, y, w, h)  # Debug printing
        center_x = x + w // 2
        center_y = y + h // 2
        detected_points.append((center_x, center_y))

    return np.array(detected_points, dtype=np.float32)


# -----------------------------------------------------------------------------------
# 4) getProjectorPoints
# -----------------------------------------------------------------------------------
def getProjectorPoints(square_size):
    """
    Returns the four expected points in projector space
    for the squares drawn by generateCornerSquares.

    Parameters
    ----------
    square_size : int
        Size of each square in pixels.

    Returns
    -------
    numpy.ndarray
        An array of four points [x, y] in projector space.
    """
    return np.array([
        [square_size // 2, square_size // 2],                               # Top-left
        [PROJECTOR_WIDTH - square_size // 2, square_size // 2],             # Top-right
        [square_size // 2, PROJECTOR_HEIGHT - square_size // 2],            # Bottom-left
        [PROJECTOR_WIDTH - square_size // 2, PROJECTOR_HEIGHT - square_size // 2]  # Bottom-right
    ], dtype=np.float32)


# -----------------------------------------------------------------------------------
# 5) computeHomography
# -----------------------------------------------------------------------------------
def computeHomography(square_size):
    """
    Computes the homography matrix mapping camera-detected points
    to the corresponding projector points.

    Parameters
    ----------
    square_size : int
        Size of each square in pixels.

    Returns
    -------
    numpy.ndarray or None
        The homography matrix (3x3). Returns None if homography
        could not be computed (e.g., not all corners detected).
    """
    # Load the captured image
    camera_frame = cv2.imread("captured_grid.jpg")
    if camera_frame is None:
        print("Error: 'captured_grid.jpg' not found.")
        return None

    # Detect corner points in camera image
    camera_points = detectCorners(camera_frame, square_size=square_size)

    # Points in projector space
    projector_points = getProjectorPoints(square_size)

    # Compute homography if we have 4 corners
    if len(camera_points) == 4:
        homography_matrix, _ = cv2.findHomography(camera_points, projector_points)
        print("Homography Matrix:\n", homography_matrix)
        return homography_matrix
    else:
        print("Error: Could not detect all 4 corners.")
        return None


# -----------------------------------------------------------------------------------
# 6) mapCameraToProjector
# -----------------------------------------------------------------------------------
def mapCameraToProjector(camera_x, camera_y, homography_matrix):
    """
    Maps a single point (x, y) from camera coordinates to projector coordinates
    using the provided homography matrix.

    Parameters
    ----------
    camera_x : float
        X-coordinate in camera space.
    camera_y : float
        Y-coordinate in camera space.
    homography_matrix : numpy.ndarray
        The 3x3 homography matrix from computeHomography().

    Returns
    -------
    (float, float)
        The point (projector_x, projector_y) in projector space.
    """
    # Represent the camera point in homogeneous coordinates
    camera_point = np.array([[camera_x], [camera_y], [1]], dtype=np.float32)
    # Multiply by the homography
    projector_point = np.dot(homography_matrix, camera_point)
    # Normalize by the third coordinate
    px = projector_point[0, 0] / projector_point[2, 0]
    py = projector_point[1, 0] / projector_point[2, 0]
    return px, py


# -----------------------------------------------------------------------------------
# 7) highlightRegion
# -----------------------------------------------------------------------------------
def highlightRegion(positionArray, HIGHLIGHTED_IMAGE_PATH):
    """
    Draws a filled rectangle on a black mask corresponding to the projector,
    then displays it on a second monitor. Useful for highlighting a region.

    Parameters
    ----------
    positionArray : list or tuple
        [x_center, y_center, width, height]
        - x_center, y_center: Center of the rectangle (projector coords)
        - width, height: Rectangle dimensions
    HIGHLIGHTED_IMAGE_PATH : str
        Full path (with extension) to save the resulting mask image.
    """
    # Create a black image matching projector resolution
    mask = np.zeros((PROJECTOR_HEIGHT, PROJECTOR_WIDTH, 3), dtype=np.uint8)

    # Extract coordinates
    x_center, y_center, width, height = positionArray
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)

    # Draw green rectangle
    cv2.rectangle(mask, (x1, y1), (x2, y2), (0, 255, 0), -1)

    # Create a resizable window on the second monitor (assuming x=1920 is your second screen)
    window_name = "Highlight Output (Debug)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, 1920, 0)  # Shift to second monitor
    cv2.resizeWindow(window_name, PROJECTOR_WIDTH, PROJECTOR_HEIGHT)

    # Show and wait
    cv2.imshow(window_name, mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the highlighted mask if a valid path is provided
    if HIGHLIGHTED_IMAGE_PATH:
        os.makedirs(os.path.dirname(HIGHLIGHTED_IMAGE_PATH), exist_ok=True)
        cv2.imwrite(HIGHLIGHTED_IMAGE_PATH, mask)
        print(f"Highlight image saved to {HIGHLIGHTED_IMAGE_PATH}")


# -----------------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------------
if __name__ == "__main__":
    # Example square size for the calibration squares
    square_size = 100

    # STEP 1: Project the calibration image with corner squares
    generateCornerSquares(square_size, save_path="calibration_image.jpg")
    print("Projecting corner squares...")

    # STEP 2: Let user capture the calibration image from the camera
    # Move in front of the camera, press 'c' to capture.
    getImageOfCalibration()

    # STEP 3: Compute the homography based on the captured image
    homography_matrix = computeHomography(square_size)

    # STEP 4: If homography is valid, map a sample camera point to projector space
    if homography_matrix is not None:
        # Example camera point
        camera_x, camera_y = 200, 300
        projector_x, projector_y = mapCameraToProjector(camera_x, camera_y, homography_matrix)
        print(f"Camera point ({camera_x}, {camera_y}) -> Projector point ({projector_x}, {projector_y})")

        # STEP 5: Optionally highlight a region around that point in the projector space
        # Here, we define a 50x50 rectangle, centered on (projector_x, projector_y)
        highlightRegion(
            positionArray=[projector_x, projector_y, 50, 50],
            HIGHLIGHTED_IMAGE_PATH="data/highlight.jpg"
        )

        print("Done!")
    else:
        print("Homography could not be computed. Please check if four corners were detected.")
