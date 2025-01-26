import cv2
import numpy as np

# Define projector dimensions
PROJECTOR_HEIGHT = 1080
PROJECTOR_WIDTH = 1920

def generateCornerSquares(square_size, save_path=None):
    """
    Creates a calibration image with 4 squares at each corner of the projector resolution.

    Parameters:
        square_size (int): Size of each square in pixels (width and height).
        save_path (str): Optional path to save the calibration image.

    Returns:
        None
    """
    # Create a blank black canvas with projector dimensions
    canvas = np.zeros((PROJECTOR_HEIGHT, PROJECTOR_WIDTH, 3), dtype=np.uint8)

    # Define corner square coordinates
    corners = [
        (0, 0),  # Top-left
        (PROJECTOR_WIDTH - square_size, 0),  # Top-right
        (0, PROJECTOR_HEIGHT - square_size),  # Bottom-left
        (PROJECTOR_WIDTH - square_size, PROJECTOR_HEIGHT - square_size),  # Bottom-right
    ]

    # Draw the squares
    for x, y in corners:
        cv2.rectangle(
            canvas, (x, y), (x + square_size, y + square_size), (0, 255, 0), -1  # Filled green square
        )

    # Display the calibration image in full-screen mode
    window_name = "Calibration Corners"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, canvas)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the calibration image if a path is provided
    if save_path:
        cv2.imwrite(save_path, canvas)
        print(f"Calibration image saved to {save_path}")

def getImageOfCalibration():
    """
    Captures an image of the projected calibration grid using the camera.
    Saves the image to disk.
    """
    cap = cv2.VideoCapture(0)  # Replace 0 with your camera index
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
        if key == ord('c'):  # Press 'c' to capture
            cv2.imwrite("captured_grid.jpg", frame)
            print("Calibration image captured and saved as 'captured_grid.jpg'.")
            break
        elif key == ord('q'):  # Press 'q' to quit
            print("Exiting without capturing.")
            break
    cap.release()
    cv2.destroyAllWindows()

def detectCorners(frame, square_size):
    """
    Detects the 4 corner points of the squares in the camera image.

    Parameters:
        frame (numpy.ndarray): Captured image from the camera.
        square_size (int): Size of each square in pixels.

    Returns:
        numpy.ndarray: Detected points in the camera space.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_points = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if abs(w - square_size) < 10 and abs(h - square_size) < 10:  # Size filter
            detected_points.append((x + w // 2, y + h // 2))  # Center of square
    return np.array(detected_points, dtype=np.float32)

def getProjectorPoints(square_size):
    """
    Returns the fixed projector points corresponding to the corners.

    Parameters:
        square_size (int): Size of each square in pixels.

    Returns:
        numpy.ndarray: Points in the projector space.
    """
    return np.array([
        [square_size // 2, square_size // 2],  # Top-left
        [PROJECTOR_WIDTH - square_size // 2, square_size // 2],  # Top-right
        [square_size // 2, PROJECTOR_HEIGHT - square_size // 2],  # Bottom-left
        [PROJECTOR_WIDTH - square_size // 2, PROJECTOR_HEIGHT - square_size // 2]  # Bottom-right
    ], dtype=np.float32)

def computeHomography(square_size):
    """
    Computes the homography matrix from the camera's detected points to the projector's points.

    Parameters:
        square_size (int): Size of each square in pixels.

    Returns:
        numpy.ndarray: Homography matrix.
    """
    camera_frame = cv2.imread("captured_grid.jpg")  # Load captured image
    if camera_frame is None:
        print("Error: 'captured_grid.jpg' not found.")
        return
    camera_points = detectCorners(camera_frame, square_size=square_size)
    projector_points = getProjectorPoints(square_size)

    if len(camera_points) == 4:
        homography_matrix, _ = cv2.findHomography(camera_points, projector_points)
        print("Homography Matrix:\n", homography_matrix)
        return homography_matrix
    else:
        print("Error: Could not detect all 4 corners.")
        return None

def mapCameraToProjector(camera_x, camera_y, homography_matrix):
    """
    Maps a point from the camera's coordinate space to the projector's coordinate space.

    Parameters:
        camera_x (float): X-coordinate in the camera space.
        camera_y (float): Y-coordinate in the camera space.
        homography_matrix (numpy.ndarray): The homography matrix.

    Returns:
        tuple: (projector_x, projector_y) in projector space.
    """
    camera_point = np.array([[camera_x, camera_y, 1]], dtype=np.float32).T
    projector_point = np.dot(homography_matrix, camera_point)
    projector_x = projector_point[0] / projector_point[2]
    projector_y = projector_point[1] / projector_point[2]
    return projector_x, projector_y

# Example usage
if __name__ == "__main__":
    square_size = 100  # Size of each square in pixels

    # Step 1: Project the calibration image
    generateCornerSquares(square_size)

    # Step 2: Capture the calibration image from the camera
    getImageOfCalibration()

    # Step 3: Compute the homography matrix
    homography_matrix = computeHomography(square_size)

    # Step 4: Example mapping from camera to projector
    if homography_matrix is not None:
        camera_x, camera_y = 400, 300  
        projector_x, projector_y = mapCameraToProjector(camera_x, camera_y, homography_matrix)
        print(f"Camera ({camera_x}, {camera_y}) -> Projector ({projector_x}, {projector_y})")
