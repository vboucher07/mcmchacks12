import cv2
import numpy as np
import os

PROJECTOR_HEIGHT, PROJECTOR_WIDTH = 720, 1280  # Resolution of the projector
HIGHLIGHTED_IMAGE_PATH = "../data/highlighted_output.jpg"  
CALIBRATION_IMAGE_PATH ="../data/calibration_output.jpg"  

def highlightRegion(positionArray, HIGHLIGHTED_IMAGE_PATH):
    """
    Highlights a region in a window and saves the image to a specified file path.

    Parameters:
        positionArray (list): A list containing [x, y, width, height].
            - x, y: Center position of the object to be highlighted.
            - width, height: Dimensions of the rectangle to highlight.
        HIGHLIGHTED_IMAGE_PATH (str): Full path to save the highlighted image, including the file name.
    """
    # Initialize a blank mask
    mask = np.zeros((PROJECTOR_HEIGHT, PROJECTOR_WIDTH, 3), dtype=np.uint8)

    # Extract coordinates
    x_center, y_center, width, height = positionArray
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)

    # Draw the rectangle on the mask
    cv2.rectangle(mask, (x1, y1), (x2, y2), (0, 255, 0), -1)  # Green highlight

    # Display mask on the current screen
    cv2.imshow("Highlight Output (Debug)", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Ensure the save directory exists
    os.makedirs(os.path.dirname(HIGHLIGHTED_IMAGE_PATH), exist_ok=True)

    # Save the image
    cv2.imwrite(HIGHLIGHTED_IMAGE_PATH, mask)
    print(f"Image saved to {HIGHLIGHTED_IMAGE_PATH}")


def generateCalibrationImage(grid_size, cell_size, output_size, save_path=None):
    """
    Generates a checkerboard calibration image.

    Parameters:
        grid_size (tuple): Number of inner corners per row and column (e.g., (7, 5)).
        cell_size (int): Size of each square in pixels.
        output_size (tuple): Dimensions of the output image (width, height) in pixels.
        save_path (str): Path to save the calibration image. If None, the image is not saved.
    """
    # Create a blank white image
    width, height = output_size
    calibration_image = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

    # Draw the checkerboard pattern
    num_rows, num_cols = grid_size
    for row in range(num_rows):
        for col in range(num_cols):
            if (row + col) % 2 == 0:  # Alternate squares
                x1 = col * cell_size
                y1 = row * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                if x2 <= width and y2 <= height:  # Ensure within bounds
                    cv2.rectangle(calibration_image, (x1, y1), (x2, y2), (0, 0, 0), -1)  # Black square

    # Display the calibration image
    cv2.imshow("Calibration Image", calibration_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Ensure the save directory exists
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, calibration_image)
        print(f"Calibration image saved to {save_path}")

def highlightRegionWithCalibration(positionArray, homography_matrix, HIGHLIGHTED_IMAGE_PATH):
    """
    Highlights a region in the projector's output using calibration data and saves the image.

    Parameters:
        positionArray (list): A list containing [x, y, width, height] in the camera's coordinate system.
            - x, y: Center position of the object to be highlighted.
            - width, height: Dimensions of the rectangle to highlight.
        homography_matrix (numpy.ndarray): The 3x3 homography matrix from calibration.
        HIGHLIGHTED_IMAGE_PATH (str): Full path to save the highlighted image, including the file name.
    """
    # Initialize a blank mask for the projector
    mask = np.zeros((PROJECTOR_HEIGHT, PROJECTOR_WIDTH, 3), dtype=np.uint8)

    # Extract coordinates in the camera's coordinate system
    x_center, y_center, width, height = positionArray
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    # Define the rectangle corners in the camera's coordinate space
    camera_corners = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ], dtype=np.float32)

    # Transform the corners to the projector's coordinate space
    projector_corners = cv2.perspectiveTransform(camera_corners.reshape(-1, 1, 2), homography_matrix)

    # Draw the transformed rectangle on the mask
    projector_corners = projector_corners.reshape(-1, 2).astype(int)  # Reshape and convert to integers
    cv2.fillPoly(mask, [projector_corners], (0, 255, 0))  # Green highlight

    # Display the mask on the projector
    cv2.imshow("Highlight Output (Adjusted)", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the adjusted image
    os.makedirs(os.path.dirname(HIGHLIGHTED_IMAGE_PATH), exist_ok=True)
    cv2.imwrite(HIGHLIGHTED_IMAGE_PATH, mask)
    print(f"Image saved to {HIGHLIGHTED_IMAGE_PATH}")

# Example usage for highlighting
# object_coords = [640, 360, 100, 50]  # (x, y, width, height)
# highlightRegion(object_coords, HIGHLIGHTED_IMAGE_PATH)

# Example usage for calibration image
grid_size = (9, 9)  # Inner corners (rows, cols)
cell_size = 100      # Each square is 50x50 pixels
output_size = (1920, 1080)  # Output image dimensions
generateCalibrationImage(grid_size, cell_size, output_size, CALIBRATION_IMAGE_PATH)

# Example Homography Matrix (replace with your actual calibration result)
homography_matrix = np.array([
    [1.2, 0.1, 30],
    [0.0, 1.5, 20],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

# # Example usagepg"
# object_coords = [400, 300, 100, 50]  # (x, y, width, height) in the camera's coordinate system
# highlightRegionWithCalibration(object_coords, homography_matrix, HIGHLIGHTED_IMAGE_PATH)

