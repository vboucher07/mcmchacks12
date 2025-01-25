import cv2
import numpy as np
import os

PROJECTOR_HEIGHT, PROJECTOR_HEIGHT = 720, 1080 # Resolution of projector
SAVE_FILE_PATH = "../data/highlighted_output.jpg"  # Specify your target directory and filename

def highlightRegion(positionArray, SAVE_FILE_PATH):
    """
    Highlights a region in a window and saves the image to a specified file path.

    Parameters:
        positionArray (list): A list containing [x, y, width, height].
            - x, y: Center position of the object to be highlighted.
            - width, height: Dimensions of the rectangle to highlight.
        SAVE_FILE_PATH (str): Full path to save the highlighted image, including the file name.
    """
    # Initialize a blank mask
    mask = np.zeros((PROJECTOR_HEIGHT, PROJECTOR_HEIGHT, 3), dtype=np.uint8)

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

    # Save the image
    cv2.imwrite(SAVE_FILE_PATH, mask)
    print(f"Image saved to {SAVE_FILE_PATH}")

# # Example usage 1
# object_coords = [640, 360, 100, 50]  # (x, y, width, height)
# highlightRegion(object_coords, SAVE_FILE_PATH)

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

    # Save the calibration image if a path is provided
    if save_path:
        cv2.imwrite(save_path, calibration_image)
        print(f"Calibration image saved to {save_path}")

# Example usage
grid_size = (9, 9)  # Inner corners (rows, cols)
cell_size = 50      # Each square is 50x50 pixels
output_size = (800, 600)  # Output image dimensions


generateCalibrationImage(grid_size, cell_size, output_size, SAVE_FILE_PATH)
