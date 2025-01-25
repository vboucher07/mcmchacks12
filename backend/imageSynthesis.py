import cv2
import numpy as np
import os

# Resolution of the projector or your main window
projector_width, projector_height = 1280, 720

# add global variables for calibration parameters 


def highlightRegion(positionArray, save_path=None):
    """
    Highlights a region in a window and optionally saves the image to a folder.

    Parameters:
        positionArray (list): A list containing [x, y, width, height].
            - x, y: Center position of the object to be highlighted.
            - width, height: Dimensions of the rectangle to highlight.
        save_path (str): Path to save the highlighted image. If None, the image is not saved.
    """
    # Initialize a blank mask
    mask = np.zeros((projector_width, projector_height, 3), dtype=np.uint8)

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

    # Save the image if a path is provided
    if save_path:
        # Ensure the save path directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, mask)
        print(f"Image saved to {save_path}")

    # Wait for user interaction and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
object_coords = [640, 360, 100, 50]  # (x, y, width, height)
save_folder = ""
image_name = "highlighted_output.jpg"
save_path = os.path.join(save_folder, image_name)

highlightRegion(object_coords)





# def projectCalibration(m:int, n:int) : # m and n are the size of the calibration square

