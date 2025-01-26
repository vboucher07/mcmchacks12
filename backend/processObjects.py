import cv2
import numpy as np
import os

# Adjust to match your projector's resolution
PROJECTOR_WIDTH = 1920
PROJECTOR_HEIGHT = 1080

def show_black_screen_on_projector(duration_ms=1000):
    """
    Displays a black screen on the second monitor (e.g., projector).
    duration_ms: How long (in milliseconds) to wait before closing automatically.
                 If duration_ms <= 0, it won't close automatically; 
                 you must close manually or use cv2.waitKey in your main flow.
    """
    window_name = "Projector Black Screen"

    # Create a black image the size of the projector
    black_image = np.zeros((PROJECTOR_HEIGHT, PROJECTOR_WIDTH, 3), dtype=np.uint8)

    # Create a resizable window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Move the window to the second monitor (adjust x=1920 for your setup)
    cv2.moveWindow(window_name, 1920, 0)

    # Optionally go fullscreen:
    # cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Or just resize to fill the projector resolution
    cv2.resizeWindow(window_name, PROJECTOR_WIDTH, PROJECTOR_HEIGHT)

    # Display the black image
    cv2.imshow(window_name, black_image)

    if duration_ms > 0:
        # Wait for the given duration, then close automatically
        cv2.waitKey(duration_ms)
        cv2.destroyWindow(window_name)
    else:
        # If duration_ms <= 0, we just show the window but do not close automatically
        pass

def capture_image_from_camera(camera_index=1, save_path="captured.jpg"):
    """
    Opens the camera, grabs one frame, saves it to 'save_path'.
    Returns the file path if successful, else None.
    """
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return None

    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from camera.")
        cap.release()
        return None

    cap.release()
    cv2.imwrite(save_path, frame)
    print(f"Image captured and saved to: {save_path}")
    return save_path


def highlightRegionsAll(bounding_boxes, highlighted_image_path=None):
    """
    Creates one mask for the projector & draws all bounding boxes in green.
    Displays on the second monitor, attempts fullscreen, waits for key press.
    """
    mask = np.zeros((PROJECTOR_HEIGHT, PROJECTOR_WIDTH, 3), dtype=np.uint8)

    # Draw each bounding box onto the same mask
    for (x, y, w, h) in bounding_boxes:
        # If you want an offset inside the bounding box, e.g. +25, do so here
        x1 = x + 25
        y1 = y + 25
        x2 = x1 + w
        y2 = y1 + h
        cv2.rectangle(mask, (x1, y1), (x2, y2), (0, 255, 0), -1)

    # Show on second monitor
    window_name = "Projector Highlights"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, 1920, 0)
    # Attempt fullscreen
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, mask)

    # Wait until user presses any key or closes the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally save
    if highlighted_image_path:
        # Must include valid extension like .jpg or .png
        dir_name = os.path.dirname(highlighted_image_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        cv2.imwrite(highlighted_image_path, mask)
        print(f"Highlight mask saved to: {highlighted_image_path}")




if __name__ == "__main__":

    show_black_screen_on_projector(duration_ms=2000)
    captured_path = capture_image_from_camera(camera_index=1, save_path="captured.jpg")
    cv2.destroyAllWindows()

    # 4) If we successfully captured an image, detect objects and highlight them
    if captured_path is not None:
        boxes = detect_objects(captured_path, min_area=5000, max_area=100000)
        print("Detected bounding boxes:", boxes)

        if boxes:
            # Show all bounding boxes at once on the projector
            highlightRegionsAll(boxes, highlighted_image_path="highlighted.jpg")
        else:
            print("No objects found or bounding boxes are empty.")
