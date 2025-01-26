import cv2
import numpy as np

def findObjects(
    camera_index=0,
    max_frames=10,
    show_frame=True,
    min_area=500,
    max_area=100000
):
    """
    Captures up to `max_frames` from the camera, detects objects (via contours),
    and returns a list of bounding boxes [(x, y, w, h)] for the final frame.

    If show_frame=True, it shows each frame with bounding boxes.
    Press 'q' to exit early.

    Parameters
    ----------
    camera_index : int
        The index of the camera (default=0 for your main webcam, try 1 if you have multiple).
    max_frames : int
        Maximum number of frames to capture before returning.
    show_frame : bool
        Whether to display the real-time bounding boxes in a window.
    min_area : float
        Minimum contour area to consider an object.
    max_area : float
        Maximum contour area to consider an object.

    Returns
    -------
    final_bboxes : list of (x, y, w, h)
        Bounding boxes of the objects detected in the last processed frame.
        (x, y) is top-left, w=width, h=height.
    """

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return []

    final_bboxes = []

    for frame_idx in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to read frame from camera.")
            continue

        # 1) Convert to grayscale & blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2) Edge detection
        edged = cv2.Canny(blurred, 50, 150)

        # 3) Close gaps with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

        # 4) Find external contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 5) Filter by area & draw bounding boxes
        object_bboxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                object_bboxes.append((x, y, w, h))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # We'll store bounding boxes from the last valid frame
        final_bboxes = object_bboxes

        # Show the frame if requested
        if show_frame:
            cv2.imshow("Detected Objects", frame)
            # Press 'q' to quit early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if show_frame:
        cv2.destroyAllWindows()

    return final_bboxes


if __name__ == "__main__":
    # Example usage:
    # Detect objects in up to 20 frames from the camera at index=0,
    # show the bounding boxes in a window, and filter out objects
    # smaller than 300 area or larger than 100000 area.
    bboxes = findObjects(
        camera_index=1,
        max_frames=200,
        show_frame=True,
        min_area=700,
        max_area=3000
    )


    print("Final bounding boxes from last frame:", bboxes)
