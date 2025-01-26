import cv2
import numpy as np

###
# Projector resolution
PROJECTOR_WIDTH = 1920
PROJECTOR_HEIGHT = 1080

# Camera resolution (ensure this matches your actual camera resolution)
camera_w, camera_h = 1920, 1080  # Updated to match the resolutions

# Margins as provided
TOP_MARGIN = 130
LEFT_MARGIN = 220
RIGHT_MARGIN = 220
BOTTOM_MARGIN = 168

# Calculate active area dimensions
ACTIVE_CAMERA_WIDTH = camera_w - LEFT_MARGIN - RIGHT_MARGIN  # 1480
ACTIVE_CAMERA_HEIGHT = camera_h - TOP_MARGIN - BOTTOM_MARGIN  # 782

# Calculate scaling factors for simple transformation
SCALE_X = PROJECTOR_WIDTH / ACTIVE_CAMERA_WIDTH  # ≈1.2973
SCALE_Y = PROJECTOR_HEIGHT / ACTIVE_CAMERA_HEIGHT  # ≈1.380

# Center of the camera frame
CENTER_X = camera_w / 2
CENTER_Y = camera_h / 2

###

def detect_objects_live_once(
    camera_index=0,
    min_area=1000,
    max_area=300000,
    max_frames=100
):
    """
    Uses a webcam (or another camera) to detect white objects on a darker background
    using thresholding + contour detection. It runs for up to `max_frames` frames
    or until the user presses 'q'. Once finished, it returns the bounding boxes
    found in the last valid frame.

    Parameters:
        camera_index (int): Index of the camera (0 for default, 1 if multiple).
        min_area (int): Minimum contour area to be considered an object.
        max_area (int): Maximum contour area to be considered an object.
        max_frames (int): How many frames to process before automatically stopping.

    Returns:
        final_bboxes (list of (x, y, w, h)):
            Bounding boxes from the final processed frame.
    """
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}.")
        return []

    # Set desired resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_h)

    # Verify if the resolution was set correctly
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera resolution set to: Width={actual_width}, Height={actual_height}")

    final_bboxes = []

    for frame_index in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to read frame from camera.")
            break

        # 1) Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2) Slight blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3) Threshold with Otsu to separate white objects from dark background
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # If your objects end up black, invert:
        # thresh = cv2.bitwise_not(thresh)

        # 4) Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 5) Contour detection
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 6) Filter by area; draw bounding boxes
        bboxes_this_frame = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                bboxes_this_frame.append((x, y, w, h))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Update final_bboxes to the bounding boxes of this latest frame
        final_bboxes = bboxes_this_frame

        # Show the live feed
        cv2.imshow("Live Object Detection", frame)
        # Press 'q' to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # final_bboxes holds bounding boxes from the *last* processed frame
    return final_bboxes

def detectMedium():
    return detect_objects_live_once(
        camera_index=0,
        min_area=15000,
        max_area=50000,
        max_frames=30
    )

def detectSmall():
    return detect_objects_live_once(
        camera_index=0,
        min_area=1000,
        max_area=5000,
        max_frames=30
    )

def detectBigBoys():
    return detect_objects_live_once(
        camera_index=0,
        min_area=125000,
        max_area=175000,
        max_frames=30
    )

def highlightRegionsAllScaled(
    bounding_boxes,
    original_width,
    original_height,
    shift_x=-170,
    shift_y=-40,
    expand_margin=10,
    highlighted_image_path=None
):
    """
    Creates one mask for the projector & draws scaled bounding boxes in green.
    The bounding boxes come from an original resolution (original_width x original_height),
    and we scale them up to the projector's resolution (PROJECTOR_WIDTH x PROJECTOR_HEIGHT).
    
    Then we apply a shift_x, shift_y in projector coordinates.
    Also optionally expand each box by `expand_margin` on each side.

    Parameters
    ----------
    bounding_boxes : list of (x, y, w, h)
        Each bounding box in camera coords.
    original_width : int
        Camera/image width used to detect bounding boxes.
    original_height : int
        Camera/image height used to detect bounding boxes.
    shift_x : int
        How many pixels to shift horizontally in projector space (+ = right, - = left).
    shift_y : int
        How many pixels to shift vertically in projector space (+ = down, - = up).
    expand_margin : int
        Additional margin to expand each bounding box on all sides (in projector coords).
    highlighted_image_path : str, optional
        If given, saves the final mask image to this path.

    Returns
    -------
    mask : np.ndarray
        The final mask (PROJECTOR_HEIGHT x PROJECTOR_WIDTH) with drawn rectangles.
    """
    # Calculate scale factors
    scale_x = PROJECTOR_WIDTH / float(original_width)
    scale_y = PROJECTOR_HEIGHT / float(original_height)

    # Create a black image (mask) matching the projector size
    mask = np.zeros((PROJECTOR_HEIGHT, PROJECTOR_WIDTH, 3), dtype=np.uint8)

    for (x, y, w, h) in bounding_boxes:
        # Scale camera coords to projector coords
        x1 = int(x * scale_x)
        y1 = int(y * scale_y)
        x2 = int((x + w) * scale_x)
        y2 = int((y + h) * scale_y)

        # Apply optional shift
        x1 += shift_x
        x2 += shift_x
        y1 += shift_y
        y2 += shift_y

        # Expand bounding box if desired
        x1 -= expand_margin
        y1 -= expand_margin
        x2 += expand_margin
        y2 += expand_margin

        # Optionally clamp coordinates to avoid negatives or going beyond the screen
        # x1 = max(0, x1);  y1 = max(0, y1)
        # x2 = min(PROJECTOR_WIDTH, x2); y2 = min(PROJECTOR_HEIGHT, y2)

        # Draw a green rectangle (filled)
        cv2.rectangle(mask, (x1, y1), (x2, y2), (0, 255, 0), -1)

    window_name = "Scaled Highlight"
    cv2.namedWindow(window_name, cv2.WINDOW_FULLSCREEN)

    cv2.imshow(window_name, mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save if requested
    if highlighted_image_path:
        import os
        dir_name = os.path.dirname(highlighted_image_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        cv2.imwrite(highlighted_image_path, mask)
        print(f"Saved scaled highlight mask to: {highlighted_image_path}")

    return mask

def apply_homography_to_point(x, y, H):
    """
    Applies the homography transformation to a single point.

    Parameters:
        x (float): X-coordinate in the camera image.
        y (float): Y-coordinate in the camera image.
        H (numpy.ndarray): 3x3 homography matrix.

    Returns:
        (float, float): Transformed (x, y) coordinates for the projector.
    """
    point = np.array([x, y, 1]).reshape(3, 1)
    transformed_point = np.dot(H, point)
    transformed_point /= transformed_point[2, 0]  # Normalize
    proj_x, proj_y = transformed_point[0, 0], transformed_point[1, 0]
    return proj_x, proj_y

def highlightRegionWithHomographyAndScaling(boundingBoxes, scaling_factor_x=1.2, scaling_factor_y=1.2, highlighted_image_path=None):
    """
    Applies homography transformation to bounding boxes and then scales their position 
    independently along x and y axes away from the center without altering their size.

    Parameters:
        boundingBoxes (list of tuples): List of bounding boxes in camera coords as (x, y, w, h).
        scaling_factor_x (float): Factor by which to scale the x-coordinates away from the center.
        scaling_factor_y (float): Factor by which to scale the y-coordinates away from the center.
        highlighted_image_path (str, optional): Path to save the highlighted mask image.

    Returns:
        mask (np.ndarray): The final mask with drawn rectangles.
    """
    # Define the homography matrix
    homography_matrix = np.array([
        [7.66839410e-01, 4.18844380e-02, 2.14153972e+02],
        [-2.71116969e-02, 7.82215021e-01, 1.27894167e+02],
        [2.17112480e-06, 1.38477082e-05, 1.00000000e+00]
    ])

    # Create a black image (mask) matching the projector size
    mask = np.zeros((PROJECTOR_HEIGHT, PROJECTOR_WIDTH, 3), dtype=np.uint8)

    for (x, y, w, h) in boundingBoxes:
        # Apply homography to the top-left corner to calculate position
        top_left_transformed = apply_homography_to_point(x, y, homography_matrix)

        # Calculate the bounding box center after homography
        bbox_center_x = top_left_transformed[0] + w / 2
        bbox_center_y = top_left_transformed[1] + h / 2

        # Compute vector from projector center to bounding box center
        vector_x = bbox_center_x - CENTER_X
        vector_y = bbox_center_y - CENTER_Y

        # Scale the vector independently along x and y axes
        scaled_vector_x = vector_x * scaling_factor_x
        scaled_vector_y = vector_y * scaling_factor_y

        # Compute new center position
        new_center_x = CENTER_X + scaled_vector_x
        new_center_y = CENTER_Y + scaled_vector_y

        # Recalculate top-left corner using the new center
        new_x_min = int(new_center_x - w / 2)
        new_y_min = int(new_center_y - h / 2)

        # Clamp the coordinates to projector bounds
        new_x_min = max(0, min(PROJECTOR_WIDTH - w, new_x_min))
        new_y_min = max(0, min(PROJECTOR_HEIGHT - h, new_y_min))

        # Draw the rectangle in the mask
        cv2.rectangle(mask, (new_x_min, new_y_min), (new_x_min + w, new_y_min + h), (0, 255, 0), -1)

    # Display the result
    window_name = "Highlight with Independent Scaling"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow(window_name, -1920, 500)  # Adjust for your display setup
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow(window_name, mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the result if requested
    if highlighted_image_path:
        import os
        dir_name = os.path.dirname(highlighted_image_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        cv2.imwrite(highlighted_image_path, mask)
        print(f"Saved scaled highlight mask to: {highlighted_image_path}")

    return mask

def highlightShape(shape):
     '''
     0 = Small ones
     1 = Medium sidez
     2 = Large
     '''
     scaling_factor_x = 1.75
     scaling_factor_y = 1.45

     if shape == 0:
        bboxes = detectSmall()
        print("Final bounding boxes:", bboxes)
        # highlightRegionsAllScaled(
        # bounding_boxes=bboxes,
        # original_width=camera_w,
        # original_height=camera_h,
        # highlighted_image_path="scaled_highlights.jpg"
        # )
        highlightRegionWithHomographyAndScaling(bboxes, scaling_factor_x=scaling_factor_x, scaling_factor_y=scaling_factor_y, highlighted_image_path="scaled_highlights.jpg")

     if shape == 1:
        bboxes = detectMedium()
        print("Final bounding boxes:", bboxes)
        # highlightRegionsAllScaled(
        # bounding_boxes=bboxes,
        # original_width=camera_w,
        # original_height=camera_h,
        # highlighted_image_path="scaled_highlights.jpg"
        # )
        highlightRegionWithHomographyAndScaling(bboxes, scaling_factor_x=scaling_factor_x, scaling_factor_y=scaling_factor_y, highlighted_image_path="scaled_highlights.jpg")

     if shape == 2:
        bboxes = detectBigBoys()
        print("Final bounding boxes:", bboxes)
        # highlightRegionsAllScaled(
        # bounding_boxes=bboxes,
        # original_width=camera_w,
        # original_height=camera_h,
        # highlighted_image_path="scaled_highlights.jpg"
        # )
        highlightRegionWithHomographyAndScaling(bboxes, scaling_factor_x=scaling_factor_x, scaling_factor_y=scaling_factor_y, highlighted_image_path="scaled_highlights.jpg")

if __name__ == "__main__":
    highlightShape(0)