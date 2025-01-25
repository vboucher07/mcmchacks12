import cv2
import numpy as np

def findAruco(
    arucoNumber,
    aruco_dict=cv2.aruco.DICT_4X4_50,
    max_frames=10,
    show_frame=False,
    min_area=500,
    max_area=100000
):
    """
    Attempts up to `max_frames` from the default camera to detect "objects" (via contours)
    AND ArUco markers. Returns a list of tuples:
        [ ((objX, objY, objW, objH), (markX, markY, markW, markH)), ... ]
    For EACH matching marker that is inside a given object's bounding box.

    'Inside' is determined by checking if the center of the marker bounding box
    lies within the object's bounding box. If no matches found, returns an empty list.

    Parameters
    ----------
    arucoNumber : int
        The ArUco marker ID to look for.
    aruco_dict : int, optional
        The ArUco dictionary, e.g. cv2.aruco.DICT_4X4_50.
    max_frames : int, optional
        How many frames to capture before giving up, default=10.
    show_frame : bool, optional
        If True, show each frame with drawn bounding boxes.
    min_area : float, optional
        Minimum contour area for object detection filter.
    max_area : float, optional
        Maximum contour area for object detection filter.

    Returns
    -------
    result_list : list of tuples
       [
         ((objX, objY, objW, objH), (markX, markY, markW, markH)),
         ...
       ]
       One tuple per ArUco marker that has ID == arucoNumber AND lies inside
       the object's bounding box. If none found, returns an empty list.
    """

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error opening camera.")
        return []

    # Prepare ArUco
    aruco_dict_obj = cv2.aruco.getPredefinedDictionary(aruco_dict)
    # Recommended approach: DetectorParameters_create()
    aruco_params = cv2.aruco.DetectorParameters()

    final_result_list = []

    for frame_idx in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            continue

        # --- (1) Detect Objects (Contours) ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

        # Find external contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter by area, get bounding boxes
        object_bboxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                object_bboxes.append((x, y, w, h))
                # Draw bounding box for visualization
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # --- (2) Detect ArUco Markers ---
        corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict_obj, parameters=aruco_params)
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # We'll store markers in the form (mx_min, my_min, m_width, m_height, center_x, center_y)
        matched_markers = []

        if ids is not None and len(ids) > 0:
            for (corner_set, marker_id) in zip(corners, ids):
                if marker_id[0] == arucoNumber:
                    # corner_set shape = (1, 4, 2)
                    corner_array = corner_set[0]
                    x_coords = corner_array[:, 0]
                    y_coords = corner_array[:, 1]

                    mx_min = int(np.min(x_coords))
                    mx_max = int(np.max(x_coords))
                    my_min = int(np.min(y_coords))
                    my_max = int(np.max(y_coords))
                    m_width = mx_max - mx_min
                    m_height = my_max - my_min

                    # The center
                    center_x = int(np.mean(x_coords))
                    center_y = int(np.mean(y_coords))

                    matched_markers.append((mx_min, my_min, m_width, m_height, center_x, center_y))

        # --- (3) Build a list of results: (object_bbox, marker_bbox) for each matching marker ---
        frame_result_list = []
        for (ox, oy, ow, oh) in object_bboxes:
            # Check each marker that matches ID = arucoNumber
            for (mx_min, my_min, m_w, m_h, cx, cy) in matched_markers:
                # If marker's center is inside the object bounding box
                if (ox <= cx <= ox + ow) and (oy <= cy <= oy + oh):
                    # We record a tuple: ((objX, objY, objW, objH), (markX, markY, markW, markH))
                    frame_result_list.append(
                        ((ox, oy, ow, oh), (mx_min, my_min, m_w, m_h))
                    )

        # If we found any such matches in this frame, store them
        if len(frame_result_list) > 0:
            final_result_list = frame_result_list
            # Optionally break if you only want the first successful frame
            # break

        # Show the frame if requested
        if show_frame:
            cv2.imshow("Aruco+Objects", frame)
            # Press 'q' to quit early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if show_frame:
        cv2.destroyAllWindows()
    return final_result_list

def findChessBoard(m, n, max_frames=30, show_frame=False):
    """
    Attempts up to `max_frames` frames from the default camera (using DirectShow)
    to detect a chessboard of size (m x n) inner corners.
    
    Returns a list of bounding boxes, one for each time it successfully finds
    the chessboard. If none found, returns an empty list.

    Each bounding box is a tuple: (x, y, w, h)

    Parameters
    ----------
    m : int
        Number of inner corners in the chessboard along one dimension.
    n : int
        Number of inner corners in the chessboard along the other dimension.
    max_frames : int, optional
        How many frames to capture before giving up, default=10.
    show_frame : bool, optional
        If True, display the camera feed with drawn corners (if found).
    
    Returns
    -------
    result_list : list of tuples
       [
         (x, y, w, h),
         ...
       ]
       One tuple per detected chessboard bounding box. If none found, returns [].
    """
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error opening camera.")
        return []

    result_list = []

    for frame_idx in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Attempt to find the chessboard corners
        found, corners = cv2.findChessboardCorners(gray, (m, n))
        if found and corners is not None:
            # corners typically has shape: (m*n, 1, 2). Reshape to (m*n, 2)
            corners_reshaped = corners.reshape(-1, 2)

            # Compute bounding box from min/max x and y
            x_coords = corners_reshaped[:, 0]
            y_coords = corners_reshaped[:, 1]
            x_min, x_max = int(x_coords.min()), int(x_coords.max())
            y_min, y_max = int(y_coords.min()), int(y_coords.max())
            w = x_max - x_min
            h = y_max - y_min

            # Draw the corners on the frame for visualization (optional)
            if show_frame:
                cv2.drawChessboardCorners(frame, (m, n), corners, found)
            
            # Append the bounding box to our result list
            result_list.append((x_min, y_min, w, h))
            
            # If you only want the **first** detection, uncomment the break:
            # break

        # Show the frame if requested (even if not found, so you can see what's happening)
        if show_frame:
            cv2.imshow("Chessboard Detection", frame)
            # Press 'q' to quit early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if show_frame:
        cv2.destroyAllWindows()

    return result_list


if __name__ == "__main__":
    # Example usage
    # aruco_id_to_find = 42
    # results = findAruco(aruco_id_to_find)

    results = findChessBoard(5, 5, show_frame=True)

    print("Results:", results)