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

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
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


if __name__ == "__main__":
    # Suppose you want to find ArUco marker ID=7
    # using 4x4_50 dictionary. Adjust as needed.
    results = findAruco(
        arucoNumber=2,
        aruco_dict=cv2.aruco.DICT_4X4_50,
        max_frames = 500,     # try 20 frames
        show_frame=True,   # show the camera feed with draws
        min_area=500,
        max_area=100000
    )

    # Print out each match
    if len(results) == 0:
        print("No matching objects/markers found.")
    else:
        for (object_bbox, marker_bbox) in results:
            print("Object BBox:", object_bbox, "Marker BBox:", marker_bbox)
