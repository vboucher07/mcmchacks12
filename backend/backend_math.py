import cv2
import numpy as np

def highlightRegion():
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
    mask = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Extract coordinates
    points = [(150, 100), (250, 840), (500, 500), (800, 800), (1000, 200), (1200, 700), (1600, 150), (1700, 900)]
    square_size = 140
    all_points = []

    # Draw green rectangle
    for pnt in points:
        cv2.rectangle(mask, pnt, (pnt[0] + square_size, pnt[1] + square_size), (255, 255, 255), -1)
        all_points.append(pnt)
        all_points.append((pnt[0] + square_size, pnt[1]))
        all_points.append((pnt[0], pnt[1] + square_size))
        all_points.append((pnt[0] + square_size, pnt[1] + square_size))

    # Create a resizable window on the second monitor (assuming x=1920 is your second screen)
    window_name = "Highlight Output (Debug)"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow(window_name, -1920, 500)  # Shift to second monitor
    # cv2.resizeWindow(window_name, 1920, 1080)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Show and wait
    cv2.imshow(window_name, mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the highlighted mask if a valid path is provided

def calcHomography():
    points = [(150, 100), (250, 840), (500, 500), (800, 800), (1000, 200), (1200, 700), (1600, 150), (1700, 900)]
    all_points = []
    square_size = 140
    for pnt in points:
        all_points.append(pnt)
        all_points.append((pnt[0] + square_size, pnt[1]))
        all_points.append((pnt[0], pnt[1] + square_size))
        all_points.append((pnt[0] + square_size, pnt[1] + square_size))
    src_points = np.array(all_points, dtype=np.float32)
    dst_points = np.array([
        [330, 203],
        [440, 197],
        [332, 315],
        [445, 307],
        [431, 769],
        [542, 765],
        [434, 880],
        [546, 874],
        [612, 501],
        [722, 497],
        [616, 611],
        [726, 606],
        [851, 721],
        [957, 717],
        [854, 829],
        [960, 826],
        [985, 255],
        [1091, 252],
        [987, 365],
        [1095, 362],
        [1147, 634],
        [1255, 630],
        [1151, 742],
        [1259, 738],
        [1434, 198],
        [1543, 196],
        [1437, 312],
        [1546, 307],
        [1529, 770],
        [1636, 767],
        [1531, 881],
        [1639, 876],
    ], dtype=np.float32)

    # Calculate homography matrix
    homography_matrix, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC)
    return homography_matrix

print(calcHomography())