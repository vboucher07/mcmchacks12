

# Number of pixels to crop from each side
crop_top = 125     # Pixels to remove from the top
crop_bottom = 168  # Pixels to remove from the bottom
crop_left = 220   # Pixels to remove from the left
crop_right = 220  # Pixels to remove from the right


def cameraToProjector(x, y):
    """Converts camera coordinates to projector coordinates."""
    # Projector dimensions
    projector_width = 1920
    projector_height = 1080

    # Camera dimensions
    camera_width = 1920
    camera_height = 1080

    # Crop the camera image
    x_cropped = x - crop_left  
    y_cropped = y - crop_top

    # Scale the cropped image to the projector
    x_scaled = x_cropped * projector_width / (camera_width - crop_left - crop_right)
    y_scaled = y_cropped * projector_height / (camera_height - crop_top - crop_bottom)

    return x_scaled, y_scaled