This is the backend folder

## Vision (model)

This file manages all related to the webcam. When prompted, it does one of two things:

1. Aruco Tag Search
   In this mode, it will turn on the camera and identify bounding boxes. It will also look for all aruco tags in the picture.
   If an aruco tag is found that matches the input number, it will return the bounding box that encloses the tag. If no aruco tag
   is found matching the input number, or a bounding box is not found that encloses the correct tag, the function returns None.

2. Chessboard Bounding Box
   In this mode, the camera will try and identify the chess board layout used for calibration with the projector. If found,
   it returns the corners as a vector<Point2f>, if not it returns null.
