import cv2
import numpy as np

def findChessBoard(board_size, show_frame=False):
    camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not camera.isOpened():
        raise Exception("Error opening camera.")
    
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)

        if ret:
            # Refine corner locations for better accuracy
            corners = cv2.cornerSubPix(
                gray,
                corners,
                winSize=(11, 11),
                zeroZone=(-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )

            camera.release()
    
            if show_frame:
                cv2.drawChessboardCorners(frame, board_size, corners, ret)
                cv2.imshow("Checkerboard Detected", frame)
                cv2.destroyAllWindows()
            return corners.reshape(-1, 2), (frame.shape[1], frame.shape[0])
        elif show_frame:
            cv2.imshow("Checkerboard Detection", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

        camera.release()
        if show_frame:
            cv2.destroyAllWindows()
        raise Exception("Checkerboard not found.")


if __name__ == "__main__":
    # Example usage
    # aruco_id_to_find = 42
    # results = findAruco(aruco_id_to_find)
    results = findChessBoard((), show_frame=True)
    print("Results:", results)