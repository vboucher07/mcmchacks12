import cv2

def check_camera_resolution(camera_index=0, num_frames=10):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}.")
        return

    print(f"Checking resolution for camera {camera_index}...")
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to read frame from camera.")
            break
        height, width = frame.shape[:2]
        print(f"Frame {i+1}: Width={width}, Height={height}")
        cv2.imshow("Frame", frame)
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    check_camera_resolution(camera_index=0, num_frames=10)