import cv2

video_path = "data/match.mp4"

cap = cv2.VideoCapture(video_path)



while True:
    success, frame = cap.read()
    if not success:
        break

    print(frame.shape, frame.dtype)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()