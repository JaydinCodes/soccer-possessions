import cv2

stream_url = "http://192.168.1.5:8080/video"   # <-- replace with YOUR phone's URL
cap = cv2.VideoCapture(stream_url)

while True:
    success, frame = cap.read()
    if not success:
        print("failed to read frame from stream")
        break

    cv2.imshow("phone", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()