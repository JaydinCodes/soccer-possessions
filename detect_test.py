import cv2
from ultralytics import YOLO

video_path = "data/match.mp4"
cap = cv2.VideoCapture(video_path)
success, frame = cap.read()
cap.release()

model = YOLO("yolov8n.pt")

results = model(frame)

for box in results[0].boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    if class_id == 0:
        label = "person"
        color = (0 ,255, 0)
    elif class_id == 32:
        label = "ball"
        color = (0, 0, 255)
    else:
        continue


    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

cv2.imshow("detections", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()



