import cv2
from ultralytics import YOLO

video_path = "data/match.mp4"
cap = cv2.VideoCapture(video_path)
model = YOLO("yolov8n.pt")

while True:
    success, frame = cap.read()

    if not success:
        break

    results = model.track(frame, persist=True, conf=0.15, imgsz=1280)

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        if class_id not in (0, 32):
            continue
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        track_id = int(box.id[0]) if box.id is not None else -1
        label = f"id:{track_id}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("tracking", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()