import cv2
import numpy as np
from ultralytics import YOLO

video_path = "data/match.mp4"
cap = cv2.VideoCapture(video_path)
success, frame = cap.read()
cap.release()

model = YOLO("yolov8n.pt")
results = model(frame, conf=0.15, imgsz=1280)

for box in results[0].boxes:
    class_id = int(box.cls[0])
    if class_id != 0:
        continue

    x1, y1, x2, y2 = map(int, box.xyxy[0])

    torso_y2 = y1 + int((y2 - y1) * 0.5)
    torso = frame[y1:torso_y2, x1:x2]

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    lower_green = np.array([34,40,40])
    upper_green = np.array([85])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    not_green_mask = cv2.bitwise_not(green_mask)
    
    mean_color = cv2.mean(torso, mask=not_green_mask)
    print(f"player at ({x1},{y1}): mean BGR = {mean_color[:3]}")