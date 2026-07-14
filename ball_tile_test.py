import cv2
from ultralytics import YOLO

video_path = "data/match.mp4"
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 630)
success, frame = cap.read()
cap.release()

# --- Baseline: full-frame pass ---
full_results = model(frame, conf=0.15, imgsz=1280, classes=[32])
print(f"FULL FRAME: {len(full_results[0].boxes)} ball candidates")
for box in full_results[0].boxes:
    print(f"  box {list(map(int, box.xyxy[0]))}, conf {float(box.conf[0]):.2f}")

# --- Tiled pass ---
height, width, _ = frame.shape
cols, rows = 4, 2
tile_w = width // cols
tile_h = height // rows

ball_detections = []
for row in range(rows):
    for col in range(cols):
        x_offset = col * tile_w
        y_offset = row * tile_h
        tile = frame[y_offset:y_offset + tile_h, x_offset:x_offset + tile_w]
        results = model(tile, conf=0.15, classes=[32], verbose=False)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            full_box = (x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset)
            ball_detections.append((full_box, float(box.conf[0])))

print(f"TILED: {len(ball_detections)} ball candidates")
for box, conf in ball_detections:
    print(f"  box {box}, conf {conf:.2f}")