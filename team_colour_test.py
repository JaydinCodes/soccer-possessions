import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans

video_path = "data/match.mp4"
cap = cv2.VideoCapture(video_path)
success, frame = cap.read()
cap.release()

model = YOLO("yolov8n.pt")
results = model(frame, conf=0.15, imgsz=1280)

positions = []
colours = []

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
    
    positions.append((x1,y1,x2,y2))
    colours.append(mean_color)


colours = np.array(colours)
kmeans = KMeans(n_clusters=2, n_init=10)
team_labels = kmeans.fit_predict(colours)

distances = []
for colour, team in zip(colours, team_labels):
    center = kmeans.cluster_centers_[team]
    distance = np.linalg.norm(colour - center)
    distances.append(distance)

for (x1,y1,x2,y2), team, dist in sorted(zip(positions, team_labels, distances), key=lambda p: -p[2]):
    print(f"team {team}, distance {dist:.1f}, box ({x1}, {y1})")


for (x1,y1,x2,y2), team in zip(positions, team_labels):
    if dist > 45:
        colour = (200, 200, 200) # unclassfied
    elif team == 0:
        colour = (0, 255, 255)
    else:
        colour = (255, 0, 255)
   
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

cv2.imshow("teams", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
