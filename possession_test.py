import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans

def get_team_colour(frame, box):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    torso_y2 = y1 + int((y2-y1), * 0.5)
    torso = frame[y1:torso_y2, x1:x2]

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255,255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    not_green_mask = cv2.bitwise_not(green_mask)

    return cv2.mean(torso, mask=not_green_mask)[:3]

video_path = "data/match.mp4"
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(video_path)
success, first_frame = cap.read()
results = model(first_frame, conf=0.15, imgsz=1280)

first_colours = []
for box in results[0].boxes:
    if int(box.cls[0]) == 0:
        first_colours.append(get_team_colour(first_frame, box))

kmeans = KMeans(n_clusters=2, n_init=10)
kmeans.fit(np.array(first_colours))

cap.release()
cap = cv2.VideoCapture(video_path)


while True:
    success, frame = cap.read()

    if not success:
        break


    results = model.track(frame, persist=True, conf=0.15, imgsz=1280)

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if class_id == 0:
            colour = get_team_colour(frame, box)
            team = kmeans.predict([colour])[0]
            dist = np.linalg.norm(np.array(colour) - kmeans.cluster_centers_[team])

            if dist > 65:
                box_colour = (200, 200, 200)
            
            elif team == 0:
                box_colour = (0 ,255, 255)
            else:
                box_colour = (255, 0, 255)

        elif class_id == 32:
            box_colour = (0, 0, 255)
        else:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_colour, 2)

        cv2.imshow("possession_test", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
