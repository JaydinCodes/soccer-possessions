import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans



def detect_ball(frame, model, cols=4, rows=2, conf=0.2):
    height, width, _ = frame.shape
    tile_w = width // cols
    tile_h = height // rows

    best_point = None
    best_conf = 0.0

    for row in range(rows):
        for col in range(cols):
            x_offset = col * tile_w
            y_offset = row * tile_h
            tile = frame[y_offset:y_offset + tile_h, x_offset:x_offset + tile_w]
            results = model(tile, conf=conf, classes=[32], verbose=False)

            for box in results[0].boxes:
                c = float(box.conf[0])
                if c > best_conf:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) / 2 + x_offset
                    cy = (y1 + y2) / 2 + y_offset
                    best_point = (cx, cy)
                    best_conf = c

    return best_point, best_conf


def get_team_color(frame, box):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    torso_y2 = y1 + int((y2 - y1) * 0.5)
    torso = frame[y1:torso_y2, x1:x2]

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    not_green_mask = cv2.bitwise_not(green_mask)
    return cv2.mean(torso, mask=not_green_mask)[:3]

def get_feet_point(box):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    return ((x1 + x2) / 2, y2)

video_path = "data/match.mp4"
model = YOLO("yolov8n.pt")
ball_model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(video_path)
success, first_frame = cap.read()
results = model(first_frame, conf=0.15, imgsz=1280)
frame_index = 0
first_colors = []
for box in results[0].boxes:
    if int(box.cls[0]) == 0:
        first_colors.append(get_team_color(first_frame, box))

kmeans = KMeans(n_clusters=2, n_init=10)
kmeans.fit(np.array(first_colors))

cap.release()
cap = cv2.VideoCapture(video_path)

last_ball_point = None
frames_since_ball = 0
MAX_BALL_JUMP = 200   # px; a real ball can't move further than this between frames
MAX_COAST = 15        # after this many rejected frames, allow re-acquiring anywhere

while True:
    success, frame = cap.read()
    if not success:
        break

    results = model.track(frame, persist=True, conf=0.15, imgsz=1280)
    frame_index += 1
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if class_id == 0:
            color = get_team_color(frame, box)
            team = kmeans.predict([color])[0]
            dist = np.linalg.norm(np.array(color) - kmeans.cluster_centers_[team])

            if dist > 65:
                box_color = (200, 200, 200)
            elif team == 0:
                box_color = (0, 255, 255)
            else:
                box_color = (255, 0, 255)
        elif class_id == 32:
            box_color = (0, 0, 255)
        else:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)


        ball_point = None
    
    
    
    player_points = []  # (feet_point, team)
    
    candidate = detect_ball(frame, ball_model)[0]
    ball_point = None

    if candidate is not None:
        if last_ball_point is None:
            jumped = False
        else:
            jumped = np.linalg.norm(np.array(candidate) - np.array(last_ball_point)) > MAX_BALL_JUMP

        if not jumped or frames_since_ball > MAX_COAST:
            ball_point = candidate
            last_ball_point = candidate
            frames_since_ball = 0
        else:
            frames_since_ball += 1
    else:
        frames_since_ball += 1


    for box in results[0].boxes:
        class_id = int(box.cls[0])
        if class_id == 0:
            color = get_team_color(frame, box)
            team = kmeans.predict([color])[0]
            player_points.append((get_feet_point(box), team))

    if ball_point is not None and player_points:
        closest_team = None
        closest_dist = None
        for point, team in player_points:
            dist = np.linalg.norm(np.array(point) - np.array(ball_point))
            if closest_dist is None or dist < closest_dist:
                closest_dist = dist
                closest_team = team

        print(f"frame {frame_index}: closest team {closest_team}, distance {closest_dist:.1f}")
    cv2.imshow("possession_test", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()