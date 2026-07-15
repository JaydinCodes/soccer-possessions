"""Detection + ball tracking, model-agnostic.

The ``Detector`` wraps a YOLO model and figures out which of its classes mean
"ball", "player", and "referee" from the model's own label names. That way the
exact same pipeline runs on:

  * the fine-tuned football model  -> classes: ball, goalkeeper, player, referee
  * generic COCO yolov8            -> classes: person (player), sports ball

The fine-tuned model finds the small soccer ball in one full-frame pass, so
tiling (slicing the frame into 8 pieces, 8x the cost) is off by default and only
kicked in for the generic model, which can't see a small ball full-frame.

Referees are detected but returned separately so they never pollute the team
colour clustering or steal possession.
"""

import numpy as np
import supervision as sv
from ultralytics import YOLO

from . import config


def _roles_from_names(names):
    """Map a model's class ids to role sets by reading its label names.

    Returns (ball, goalkeeper, player, referee). Generic COCO has only
    person -> player and sports ball -> ball; the rest stay empty.
    """
    ball, goalkeeper, player, referee = set(), set(), set(), set()
    for i, name in names.items():
        n = name.lower()
        if "ball" in n:
            ball.add(i)
        elif "goalkeeper" in n or n in ("gk", "keeper"):
            goalkeeper.add(i)
        elif "referee" in n or n == "ref":
            referee.add(i)
        elif n in ("player", "person"):
            player.add(i)
    return ball, goalkeeper, player, referee


class Detector:
    """YOLO model + role resolution. Emits ByteTrack-ready person detections.

    We use predict, NOT model.track(): ByteTrack (inside track()) refuses to
    start a track for detections below ~0.6 confidence, and the small soccer
    ball is almost always below that, so track() drops the ball (measured 3%
    recall vs 67% with predict). We run the player tracker ourselves, over the
    high-confidence person detections only, and keep the ball on predict.
    """

    def __init__(self, model_path, use_tiling=None, imgsz=None, device=None):
        self.model = YOLO(model_path)
        (self.ball_ids, self.goalkeeper_ids,
         self.player_ids, self.referee_ids) = _roles_from_names(self.model.names)
        self.person_ids = self.goalkeeper_ids | self.player_ids | self.referee_ids
        self.use_tiling = config.USE_TILING if use_tiling is None else use_tiling
        self.imgsz = imgsz or config.IMG_SIZE
        self.device = device        # e.g. 0 for GPU, "cpu"; None lets YOLO decide

    def role_of(self, class_id):
        """Human-readable role for a detected class id."""
        if class_id in self.goalkeeper_ids:
            return "goalkeeper"
        if class_id in self.referee_ids:
            return "referee"
        return "player"

    def detect(self, frame):
        """One predict pass. Returns (persons, ball_candidates).

        persons         : sv.Detections of players + goalkeepers + referees,
                          ready to hand to the tracker. class_id is preserved so
                          each track can vote on its role over time.
        ball_candidates : list of (point, conf)
        """
        # Pass classes explicitly every call. Ultralytics remembers the last
        # `classes` filter on its predictor, so the ball-only crop search would
        # otherwise leak into this pass and drop all the players.
        result = self.model(
            frame, conf=config.DETECT_CONF, imgsz=self.imgsz,
            classes=sorted(self.ball_ids | self.person_ids),
            device=self.device, verbose=False,
        )[0]
        det = sv.Detections.from_ultralytics(result)

        ball_mask = np.isin(det.class_id, list(self.ball_ids)) if self.ball_ids \
            else np.zeros(len(det), dtype=bool)
        ball_candidates = []
        for (x1, y1, x2, y2), conf in zip(det.xyxy[ball_mask], det.confidence[ball_mask]):
            ball_candidates.append((((x1 + x2) / 2.0, (y1 + y2) / 2.0), float(conf)))

        person_mask = np.isin(det.class_id, list(self.person_ids))
        persons = det[person_mask]

        if not ball_candidates and self.use_tiling:
            ball_candidates = self._detect_ball_tiled(frame)
        return persons, ball_candidates

    def detect_ball_crop(self, frame, center):
        """Zoom into a window around `center` and detect the ball there.

        Used as a fallback when the full-frame pass missed the ball: cropping to
        a small window and re-running at BALL_CROP_IMGSZ effectively magnifies
        the ball, so the detector that couldn't see it full-frame often can now.
        Returns candidates in FULL-frame coordinates.
        """
        if center is None:
            return []
        h, w = frame.shape[:2]
        half = config.BALL_CROP_HALF
        cx, cy = int(center[0]), int(center[1])
        x1, y1 = max(0, cx - half), max(0, cy - half)
        x2, y2 = min(w, cx + half), min(h, cy + half)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return []

        res = self.model(
            crop, conf=config.BALL_CROP_CONF, imgsz=config.BALL_CROP_IMGSZ,
            classes=sorted(self.ball_ids), device=self.device, verbose=False,
        )[0]
        candidates = []
        for box in res.boxes:
            bx1, by1, bx2, by2 = map(float, box.xyxy[0])
            pt = ((bx1 + bx2) / 2.0 + x1, (by1 + by2) / 2.0 + y1)
            candidates.append((pt, float(box.conf[0])))
        return candidates

    def _detect_ball_tiled(self, frame):
        """Slice the frame into tiles and return EVERY ball candidate found."""
        height, width = frame.shape[:2]
        tile_w = width // config.TILE_COLS
        tile_h = height // config.TILE_ROWS
        ball_list = sorted(self.ball_ids)

        candidates = []
        for row in range(config.TILE_ROWS):
            for col in range(config.TILE_COLS):
                xo, yo = col * tile_w, row * tile_h
                tile = frame[yo:yo + tile_h, xo:xo + tile_w]
                res = self.model(tile, conf=config.TILE_CONF,
                                 classes=ball_list, verbose=False)
                for box in res[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    pt = ((x1 + x2) / 2.0 + xo, (y1 + y2) / 2.0 + yo)
                    candidates.append((pt, float(box.conf[0])))
        return candidates


def select_ball(candidates, player_feet, last_point):
    """Pick the best ball candidate given the game context.

    Rather than trusting raw confidence (which lets a blob on the penalty spot
    win), we score each candidate on three things:

        score = confidence
              + PROX_WEIGHT * (how close it is to a player's feet)
              + CONT_WEIGHT * (how close it is to where the ball just was)

    A faint blob at a player's feet beats a crisp blob stranded in open space --
    the "ball by a player's feet" idea. A real loose ball still wins when nothing
    is around, carried by confidence + continuity.

    candidates  : list of (point, conf)
    player_feet : list of feet points (x, y)
    last_point  : last accepted ball point, or None
    """
    if not candidates:
        return None

    best_point, best_score = None, None
    for pt, conf in candidates:
        score = conf
        if player_feet:
            prox = min(np.hypot(pt[0] - fx, pt[1] - fy) for fx, fy in player_feet)
            score += config.BALL_PROX_WEIGHT * max(0.0, 1.0 - prox / config.BALL_PROX_SCALE)
        if last_point is not None:
            jump = np.hypot(pt[0] - last_point[0], pt[1] - last_point[1])
            score += config.BALL_CONT_WEIGHT * max(0.0, 1.0 - jump / config.BALL_CONT_SCALE)
        if best_score is None or score > best_score:
            best_score, best_point = score, pt
    return best_point


class BallTracker:
    """Kalman-filtered ball tracker with gap interpolation.

    The detector loses the ball on ~a third of frames (motion blur, occlusion,
    the ball leaving the pitch plane). A raw per-frame position therefore blinks
    on and off, breaking the trail and dropping possession mid-dribble.

    A constant-velocity Kalman filter fixes this. It models the ball's state as
    (x, y, vx, vy):
      * on a detection -> "correct" the estimate toward the measurement (this
        also smooths detector jitter), after rejecting implausible teleports;
      * on a miss -> "coast": predict where the ball went from its last velocity
        and report that, so the trail and possession stay continuous.

    Coasting is capped (BALL_MAX_INTERP) so we never invent a ball for long --
    after that the track is dropped and re-acquired fresh.

    update() returns (point, interpolated): interpolated=True means the point
    was predicted through a gap rather than actually detected this frame.
    """

    def __init__(self):
        import cv2
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.initialized = False
        self.misses = 0
        self.last_point = None

    def _init(self, pt):
        self.kf.statePost = np.array([[pt[0]], [pt[1]], [0], [0]], np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = True
        self.misses = 0
        self.last_point = pt

    def predicted_center(self):
        """Where the ball is expected THIS frame (position + velocity), without
        advancing the filter. Best crop centre for the targeted ball search."""
        if not self.initialized:
            return self.last_point
        s = self.kf.statePost
        return (float(s[0, 0] + s[2, 0]), float(s[1, 0] + s[3, 0]))

    def update(self, candidate):
        """Feed one raw candidate point (or None). Return (point, interpolated)."""
        if not self.initialized:
            if candidate is not None:
                self._init(candidate)
                return candidate, False
            return None, False

        predicted = self.kf.predict()
        pred_pt = (float(predicted[0, 0]), float(predicted[1, 0]))

        if candidate is not None:
            jump = np.hypot(candidate[0] - pred_pt[0], candidate[1] - pred_pt[1])
            budget = config.MAX_JUMP_PER_FRAME * (self.misses + 1)
            if jump < budget or self.misses > config.MAX_COAST:
                measurement = np.array([[np.float32(candidate[0])],
                                        [np.float32(candidate[1])]])
                corrected = self.kf.correct(measurement)
                self.misses = 0
                self.last_point = (float(corrected[0, 0]), float(corrected[1, 0]))
                return self.last_point, False

        # Miss (or rejected teleport): coast on the prediction, up to the cap.
        self.misses += 1
        if self.misses <= config.BALL_MAX_INTERP:
            self.kf.statePost = predicted.copy()
            self.last_point = pred_pt
            return pred_pt, True

        # Lost for too long -- drop the track and re-acquire next detection.
        self.initialized = False
        self.last_point = None
        return None, False
