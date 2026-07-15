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
from ultralytics import YOLO

from . import config


def _roles_from_names(names):
    """Map a model's class ids to roles by reading its label names."""
    ball, player, referee = set(), set(), set()
    for i, name in names.items():
        n = name.lower()
        if "ball" in n:
            ball.add(i)
        elif "referee" in n or n == "ref":
            referee.add(i)
        elif n in ("player", "goalkeeper", "person"):
            player.add(i)
    return ball, player, referee


class Detector:
    """YOLO model + role resolution + optional tiled ball fallback."""

    def __init__(self, model_path, use_tiling=None):
        self.model = YOLO(model_path)
        self.ball_ids, self.player_ids, self.referee_ids = _roles_from_names(self.model.names)
        self.use_tiling = config.USE_TILING if use_tiling is None else use_tiling

    def detect(self, frame):
        """One detection pass. Returns (players, referees, ball_candidates).

        players         : list of (xyxy_tuple, track_id)   -- players + goalkeepers
        referees        : list of xyxy_tuple                -- drawn but ignored by logic
        ball_candidates : list of (point, conf)

        Note: we use predict, NOT model.track(). ByteTrack refuses to start a
        track for detections below ~0.6 confidence, and the small soccer ball is
        almost always detected below that -- so track() silently drops the ball
        (measured: 3% ball recall with track vs 67% with predict on this clip).
        We don't use player track IDs anywhere, so predict is strictly better.
        """
        results = self.model(
            frame, conf=config.DETECT_CONF, imgsz=config.IMG_SIZE, verbose=False,
        )
        players, referees, ball_candidates = [], [], []
        for box in results[0].boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cls in self.ball_ids:
                ball_candidates.append((((x1 + x2) / 2.0, (y1 + y2) / 2.0), float(box.conf[0])))
            elif cls in self.referee_ids:
                referees.append((x1, y1, x2, y2))
            elif cls in self.player_ids:
                players.append(((x1, y1, x2, y2), -1))

        if not ball_candidates and self.use_tiling:
            ball_candidates = self._detect_ball_tiled(frame)
        return players, referees, ball_candidates

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
    """Accepts/rejects raw ball detections using physical plausibility.

    A detection is accepted if the ball didn't "teleport" further than it
    plausibly could since we last saw it. After a long miss (MAX_COAST) we drop
    continuity and accept anywhere, so we can re-acquire after an occlusion.
    """

    def __init__(self):
        self.last_point = None
        self.frames_since = 0

    def update(self, candidate):
        """Feed one raw candidate point (or None). Return the accepted point or None."""
        if candidate is None:
            self.frames_since += 1
            return None

        if self.last_point is None:
            accept = True
        else:
            gap = self.frames_since + 1
            budget = config.MAX_JUMP_PER_FRAME * gap
            jump = np.linalg.norm(np.array(candidate) - np.array(self.last_point))
            accept = (jump < budget) or (self.frames_since > config.MAX_COAST)

        if accept:
            self.last_point = candidate
            self.frames_since = 0
            return candidate

        self.frames_since += 1
        return None
