"""The main loop: source -> detections -> teams -> possession -> overlay -> JSON.

Writes ``outputs/possession_log.json`` incrementally (atomically) as it runs,
so the Streamlit dashboard shows live stats while the video is still playing.
"""

import json
import logging
import os
import tempfile
from collections import deque

# ByteTrack's camera-motion compensation spams a harmless warning on OpenCV 5
# (it falls back to identity). Quiet it so the console stays readable.
logging.getLogger("ultralytics").setLevel(logging.ERROR)

from . import config
from . import detection
from . import overlay
from . import possession
from .teams import TeamClassifier, jersey_feature


def _open_source(source):
    """Accept an int-like webcam index, a phone URL, or a file path."""
    import cv2
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    return cv2.VideoCapture(source)


def _write_json(path, summary, timeline, frames_processed, teams_ready):
    """Atomic write so the dashboard never reads a half-written file."""
    payload = {
        "summary": summary,
        "timeline": timeline,
        "meta": {
            "frames_processed": frames_processed,
            "teams_ready": teams_ready,
        },
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def run(source, show=True, save_path=None, output_json=None):
    """Process a video source end to end.

    source      : file path, phone stream URL, or webcam index ("0").
    show        : open an OpenCV preview window (press q to quit).
    save_path   : optional path to write the annotated video.
    output_json : where to write live possession stats (defaults to config).
    """
    import cv2

    output_json = output_json or config.OUTPUT_JSON

    model_path = config.MODEL_PATH
    if not os.path.exists(model_path):
        # Fine-tuned model not present -- fall back to generic COCO + tiling.
        print(f"[warn] {model_path} not found, falling back to "
              f"{config.FALLBACK_MODEL_PATH} with tiling.")
        model_path = config.FALLBACK_MODEL_PATH
        detector = detection.Detector(model_path, use_tiling=True)
    else:
        detector = detection.Detector(model_path)

    cap = _open_source(source)
    if not cap.isOpened():
        raise RuntimeError(f"could not open source: {source!r}")

    classifier = TeamClassifier()
    ball_tracker = detection.BallTracker()
    smoother = possession.PossessionSmoother()
    stats = possession.PossessionStats()
    trail = deque(maxlen=config.BALL_TRAIL_LEN)

    writer = None
    frame_index = 0
    processed = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            if frame_index % config.PROCESS_EVERY != 0:
                continue
            processed += 1

            players, referees, ball_candidates = detector.detect(frame)

            # -- teams: warm up the classifier, then classify --------------
            # Referees are excluded here, so their kit never skews the clusters.
            features = [jersey_feature(frame, box) for box, _ in players]
            if not classifier.ready:
                for feat in features:
                    classifier.add_sample(feat)
                if classifier.n_samples >= config.TEAM_FIT_SAMPLES:
                    classifier.fit()

            # -- players first: we need their feet to judge the ball --------
            for ref_box in referees:
                overlay.draw_referee(frame, ref_box)
            player_points = []
            for (box, _tid), feat in zip(players, features):
                team, confident = classifier.predict(feat)
                overlay.draw_player(frame, box, team, confident)
                feet = ((box[0] + box[2]) / 2.0, box[3])
                height = box[3] - box[1]
                player_points.append((feet, team if confident else None, height))
            feet_points = [feet for feet, _team, _h in player_points]

            # -- ball: pick the candidate that fits the game (near a player's
            # feet / near the last position), then apply the jump filter.
            chosen = detection.select_ball(ball_candidates, feet_points,
                                           ball_tracker.last_point)
            ball_point = ball_tracker.update(chosen)
            trail.append(ball_point)

            # -- possession -------------------------------------------------
            raw = possession.raw_possessor(ball_point, player_points)
            smoothed = smoother.update(raw)
            if classifier.ready:
                stats.record(frame_index, smoothed)

            # -- draw -------------------------------------------------------
            overlay.draw_ball(frame, trail)
            overlay.draw_hud(frame, stats.summary(), smoothed)

            # -- live JSON --------------------------------------------------
            if processed % config.JSON_FLUSH_EVERY == 0:
                _write_json(output_json, stats.summary(), stats.timeline,
                            processed, classifier.ready)

            if save_path is not None:
                if writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25
                    writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
                writer.write(frame)

            if show:
                cv2.imshow("soccer possession", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    # Final flush.
    _write_json(output_json, stats.summary(), stats.timeline, processed, classifier.ready)
    print(stats.summary())
    return stats.summary()
