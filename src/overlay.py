"""All drawing on the video frame lives here.

  * Players: box tinted by team (yellow / magenta), grey if unclassified.
  * Ball: BLUE box + a fading BLUE trail of where it's been.
  * HUD: a live possession bar across the top so the video is self-explanatory.
"""

import cv2
import numpy as np

from . import config


def draw_person(frame, box_xyxy, role, team, track_id=None):
    """Draw one tracked person, coloured/labelled by their voted role.

    role : "player" | "goalkeeper" | "referee"
    team : 0/1 for a classified player, else None
    """
    x1, y1, x2, y2 = box_xyxy
    if role == "referee":
        color, label = config.REFEREE_BGR, "REF"
    elif role == "goalkeeper":
        color, label = config.GOALKEEPER_BGR, "GK"
    elif team is None:
        color, label = config.UNCLASSIFIED_BGR, "?"
    else:
        color, label = config.TEAM_BGR[team], config.TEAM_NAMES[team]

    if track_id is not None and role != "referee":
        label = f"{label} #{track_id}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_ball(frame, trail):
    """Draw the blue ball box + a fading blue trail.

    trail is a sequence of recent (x, y) points, oldest first. The most recent
    point gets the box; older points fade out along the trail.
    """
    pts = [p for p in trail if p is not None]
    if not pts:
        return

    # Fading trail: older segments are thinner + darker.
    n = len(pts)
    for i in range(1, n):
        frac = i / n
        thickness = max(1, int(3 * frac))
        b, g, r = config.BALL_BGR
        col = (int(b * frac), int(g * frac), int(r * frac))
        cv2.line(frame, (int(pts[i - 1][0]), int(pts[i - 1][1])),
                 (int(pts[i][0]), int(pts[i][1])), col, thickness)

    # Current ball position: blue box + dot.
    cx, cy = pts[-1]
    cx, cy = int(cx), int(cy)
    cv2.rectangle(frame, (cx - 12, cy - 12), (cx + 12, cy + 12), config.BALL_BGR, 2)
    cv2.circle(frame, (cx, cy), 3, config.BALL_BGR, -1)
    cv2.putText(frame, "ball", (cx - 12, cy - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, config.BALL_BGR, 1)


def draw_hud(frame, summary, possessor):
    """A translucent top bar: live possession split + current possessor."""
    h, w = frame.shape[:2]
    pct_a = summary["team_a_pct"]
    pct_b = summary["team_b_pct"]

    bar_h = 34
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    # Split bar showing the live possession ratio.
    split = int(w * pct_a / 100.0) if (pct_a + pct_b) > 0 else w // 2
    cv2.rectangle(frame, (0, 0), (split, 6), config.TEAM_BGR[0], -1)
    cv2.rectangle(frame, (split, 0), (w, 6), config.TEAM_BGR[1], -1)

    cv2.putText(frame, f"Team A {pct_a:.0f}%", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.TEAM_BGR[0], 2)
    txt_b = f"Team B {pct_b:.0f}%"
    (tw, _), _ = cv2.getTextSize(txt_b, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(frame, txt_b, (w - tw - 10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.TEAM_BGR[1], 2)

    if possessor in (0, 1):
        who = config.TEAM_NAMES[possessor]
        col = config.TEAM_BGR[possessor]
    else:
        who = "Loose ball"
        col = config.UNCLASSIFIED_BGR
    (tw, _), _ = cv2.getTextSize(who, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(frame, who, (w // 2 - tw // 2, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
