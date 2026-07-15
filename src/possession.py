"""Who owns the ball, plus temporal smoothing.

Raw per-frame possession flickers: one noisy frame where a defender's foot is
momentarily closest flips the whole possession count. The plan called for
smoothing that never got added -- here it is.

We keep a sliding window of recent *raw* possessors and report the majority
vote. A single odd frame can't flip the smoothed result; possession only
changes hands when a team is consistently closest for most of the window.
"""

from collections import deque, Counter

import numpy as np

from . import config


def raw_possessor(ball_point, player_points):
    """Nearest player's team, if the ball is within that player's reach.

    player_points : list of (feet_point, team, height). ``height`` is the
    player's box height in px, used for a perspective-aware reach radius (see
    config.POSSESSION_HEIGHT_RATIO) instead of one fixed pixel threshold.

    Players with an unknown team (team is None) still block the ball -- they can
    "own" a loose ball so we don't wrongly credit a farther, classified player --
    but they don't count toward either team's total.
    """
    if ball_point is None or not player_points:
        return None

    ball = np.array(ball_point)
    best_team, best_slack = None, None
    for point, team, height in player_points:
        reach = max(config.POSSESSION_DIST_FLOOR,
                    config.POSSESSION_HEIGHT_RATIO * height)
        dist = np.linalg.norm(np.array(point) - ball)
        slack = dist - reach          # <= 0 means the ball is within reach
        if slack <= 0 and (best_slack is None or slack < best_slack):
            best_slack = slack
            best_team = team
    return best_team                  # None if no player is within reach


class PossessionSmoother:
    """Majority vote over a sliding window of raw possessors."""

    def __init__(self, window=None):
        self.window = window or config.SMOOTH_WINDOW
        self._buf = deque(maxlen=self.window)

    def update(self, raw_team):
        """Add one raw possessor (0, 1, or None) and return the smoothed one."""
        self._buf.append(raw_team)
        counts = Counter(self._buf)
        team, _ = counts.most_common(1)[0]
        return team


class PossessionStats:
    """Accumulates smoothed possession into per-team frame counts + a timeline."""

    def __init__(self):
        self.counts = {0: 0, 1: 0}
        self.loose = 0
        self.timeline = []

    def record(self, frame_index, team):
        if team in (0, 1):
            self.counts[team] += 1
        else:
            self.loose += 1
        self.timeline.append({"frame": frame_index, "team": team})

    def summary(self):
        owned = self.counts[0] + self.counts[1]
        pct_a = 100 * self.counts[0] / owned if owned else 0.0
        pct_b = 100 * self.counts[1] / owned if owned else 0.0
        return {
            "team_a_pct": round(pct_a, 1),
            "team_b_pct": round(pct_b, 1),
            "team_a_frames": self.counts[0],
            "team_b_frames": self.counts[1],
            "loose_frames": self.loose,
        }
