"""Hue-based team classification.

The old version clustered on raw BGR, which collapses under lighting changes
(the night clip read everything as one team). Colour in BGR mixes "what colour
is it" with "how bright is the light" into the same three numbers, so k-means
ends up splitting on brightness instead of jersey colour.

Fix: work in HSV and build a lighting-robust feature per player:

    feature = [ S*cos(H), S*sin(H), V ]

  * H (hue) is *what colour* the jersey is. We encode it as a point on a circle
    (cos/sin) so that red at hue 0 and red at hue 179 are neighbours, not
    opposites -- k-means can't cluster on a raw hue number because of that wrap.
  * We scale the circle by S (saturation) so washed-out / white kits collapse
    toward the origin instead of getting a random hue from sensor noise.
  * V (value/brightness) is kept as its own axis so we can still tell a white
    kit from a black kit (both low saturation, but very different brightness).

Grass is masked out first so the pitch never votes on the jersey colour.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans

from . import config


def _torso_patch(frame, box_xyxy):
    """Crop the upper half of a player box -- that's where the jersey is."""
    x1, y1, x2, y2 = box_xyxy
    torso_y2 = y1 + int((y2 - y1) * 0.5)
    return frame[y1:torso_y2, x1:x2]


def jersey_feature(frame, box_xyxy):
    """Return a lighting-robust colour feature for one player, or None.

    None means the crop was empty or entirely grass/skin (no jersey pixels to
    read), and the caller should skip this player rather than guess.
    """
    torso = _torso_patch(frame, box_xyxy)
    if torso.size == 0:
        return None

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

    # Mask out the green pitch so grass doesn't drag the average.
    green = cv2.inRange(hsv, np.array(config.GREEN_LOWER), np.array(config.GREEN_UPPER))
    jersey = cv2.bitwise_not(green)
    if cv2.countNonZero(jersey) < 10:      # basically all grass -> useless
        return None

    h, s, v = cv2.split(hsv)
    mask = jersey > 0
    # OpenCV hue is 0..179; map to a full circle in radians.
    hue_rad = h[mask].astype(np.float32) * (2.0 * np.pi / 180.0)
    sat = s[mask].astype(np.float32) / 255.0
    val = v[mask].astype(np.float32) / 255.0

    x = float(np.mean(sat * np.cos(hue_rad)))
    y = float(np.mean(sat * np.sin(hue_rad)))
    z = float(np.mean(val))
    return np.array([x, y, z], dtype=np.float32)


class TeamClassifier:
    """Learns two team clusters from early frames, then labels each player.

    Usage:
        tc = TeamClassifier()
        # feed features until tc.ready
        tc.add_sample(feat); ...
        tc.fit()
        team, confident = tc.predict(feat)
    """

    def __init__(self):
        self._samples = []
        self.kmeans = None
        self._order = (0, 1)   # remap raw cluster ids -> stable team ids

    # -- fitting -----------------------------------------------------------
    @property
    def ready(self):
        return self.kmeans is not None

    @property
    def n_samples(self):
        return len(self._samples)

    def add_sample(self, feature):
        if feature is not None:
            self._samples.append(feature)

    def fit(self):
        """Cluster the collected samples into two teams.

        Cluster ids are re-ordered by hue angle so "Team A" is deterministic
        across runs instead of flipping randomly with k-means init.
        """
        if len(self._samples) < 2:
            raise ValueError("need at least 2 samples to fit teams")
        data = np.array(self._samples, dtype=np.float32)
        km = KMeans(n_clusters=2, n_init=10, random_state=0)
        km.fit(data)

        # Order teams by hue angle of the centroid (atan2 of the x,y axes),
        # so the assignment is stable and reproducible.
        angles = [np.arctan2(c[1], c[0]) for c in km.cluster_centers_]
        order = sorted(range(2), key=lambda i: angles[i])
        self._order = tuple(order)
        self.kmeans = km

    # -- inference ---------------------------------------------------------
    def predict(self, feature):
        """Return (team_id, confident).

        confident is False when the player's colour is far from either team
        centroid (referee, keeper, weird lighting) -- caller can grey them out
        instead of forcing them onto a team and skewing possession.
        """
        if feature is None or self.kmeans is None:
            return None, False
        raw = int(self.kmeans.predict(feature.reshape(1, -1))[0])
        dist = float(np.linalg.norm(feature - self.kmeans.cluster_centers_[raw]))
        team = self._order.index(raw)     # remap to stable id
        return team, dist <= config.UNCLASSIFIED_DIST
