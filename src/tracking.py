"""Player tracking + per-track voting.

The detector judges every frame from scratch, so a player's team and even their
role (player / goalkeeper / referee) can flip on a single noisy frame -- that's
why a referee occasionally showed up boxed as "Team B".

The fix is memory. We run ByteTrack over the person detections to give each
player a persistent id, then decide two things by MAJORITY VOTE over the whole
track, not per frame:

  * role  -- is this track usually a player, a goalkeeper, or a referee?
  * team  -- which team colour has this track worn most often?

One bad frame can't outvote a hundred good ones, so identities stay stable.
"""

import warnings
from collections import Counter, defaultdict

import numpy as np

# supervision 0.29 warns that sv.ByteTrack is deprecated; it still works and we
# pin the version, so silence the cosmetic warning.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import supervision as sv

from . import config


class PlayerTracker:
    """ByteTrack ids + rolling team/role votes per id."""

    def __init__(self, detector):
        self.detector = detector
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.tracker = sv.ByteTrack(frame_rate=config.TRACK_FRAME_RATE)
        self.role_votes = defaultdict(Counter)   # track_id -> Counter(role)
        self.team_votes = defaultdict(Counter)   # track_id -> Counter(team)

    def update(self, persons):
        """Advance the tracker one frame.

        persons : sv.Detections (players + goalkeepers + referees).
        Returns a list of dicts: {id, xyxy, role} with role already majority-voted.
        Team is voted separately via vote_team()/team_of() once we've read the
        jersey colour (which needs the frame).
        """
        tracked = self.tracker.update_with_detections(persons)
        out = []
        for xyxy, class_id, tid in zip(tracked.xyxy, tracked.class_id, tracked.tracker_id):
            tid = int(tid)
            self.role_votes[tid][self.detector.role_of(int(class_id))] += 1
            role = self.role_votes[tid].most_common(1)[0][0]
            box = tuple(map(int, xyxy))
            out.append({"id": tid, "xyxy": box, "role": role})
        return out

    def vote_team(self, track_id, team):
        if team is not None:
            self.team_votes[track_id][team] += 1

    def team_of(self, track_id):
        votes = self.team_votes.get(track_id)
        if not votes:
            return None
        return votes.most_common(1)[0][0]
