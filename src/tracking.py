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
        self.team_votes = defaultdict(Counter)   # track_id -> Counter(team) [confident only]
        self.team_reads = defaultdict(int)       # track_id -> total colour reads
        self.hivis_reads = defaultdict(int)      # track_id -> reads that were hi-vis

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

    def observe_team(self, track_id, team, confident, is_hivis=False):
        """Record one jersey-colour read for a track.

        Every read (that produced a usable colour) counts toward the total;
        only confident reads vote for a team, and hi-vis reads are tallied too.
        The ratios drive the referee heuristics in effective_role().
        """
        self.team_reads[track_id] += 1
        if confident and team is not None:
            self.team_votes[track_id][team] += 1
        if is_hivis:
            self.hivis_reads[track_id] += 1

    def team_of(self, track_id):
        votes = self.team_votes.get(track_id)
        if not votes:
            return None
        return votes.most_common(1)[0][0]

    def effective_role(self, track_id, model_role):
        """Final role, applying the referee-by-colour heuristic.

        The model's goalkeeper/referee labels are trusted as-is. A "player" that
        has been colour-read enough times but almost never matched either team is
        relabelled a referee (hi-vis kit that fools the detector).
        """
        if model_role != "player":
            return model_role
        reads = self.team_reads.get(track_id, 0)
        if reads < config.REF_MIN_OBS:
            return "player"
        # Signal 2: consistently hi-vis kit -> official.
        if self.hivis_reads.get(track_id, 0) / reads >= config.REF_HIVIS_FRAC:
            return "referee"
        # Signal 1: colour matches neither team.
        matched = sum(self.team_votes[track_id].values())
        if matched / reads < config.REF_MAX_TEAM_FRAC:
            return "referee"
        return "player"
