"""Central place for every tunable knob and the shared colour scheme.

Keeping these in one file means the video overlay and the Streamlit dashboard
can import the SAME team colours, so what you see in the video matches the
dashboard exactly.
"""

# ---------------------------------------------------------------------------
# Model / inference
# ---------------------------------------------------------------------------
# Default is the fine-tuned football model (classes: ball, goalkeeper, player,
# referee). It finds the small soccer ball in a SINGLE full-frame pass, so we no
# longer need the slow 8-tile search. Fall back to "yolov8n.pt" (generic COCO)
# if the file is missing -- see detection.Detector, which auto-detects which
# classes mean ball / player / referee, so both models just work.
MODEL_PATH = "models/football.pt"
FALLBACK_MODEL_PATH = "yolov8n.pt"

IMG_SIZE = 1280                  # YOLO inference size (bigger = better small objects, slower)
DETECT_CONF = 0.15               # confidence threshold for the detection pass

# Tiling: only needed for the generic COCO model, which can't see a small ball
# full-frame. The fine-tuned model doesn't need it, so it's off by default.
USE_TILING = False

# Process every Nth frame. 1 = every frame (needed for smooth live tracking),
# higher = faster but choppier. For offline file processing 3-5 is a good balance.
PROCESS_EVERY = 2

# ---------------------------------------------------------------------------
# Ball detection (tiling fallback, only used when USE_TILING is on)
# ---------------------------------------------------------------------------
TILE_COLS = 4
TILE_ROWS = 2
TILE_CONF = 0.15

# Temporal continuity: reject ball jumps that are physically implausible.
MAX_JUMP_PER_FRAME = 150   # px the ball can plausibly move in ONE processed frame
MAX_COAST = 15             # after this many misses, allow re-acquiring anywhere
BALL_TRAIL_LEN = 25        # how many past ball points to draw as a trail

# Ball candidate scoring: prefer the blob near a player's feet / near where the
# ball just was, not just the most confident blob (kills penalty-spot false
# positives). See detection.select_ball.
BALL_PROX_WEIGHT = 0.6     # how much "near a player's feet" boosts a candidate
BALL_PROX_SCALE = 200.0    # px: proximity bonus fades to 0 by this distance
BALL_CONT_WEIGHT = 0.5     # how much "near the last ball position" boosts it
BALL_CONT_SCALE = 300.0    # px: continuity bonus fades to 0 by this distance

# ---------------------------------------------------------------------------
# Team clustering (hue based)
# ---------------------------------------------------------------------------
TEAM_FIT_SAMPLES = 60      # torso colour samples to collect before fitting k-means
UNCLASSIFIED_DIST = 0.55   # distance (in normalised hue-feature space) beyond
                           # which a player is "unknown" rather than forced onto a team

# Green pitch mask (HSV) so grass never pollutes the jersey colour.
GREEN_LOWER = (35, 40, 40)
GREEN_UPPER = (85, 255, 255)

# ---------------------------------------------------------------------------
# Possession
# ---------------------------------------------------------------------------
# Perspective-aware possession. A fixed pixel threshold is wrong: a ball far
# from camera spans fewer pixels than one near it. So instead of a constant, we
# scale the "on the ball" distance by the player's OWN box height -- a player
# near the camera is tall (big threshold), a distant player is short (small
# threshold). The player's pixel-height is a free depth cue.
POSSESSION_HEIGHT_RATIO = 1.1   # ball within 1.1x a player's box-height of their feet
POSSESSION_DIST_FLOOR = 55      # px floor so tiny/far boxes still get a fair radius
SMOOTH_WINDOW = 9               # processed frames to majority-vote over (kills flicker)

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUTPUT_JSON = "outputs/possession_log.json"
JSON_FLUSH_EVERY = 15      # write JSON every N processed frames so the dashboard is live

# ---------------------------------------------------------------------------
# Shared colours  (BGR for OpenCV video, HEX for the Streamlit dashboard)
# Ball is BLUE (box + trail). Teams are yellow / magenta so nothing clashes
# with the blue ball or the green pitch.
# ---------------------------------------------------------------------------
BALL_BGR = (255, 128, 0)          # blue (OpenCV is BGR)
BALL_HEX = "#0080ff"

TEAM_BGR = {
    0: (0, 215, 255),             # team A -> yellow/amber
    1: (255, 0, 255),             # team B -> magenta
}
TEAM_HEX = {
    0: "#ffd700",                 # team A
    1: "#ff00ff",                 # team B
}
UNCLASSIFIED_BGR = (190, 190, 190)
REFEREE_BGR = (255, 255, 255)     # referees drawn in white, excluded from teams
TEAM_NAMES = {0: "Team A", 1: "Team B"}
