"""Central place for every tunable knob and the shared colour scheme.

Keeping these in one file means the video overlay and the Streamlit dashboard
can import the SAME team colours, so what you see in the video matches the
dashboard exactly.
"""

# ---------------------------------------------------------------------------
# Model / inference
# ---------------------------------------------------------------------------
MODEL_PATH = "yolov8n.pt"        # swap for a fine-tuned soccer model when ready
IMG_SIZE = 1280                  # YOLO inference size (bigger = better small objects, slower)
PLAYER_CONF = 0.15               # confidence threshold for players
BALL_CONF = 0.20                 # confidence threshold for the ball

PERSON_CLASS = 0                 # COCO class id for "person"
BALL_CLASS = 32                  # COCO class id for "sports ball"

# Process every Nth frame. 1 = every frame (needed for smooth live tracking),
# higher = faster but choppier. For offline file processing 3-5 is a good balance.
PROCESS_EVERY = 2

# ---------------------------------------------------------------------------
# Ball detection (tiling fallback)
# ---------------------------------------------------------------------------
# We first try to find the ball in the normal full-frame pass (free, since we
# already run it for players). Only if that misses do we pay for a tiled search.
TILE_COLS = 4
TILE_ROWS = 2
TILE_CONF = 0.15

# Temporal continuity: reject ball jumps that are physically implausible.
MAX_JUMP_PER_FRAME = 150   # px the ball can plausibly move in ONE processed frame
MAX_COAST = 15             # after this many misses, allow re-acquiring anywhere
BALL_TRAIL_LEN = 25        # how many past ball points to draw as a trail

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
POSSESSION_DIST = 70       # px from ball to a player's feet to count as "on the ball"
SMOOTH_WINDOW = 9          # processed frames to majority-vote over (kills flicker)

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
TEAM_NAMES = {0: "Team A", 1: "Team B"}
