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
#
# We prefer the OpenVINO export (models/football_openvino_model/) when present:
# on an Intel CPU it runs the SAME model ~1.25x faster with no accuracy loss.
# Create it once with:  python -c "from ultralytics import YOLO; \
#   YOLO('models/football.pt').export(format='openvino', dynamic=True)"
MODEL_PATH = "models/football.pt"
OPENVINO_PATH = "models/football_openvino_model"
FALLBACK_MODEL_PATH = "yolov8n.pt"

# Optional dedicated ball model (fine-tuned on SNMOT ball labels). When present
# it replaces the main model's generic ball detection -- ~2x the recall on real
# footage. Runs as a second small pass; on GPU that's fine. Set to None to use
# only the main model's ball class.
BALL_MODEL_PATH = "models/ball.pt"
BALL_MODEL_IMGSZ = 960

# Inference size. The ball is small and resolution-hungry: ball recall is ~69%
# at 1280 but ~42% at 960. Players survive low res fine. 1280 = accurate/slower,
# 960 = faster/live (Kalman interpolation cushions the lost ball frames).
IMG_SIZE = 1280
DETECT_CONF = 0.15               # confidence threshold for the detection pass

# Tiling: only needed for the generic COCO model, which can't see a small ball
# full-frame. The fine-tuned model doesn't need it, so it's off by default.
USE_TILING = False

# Process every Nth frame. 1 = every frame (needed for smooth live tracking),
# higher = faster but choppier. For offline file processing 3-5 is a good balance.
PROCESS_EVERY = 2

# ---------------------------------------------------------------------------
# Player tracking (ByteTrack) + per-track voting
# ---------------------------------------------------------------------------
# We only process every PROCESS_EVERY-th frame, so the effective frame rate the
# tracker sees is source_fps / PROCESS_EVERY. ByteTrack uses this to decide how
# long to keep a lost track alive.
TRACK_FRAME_RATE = 12

# ---------------------------------------------------------------------------
# Ball detection (tiling fallback, only used when USE_TILING is on)
# ---------------------------------------------------------------------------
TILE_COLS = 4
TILE_ROWS = 2
TILE_CONF = 0.15

# Temporal continuity: reject ball jumps that are physically implausible.
MAX_JUMP_PER_FRAME = 150   # px the ball can plausibly move in ONE processed frame
MAX_COAST = 15             # after this many misses, allow re-acquiring anywhere
BALL_MAX_INTERP = 8        # max consecutive frames to Kalman-interpolate through a gap
BALL_TRAIL_LEN = 25        # how many past ball points to draw as a trail

# Targeted crop search: when the full-frame pass misses the ball, zoom into a
# window around the Kalman-predicted position and re-detect there. The ball
# fills more pixels in the crop, so recall goes up. Only one small extra
# inference, and only on frames that missed -- cheap, especially on GPU.
BALL_CROP_SEARCH = True
BALL_CROP_HALF = 160       # half-size of the search window (px) around the prediction
BALL_CROP_IMGSZ = 640      # inference size for the crop (upscales the small ball)
BALL_CROP_CONF = 0.10      # lower conf ok -- the ball is clearer when zoomed in

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

# Referee heuristics: the model sometimes labels a referee as a "player".
# Two independent signals relabel such a track as a referee, once it's been read
# enough times (REF_MIN_OBS):
#   1. Colour: a real outfield player matches one of the two team colours most
#      of the time; a ref in a neutral kit matches neither.
#   2. Hi-vis: a fluorescent yellow-green ref shirt is much brighter/more
#      saturated than night grass, so a torso that's consistently hi-vis is an
#      official (this survives the grass mask, which eats the plain-green hue).
REF_MIN_OBS = 8            # reads before the heuristics may fire
REF_MAX_TEAM_FRAC = 0.2    # signal 1: <20% of reads matched a team -> referee
REF_HIVIS_FRAC = 0.5       # signal 2: >50% of reads were hi-vis -> referee

# Hi-vis mask: bright, saturated yellow-green (fluorescent official kit).
HIVIS_LOWER = (25, 120, 170)
HIVIS_UPPER = (50, 255, 255)
HIVIS_MIN_FRAC = 0.1       # torso fraction in the hi-vis band to count as hi-vis

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
GOALKEEPER_BGR = (0, 140, 255)    # goalkeepers drawn in orange, labelled "GK"
TEAM_NAMES = {0: "Team A", 1: "Team B"}
