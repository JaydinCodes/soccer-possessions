# Soccer Possession Tracker

Detects players and the ball in soccer footage, auto-assigns teams by jersey
colour, tracks who has the ball, and shows live possession stats on a Streamlit
dashboard. Works on a video file, a phone camera stream, or a webcam.

## Model

Detection uses a **fine-tuned football model** (`models/football.pt`, classes:
`ball, goalkeeper, player, referee`). It finds the small soccer ball in a single
full-frame pass — no tiling needed — and separates referees so they don't skew
team colours. It's git-ignored (22 MB); download it once:

```bash
mkdir -p models
curl -L -o models/football.pt \
  https://huggingface.co/uisikdag/yolo-v8-football-players-detection/resolve/main/best.pt
```

If `models/football.pt` is missing, the pipeline automatically falls back to
generic `yolov8n.pt` + tiling. `detection.Detector` reads the model's own class
names, so either model just works — swap `config.MODEL_PATH` to try another.

**Speed.** The pipeline auto-detects the best backend:

- **NVIDIA GPU (best)** — PyTorch on CUDA, **~33 fps @ 1280** (real-time at full
  ball recall). Needs the CUDA build of torch, not the `+cpu` one:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
  python -c "import torch; print(torch.cuda.is_available())"   # -> True
  ```
- **CPU only** — inference is the bottleneck (~1 fps @ 1280). Use `--fast` for a
  live-usable ~6 fps: it runs OpenVINO @ 960 (export it once with
  `python -c "from ultralytics import YOLO; YOLO('models/football.pt').export(format='openvino', dynamic=True)"`).
  Ball recall drops to ~42% but Kalman interpolation cushions it.

Tune either path with `--imgsz` / `--every`.

## Run it

```bash
# 1. process a source (opens a preview window; press q to quit)
python run.py --source data/match.mp4                     # video file
python run.py --source http://192.168.1.5:8080/video      # phone (IP Webcam app)
python run.py --source 0                                   # laptop webcam

# headless + save an annotated video
python run.py --source data/match.mp4 --no-show --save outputs/annotated.mp4

# FAST preset for live: OpenVINO @ 960 + skip more frames (~6x faster on Intel CPU)
python run.py --source 0 --fast

# 2. in another terminal, open the live dashboard
streamlit run app.py
```

The tracker writes `outputs/possession_log.json` as it runs; the dashboard
re-reads it every second, so possession updates live.

## Layout (`src/`)

| File | Responsibility |
|---|---|
| `config.py` | Every tunable knob + the shared team colour scheme (video ↔ dashboard) |
| `teams.py` | **Hue-based** team clustering (HSV, green-masked, circular-hue k-means) |
| `detection.py` | Combined player+ball YOLO pass, tiling fallback, ball continuity filter |
| `possession.py` | Nearest-player possession + **majority-vote smoothing** |
| `overlay.py` | Team-tinted player boxes, blue ball box + trail, live HUD |
| `pipeline.py` | Main loop, wires it together, writes live JSON |

## What changed from the `*_test.py` prototypes

- **Teams cluster on hue, not raw BGR** — no longer collapses under stadium
  lighting (the old night-clip failure). See the docstring in `teams.py`.
- **Possession smoothing** — majority vote over a window kills frame-to-frame
  flicker (`config.SMOOTH_WINDOW`).
- **Perspective-aware possession** — the "on the ball" radius scales with each
  player's box height instead of a fixed 70px (`config.POSSESSION_HEIGHT_RATIO`).
- **Blue ball box + fading trail**, and a live possession HUD burned into the video.
- **Live dashboard** — auto-refreshes instead of needing a manual reload.
- **Faster** — one combined YOLO pass for players+ball; the 8-tile ball search
  only runs on frames where the cheap full-frame pass missed the ball.

## Known limitation / next lever

Ball detection uses generic YOLO (`yolov8n`, COCO "sports ball"). It's sparse
and produces false positives far from players (penalty spot, pitch markings),
which under-counts possession. The single biggest remaining accuracy win is a
**fine-tuned soccer-ball model** (e.g. a Roboflow one) dropped in via
`config.MODEL_PATH` — no other code needs to change.
