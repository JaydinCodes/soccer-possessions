"""Turn the SoccerNet (SNMOT) sequences into a YOLO ball-detection dataset.

Each SNMOT-XXX/ has:
  img1/000001.jpg ...           frames
  gt/gt.txt                     MOT labels: frame,track_id,x,y,w,h,...
  gameinfo.ini                  maps track_id -> role (incl. which id is the ball)

We find the ball's track_id from gameinfo.ini (it differs per sequence), pull its
box on every frame, and write a YOLO label (class 0 = ball, normalised cx cy w h)
NEXT TO each image. Then we list the frames in train.txt / val.txt and point a
dataset.yaml at them. Images are referenced in place -- nothing is copied.

Split: a couple of whole sequences are held out for val (no frame leakage).

    python prepare_ball_dataset.py
"""

import glob
import os
import re

EXTRACT_DIR = "data/snmot_extract"
DATASET_DIR = "datasets/ball"
VAL_SEQUENCES = {"SNMOT-099", "SNMOT-100"}   # held out for validation


def ball_track_id(gameinfo_path):
    """Return the track_id whose role is 'ball', or None."""
    for line in open(gameinfo_path):
        m = re.match(r"\s*trackletID_(\d+)\s*=\s*ball", line, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def seq_dims(seqinfo_path):
    w = h = None
    for line in open(seqinfo_path):
        if line.startswith("imWidth"):
            w = int(line.split("=")[1])
        elif line.startswith("imHeight"):
            h = int(line.split("=")[1])
    return w, h


def process_sequence(seq_dir):
    """Write ball YOLO labels into seq_dir/img1/. Return list of image paths."""
    name = os.path.basename(seq_dir.rstrip("/\\"))
    bid = ball_track_id(os.path.join(seq_dir, "gameinfo.ini"))
    if bid is None:
        print(f"[skip] {name}: no ball tracklet")
        return []
    W, H = seq_dims(os.path.join(seq_dir, "seqinfo.ini"))
    img_dir = os.path.join(seq_dir, "img1")

    # frame -> ball box (x, y, w, h)
    ball_by_frame = {}
    for line in open(os.path.join(seq_dir, "gt", "gt.txt")):
        p = line.strip().split(",")
        if len(p) < 6:
            continue
        if int(p[1]) == bid:
            fr = int(p[0])
            ball_by_frame[fr] = (float(p[2]), float(p[3]), float(p[4]), float(p[5]))

    images = []
    for img_path in sorted(glob.glob(os.path.join(img_dir, "*.jpg"))):
        fr = int(os.path.splitext(os.path.basename(img_path))[0])
        label_path = os.path.splitext(img_path)[0] + ".txt"
        if fr in ball_by_frame:
            x, y, bw, bh = ball_by_frame[fr]
            cx, cy = (x + bw / 2) / W, (y + bh / 2) / H
            with open(label_path, "w") as f:
                f.write(f"0 {cx:.6f} {cy:.6f} {bw / W:.6f} {bh / H:.6f}\n")
        else:
            open(label_path, "w").close()   # no ball this frame -> background
        images.append(os.path.abspath(img_path))
    print(f"[ok] {name}: ball id {bid}, {len(ball_by_frame)}/{len(images)} frames with ball")
    return images


def main():
    seqs = sorted(glob.glob(os.path.join(EXTRACT_DIR, "SNMOT-*")))
    if not seqs:
        raise SystemExit(f"no sequences under {EXTRACT_DIR}")

    # Consecutive video frames are nearly identical, so subsample the training
    # frames (keep every TRAIN_STRIDE-th) -- much faster training, negligible loss.
    TRAIN_STRIDE = 3
    train, val = [], []
    for seq in seqs:
        imgs = process_sequence(seq)
        if os.path.basename(seq) in VAL_SEQUENCES:
            val.extend(imgs[::2])          # val: every 2nd frame is plenty
        else:
            train.extend(imgs[::TRAIN_STRIDE])

    os.makedirs(DATASET_DIR, exist_ok=True)
    train_txt = os.path.abspath(os.path.join(DATASET_DIR, "train.txt"))
    val_txt = os.path.abspath(os.path.join(DATASET_DIR, "val.txt"))
    open(train_txt, "w").write("\n".join(train) + "\n")
    open(val_txt, "w").write("\n".join(val) + "\n")

    yaml_path = os.path.abspath(os.path.join(DATASET_DIR, "dataset.yaml"))
    with open(yaml_path, "w") as f:
        f.write(f"train: {train_txt}\n")
        f.write(f"val: {val_txt}\n")
        f.write("nc: 1\n")
        f.write("names: [ball]\n")
    print(f"\ntrain: {len(train)} images | val: {len(val)} images")
    print(f"dataset.yaml -> {yaml_path}")


if __name__ == "__main__":
    main()
