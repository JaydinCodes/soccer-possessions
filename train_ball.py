"""Fine-tune a dedicated ball detector on the SNMOT ball labels.

The soccer ball is tiny (~15px), which is why the generic model only sees it
~2/3 of the time. A single-class detector trained at high resolution on real
labels does much better. Run prepare_ball_dataset.py first.

    python train_ball.py
"""

from ultralytics import YOLO


def main():
    model = YOLO("yolov8s.pt")   # more capacity for the hard tiny-ball task; still live-fast on GPU
    model.train(
        data="datasets/ball/dataset.yaml",
        imgsz=960,               # fits the 6GB GTX 1660 Ti (1280 spills to slow shared RAM)
        epochs=60,
        batch=4,                 # yolov8s @ 960 fits the 6GB card at batch 4
        device=0,
        workers=4,
        patience=20,             # let it train longer before early-stopping
        project="runs", name="ball_detector",
        exist_ok=True,
        # small-object friendly aug: heavier scale/translate, no big mosaic distort
        mosaic=1.0, close_mosaic=8, scale=0.5, translate=0.1,
    )
    print("done -> runs/ball_detector/weights/best.pt")


if __name__ == "__main__":
    main()
