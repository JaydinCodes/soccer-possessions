"""Entry point for the possession tracker.

Examples:
    # process the bundled match file with a live preview window
    python run.py --source data/match.mp4

    # phone camera (IP Webcam app) -- live tracking
    python run.py --source http://192.168.1.5:8080/video

    # laptop webcam
    python run.py --source 0

    # headless (no window) + save an annotated video
    python run.py --source data/match.mp4 --no-show --save outputs/annotated.mp4

Run the dashboard alongside it in another terminal:
    streamlit run app.py
"""

import argparse

from src.pipeline import run


def main():
    p = argparse.ArgumentParser(description="Soccer possession tracker")
    p.add_argument("--source", default="data/match.mp4",
                   help="video file, phone stream URL, or webcam index (e.g. 0)")
    p.add_argument("--no-show", action="store_true", help="don't open a preview window")
    p.add_argument("--save", default=None, help="path to save an annotated video")
    p.add_argument("--json", default=None, help="override output JSON path")
    args = p.parse_args()

    run(args.source, show=not args.no_show, save_path=args.save, output_json=args.json)


if __name__ == "__main__":
    main()
