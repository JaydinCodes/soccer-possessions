"""Stitch a folder of ordered frames (.jpg/.png) into an mp4.

The mirror of VideoCapture: a VideoWriter reads frames from disk and writes them
into a video. Frames are used in sorted filename order, so name them so they
sort correctly (e.g. 000001.jpg, 000002.jpg).

Usage:
    python build_video.py                                   # data/img1 -> data/match.mp4
    python build_video.py data/img2                         # -> data/img2.mp4
    python build_video.py data/img2 data/game2.mp4 --fps 30
"""

import argparse
import os

import cv2

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def build(image_folder, output_path, fps):
    names = sorted(f for f in os.listdir(image_folder)
                   if f.lower().endswith(IMG_EXTS))
    if not names:
        raise SystemExit(f"no images ({', '.join(IMG_EXTS)}) found in {image_folder}")

    first = cv2.imread(os.path.join(image_folder, names[0]))
    if first is None:
        raise SystemExit(f"could not read {names[0]}")
    height, width = first.shape[:2]

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    for name in names:
        frame = cv2.imread(os.path.join(image_folder, name))
        if frame is None:
            print(f"[skip] unreadable: {name}")
            continue
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)
    writer.release()
    print(f"done: {len(names)} frames -> {output_path} ({width}x{height} @ {fps}fps)")


def main():
    p = argparse.ArgumentParser(description="Stitch ordered frames into an mp4")
    p.add_argument("folder", nargs="?", default="data/img1", help="folder of frames")
    p.add_argument("output", nargs="?", default=None,
                   help="output mp4 (default: <folder>.mp4)")
    p.add_argument("--fps", type=int, default=25)
    args = p.parse_args()

    output = args.output or (args.folder.rstrip("/\\") + ".mp4")
    build(args.folder, output, args.fps)


if __name__ == "__main__":
    main()
