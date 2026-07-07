#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]


SOURCES = [
    (
        "deepsort_tuned_f300.jpg",
        "runs/new_video_report/track_compare/deepsort_tuned2/video.mp4",
    ),
    (
        "bytetrack_tuned_f300.jpg",
        "runs/new_video_report/track_compare/bytetrack_tuned/video_bytetrack_lite.mp4",
    ),
    (
        "deepsort_default_f300.jpg",
        "runs/new_video_report/tracker_param_compare/deepsort_default2/video.mp4",
    ),
    (
        "deepsort_tuned_param_f300.jpg",
        "runs/new_video_report/tracker_param_compare/deepsort_tuned2/video.mp4",
    ),
    (
        "bytetrack_default_f300.jpg",
        "runs/new_video_report/tracker_param_compare/bytetrack_default/video_bytetrack_lite.mp4",
    ),
    (
        "bytetrack_tuned_param_f300.jpg",
        "runs/new_video_report/tracker_param_compare/bytetrack_tuned/video_bytetrack_lite.mp4",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract same-frame runtime screenshots for the report.")
    parser.add_argument("--frame", type=int, default=300, help="1-based frame index to extract.")
    parser.add_argument("--output", default="report_assets/new_video_report/runtime_frames", help="Output directory.")
    return parser.parse_args()


def extract_frame(video_path: Path, output_path: Path, frame_number: int) -> bool:
    if not video_path.exists():
        print(f"warning: missing video: {video_path}")
        return False

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"warning: failed to open video: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target = max(1, frame_number)
    if total_frames > 0:
        target = min(target, total_frames)

    cap.set(cv2.CAP_PROP_POS_FRAMES, target - 1)
    ok, frame = cap.read()
    if not ok and total_frames > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ok, frame = cap.read()
        target = total_frames
    cap.release()

    if not ok:
        print(f"warning: failed to read frame from: {video_path}")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)
    print(f"saved frame {target}: {output_path}")
    return True


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    written = 0
    for filename, rel_video in SOURCES:
        video_path = ROOT / rel_video
        output_path = output_dir / filename
        if extract_frame(video_path, output_path, args.frame):
            written += 1

    print(f"wrote {written}/{len(SOURCES)} runtime frames to {output_dir}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
