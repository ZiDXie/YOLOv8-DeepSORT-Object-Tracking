#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
MYENV_PYTHON = Path("/home/xie/miniconda3/envs/myenv/bin/python")
DEFAULT_MODEL = ROOT / "yolov8m.pt"
DEFAULT_SOURCE = ROOT / "video" / "video.mp4"
BYTETRACK_SCRIPT = ROOT / "tools" / "bytetrack_verify.py"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "track_compare" / "bytetrack_main"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv8 + ByteTrackLite person tracking.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Video file to process.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="YOLOv8 weights file.")
    parser.add_argument("--device", default=None, help="Inference device, e.g. cpu or 0.")
    parser.add_argument("--imgsz", type=int, default=960, help="Input image size.")
    parser.add_argument("--low-conf", type=float, default=0.03, help="Low-score detection threshold.")
    parser.add_argument("--track-thresh", type=float, default=0.50, help="Minimum score to initialize a new track.")
    parser.add_argument("--iou", type=float, default=0.85, help="NMS IoU threshold.")
    parser.add_argument("--match-thresh", type=float, default=0.80, help="IoU threshold for track association.")
    parser.add_argument("--max-age", type=int, default=60, help="Frames to keep unmatched tracks alive.")
    parser.add_argument("--max-frames", type=int, default=0, help="Frames to process; 0 means full video.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--show", action="store_true", help="Display the video during inference.")
    return parser.parse_args()


def resolve_device(device: str | None) -> str:
    if device:
        return device
    return "0" if torch.cuda.is_available() else "cpu"


def main() -> int:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    model = Path(args.model).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    device = resolve_device(args.device)

    python_bin = MYENV_PYTHON if MYENV_PYTHON.exists() else Path(sys.executable)

    if not BYTETRACK_SCRIPT.exists():
        print(f"Missing ByteTrack script: {BYTETRACK_SCRIPT}", file=sys.stderr)
        return 1
    if not source.exists():
        print(f"Missing source video: {source}", file=sys.stderr)
        return 1
    if not model.exists():
        print(f"Missing model weights: {model}", file=sys.stderr)
        return 1

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    print(f"Using python: {python_bin}")
    print(f"Using source: {source}")
    print(f"Using model: {model}")
    print(f"Using device: {device}")
    print(f"Using imgsz: {args.imgsz}")
    print(f"Using low_conf: {args.low_conf}")
    print(f"Using track_thresh: {args.track_thresh}")
    print(f"Using iou: {args.iou}")
    print(f"Using match_thresh: {args.match_thresh}")
    print(f"Using max_age: {args.max_age}")
    print(f"Using max_frames: {args.max_frames}")
    print(f"Using output_dir: {output_dir}")

    cmd = [
        str(python_bin),
        str(BYTETRACK_SCRIPT),
        "--source",
        str(source),
        "--model",
        str(model),
        "--device",
        device,
        "--imgsz",
        str(args.imgsz),
        "--low-conf",
        str(args.low_conf),
        "--track-thresh",
        str(args.track_thresh),
        "--iou",
        str(args.iou),
        "--match-thresh",
        str(args.match_thresh),
        "--max-age",
        str(args.max_age),
        "--max-frames",
        str(args.max_frames),
        "--output-dir",
        str(output_dir),
    ]
    if args.show:
        cmd.append("--show")

    return subprocess.run(cmd, cwd=ROOT, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
