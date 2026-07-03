#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "yolov8m.pt"
DEFAULT_SOURCE = ROOT / "ultralytics" / "yolo" / "v8" / "detect" / "deep_sort_pytorch" / "deep_sort" / "deep" / "checkpoint" / "demo.avi"
PREDICT_SCRIPT = ROOT / "ultralytics" / "yolo" / "v8" / "detect" / "predict.py"


def hydra_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv8 + DeepSORT tracking.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Video file to process.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="YOLOv8 weights file.")
    parser.add_argument("--device", default=None, help="Inference device, e.g. cpu or 0.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Input image size.")
    parser.add_argument("--conf", type=float, default=0.05, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.85, help="NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=1000, help="Maximum detections per frame.")
    parser.add_argument("--deepsort-min-conf", type=float, default=0.05, help="DeepSORT confidence threshold.")
    parser.add_argument("--deepsort-nms-overlap", type=float, default=1.00, help="DeepSORT NMS overlap threshold.")
    parser.add_argument("--deepsort-max-age", type=int, default=150, help="DeepSORT max missed frames before deleting a track.")
    parser.add_argument("--deepsort-n-init", type=int, default=1, help="DeepSORT frames required to confirm a track.")
    parser.add_argument("--debug-counts", action="store_true", help="Print YOLO/DeepSORT person counts per frame.")
    parser.add_argument("--counts-csv", default=None, help="Optional CSV path for per-frame DeepSORT counts.")
    parser.add_argument("--augment", action="store_true", help="Use augmented YOLO inference for higher recall at lower speed.")
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
    device = resolve_device(args.device)

    if not PREDICT_SCRIPT.exists():
        print(f"Missing predict script: {PREDICT_SCRIPT}", file=sys.stderr)
        return 1
    if not source.exists():
        print(f"Missing source video: {source}", file=sys.stderr)
        return 1
    if not model.exists():
        print(f"Missing model weights: {model}", file=sys.stderr)
        return 1

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env["DEEPSORT_MIN_CONFIDENCE"] = str(args.deepsort_min_conf)
    env["DEEPSORT_NMS_MAX_OVERLAP"] = str(args.deepsort_nms_overlap)
    env["DEEPSORT_MAX_AGE"] = str(args.deepsort_max_age)
    env["DEEPSORT_N_INIT"] = str(args.deepsort_n_init)
    if args.debug_counts:
        env["TRACK_DEBUG_COUNTS"] = "1"
    if args.counts_csv:
        counts_csv = Path(args.counts_csv).expanduser().resolve()
        counts_csv.parent.mkdir(parents=True, exist_ok=True)
        if counts_csv.exists():
            counts_csv.unlink()
        env["TRACK_COUNTS_CSV"] = str(counts_csv)

    print(f"Using source: {source}")
    print(f"Using model: {model}")
    print(f"Using device: {device}")
    print(f"Using imgsz: {args.imgsz}")
    print(f"Using conf: {args.conf}")
    print(f"Using iou: {args.iou}")
    print(f"Using max_det: {args.max_det}")
    print(f"Using DeepSORT min_confidence: {args.deepsort_min_conf}")
    print(f"Using DeepSORT nms_max_overlap: {args.deepsort_nms_overlap}")
    print(f"Using DeepSORT max_age: {args.deepsort_max_age}")
    print(f"Using DeepSORT n_init: {args.deepsort_n_init}")
    print(f"Using augment: {args.augment}")
    print(f"Debug counts: {args.debug_counts}")
    print(f"Counts CSV: {args.counts_csv}")

    cmd = [
        sys.executable,
        str(PREDICT_SCRIPT),
        f"model={hydra_quote(str(model))}",
        f"source={hydra_quote(str(source))}",
        f"imgsz={args.imgsz}",
        f"conf={args.conf}",
        f"iou={args.iou}",
        f"max_det={args.max_det}",
        f"augment={str(args.augment)}",
        f"show={str(args.show)}",
        f"device={device}",
    ]
    return subprocess.run(cmd, cwd=ROOT, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
