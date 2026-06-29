#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "yolov8n.pt"
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

    print(f"Using source: {source}")
    print(f"Using model: {model}")
    print(f"Using device: {device}")

    cmd = [
        sys.executable,
        str(PREDICT_SCRIPT),
        f"model={hydra_quote(str(model))}",
        f"source={hydra_quote(str(source))}",
        f"show={str(args.show)}",
        f"device={device}",
    ]
    return subprocess.run(cmd, cwd=ROOT, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
