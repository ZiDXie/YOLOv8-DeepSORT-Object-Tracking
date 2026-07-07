#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
DEEPSORT_CKPT = ROOT / "ultralytics" / "yolo" / "v8" / "detect" / "deep_sort_pytorch" / "deep_sort" / "deep" / "checkpoint" / "ckpt.t7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepSORT/ByteTrackLite default-vs-tuned parameter comparison.")
    parser.add_argument("--source", default="video/video.mp4", help="Input video path.")
    parser.add_argument("--model", default="yolov8m.pt", help="YOLOv8 weights path.")
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Fixed YOLO input size for all experiments.")
    parser.add_argument("--conf", type=float, default=0.05, help="Fixed YOLO confidence/low-confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.85, help="Fixed YOLO NMS IoU threshold.")
    parser.add_argument("--max-frames", type=int, default=0, help="Frames to process for ByteTrackLite; 0 means full video.")
    parser.add_argument("--runs-root", default="runs/new_video_report/tracker_param_compare", help="Tracking run output directory.")
    parser.add_argument("--output", default="report_assets/new_video_report/tracker_param_compare", help="Report asset output directory.")
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="Do not rerun trackers; only regenerate evaluation from existing count CSV files.",
    )
    return parser.parse_args()


def rel_or_abs(path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else ROOT / candidate


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def video_info(path: Path) -> dict[str, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    info = {
        "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frames": cap.get(cv2.CAP_PROP_FRAME_COUNT),
    }
    cap.release()
    return info


def run_step(name: str, cmd: list[str], env: dict[str, str]) -> None:
    print(f"\n==== {name} ====")
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def deepsort_cmd(
    source: Path,
    model: Path,
    args: argparse.Namespace,
    run_dir: Path,
    run_name: str,
    min_conf: float,
    max_age: int,
    n_init: int,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "run_tracking.py"),
        "--source",
        str(source),
        "--model",
        str(model),
        "--device",
        args.device,
        "--imgsz",
        str(args.imgsz),
        "--conf",
        str(args.conf),
        "--iou",
        str(args.iou),
        "--max-det",
        "1000",
        "--deepsort-min-conf",
        str(min_conf),
        "--deepsort-nms-overlap",
        "1.0",
        "--deepsort-max-age",
        str(max_age),
        "--deepsort-n-init",
        str(n_init),
        "--counts-csv",
        str(run_dir / f"{source.stem}_deepsort_counts.csv"),
        "--project",
        str(run_dir.parent),
        "--name",
        run_name,
    ]


def bytetrack_cmd(
    source: Path,
    model: Path,
    args: argparse.Namespace,
    run_dir: Path,
    track_thresh: float,
    max_age: int,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "run_bytetrack.py"),
        "--source",
        str(source),
        "--model",
        str(model),
        "--device",
        args.device,
        "--imgsz",
        str(args.imgsz),
        "--low-conf",
        str(args.conf),
        "--track-thresh",
        str(track_thresh),
        "--iou",
        str(args.iou),
        "--match-thresh",
        "0.80",
        "--max-age",
        str(max_age),
        "--max-frames",
        str(args.max_frames),
        "--output-dir",
        str(run_dir),
    ]


def write_summary(args: argparse.Namespace, source: Path, runs_root: Path, output: Path) -> None:
    info = video_info(source)
    lines = [
        "# DeepSORT 与 ByteTrackLite 默认/调整参数对比实验",
        "",
        "## 视频信息",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 视频路径 | `{source}` |",
        f"| 分辨率 | {int(info['width'])} x {int(info['height'])} |",
        f"| FPS | {info['fps']:.2f} |",
        f"| 总帧数 | {int(info['frames'])} |",
        f"| 时长 | {info['frames'] / max(info['fps'], 1):.2f} s |",
        "",
        "## 固定检测参数",
        "",
        "| 参数 | 数值 |",
        "|---|---:|",
        f"| model | `{args.model}` |",
        f"| imgsz | {args.imgsz} |",
        f"| conf / low-conf | {args.conf} |",
        f"| iou | {args.iou} |",
        f"| device | `{args.device}` |",
        "",
        "## 跟踪器参数分组",
        "",
        "| 实验 | 跟踪器 | 关键参数 |",
        "|---|---|---|",
        "| deepsort_default | DeepSORT | min_conf=0.30, max_age=70, n_init=3 |",
        "| deepsort_tuned | DeepSORT | min_conf=0.05, max_age=150, n_init=1 |",
        "| bytetrack_default | ByteTrackLite | track_thresh=0.50, max_age=60 |",
        "| bytetrack_tuned | ByteTrackLite | track_thresh=0.05, max_age=150 |",
        "",
        "## 输出位置",
        "",
        f"- 跟踪视频和逐帧 CSV：`{runs_root}`",
        f"- 汇总表和图：`{output}`",
        f"- 核心汇总表：`{output / 'performance_summary.md'}`",
        f"- 检测/轨迹数量对比图：`{output / 'count_comparison.png'}`",
        f"- 稳定性对比图：`{output / 'stability_comparison.png'}`",
        f"- 轨迹数时间序列：`{output / 'timeseries_overview.png'}`",
        "",
        "## 报告使用建议",
        "",
        "- 该实验固定 YOLO 检测参数，只改变跟踪器参数，因此适合写入报告的“跟踪器参数消融”部分。",
        "- 对 DeepSORT，重点比较零轨迹帧率、轨迹跳变均值、大跳变帧率和平均轨迹数。",
        "- 对 ByteTrackLite，重点比较 `track_thresh` 降低后是否带来更多轨迹、更低零轨迹帧率，以及是否增加轨迹跳变。",
        "- 本实验仍是无人工标注代理指标，不能替代 MOTA、IDF1、Precision、Recall。",
    ]
    output.mkdir(parents=True, exist_ok=True)
    (output / "tracker_param_compare_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    source = rel_or_abs(args.source)
    model = rel_or_abs(args.model)
    runs_root = rel_or_abs(args.runs_root)
    output = rel_or_abs(args.output)

    require_file(source, "source video")
    require_file(model, "YOLO model")
    require_file(DEEPSORT_CKPT, "DeepSORT checkpoint")
    runs_root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    experiments = [
        (
            "DeepSORT default parameters",
            deepsort_cmd(source, model, args, runs_root / "deepsort_default", "deepsort_default", 0.30, 70, 3),
        ),
        (
            "DeepSORT tuned parameters",
            deepsort_cmd(source, model, args, runs_root / "deepsort_tuned", "deepsort_tuned", 0.05, 150, 1),
        ),
        (
            "ByteTrackLite default parameters",
            bytetrack_cmd(source, model, args, runs_root / "bytetrack_default", 0.50, 60),
        ),
        (
            "ByteTrackLite tuned parameters",
            bytetrack_cmd(source, model, args, runs_root / "bytetrack_tuned", 0.05, 150),
        ),
    ]

    if args.skip_videos:
        print("Skipping tracker execution; regenerating evaluation from existing CSV files.")
    else:
        for name, cmd in experiments:
            run_step(name, cmd, env)

    run_step(
        "Evaluate tracker parameter comparison",
        [
            sys.executable,
            str(ROOT / "tools" / "evaluate_tracking_performance.py"),
            "--runs",
            str(runs_root),
            "--video",
            str(source),
            "--output",
            str(output),
        ],
        env,
    )
    write_summary(args, source, runs_root, output)

    print("\n==== DONE ====")
    print(f"Summary: {output / 'tracker_param_compare_summary.md'}")
    print(f"Performance table: {output / 'performance_summary.md'}")
    print(f"Runs: {runs_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
