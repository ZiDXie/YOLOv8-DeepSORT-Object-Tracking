#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report metrics and figures.")
    parser.add_argument("--video", default="video/video.mp4", help="Input video path.")
    parser.add_argument("--runs", default="runs/track_compare", help="Directory containing tracking CSV files.")
    parser.add_argument("--output", default="report_assets", help="Output directory for report assets.")
    return parser.parse_args()


def read_counts(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            parsed: dict[str, float] = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = 0.0
            rows.append(parsed)
    return rows


def summarize(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not rows:
        return {}
    keys = [key for key in rows[0] if key != "frame"]
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [row[key] for row in rows]
        summary[key] = {
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "zero": sum(1 for value in values if value == 0),
        }
    return summary


def video_info(path: Path) -> dict[str, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    info = {
        "frames": cap.get(cv2.CAP_PROP_FRAME_COUNT),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
    }
    cap.release()
    return info


def save_frame(path: Path, output: Path, frame_index: int = 0) -> None:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if ok:
        cv2.imwrite(str(output), frame)


def plot_series(csv_path: Path, rows: list[dict[str, float]], output_dir: Path) -> None:
    if not rows:
        return
    frames = [row["frame"] for row in rows]
    keys = [key for key in rows[0] if key != "frame"]
    plt.figure(figsize=(10, 5))
    for key in keys:
        plt.plot(frames, [row[key] for row in rows], label=key)
    plt.xlabel("Frame")
    plt.ylabel("Count")
    plt.title(csv_path.parent.name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{csv_path.parent.name}_counts.png", dpi=160)
    plt.close()


def plot_comparison(summaries: dict[str, dict[str, dict[str, float]]], output: Path) -> None:
    labels = []
    det_avgs = []
    track_avgs = []
    for name, summary in summaries.items():
        labels.append(name)
        det_avgs.append(summary.get("detections", {}).get("avg", 0.0))
        track_key = "active_tracks" if "active_tracks" in summary else "bytetrack_tracks"
        if "deepsort_tracks" in summary and "bytetrack_tracks" not in summary:
            track_key = "deepsort_tracks"
        track_avgs.append(summary.get(track_key, {}).get("avg", 0.0))
    if not labels:
        return
    x = range(len(labels))
    plt.figure(figsize=(max(10, len(labels) * 1.6), 5))
    plt.bar([i - 0.2 for i in x], det_avgs, width=0.4, label="avg detections")
    plt.bar([i + 0.2 for i in x], track_avgs, width=0.4, label="avg tracks")
    plt.xticks(list(x), labels, rotation=30, ha="right")
    plt.ylabel("Average count")
    plt.title("Tracking Experiment Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def write_markdown(info: dict[str, float], summaries: dict[str, dict[str, dict[str, float]]], output: Path) -> None:
    lines = [
        "# 实验指标汇总",
        "",
        "## 视频信息",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 分辨率 | {int(info['width'])} x {int(info['height'])} |",
        f"| 帧率 | {info['fps']:.2f} FPS |",
        f"| 总帧数 | {int(info['frames'])} |",
        f"| 时长 | {info['frames'] / max(info['fps'], 1):.2f} s |",
        "",
        "## 跟踪实验统计",
        "",
        "| 实验 | 指标 | 平均值 | 最小值 | 最大值 | 零值帧数 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, summary in summaries.items():
        for key, stats in summary.items():
            lines.append(
                f"| {name} | {key} | {stats['avg']:.2f} | {stats['min']:.0f} | {stats['max']:.0f} | {stats['zero']:.0f} |"
            )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    video = Path(args.video)
    runs = Path(args.runs)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    info = video_info(video)
    save_frame(video, output / "scene_frame.jpg", frame_index=0)

    summaries: dict[str, dict[str, dict[str, float]]] = {}
    for csv_path in sorted(runs.glob("**/*counts.csv")):
        rows = read_counts(csv_path)
        summaries[csv_path.parent.name] = summarize(rows)
        plot_series(csv_path, rows, output)

    plot_comparison(summaries, output / "experiment_comparison.png")
    write_markdown(info, summaries, output / "metrics_summary.md")
    print(f"wrote assets to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
