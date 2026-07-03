#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


TRACK_COLUMNS = ("active_tracks", "deepsort_tracks", "bytetrack_tracks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate tracking CSV files with no-label proxy metrics.")
    parser.add_argument("--runs", default="runs/track_compare", help="Directory containing tracking count CSV files.")
    parser.add_argument("--video", default="video/video.mp4", help="Source video used by the runs.")
    parser.add_argument("--output", default="report_assets/performance_eval", help="Output directory for reports and plots.")
    parser.add_argument("--low-det-threshold", type=float, default=5.0, help="Frames at or below this detection count are low-detection frames.")
    parser.add_argument("--jump-ratio", type=float, default=0.25, help="Large jump threshold as a ratio of mean track count.")
    parser.add_argument("--jump-min", type=float, default=5.0, help="Minimum large jump threshold in tracks.")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, float]]:
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


def video_info(path: Path) -> dict[str, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"frames": 0.0, "fps": 0.0, "width": 0.0, "height": 0.0}
    info = {
        "frames": cap.get(cv2.CAP_PROP_FRAME_COUNT),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
    }
    cap.release()
    return info


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def max_zero_run(values: list[float]) -> int:
    longest = 0
    current = 0
    for value in values:
        if value == 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def diffs(values: list[float]) -> list[float]:
    return [abs(values[i] - values[i - 1]) for i in range(1, len(values))]


def infer_track_columns(row: dict[str, float]) -> list[str]:
    return [name for name in TRACK_COLUMNS if name in row]


def summarize_series(
    run_name: str,
    tracker_name: str,
    rows: list[dict[str, float]],
    track_col: str,
    low_det_threshold: float,
    jump_ratio: float,
    jump_min: float,
) -> dict[str, float | str]:
    detections = [row.get("detections", row.get("yolo_person", 0.0)) for row in rows]
    tracks = [row.get(track_col, 0.0) for row in rows]
    high = [row.get("high_detections", 0.0) for row in rows]
    fps_values = [row["fps"] for row in rows if row.get("fps", 0.0) > 0]
    frame_ms_values = [row["frame_time_ms"] for row in rows if row.get("frame_time_ms", 0.0) > 0]

    track_diffs = diffs(tracks)
    avg_tracks = mean(tracks)
    large_jump_threshold = max(jump_min, avg_tracks * jump_ratio)
    large_jumps = [value for value in track_diffs if value >= large_jump_threshold]
    valid_ratios = [tracks[i] / detections[i] for i in range(len(rows)) if detections[i] > 0]
    high_ratios = [high[i] / detections[i] for i in range(len(rows)) if detections[i] > 0 and high]

    return {
        "run": run_name,
        "tracker": tracker_name,
        "frames": len(rows),
        "avg_detections": mean(detections),
        "min_detections": min(detections) if detections else 0.0,
        "max_detections": max(detections) if detections else 0.0,
        "zero_detection_frames": sum(1 for value in detections if value == 0),
        "zero_detection_ratio": sum(1 for value in detections if value == 0) / len(rows) if rows else 0.0,
        "low_detection_ratio": sum(1 for value in detections if value <= low_det_threshold) / len(rows) if rows else 0.0,
        "avg_tracks": avg_tracks,
        "min_tracks": min(tracks) if tracks else 0.0,
        "max_tracks": max(tracks) if tracks else 0.0,
        "zero_track_frames": sum(1 for value in tracks if value == 0),
        "zero_track_ratio": sum(1 for value in tracks if value == 0) / len(rows) if rows else 0.0,
        "longest_zero_track_run": max_zero_run(tracks),
        "track_jump_mean": mean(track_diffs),
        "track_jump_std": std(track_diffs),
        "track_jump_max": max(track_diffs) if track_diffs else 0.0,
        "large_jump_threshold": large_jump_threshold,
        "large_jump_ratio": len(large_jumps) / len(track_diffs) if track_diffs else 0.0,
        "track_cv": std(tracks) / avg_tracks if avg_tracks > 0 else 0.0,
        "track_detection_ratio": mean(valid_ratios),
        "high_detection_ratio": mean(high_ratios),
        "avg_fps": mean(fps_values),
        "avg_frame_time_ms": mean(frame_ms_values),
    }


def collect_summaries(runs: Path, low_det_threshold: float, jump_ratio: float, jump_min: float) -> tuple[list[dict[str, float | str]], dict[str, list[dict[str, float]]]]:
    summaries: list[dict[str, float | str]] = []
    series: dict[str, list[dict[str, float]]] = {}
    for csv_path in sorted(runs.glob("**/*counts.csv")):
        rows = read_rows(csv_path)
        if not rows:
            continue
        track_cols = infer_track_columns(rows[0])
        if not track_cols:
            continue
        run_name = csv_path.parent.name
        series[run_name] = rows
        for track_col in track_cols:
            tracker_name = track_col.replace("_tracks", "").replace("active", "bytetrack_lite")
            summaries.append(
                summarize_series(run_name, tracker_name, rows, track_col, low_det_threshold, jump_ratio, jump_min)
            )
    return summaries, series


def write_summary_csv(summaries: list[dict[str, float | str]], output: Path) -> None:
    if not summaries:
        output.write_text("", encoding="utf-8")
        return
    fields = list(summaries[0].keys())
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)


def format_value(value: float | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if abs(value) >= 100:
        return f"{value:.1f}"
    if 0 < abs(value) < 1:
        return f"{value:.3f}"
    return f"{value:.2f}"


def write_summary_md(info: dict[str, float], summaries: list[dict[str, float | str]], output: Path) -> None:
    lines = [
        "# 跟踪性能评估汇总",
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
        "## 核心指标",
        "",
        "| 实验 | 跟踪器 | 帧数 | 平均检测 | 平均轨迹 | 零轨迹帧率 | 最长零轨迹段 | 轨迹跳变均值 | 大跳变帧率 | 轨迹变异系数 | 平均FPS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {run} | {tracker} | {frames} | {avg_detections} | {avg_tracks} | {zero_track_ratio} | {longest_zero_track_run} | {track_jump_mean} | {large_jump_ratio} | {track_cv} | {avg_fps} |".format(
                run=row["run"],
                tracker=row["tracker"],
                frames=int(float(row["frames"])),
                avg_detections=format_value(row["avg_detections"]),
                avg_tracks=format_value(row["avg_tracks"]),
                zero_track_ratio=format_value(row["zero_track_ratio"]),
                longest_zero_track_run=int(float(row["longest_zero_track_run"])),
                track_jump_mean=format_value(row["track_jump_mean"]),
                large_jump_ratio=format_value(row["large_jump_ratio"]),
                track_cv=format_value(row["track_cv"]),
                avg_fps=format_value(row["avg_fps"]),
            )
        )
    lines.extend(
        [
            "",
            "## 指标说明",
            "",
            "- 零轨迹帧率越低，说明完全断轨的帧越少。",
            "- 轨迹跳变均值和大跳变帧率越低，说明输出轨迹数量越平稳。",
            "- 轨迹变异系数为轨迹数标准差除以平均轨迹数，用于衡量相对波动。",
            "- 平均 FPS 只有在输入 CSV 包含 `fps` 或 `frame_time_ms` 时才有意义；旧 CSV 会显示为 0。",
            "- 这些是无人工标注条件下的代理指标，不能等价于 MOTA、IDF1、Precision 或 Recall。",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_bars(summaries: list[dict[str, float | str]], output: Path) -> None:
    if not summaries:
        return
    labels = [f"{row['run']}\n{row['tracker']}" for row in summaries]
    avg_dets = [float(row["avg_detections"]) for row in summaries]
    avg_tracks = [float(row["avg_tracks"]) for row in summaries]
    x = range(len(labels))
    plt.figure(figsize=(max(10, len(labels) * 1.8), 5))
    plt.bar([i - 0.2 for i in x], avg_dets, width=0.4, label="avg detections")
    plt.bar([i + 0.2 for i in x], avg_tracks, width=0.4, label="avg tracks")
    plt.xticks(list(x), labels, rotation=25, ha="right")
    plt.ylabel("Count")
    plt.title("Detection and Track Count Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def plot_stability(summaries: list[dict[str, float | str]], output: Path) -> None:
    if not summaries:
        return
    labels = [f"{row['run']}\n{row['tracker']}" for row in summaries]
    jump = [float(row["track_jump_mean"]) for row in summaries]
    cv = [float(row["track_cv"]) for row in summaries]
    x = range(len(labels))
    plt.figure(figsize=(max(10, len(labels) * 1.8), 5))
    plt.bar([i - 0.2 for i in x], jump, width=0.4, label="mean abs track jump")
    plt.bar([i + 0.2 for i in x], cv, width=0.4, label="track count CV")
    plt.xticks(list(x), labels, rotation=25, ha="right")
    plt.ylabel("Stability proxy")
    plt.title("Tracking Stability Proxy Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def plot_timeseries(series: dict[str, list[dict[str, float]]], output: Path) -> None:
    if not series:
        return
    plt.figure(figsize=(12, 6))
    for run_name, rows in series.items():
        if not rows:
            continue
        frames = [row.get("frame", i + 1) for i, row in enumerate(rows)]
        track_cols = infer_track_columns(rows[0])
        if not track_cols:
            continue
        track_col = track_cols[0]
        tracks = [row.get(track_col, 0.0) for row in rows]
        plt.plot(frames, tracks, label=f"{run_name}:{track_col}", linewidth=1.2)
    plt.xlabel("Frame")
    plt.ylabel("Track count")
    plt.title("Track Count Time Series")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def safe_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def plot_run_counts(run_name: str, rows: list[dict[str, float]], output_dir: Path) -> None:
    if not rows:
        return
    frames = [row.get("frame", i + 1) for i, row in enumerate(rows)]
    count_keys = [
        key for key in ("detections", "high_detections", "deepsort_in", "active_tracks", "deepsort_tracks", "bytetrack_tracks")
        if key in rows[0]
    ]
    if count_keys:
        plt.figure(figsize=(10, 5))
        for key in count_keys:
            plt.plot(frames, [row.get(key, 0.0) for row in rows], label=key, linewidth=1.3)
        plt.xlabel("Frame")
        plt.ylabel("Count")
        plt.title(f"{run_name} count series")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{safe_filename(run_name)}_counts.png", dpi=160)
        plt.close()

    if "fps" in rows[0] and any(row.get("fps", 0.0) > 0 for row in rows):
        plt.figure(figsize=(10, 4))
        plt.plot(frames, [row.get("fps", 0.0) for row in rows], label="fps", color="#c4572e", linewidth=1.2)
        plt.xlabel("Frame")
        plt.ylabel("FPS")
        plt.title(f"{run_name} FPS series")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{safe_filename(run_name)}_fps.png", dpi=160)
        plt.close()


def plot_individual_runs(series: dict[str, list[dict[str, float]]], output_dir: Path) -> None:
    for run_name, rows in series.items():
        plot_run_counts(run_name, rows, output_dir)


def main() -> int:
    args = parse_args()
    runs = Path(args.runs)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    summaries, series = collect_summaries(runs, args.low_det_threshold, args.jump_ratio, args.jump_min)
    info = video_info(Path(args.video))
    write_summary_csv(summaries, output / "performance_summary.csv")
    write_summary_md(info, summaries, output / "performance_summary.md")
    plot_bars(summaries, output / "count_comparison.png")
    plot_stability(summaries, output / "stability_comparison.png")
    plot_timeseries(series, output / "timeseries_overview.png")
    plot_individual_runs(series, output)
    print(f"wrote performance evaluation to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
