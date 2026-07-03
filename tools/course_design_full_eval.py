#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DETECT_ROOT = ROOT / "ultralytics" / "yolo" / "v8" / "detect"
DEEPSORT_CKPT = DETECT_ROOT / "deep_sort_pytorch" / "deep_sort" / "deep" / "checkpoint" / "ckpt.t7"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(DETECT_ROOT) not in sys.path:
    sys.path.insert(0, str(DETECT_ROOT))

from deep_sort_pytorch.deep_sort import DeepSort
from tools.bytetrack_verify import ByteTrackLite, YoloPersonDetector, xyxy_iou


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full no-label course-design evaluation for YOLOv8 + DeepSORT/ByteTrackLite.")
    parser.add_argument("--source", default="video/video.mp4", help="Input video path.")
    parser.add_argument("--model", default="yolov8m.pt", help="YOLOv8 weights path.")
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO input image size.")
    parser.add_argument("--conf", type=float, default=0.05, help="YOLO low confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.85, help="YOLO NMS IoU threshold.")
    parser.add_argument("--max-frames", type=int, default=0, help="Frames to process; 0 means full video.")
    parser.add_argument("--output", default="report_assets/course_eval_gpu", help="Output directory.")
    parser.add_argument("--deepsort-min-conf", type=float, default=0.05)
    parser.add_argument("--deepsort-max-age", type=int, default=150)
    parser.add_argument("--deepsort-n-init", type=int, default=1)
    parser.add_argument("--bytetrack-track-thresh", type=float, default=0.05)
    parser.add_argument("--bytetrack-match-thresh", type=float, default=0.80)
    parser.add_argument("--bytetrack-max-age", type=int, default=150)
    parser.add_argument("--large-jump-ratio", type=float, default=0.25)
    parser.add_argument("--large-jump-min", type=float, default=5.0)
    parser.add_argument("--high-iou-thresh", type=float, default=0.50)
    return parser.parse_args()


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)
    xywh = boxes.copy().astype(np.float32)
    xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
    xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
    xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
    xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
    return xywh


def box_area(boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.float32)
    return np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float32), ddof=1))


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


def pairwise_overlap_ratio(boxes: np.ndarray, high_iou_thresh: float) -> tuple[float, float]:
    if len(boxes) < 2:
        return 0.0, 0.0
    ious = xyxy_iou(boxes.astype(np.float32), boxes.astype(np.float32))
    upper = ious[np.triu_indices(len(boxes), k=1)]
    if len(upper) == 0:
        return 0.0, 0.0
    return float(upper.mean()), float((upper >= high_iou_thresh).sum() / len(upper))


def cuda_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024 * 1024))


class TrackerStats:
    def __init__(self, name: str):
        self.name = name
        self.counts: list[float] = []
        self.ids_by_frame: list[set[int]] = []
        self.lifetimes: defaultdict[int, int] = defaultdict(int)
        self.first_seen: dict[int, int] = {}
        self.last_center: dict[int, tuple[float, float]] = {}
        self.last_area: dict[int, float] = {}
        self.center_moves: list[float] = []
        self.area_change_ratios: list[float] = []
        self.mean_pair_iou: list[float] = []
        self.high_iou_pair_ratio: list[float] = []

    def update(self, frame_idx: int, track_rows: list[tuple[int, np.ndarray, float]], high_iou_thresh: float) -> None:
        ids = {track_id for track_id, _, _ in track_rows}
        self.ids_by_frame.append(ids)
        self.counts.append(float(len(track_rows)))
        boxes = np.asarray([box for _, box, _ in track_rows], dtype=np.float32) if track_rows else np.empty((0, 4), dtype=np.float32)
        avg_iou, high_ratio = pairwise_overlap_ratio(boxes, high_iou_thresh)
        self.mean_pair_iou.append(avg_iou)
        self.high_iou_pair_ratio.append(high_ratio)
        for track_id, box, _score in track_rows:
            self.lifetimes[track_id] += 1
            self.first_seen.setdefault(track_id, frame_idx)
            x1, y1, x2, y2 = [float(v) for v in box]
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            area = max(0.0, (x2 - x1) * (y2 - y1))
            if track_id in self.last_center:
                px, py = self.last_center[track_id]
                self.center_moves.append(math.hypot(center[0] - px, center[1] - py))
            if track_id in self.last_area and self.last_area[track_id] > 0:
                self.area_change_ratios.append(abs(area - self.last_area[track_id]) / self.last_area[track_id])
            self.last_center[track_id] = center
            self.last_area[track_id] = area

    def summarize(self, total_frames: int, large_jump_ratio: float, large_jump_min: float) -> dict[str, float | str]:
        diffs = [abs(self.counts[i] - self.counts[i - 1]) for i in range(1, len(self.counts))]
        avg_tracks = mean(self.counts)
        large_threshold = max(large_jump_min, avg_tracks * large_jump_ratio)
        large_jumps = [value for value in diffs if value >= large_threshold]
        lifetimes = list(self.lifetimes.values())
        short_tracks = [value for value in lifetimes if value <= 3]
        new_ids = []
        prev: set[int] = set()
        for ids in self.ids_by_frame:
            new_ids.append(len(ids - prev))
            prev = ids
        return {
            "tracker": self.name,
            "frames": total_frames,
            "avg_tracks": avg_tracks,
            "min_tracks": min(self.counts) if self.counts else 0.0,
            "max_tracks": max(self.counts) if self.counts else 0.0,
            "zero_track_ratio": sum(1 for value in self.counts if value == 0) / total_frames if total_frames else 0.0,
            "longest_zero_track_run": max_zero_run(self.counts),
            "track_jump_mean": mean(diffs),
            "track_jump_max": max(diffs) if diffs else 0.0,
            "large_jump_ratio": len(large_jumps) / len(diffs) if diffs else 0.0,
            "track_cv": std(self.counts) / avg_tracks if avg_tracks > 0 else 0.0,
            "unique_ids": len(lifetimes),
            "avg_track_life": mean([float(v) for v in lifetimes]),
            "median_track_life": percentile([float(v) for v in lifetimes], 50),
            "short_track_ratio": len(short_tracks) / len(lifetimes) if lifetimes else 0.0,
            "new_ids_per_frame": mean([float(v) for v in new_ids]),
            "avg_center_move": mean(self.center_moves),
            "avg_area_change_ratio": mean(self.area_change_ratios),
            "mean_pair_iou": mean(self.mean_pair_iou),
            "high_iou_pair_ratio": mean(self.high_iou_pair_ratio),
        }


def write_dict_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_series(path: Path, frames: list[int], series: dict[str, list[float]], title: str, ylabel: str) -> None:
    plt.figure(figsize=(11, 5))
    for label, values in series.items():
        plt.plot(frames, values, label=label, linewidth=1.2)
    plt.xlabel("Frame")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color=["#356f8c", "#c4572e"][: len(labels)])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def format_float(value: float | str) -> str:
    if isinstance(value, str):
        return value
    if abs(value) >= 100:
        return f"{value:.1f}"
    if 0 < abs(value) < 1:
        return f"{value:.3f}"
    return f"{value:.2f}"


def write_summary_md(path: Path, video_meta: dict[str, float | str], detection_summary: dict[str, float], tracker_summaries: list[dict[str, float | str]]) -> None:
    lines = [
        "# 课程设计综合性能评估",
        "",
        "## 视频与环境",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 视频帧数 | {int(video_meta['frames'])} |",
        f"| 分辨率 | {int(video_meta['width'])} x {int(video_meta['height'])} |",
        f"| 原视频FPS | {float(video_meta['fps']):.2f} |",
        f"| CUDA可用 | {video_meta['cuda_available']} |",
        f"| GPU | {video_meta['gpu_name']} |",
        "",
        "## 检测指标",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
    ]
    for key, value in detection_summary.items():
        lines.append(f"| {key} | {format_float(value)} |")
    lines.extend(
        [
            "",
            "## 跟踪器指标",
            "",
            "| 跟踪器 | 平均轨迹 | 零轨迹帧率 | 跳变均值 | 大跳变帧率 | 唯一ID数 | 平均寿命 | 短轨迹比例 | 新ID/帧 | 平均FPS |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    fps = detection_summary.get("avg_pipeline_fps", 0.0)
    for row in tracker_summaries:
        lines.append(
            f"| {row['tracker']} | {format_float(row['avg_tracks'])} | {format_float(row['zero_track_ratio'])} | "
            f"{format_float(row['track_jump_mean'])} | {format_float(row['large_jump_ratio'])} | "
            f"{format_float(row['unique_ids'])} | {format_float(row['avg_track_life'])} | "
            f"{format_float(row['short_track_ratio'])} | {format_float(row['new_ids_per_frame'])} | {format_float(fps)} |"
        )
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 本脚本在无人工标注条件下生成代理指标，不能替代 MOTA、IDF1、Precision、Recall。",
            "- 轨迹跳变、大跳变帧率、短轨迹比例和新ID/帧越低，通常说明轨迹更稳定。",
            "- 小目标比例可用于支撑远距离行人更难检测的场景分析。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {source}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = total_frames if args.max_frames == 0 else min(args.max_frames, total_frames)

    detector = YoloPersonDetector(args.model, args.device, args.imgsz, args.conf, args.iou)
    deepsort = DeepSort(
        str(DEEPSORT_CKPT),
        min_confidence=args.deepsort_min_conf,
        nms_max_overlap=1.0,
        max_age=args.deepsort_max_age,
        n_init=args.deepsort_n_init,
        use_cuda=args.device != "cpu" and torch.cuda.is_available(),
    )
    bytetrack = ByteTrackLite(args.bytetrack_match_thresh, args.bytetrack_track_thresh, args.bytetrack_max_age)

    frame_rows: list[dict[str, float | int]] = []
    deep_track_rows: list[dict[str, float | int]] = []
    byte_track_rows: list[dict[str, float | int]] = []
    deep_stats = TrackerStats("deepsort")
    byte_stats = TrackerStats("bytetrack_lite")
    all_scores: list[float] = []
    area_ratios: list[float] = []
    small_count = medium_count = large_count = 0
    frames: list[int] = []
    detect_counts: list[float] = []
    deep_counts: list[float] = []
    byte_counts: list[float] = []
    fps_values: list[float] = []
    latency_values: list[float] = []
    memory_values: list[float] = []

    start_all = time.time()
    for frame_idx in range(1, limit + 1):
        ok, frame = cap.read()
        if not ok:
            break
        frame_start = time.time()
        detections = detector(frame)
        det_boxes = detections[:, :4] if len(detections) else np.empty((0, 4), dtype=np.float32)
        det_scores = detections[:, 4] if len(detections) else np.empty((0,), dtype=np.float32)
        all_scores.extend([float(v) for v in det_scores])
        frame_area = max(1, width * height)
        ratios = [float(v / frame_area) for v in box_area(det_boxes)]
        area_ratios.extend(ratios)
        small_count += sum(1 for value in ratios if value < 0.01)
        medium_count += sum(1 for value in ratios if 0.01 <= value < 0.05)
        large_count += sum(1 for value in ratios if value >= 0.05)

        if len(detections):
            xywh = torch.from_numpy(xyxy_to_xywh(det_boxes)).float()
            confs = torch.from_numpy(det_scores[:, None]).float()
            classes = [0] * len(detections)
        else:
            xywh = torch.empty((0, 4), dtype=torch.float32)
            confs = torch.empty((0, 1), dtype=torch.float32)
            classes = []

        deep_outputs, _ = deepsort.update(xywh, confs, classes, frame)
        deep_tracks: list[tuple[int, np.ndarray, float]] = []
        if len(deep_outputs):
            for item in deep_outputs:
                box = np.asarray(item[:4], dtype=np.float32)
                track_id = int(item[-1])
                deep_tracks.append((track_id, box, 0.0))
                deep_track_rows.append({"frame": frame_idx, "track_id": track_id, "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3], "score": 0.0})

        byte_outputs = bytetrack.update(detections)
        byte_tracks: list[tuple[int, np.ndarray, float]] = []
        for track in byte_outputs:
            box = track.box.astype(np.float32)
            byte_tracks.append((int(track.track_id), box, float(track.score)))
            byte_track_rows.append({"frame": frame_idx, "track_id": int(track.track_id), "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3], "score": float(track.score)})

        deep_stats.update(frame_idx, deep_tracks, args.high_iou_thresh)
        byte_stats.update(frame_idx, byte_tracks, args.high_iou_thresh)
        frame_time_ms = (time.time() - frame_start) * 1000.0
        current_fps = 1000.0 / frame_time_ms if frame_time_ms > 0 else 0.0
        gpu_mem = cuda_memory_mb()

        frame_rows.append(
            {
                "frame": frame_idx,
                "detections": len(detections),
                "avg_conf": float(det_scores.mean()) if len(det_scores) else 0.0,
                "small_boxes": sum(1 for value in ratios if value < 0.01),
                "medium_boxes": sum(1 for value in ratios if 0.01 <= value < 0.05),
                "large_boxes": sum(1 for value in ratios if value >= 0.05),
                "deepsort_tracks": len(deep_tracks),
                "bytetrack_tracks": len(byte_tracks),
                "frame_time_ms": frame_time_ms,
                "fps": current_fps,
                "gpu_memory_mb": gpu_mem,
            }
        )
        frames.append(frame_idx)
        detect_counts.append(float(len(detections)))
        deep_counts.append(float(len(deep_tracks)))
        byte_counts.append(float(len(byte_tracks)))
        latency_values.append(frame_time_ms)
        fps_values.append(current_fps)
        memory_values.append(gpu_mem)

        if frame_idx % 50 == 0:
            print(f"processed {frame_idx}/{limit} frames in {time.time() - start_all:.1f}s")

    cap.release()
    processed = len(frame_rows)
    total_boxes = max(1, small_count + medium_count + large_count)
    detection_summary = {
        "frames": processed,
        "avg_detections": mean(detect_counts),
        "min_detections": min(detect_counts) if detect_counts else 0.0,
        "max_detections": max(detect_counts) if detect_counts else 0.0,
        "zero_detection_ratio": sum(1 for value in detect_counts if value == 0) / processed if processed else 0.0,
        "avg_conf": mean(all_scores),
        "conf_p10": percentile(all_scores, 10),
        "conf_p50": percentile(all_scores, 50),
        "conf_p90": percentile(all_scores, 90),
        "avg_box_area_ratio": mean(area_ratios),
        "small_box_ratio": small_count / total_boxes,
        "medium_box_ratio": medium_count / total_boxes,
        "large_box_ratio": large_count / total_boxes,
        "avg_latency_ms": mean(latency_values),
        "p50_latency_ms": percentile(latency_values, 50),
        "p95_latency_ms": percentile(latency_values, 95),
        "p99_latency_ms": percentile(latency_values, 99),
        "avg_pipeline_fps": mean(fps_values),
        "peak_gpu_memory_mb": max(memory_values) if memory_values else 0.0,
    }
    tracker_summaries = [
        deep_stats.summarize(processed, args.large_jump_ratio, args.large_jump_min),
        byte_stats.summarize(processed, args.large_jump_ratio, args.large_jump_min),
    ]

    write_dict_csv(output / "frame_metrics.csv", frame_rows)
    write_dict_csv(output / "tracks_deepsort.csv", deep_track_rows)
    write_dict_csv(output / "tracks_bytetrack.csv", byte_track_rows)
    write_dict_csv(output / "detection_summary.csv", [detection_summary])
    write_dict_csv(output / "tracker_summary.csv", tracker_summaries)

    video_meta = {
        "frames": processed,
        "width": width,
        "height": height,
        "fps": video_fps,
        "cuda_available": str(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    }
    write_summary_md(output / "summary.md", video_meta, detection_summary, tracker_summaries)

    plot_series(output / "counts_timeseries.png", frames, {"detections": detect_counts, "deepsort": deep_counts, "bytetrack_lite": byte_counts}, "Detection and Track Count", "Count")
    plot_series(output / "fps_timeseries.png", frames, {"fps": fps_values}, "Pipeline FPS", "FPS")
    plot_series(output / "latency_timeseries.png", frames, {"latency_ms": latency_values}, "Pipeline Latency", "ms")
    plot_series(output / "gpu_memory_timeseries.png", frames, {"gpu_memory_mb": memory_values}, "GPU Memory", "MB")
    plot_bar(output / "track_jump_comparison.png", [row["tracker"] for row in tracker_summaries], [float(row["track_jump_mean"]) for row in tracker_summaries], "Mean Track Count Jump", "tracks/frame")
    plot_bar(output / "short_track_ratio.png", [row["tracker"] for row in tracker_summaries], [float(row["short_track_ratio"]) for row in tracker_summaries], "Short Track Ratio", "ratio")
    plot_bar(output / "small_medium_large_boxes.png", ["small", "medium", "large"], [detection_summary["small_box_ratio"], detection_summary["medium_box_ratio"], detection_summary["large_box_ratio"]], "Box Size Distribution", "ratio")

    print(f"wrote full evaluation to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
