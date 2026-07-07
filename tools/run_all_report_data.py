#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
DEEPSORT_CKPT = ROOT / "ultralytics" / "yolo" / "v8" / "detect" / "deep_sort_pytorch" / "deep_sort" / "deep" / "checkpoint" / "ckpt.t7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run every data-generation step required by the course-design report.")
    parser.add_argument("--source", default="video/video.mp4", help="Input video path.")
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--model", default="yolov8m.pt", help="Main YOLOv8 model for tuned experiments.")
    parser.add_argument("--max-frames", type=int, default=0, help="Frames for scripts that support limiting; 0 means full video.")
    parser.add_argument("--output-root", default="report_assets/new_video_report", help="Root directory for report assets.")
    parser.add_argument("--runs-root", default="runs/new_video_report", help="Root directory for tracking videos and count CSV files.")
    parser.add_argument("--skip-visual-videos", action="store_true", help="Skip separate DeepSORT/ByteTrack visual video generation.")
    parser.add_argument("--skip-ablation", action="store_true", help="Skip 4-group YOLOv8n/m ablation runs.")
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


def copy_if_exists(source: Path, target: Path) -> None:
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def read_one_csv(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None


def read_tracker_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fmt(row: dict[str, str] | None, key: str) -> str:
    if not row:
        return ""
    value = row.get(key, "")
    if value == "":
        return ""
    try:
        return f"{float(value):.3f}"
    except ValueError:
        return value


def write_ablation_summary(experiments: list[dict[str, object]], output_dir: Path) -> None:
    rows: list[dict[str, object]] = []
    for exp in experiments:
        exp_output = Path(exp["output"])
        det = read_one_csv(exp_output / "detection_summary.csv")
        trackers = read_tracker_rows(exp_output / "tracker_summary.csv")
        by_tracker = {row.get("tracker"): row for row in trackers}
        deep = by_tracker.get("deepsort")
        byte = by_tracker.get("bytetrack_lite")
        rows.append(
            {
                "实验": exp["id"],
                "名称": exp["name"],
                "模型": exp["model"],
                "imgsz": exp["imgsz"],
                "conf": exp["conf"],
                "iou": exp["iou"],
                "平均检测数": fmt(det, "avg_detections"),
                "零检测帧率": fmt(det, "zero_detection_ratio"),
                "平均置信度": fmt(det, "avg_conf"),
                "小目标比例": fmt(det, "small_box_ratio"),
                "DeepSORT平均轨迹": fmt(deep, "avg_tracks"),
                "ByteTrack平均轨迹": fmt(byte, "avg_tracks"),
                "DeepSORT跳变均值": fmt(deep, "track_jump_mean"),
                "ByteTrack跳变均值": fmt(byte, "track_jump_mean"),
                "DeepSORT短轨迹比例": fmt(deep, "short_track_ratio"),
                "ByteTrack短轨迹比例": fmt(byte, "short_track_ratio"),
                "平均FPS": fmt(det, "avg_pipeline_fps"),
                "峰值显存MB": fmt(det, "peak_gpu_memory_mb"),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ablation_summary.csv"
    md_path = output_dir / "ablation_summary.md"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# YOLOv8n/YOLOv8m 默认与调整参数消融实验汇总",
        "",
        "| 实验 | 名称 | 模型 | imgsz | conf | iou | 平均检测数 | 零检测帧率 | 平均置信度 | 小目标比例 | DeepSORT平均轨迹 | ByteTrack平均轨迹 | DeepSORT跳变均值 | ByteTrack跳变均值 | DeepSORT短轨迹比例 | ByteTrack短轨迹比例 | 平均FPS | 峰值显存MB |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {实验} | {名称} | {模型} | {imgsz} | {conf} | {iou} | {平均检测数} | {零检测帧率} | {平均置信度} | {小目标比例} | {DeepSORT平均轨迹} | {ByteTrack平均轨迹} | {DeepSORT跳变均值} | {ByteTrack跳变均值} | {DeepSORT短轨迹比例} | {ByteTrack短轨迹比例} | {平均FPS} | {峰值显存MB} |".format(**row)
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_master_summary(args: argparse.Namespace, paths: dict[str, Path], output: Path) -> None:
    info = video_info(paths["source"])
    lines = [
        "# 新视频课程设计报告数据总输出",
        "",
        "## 输入视频",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 路径 | `{paths['source']}` |",
        f"| 分辨率 | {int(info['width'])} x {int(info['height'])} |",
        f"| FPS | {info['fps']:.2f} |",
        f"| 总帧数 | {int(info['frames'])} |",
        f"| 时长 | {info['frames'] / max(info['fps'], 1):.2f} s |",
        "",
        "## 关键输出",
        "",
        f"- 视频基础信息与首帧图：`{paths['report_assets']}`",
        f"- 单独运行统计与可视化视频：`{paths['runs_root']}`",
        f"- 单独运行性能评估：`{paths['performance_eval']}`",
        f"- DeepSORT/ByteTrack 公平综合评估：`{paths['course_eval']}`",
        f"- 4 组消融实验：`{paths['ablation']}`",
        f"- 消融汇总表：`{paths['ablation'] / 'ablation_summary.md'}`",
        "",
        "## 运行参数",
        "",
        f"- `source`: `{args.source}`",
        f"- `device`: `{args.device}`",
        f"- `model`: `{args.model}`",
        f"- `max_frames`: `{args.max_frames}`",
        f"- `skip_visual_videos`: `{args.skip_visual_videos}`",
        f"- `skip_ablation`: `{args.skip_ablation}`",
        "",
        "## 报告填写建议",
        "",
        "- 第 1 章视频信息和场景图使用 `report_assets/new_video_report/base/metrics_summary.md` 与 `scene_frame.jpg`。",
        "- 第 6 章 DeepSORT/ByteTrack 单独运行对比使用 `performance_eval/performance_summary.md` 和对应 PNG 图。",
        "- 第 6.11 综合评估使用 `course_eval/summary.md`、`tracker_summary.csv` 和各类曲线图。",
        "- YOLOv8n/YOLOv8m 默认/调整参数对比使用 `ablation/ablation_summary.md`。",
        "- 若需要把新结果替换报告旧图，可将本目录中的 PNG 路径更新到 `课程设计报告.md`。",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    source = rel_or_abs(args.source)
    model = rel_or_abs(args.model)
    output_root = rel_or_abs(args.output_root)
    runs_root = rel_or_abs(args.runs_root)
    base_assets = output_root / "base"
    performance_eval = output_root / "performance_eval"
    course_eval = output_root / "course_eval"
    ablation = output_root / "ablation"
    visual_runs = runs_root / "track_compare"

    require_file(source, "source video")
    require_file(model, "main YOLO model")
    require_file(ROOT / "yolov8n.pt", "yolov8n weights")
    require_file(ROOT / "yolov8m.pt", "yolov8m weights")
    require_file(DEEPSORT_CKPT, "DeepSORT checkpoint")

    output_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    if not args.skip_visual_videos:
        deep_counts = visual_runs / "deepsort_tuned" / f"{source.stem}_deepsort_counts.csv"
        run_step(
            "DeepSORT visual tracking",
            [
                sys.executable,
                str(ROOT / "run_tracking.py"),
                "--source",
                str(source),
                "--model",
                str(model),
                "--device",
                args.device,
                "--imgsz",
                "1280",
                "--conf",
                "0.05",
                "--iou",
                "0.85",
                "--deepsort-min-conf",
                "0.05",
                "--deepsort-nms-overlap",
                "1.0",
                "--deepsort-max-age",
                "150",
                "--deepsort-n-init",
                "1",
                "--counts-csv",
                str(deep_counts),
                "--project",
                str(visual_runs),
                "--name",
                "deepsort_tuned",
            ],
            env,
        )

        run_step(
            "ByteTrackLite visual tracking",
            [
                sys.executable,
                str(ROOT / "run_bytetrack.py"),
                "--source",
                str(source),
                "--model",
                str(model),
                "--device",
                args.device,
                "--imgsz",
                "1280",
                "--low-conf",
                "0.05",
                "--track-thresh",
                "0.05",
                "--iou",
                "0.85",
                "--match-thresh",
                "0.80",
                "--max-age",
                "150",
                "--max-frames",
                str(args.max_frames),
                "--output-dir",
                str(visual_runs / "bytetrack_tuned"),
            ],
            env,
        )

    run_step(
        "Base report metrics",
        [
            sys.executable,
            str(ROOT / "tools" / "report_metrics.py"),
            "--video",
            str(source),
            "--runs",
            str(visual_runs),
            "--output",
            str(base_assets),
        ],
        env,
    )
    copy_if_exists(base_assets / "scene_frame.jpg", output_root / "scene_frame.jpg")

    run_step(
        "Tracking performance evaluation",
        [
            sys.executable,
            str(ROOT / "tools" / "evaluate_tracking_performance.py"),
            "--video",
            str(source),
            "--runs",
            str(visual_runs),
            "--output",
            str(performance_eval),
        ],
        env,
    )

    run_step(
        "Fair DeepSORT vs ByteTrackLite full evaluation",
        [
            sys.executable,
            str(ROOT / "tools" / "course_design_full_eval.py"),
            "--source",
            str(source),
            "--model",
            str(model),
            "--device",
            args.device,
            "--imgsz",
            "1280",
            "--conf",
            "0.05",
            "--iou",
            "0.85",
            "--max-frames",
            str(args.max_frames),
            "--output",
            str(course_eval),
            "--deepsort-min-conf",
            "0.05",
            "--deepsort-max-age",
            "150",
            "--deepsort-n-init",
            "1",
            "--bytetrack-track-thresh",
            "0.05",
            "--bytetrack-match-thresh",
            "0.80",
            "--bytetrack-max-age",
            "150",
        ],
        env,
    )

    experiments: list[dict[str, object]] = [
        {"id": "E1", "name": "yolov8n_default", "model": "yolov8n.pt", "imgsz": 640, "conf": 0.25, "iou": 0.70, "output": ablation / "yolov8n_default"},
        {"id": "E2", "name": "yolov8n_tuned", "model": "yolov8n.pt", "imgsz": 1280, "conf": 0.05, "iou": 0.85, "output": ablation / "yolov8n_tuned"},
        {"id": "E3", "name": "yolov8m_default", "model": "yolov8m.pt", "imgsz": 640, "conf": 0.25, "iou": 0.70, "output": ablation / "yolov8m_default"},
        {"id": "E4", "name": "yolov8m_tuned", "model": "yolov8m.pt", "imgsz": 1280, "conf": 0.05, "iou": 0.85, "output": ablation / "yolov8m_tuned"},
    ]
    if not args.skip_ablation:
        for exp in experiments:
            run_step(
                f"Ablation {exp['id']} {exp['name']}",
                [
                    sys.executable,
                    str(ROOT / "tools" / "course_design_full_eval.py"),
                    "--source",
                    str(source),
                    "--model",
                    str(ROOT / str(exp["model"])),
                    "--device",
                    args.device,
                    "--imgsz",
                    str(exp["imgsz"]),
                    "--conf",
                    str(exp["conf"]),
                    "--iou",
                    str(exp["iou"]),
                    "--max-frames",
                    str(args.max_frames),
                    "--output",
                    str(exp["output"]),
                ],
                env,
            )
        write_ablation_summary(experiments, ablation)

    paths = {
        "source": source,
        "report_assets": base_assets,
        "runs_root": runs_root,
        "performance_eval": performance_eval,
        "course_eval": course_eval,
        "ablation": ablation,
    }
    write_master_summary(args, paths, output_root / "all_report_data_summary.md")
    print("\n==== DONE ====")
    print(f"Master summary: {output_root / 'all_report_data_summary.md'}")
    print(f"Base assets: {base_assets}")
    print(f"Performance evaluation: {performance_eval}")
    print(f"Fair evaluation: {course_eval}")
    print(f"Ablation: {ablation}")
    print(f"Visual runs: {visual_runs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
