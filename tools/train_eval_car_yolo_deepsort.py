#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import yaml


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SOURCES = [
    ("car_train_1", ROOT / "video" / "car" / "train" / "428332246.mp4", ROOT / "video" / "car" / "train" / "1"),
    ("car_train_2", ROOT / "video" / "car" / "train" / "570162587.mp4", ROOT / "video" / "car" / "train" / "2"),
]
TEST_SOURCES = [
    ("1675903118", ROOT / "video" / "car" / "test" / "1675903118.mp4", ROOT / "video" / "car" / "test" / "3"),
]
CLASS_NAMES = {0: "car"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train car YOLO, evaluate labeled test data, and run YOLO + DeepSORT with vehicle counts."
    )
    parser.add_argument("--dataset-dir", default="datasets/car_yolo_eval", help="Output YOLO dataset directory.")
    parser.add_argument("--train-project", default="runs/car_yolo_eval_train", help="YOLO training project directory.")
    parser.add_argument("--train-name", default="car_yolov8n", help="YOLO training run name.")
    parser.add_argument("--val-project", default="runs/car_yolo_eval_val", help="YOLO test evaluation project directory.")
    parser.add_argument("--val-name", default="car_yolov8n_test", help="YOLO test evaluation run name.")
    parser.add_argument("--track-project", default="runs/car_yolo_eval_deepsort", help="DeepSORT output directory.")
    parser.add_argument("--report-dir", default="runs/car_yolo_eval_report", help="Paper-ready summary output directory.")
    parser.add_argument("--model", default="yolov8n.pt", help="Initial YOLO model/weights for training.")
    parser.add_argument("--weights", default=None, help="Weights for --skip-train mode. Defaults to train best.pt.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training, validation, and inference image size.")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size.")
    parser.add_argument("--device", default="0", help="CUDA device or cpu.")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO inference confidence for DeepSORT video tests.")
    parser.add_argument("--iou", type=float, default=0.70, help="YOLO NMS IoU for validation and DeepSORT video tests.")
    parser.add_argument("--max-det", type=int, default=1000, help="Maximum detections per frame.")
    parser.add_argument("--deepsort-min-conf", type=float, default=0.25, help="DeepSORT min confidence.")
    parser.add_argument("--deepsort-max-age", type=int, default=70, help="DeepSORT max missed frames before deleting a track.")
    parser.add_argument("--deepsort-n-init", type=int, default=3, help="Frames required to confirm a DeepSORT track.")
    parser.add_argument("--track-frame-index", type=int, default=300, help="Frame index extracted from tracking video for paper figures.")
    parser.add_argument("--prepare-only", action="store_true", help="Only build dataset and data yaml.")
    parser.add_argument("--skip-prepare", action="store_true", help="Use existing dataset-dir.")
    parser.add_argument("--skip-train", action="store_true", help="Skip YOLO training and use --weights or train best.pt.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip YOLO test-set metric evaluation.")
    parser.add_argument("--skip-track", action="store_true", help="Skip DeepSORT video tracking and counting.")
    parser.add_argument("--force", action="store_true", help="Overwrite extracted dataset files and paper artifacts.")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {label}: {path}")


def video_info(video_path: Path) -> dict[str, int | float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    info = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def count_label_objects(label_dir: Path) -> int:
    total = 0
    for path in label_dir.glob("*.txt"):
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            total += len([line for line in text.splitlines() if line.strip()])
    return total


def copy_label(src: Path, dst: Path, force: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        return
    if src.exists():
        shutil.copy2(src, dst)
    else:
        dst.write_text("", encoding="utf-8")


def extract_video_to_split(
    source_name: str,
    video_path: Path,
    label_root: Path,
    dataset_dir: Path,
    split: str,
    force: bool,
) -> dict[str, int | str]:
    require_file(video_path, f"{split} video")
    label_dir = label_root / "obj_train_data"
    require_dir(label_dir, f"{split} label directory")

    info = video_info(video_path)
    labels = sorted(label_dir.glob("frame_*.txt"))
    if len(labels) != info["frames"]:
        raise RuntimeError(
            f"Frame/label mismatch for {video_path}: frames={info['frames']}, labels={len(labels)}"
        )

    cap = cv2.VideoCapture(str(video_path))
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        stem = f"{source_name}_frame_{written:06d}"
        image_path = dataset_dir / "images" / split / f"{stem}.jpg"
        label_src = label_dir / f"frame_{written:06d}.txt"
        label_dst = dataset_dir / "labels" / split / f"{stem}.txt"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        if force or not image_path.exists():
            cv2.imwrite(str(image_path), frame)
        copy_label(label_src, label_dst, force)
        written += 1
    cap.release()

    if written != info["frames"]:
        raise RuntimeError(f"Decoded frame count mismatch for {video_path}: decoded={written}, metadata={info['frames']}")

    objects = count_label_objects(label_dir)
    print(f"{split}: {source_name}, frames={written}, labels={len(labels)}, objects={objects}")
    return {
        "source": source_name,
        "video": str(video_path),
        "split": split,
        "frames": written,
        "labels": len(labels),
        "objects": objects,
        "width": int(info["width"]),
        "height": int(info["height"]),
        "fps": float(info["fps"]),
    }


def write_data_yaml(dataset_dir: Path) -> Path:
    data = {
        "path": str(dataset_dir),
        "train": "images/train",
        "val": "images/test",
        "test": "images/test",
        "names": CLASS_NAMES,
        "nc": len(CLASS_NAMES),
    }
    yaml_path = dataset_dir / "car.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return yaml_path


def prepare_dataset(args: argparse.Namespace) -> tuple[Path, list[dict[str, int | float | str]]]:
    dataset_dir = resolve(args.dataset_dir)
    for split in ("train", "test"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, int | float | str]] = []
    for source_name, video_path, label_root in TRAIN_SOURCES:
        rows.append(extract_video_to_split(source_name, video_path, label_root, dataset_dir, "train", args.force))
    for source_name, video_path, label_root in TEST_SOURCES:
        rows.append(extract_video_to_split(source_name, video_path, label_root, dataset_dir, "test", args.force))

    yaml_path = write_data_yaml(dataset_dir)
    print(f"dataset: {dataset_dir}")
    print(f"data yaml: {yaml_path}")
    return yaml_path, rows


def collect_existing_dataset_rows(dataset_dir: Path) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []
    for split, sources in (("train", TRAIN_SOURCES), ("test", TEST_SOURCES)):
        for source_name, video_path, label_root in sources:
            info = video_info(video_path)
            label_dir = label_root / "obj_train_data"
            rows.append(
                {
                    "source": source_name,
                    "video": str(video_path),
                    "split": split,
                    "frames": int(info["frames"]),
                    "labels": len(list(label_dir.glob("frame_*.txt"))),
                    "objects": count_label_objects(label_dir),
                    "width": int(info["width"]),
                    "height": int(info["height"]),
                    "fps": float(info["fps"]),
                }
            )
    require_file(dataset_dir / "car.yaml", "existing data yaml")
    return rows


def run_command(name: str, cmd: list[str]) -> None:
    print(f"\n==== {name} ====")
    print("$", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def run_command_capture(name: str, cmd: list[str]) -> str:
    print(f"\n==== {name} ====")
    print("$", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    result = subprocess.run(cmd, cwd=ROOT, check=True, env=env, text=True, capture_output=True)
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if output:
        print(output)
    return output


def train_model(args: argparse.Namespace, data_yaml: Path) -> Path:
    train_project = resolve(args.train_project)
    model = resolve(args.model) if Path(args.model).parent != Path(".") else Path(args.model)
    cmd = [
        sys.executable,
        str(ROOT / "ultralytics" / "yolo" / "v8" / "detect" / "train.py"),
        f"model={model}",
        f"data={data_yaml}",
        f"epochs={args.epochs}",
        f"imgsz={args.imgsz}",
        f"batch={args.batch}",
        f"device={args.device}",
        f"workers={args.workers}",
        f"project={train_project}",
        f"name={args.train_name}",
        "exist_ok=True",
        "plots=True",
    ]
    run_command("Train custom car YOLO", [str(v) for v in cmd])
    weights = train_project / args.train_name / "weights" / "best.pt"
    require_file(weights, "trained best.pt")
    return weights


def resolve_weights(args: argparse.Namespace) -> Path:
    if args.weights:
        weights = resolve(args.weights)
    else:
        weights = resolve(args.train_project) / args.train_name / "weights" / "best.pt"
    require_file(weights, "YOLO weights")
    return weights


def parse_yolo_val_metrics(log_text: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for line in log_text.splitlines():
        clean = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
        match = re.search(
            r"\ball\s+(\d+)\s+(\d+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)",
            clean,
        )
        if match:
            metrics = {
                "test_images": match.group(1),
                "test_instances": match.group(2),
                "test_precision": match.group(3),
                "test_recall": match.group(4),
                "test_mAP50": match.group(5),
                "test_mAP50_95": match.group(6),
            }
    return metrics


def evaluate_yolo(args: argparse.Namespace, data_yaml: Path, weights: Path) -> tuple[Path, dict[str, str]]:
    val_project = resolve(args.val_project)
    val_dir = val_project / args.val_name
    cmd = [
        sys.executable,
        str(ROOT / "ultralytics" / "yolo" / "v8" / "detect" / "val.py"),
        f"model={weights}",
        f"data={data_yaml}",
        f"imgsz={args.imgsz}",
        f"batch={args.batch}",
        f"device={args.device}",
        f"workers={args.workers}",
        f"iou={args.iou}",
        f"max_det={args.max_det}",
        f"project={val_project}",
        f"name={args.val_name}",
        "exist_ok=True",
        "plots=True",
    ]
    log_text = run_command_capture("Evaluate YOLO on labeled test set", [str(v) for v in cmd])
    val_dir.mkdir(parents=True, exist_ok=True)
    (val_dir / "val_output.log").write_text(log_text, encoding="utf-8")
    return val_dir, parse_yolo_val_metrics(log_text)


def load_val_metrics_if_exists(val_dir: Path | None) -> dict[str, str]:
    if not val_dir:
        return {}
    log_path = val_dir / "val_output.log"
    if not log_path.exists():
        return {}
    return parse_yolo_val_metrics(log_path.read_text(encoding="utf-8", errors="ignore"))


def run_deepsort_tests(args: argparse.Namespace, weights: Path) -> list[Path]:
    track_project = resolve(args.track_project)
    output_videos: list[Path] = []
    for source_name, video_path, _ in TEST_SOURCES:
        require_file(video_path, "test video")
        output_dir = track_project / source_name
        counts_csv = output_dir / f"{source_name}_deepsort_counts.csv"
        cmd = [
            sys.executable,
            str(ROOT / "run_tracking.py"),
            "--source",
            str(video_path),
            "--model",
            str(weights),
            "--device",
            str(args.device),
            "--imgsz",
            str(args.imgsz),
            "--conf",
            str(args.conf),
            "--iou",
            str(args.iou),
            "--max-det",
            str(args.max_det),
            "--deepsort-min-conf",
            str(args.deepsort_min_conf),
            "--deepsort-max-age",
            str(args.deepsort_max_age),
            "--deepsort-n-init",
            str(args.deepsort_n_init),
            "--counts-csv",
            str(counts_csv),
            "--project",
            str(track_project),
            "--name",
            source_name,
        ]
        run_command(f"YOLO + DeepSORT vehicle tracking: {video_path.name}", [str(v) for v in cmd])
        output_video = find_tracking_video(track_project, source_name, video_path.stem)
        require_file(output_video, "DeepSORT visualization video")
        require_file(counts_csv, "DeepSORT counts csv")
        counted_video = output_dir / f"{source_name}_with_counts.mp4"
        overlay_counts_on_video(output_video, counts_csv, counted_video)
        output_videos.append(counted_video)
    return output_videos


def find_tracking_video(track_project: Path, run_name: str, video_stem: str) -> Path:
    candidates = sorted(
        track_project.glob(f"{run_name}*/{video_stem}.mp4"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    if not candidates:
        return track_project / run_name / f"{video_stem}.mp4"
    return candidates[0]


def load_counts_by_frame(counts_csv: Path) -> dict[int, dict[str, str]]:
    with counts_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    counts: dict[int, dict[str, str]] = {}
    for row in rows:
        try:
            frame = int(float(row.get("frame", "0")))
        except ValueError:
            continue
        counts[frame] = row
    return counts


def draw_count_panel(frame, row: dict[str, str] | None, frame_index: int) -> None:
    detections = row.get("detections", "0") if row else "0"
    deepsort_in = row.get("deepsort_in", "0") if row else "0"
    tracks = row.get("deepsort_tracks", "0") if row else "0"
    fps = row.get("fps", "0") if row else "0"
    lines = [
        f"Frame: {frame_index}",
        f"Detected vehicles: {detections}",
        f"Tracked vehicles: {tracks}",
        f"DeepSORT input: {deepsort_in}",
        f"FPS: {fps}",
    ]
    x, y = 18, 22
    line_h = 28
    panel_w = 330
    panel_h = line_h * len(lines) + 18
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (x, y + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def overlay_counts_on_video(video_path: Path, counts_csv: Path, output_path: Path) -> None:
    counts = load_counts_by_frame(counts_csv)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open DeepSORT video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frame_index = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        draw_count_panel(frame, counts.get(frame_index), frame_index)
        writer.write(frame)
        frame_index += 1
    cap.release()
    writer.release()
    require_file(output_path, "DeepSORT video with vehicle counts")
    print(f"count overlay video: {output_path}")


def read_last_training_metrics(train_dir: Path) -> dict[str, str]:
    results_csv = train_dir / "results.csv"
    if not results_csv.exists():
        return {}
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    row = rows[-1]
    return {k.strip(): v.strip() for k, v in row.items()}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_counts_summary(counts_csv: Path, source_name: str) -> dict[str, object]:
    with counts_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"source": source_name, "frames": 0}

    def values(*columns: str) -> list[float]:
        out = []
        for row in rows:
            raw = ""
            for column in columns:
                raw = row.get(column, "")
                if raw not in ("", None):
                    break
            try:
                out.append(float(raw or 0))
            except ValueError:
                out.append(0.0)
        return out

    yolo_counts = values("detections", "yolo_person_count")
    deepsort_inputs = values("deepsort_in", "deepsort_input_count")
    track_counts = values("deepsort_tracks", "track_count")
    frame_times = values("frame_time_ms")
    fps_values = values("fps")
    return {
        "source": source_name,
        "frames": len(rows),
        "avg_yolo_vehicle_detections": f"{sum(yolo_counts) / len(yolo_counts):.3f}",
        "max_yolo_vehicle_detections": int(max(yolo_counts) if yolo_counts else 0),
        "avg_deepsort_input_boxes": f"{sum(deepsort_inputs) / len(deepsort_inputs):.3f}",
        "avg_tracked_vehicles": f"{sum(track_counts) / len(track_counts):.3f}",
        "max_tracked_vehicles": int(max(track_counts) if track_counts else 0),
        "avg_frame_time_ms": f"{sum(frame_times) / len(frame_times):.3f}",
        "avg_fps": f"{sum(fps_values) / len(fps_values):.3f}",
    }


def extract_tracking_frame(video_path: Path, output_path: Path, requested_index: int) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open tracking video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = requested_index if 0 <= requested_index < frame_count else max(frame_count // 2, 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to extract frame {frame_index} from {video_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)
    return frame_index


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def write_report(
    args: argparse.Namespace,
    data_yaml: Path,
    dataset_rows: list[dict[str, int | float | str]],
    weights: Path | None,
    val_dir: Path | None,
    val_metrics: dict[str, str],
    output_videos: list[Path],
) -> None:
    report_dir = resolve(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    write_csv(report_dir / "dataset_summary.csv", [dict(row) for row in dataset_rows])

    train_dir = resolve(args.train_project) / args.train_name
    train_metrics = read_last_training_metrics(train_dir)
    metrics_row: dict[str, object] = {
        "data_yaml": str(data_yaml),
        "weights": str(weights) if weights else "",
        "train_frames": sum(int(r["frames"]) for r in dataset_rows if r["split"] == "train"),
        "test_frames": sum(int(r["frames"]) for r in dataset_rows if r["split"] == "test"),
        "train_objects": sum(int(r["objects"]) for r in dataset_rows if r["split"] == "train"),
        "test_objects": sum(int(r["objects"]) for r in dataset_rows if r["split"] == "test"),
        "train_results_csv": str(train_dir / "results.csv") if (train_dir / "results.csv").exists() else "",
        "test_eval_dir": str(val_dir) if val_dir else "",
    }
    for key in ("epoch", "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"):
        if key in train_metrics:
            metrics_row[key] = train_metrics[key]
    for key, value in val_metrics.items():
        metrics_row[key] = value
    write_csv(report_dir / "metrics_summary.csv", [metrics_row])

    count_rows: list[dict[str, object]] = []
    for source_name, _, _ in TEST_SOURCES:
        counts_csv = resolve(args.track_project) / source_name / f"{source_name}_deepsort_counts.csv"
        if counts_csv.exists():
            count_rows.append(read_counts_summary(counts_csv, source_name))
    write_csv(report_dir / "deepsort_count_summary.csv", count_rows)

    copied_figures = []
    for figure in ("results.png", "confusion_matrix.png", "PR_curve.png", "F1_curve.png", "P_curve.png", "R_curve.png"):
        src = train_dir / figure
        dst = report_dir / f"train_{figure}"
        copy_if_exists(src, dst)
        if dst.exists():
            copied_figures.append(dst)
        if val_dir:
            src = val_dir / figure
            dst = report_dir / f"test_{figure}"
            copy_if_exists(src, dst)
            if dst.exists():
                copied_figures.append(dst)

    extracted_frames: list[tuple[Path, int]] = []
    for video in output_videos:
        out = report_dir / f"{video.stem}_deepsort_frame_{args.track_frame_index}.jpg"
        frame_index = extract_tracking_frame(video, out, args.track_frame_index)
        extracted_frames.append((out, frame_index))

    lines = [
        "# Car YOLO + DeepSORT 论文实验结果汇总",
        "",
        "## 数据集",
        "",
        f"- 数据配置：`{data_yaml}`",
        f"- 训练帧数：{metrics_row['train_frames']}",
        f"- 测试帧数：{metrics_row['test_frames']}",
        f"- 训练目标框数：{metrics_row['train_objects']}",
        f"- 测试目标框数：{metrics_row['test_objects']}",
        "",
        "## YOLO 检测指标",
        "",
        f"- 权重：`{weights}`" if weights else "- 权重：未生成或未指定",
        f"- Test Precision：{metrics_row.get('test_precision', metrics_row.get('metrics/precision(B)', ''))}",
        f"- Test Recall：{metrics_row.get('test_recall', metrics_row.get('metrics/recall(B)', ''))}",
        f"- Test mAP@0.5：{metrics_row.get('test_mAP50', metrics_row.get('metrics/mAP50(B)', ''))}",
        f"- Test mAP@0.5:0.95：{metrics_row.get('test_mAP50_95', metrics_row.get('metrics/mAP50-95(B)', ''))}",
        f"- 测试评估目录：`{val_dir}`" if val_dir else "- 测试评估目录：未运行",
        "",
        "## DeepSORT 车辆计数与视频",
        "",
    ]
    for video in output_videos:
        lines.append(f"- 可观看跟踪视频：`{video}`")
    for source_name, _, _ in TEST_SOURCES:
        counts_csv = resolve(args.track_project) / source_name / f"{source_name}_deepsort_counts.csv"
        if counts_csv.exists():
            lines.append(f"- 逐帧车辆计数：`{counts_csv}`")
    for frame_path, frame_index in extracted_frames:
        lines.append(f"- 论文截图帧：`{frame_path}`，实际帧号：{frame_index}")
    lines += [
        "",
        "## 表格与图表文件",
        "",
        f"- 数据集统计：`{report_dir / 'dataset_summary.csv'}`",
        f"- 检测指标汇总：`{report_dir / 'metrics_summary.csv'}`",
        f"- DeepSORT 计数汇总：`{report_dir / 'deepsort_count_summary.csv'}`",
    ]
    for figure in copied_figures:
        lines.append(f"- 图表：`{figure}`")

    (report_dir / "metrics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report: {report_dir / 'metrics_summary.md'}")


def main() -> int:
    args = parse_args()
    dataset_dir = resolve(args.dataset_dir)
    data_yaml = dataset_dir / "car.yaml"

    if args.skip_prepare:
        require_file(data_yaml, "existing data yaml")
        dataset_rows = collect_existing_dataset_rows(dataset_dir)
    else:
        data_yaml, dataset_rows = prepare_dataset(args)

    if args.prepare_only:
        write_report(args, data_yaml, dataset_rows, None, None, {}, [])
        return 0

    weights = resolve_weights(args) if args.skip_train else train_model(args, data_yaml)
    if args.skip_eval:
        val_dir = None
        val_metrics = {}
    else:
        val_dir, val_metrics = evaluate_yolo(args, data_yaml, weights)
    output_videos = [] if args.skip_track else run_deepsort_tests(args, weights)
    if not val_metrics:
        val_metrics = load_val_metrics_if_exists(val_dir)
    write_report(args, data_yaml, dataset_rows, weights, val_dir, val_metrics, output_videos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
