#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
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
TEST_VIDEOS = [
    ROOT / "video" / "car" / "test" / "1675903118.mp4",
    ROOT / "video" / "car" / "test" / "597626025.mp4",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare car YOLO data, train YOLOv8, then test with YOLO + DeepSORT.")
    parser.add_argument("--dataset-dir", default="datasets/car_yolo", help="Output YOLOv8 dataset directory.")
    parser.add_argument("--train-project", default="runs/car_yolo_train", help="YOLO training project directory.")
    parser.add_argument("--train-name", default="car_yolov8n", help="YOLO training run name.")
    parser.add_argument("--track-project", default="runs/car_yolo_deepsort", help="DeepSORT test output project directory.")
    parser.add_argument("--model", default="yolov8n.pt", help="Initial YOLO model/weights for training.")
    parser.add_argument("--weights", default=None, help="Trained weights for --skip-train mode. Defaults to project/name/weights/best.pt.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training and inference image size.")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size.")
    parser.add_argument("--device", default="0", help="CUDA device or cpu.")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--val-stride", type=int, default=7, help="Every Nth frame goes to val; 7 is about 14%%.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO inference confidence for test videos.")
    parser.add_argument("--iou", type=float, default=0.70, help="YOLO inference NMS IoU for test videos.")
    parser.add_argument("--deepsort-min-conf", type=float, default=0.25, help="DeepSORT min confidence for test videos.")
    parser.add_argument("--deepsort-max-age", type=int, default=70, help="DeepSORT max age for test videos.")
    parser.add_argument("--deepsort-n-init", type=int, default=3, help="DeepSORT n_init for test videos.")
    parser.add_argument("--prepare-only", action="store_true", help="Only build YOLOv8 dataset and data yaml.")
    parser.add_argument("--skip-prepare", action="store_true", help="Do not rebuild dataset; use existing dataset-dir.")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and use --weights or best.pt.")
    parser.add_argument("--skip-test", action="store_true", help="Skip DeepSORT testing on test videos.")
    parser.add_argument("--force", action="store_true", help="Overwrite extracted images and labels.")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def ensure_empty_or_ready(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def split_name(global_index: int, val_stride: int) -> str:
    if val_stride <= 0:
        return "train"
    return "val" if global_index % val_stride == 0 else "train"


def copy_label(src: Path, dst: Path, force: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        return
    if src.exists():
        shutil.copy2(src, dst)
    else:
        dst.write_text("", encoding="utf-8")


def extract_video_frames(
    source_name: str,
    video_path: Path,
    label_root: Path,
    dataset_dir: Path,
    start_index: int,
    val_stride: int,
    force: bool,
) -> tuple[int, int, int]:
    require_file(video_path, "train video")
    require_file(label_root / "obj_train_data", "label directory")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open train video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    written = 0
    train_count = 0
    val_count = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        global_index = start_index + frame_idx
        subset = split_name(global_index, val_stride)
        if subset == "train":
            train_count += 1
        else:
            val_count += 1

        stem = f"{source_name}_frame_{frame_idx:06d}"
        image_path = dataset_dir / "images" / subset / f"{stem}.jpg"
        label_src = label_root / "obj_train_data" / f"frame_{frame_idx:06d}.txt"
        label_dst = dataset_dir / "labels" / subset / f"{stem}.txt"

        image_path.parent.mkdir(parents=True, exist_ok=True)
        if force or not image_path.exists():
            cv2.imwrite(str(image_path), frame)
        copy_label(label_src, label_dst, force)

        written += 1
        frame_idx += 1

    cap.release()
    if total and written != total:
        print(f"warning: extracted {written} frames but video metadata says {total}: {video_path}")
    print(f"{source_name}: frames={written}, train={train_count}, val={val_count}")
    return written, train_count, val_count


def write_data_yaml(dataset_dir: Path) -> Path:
    data = {
        "path": str(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "car"},
        "nc": 1,
    }
    yaml_path = dataset_dir / "car.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return yaml_path


def count_files(path: Path, suffix: str) -> int:
    return len(list(path.rglob(f"*{suffix}"))) if path.exists() else 0


def prepare_dataset(args: argparse.Namespace) -> Path:
    dataset_dir = resolve(args.dataset_dir)
    ensure_empty_or_ready(dataset_dir)
    for subset in ("train", "val"):
        (dataset_dir / "images" / subset).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / subset).mkdir(parents=True, exist_ok=True)

    global_index = 0
    total_written = 0
    total_train = 0
    total_val = 0
    for source_name, video_path, label_root in TRAIN_SOURCES:
        written, train_count, val_count = extract_video_frames(
            source_name,
            video_path,
            label_root,
            dataset_dir,
            global_index,
            args.val_stride,
            args.force,
        )
        global_index += written
        total_written += written
        total_train += train_count
        total_val += val_count

    yaml_path = write_data_yaml(dataset_dir)
    image_count = count_files(dataset_dir / "images", ".jpg")
    label_count = count_files(dataset_dir / "labels", ".txt")
    print(f"dataset: {dataset_dir}")
    print(f"data yaml: {yaml_path}")
    print(f"frames={total_written}, train={total_train}, val={total_val}, images={image_count}, labels={label_count}")
    return yaml_path


def run_command(name: str, cmd: list[str]) -> None:
    print(f"\n==== {name} ====")
    print("$", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def train_model(args: argparse.Namespace, data_yaml: Path) -> Path:
    train_project = resolve(args.train_project)
    model = resolve(args.model) if not Path(args.model).name == args.model else Path(args.model)
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


def run_deepsort_tests(args: argparse.Namespace, weights: Path) -> None:
    track_project = resolve(args.track_project)
    for source in TEST_VIDEOS:
        require_file(source, "test video")
        run_name = source.stem
        output_dir = track_project / run_name
        counts_csv = output_dir / f"{source.stem}_deepsort_counts.csv"
        cmd = [
            sys.executable,
            str(ROOT / "run_tracking.py"),
            "--source",
            str(source),
            "--model",
            str(weights),
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
            run_name,
        ]
        run_command(f"YOLO + DeepSORT test: {source.name}", [str(v) for v in cmd])


def write_summary(args: argparse.Namespace, data_yaml: Path, weights: Path | None) -> None:
    summary_dir = resolve(args.track_project)
    summary_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Car YOLO + DeepSORT 训练与测试",
        "",
        "## 数据",
        "",
        f"- 数据集配置：`{data_yaml}`",
        f"- 训练视频：`{TRAIN_SOURCES[0][1]}`、`{TRAIN_SOURCES[1][1]}`",
        f"- 测试视频：`{TEST_VIDEOS[0]}`、`{TEST_VIDEOS[1]}`",
        "- 类别：`car`",
        "",
        "## 训练",
        "",
        f"- 初始模型：`{args.model}`",
        f"- epochs：{args.epochs}",
        f"- imgsz：{args.imgsz}",
        f"- batch：{args.batch}",
        f"- device：`{args.device}`",
        f"- 权重：`{weights}`" if weights else "- 权重：未训练或未指定",
        "",
        "## 测试输出",
        "",
        f"- DeepSORT 测试输出目录：`{resolve(args.track_project)}`",
        "- 每个测试视频会生成可视化视频和 `*_deepsort_counts.csv`。",
        "",
        "## 说明",
        "",
        "- `video/car/test` 没有标注文件，因此这里只做可视化测试和逐帧统计，不计算 mAP。",
        "- DeepSORT 的 ReID 权重原本面向行人，车辆 ID 稳定性可能不如专用车辆 ReID 或 ByteTrack/OC-SORT。",
    ]
    (summary_dir / "car_yolo_deepsort_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summary: {summary_dir / 'car_yolo_deepsort_summary.md'}")


def main() -> int:
    args = parse_args()
    data_yaml = resolve(args.dataset_dir) / "car.yaml"

    if not args.skip_prepare:
        data_yaml = prepare_dataset(args)
    else:
        require_file(data_yaml, "existing data yaml")

    if args.prepare_only:
        write_summary(args, data_yaml, None)
        return 0

    if args.skip_train:
        weights = resolve_weights(args)
    else:
        weights = train_model(args, data_yaml)

    if not args.skip_test:
        run_deepsort_tests(args, weights)

    write_summary(args, data_yaml, weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
