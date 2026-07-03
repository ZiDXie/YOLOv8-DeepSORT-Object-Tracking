#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

if not hasattr(np, "float"):
    np.float = float  # ByteTrack compatibility with NumPy 2.x

import torch


ROOT = Path(__file__).resolve().parents[1]
BYTETRACK_ROOT = ROOT / "third_party" / "ByteTrack"
DETECT_ROOT = ROOT / "ultralytics" / "yolo" / "v8" / "detect"
DEEPSORT_ROOT = DETECT_ROOT / "deep_sort_pytorch"

sys.path.insert(0, str(BYTETRACK_ROOT))
sys.path.insert(0, str(DETECT_ROOT))

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.tracker.byte_tracker import BYTETracker
from yolox.utils import fuse_model, get_model_info, postprocess

from deep_sort_pytorch.deep_sort import DeepSort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare MOT-pretrained YOLOX detector with DeepSORT and ByteTrack.")
    parser.add_argument("--source", default="video/video.mp4", help="Input video path.")
    parser.add_argument("--exp-file", default="third_party/ByteTrack/exps/example/mot/yolox_m_mix_det.py")
    parser.add_argument("--ckpt", default="weights/bytetrack_m_mot17.pth.tar", help="ByteTrack/YOLOX MOT checkpoint.")
    parser.add_argument("--reid-ckpt", default="ultralytics/yolo/v8/detect/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7")
    parser.add_argument("--output-dir", default="runs/track_compare/mot_detector_300")
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"], help="Use cpu unless CUDA is available.")
    parser.add_argument("--max-frames", type=int, default=300, help="Frames to process; 0 means full video.")
    parser.add_argument("--conf", type=float, default=0.01, help="YOLOX detector confidence threshold.")
    parser.add_argument("--nms", type=float, default=0.7, help="YOLOX detector NMS threshold.")
    parser.add_argument("--tsize", type=int, default=960, help="Square test size for faster CPU verification.")
    parser.add_argument("--track-thresh", type=float, default=0.50, help="ByteTrack high-score threshold.")
    parser.add_argument("--track-buffer", type=int, default=60, help="ByteTrack lost track buffer.")
    parser.add_argument("--match-thresh", type=float, default=0.80, help="ByteTrack match threshold.")
    parser.add_argument("--max-dets", type=int, default=50, help="Keep top-N detections per frame before tracking; 0 keeps all.")
    parser.add_argument("--deepsort-min-conf", type=float, default=0.05)
    parser.add_argument("--deepsort-max-age", type=int, default=150)
    parser.add_argument("--deepsort-n-init", type=int, default=1)
    parser.add_argument("--fuse", action="store_true", help="Fuse YOLOX conv/bn layers.")
    return parser.parse_args()


class MotYoloxDetector:
    def __init__(self, args: argparse.Namespace):
        self.exp = get_exp(args.exp_file, None)
        self.exp.test_conf = args.conf
        self.exp.nmsthre = args.nms
        self.exp.test_size = (args.tsize, args.tsize)
        use_cuda = args.device == "gpu" and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.model = self.exp.get_model().to(self.device)
        print(f"YOLOX model: {get_model_info(self.model, self.exp.test_size)}")
        ckpt = torch.load(args.ckpt, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
        if args.fuse:
            self.model = fuse_model(self.model)
        self.model.eval()
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    @torch.inference_mode()
    def __call__(self, frame: np.ndarray) -> tuple[np.ndarray, torch.Tensor | None]:
        h, w = frame.shape[:2]
        image, ratio = preproc(frame, self.exp.test_size, self.rgb_means, self.std)
        tensor = torch.from_numpy(image).unsqueeze(0).float().to(self.device)
        outputs = self.model(tensor)
        outputs = postprocess(outputs, self.exp.num_classes, self.exp.test_conf, self.exp.nmsthre)
        if outputs[0] is None:
            return np.empty((0, 5), dtype=np.float32), None

        bt_output = outputs[0].detach()
        raw = bt_output.cpu().numpy()
        scores = raw[:, 4] * raw[:, 5]
        boxes = raw[:, :4] / ratio
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1)
        detections = np.concatenate([boxes, scores[:, None]], axis=1).astype(np.float32)
        return detections, bt_output


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    xywh = boxes.copy()
    xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
    xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
    xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
    xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
    return xywh


def draw_xyxy_tracks(frame: np.ndarray, tracks: np.ndarray, label_prefix: str) -> None:
    if tracks is None or len(tracks) == 0:
        return
    for track in tracks:
        x1, y1, x2, y2 = [int(v) for v in track[:4]]
        track_id = int(track[-1])
        color = ((track_id * 37) % 255, (track_id * 17) % 255, (track_id * 97) % 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label_prefix} {track_id}", (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_tlwh_tracks(frame: np.ndarray, tracks: list, label_prefix: str) -> None:
    for track in tracks:
        x, y, w, h = track.tlwh
        track_id = int(track.track_id)
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        color = ((track_id * 37) % 255, (track_id * 17) % 255, (track_id * 97) % 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label_prefix} {track_id}", (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(source)
    if not Path(args.ckpt).exists():
        raise FileNotFoundError(args.ckpt)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = total_frames if args.max_frames == 0 else min(args.max_frames, total_frames)

    deep_video = output_dir / f"{source.stem}_mot_yolox_deepsort.mp4"
    byte_video = output_dir / f"{source.stem}_mot_yolox_bytetrack.mp4"
    counts_csv = output_dir / f"{source.stem}_mot_yolox_counts.csv"
    deep_writer = cv2.VideoWriter(str(deep_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    byte_writer = cv2.VideoWriter(str(byte_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    detector = MotYoloxDetector(args)
    deepsort = DeepSort(
        args.reid_ckpt,
        min_confidence=args.deepsort_min_conf,
        nms_max_overlap=1.0,
        max_age=args.deepsort_max_age,
        n_init=args.deepsort_n_init,
        use_cuda=args.device == "gpu" and torch.cuda.is_available(),
    )
    bt_args = SimpleNamespace(
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        mot20=False,
    )
    bytetrack = BYTETracker(bt_args, frame_rate=int(round(fps)) or 30)

    start = time.time()
    with counts_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "detections", "high_detections", "deepsort_tracks", "bytetrack_tracks"])
        for frame_idx in range(limit):
            ok, frame = cap.read()
            if not ok:
                break

            detections, bt_output = detector(frame)
            if args.max_dets > 0 and len(detections) > args.max_dets:
                order = np.argsort(-detections[:, 4])[:args.max_dets]
                detections = detections[order]
                if bt_output is not None:
                    bt_output = bt_output[torch.as_tensor(order, device=bt_output.device)]
            high_count = int((detections[:, 4] >= args.track_thresh).sum()) if len(detections) else 0

            deep_frame = frame.copy()
            byte_frame = frame.copy()

            if len(detections):
                xywh = torch.from_numpy(xyxy_to_xywh(detections[:, :4])).float()
                confs = torch.from_numpy(detections[:, 4:5]).float()
                classes = [0] * len(detections)
            else:
                xywh = torch.empty((0, 4), dtype=torch.float32)
                confs = torch.empty((0, 1), dtype=torch.float32)
                classes = []

            deep_outputs, _ = deepsort.update(xywh, confs, classes, frame)
            deep_count = len(deep_outputs) if len(deep_outputs) else 0
            draw_xyxy_tracks(deep_frame, deep_outputs, "D")

            if bt_output is not None:
                byte_tracks = bytetrack.update(bt_output, [height, width], detector.exp.test_size)
            else:
                empty = torch.empty((0, 7), dtype=torch.float32)
                byte_tracks = bytetrack.update(empty, [height, width], detector.exp.test_size)
            byte_count = len(byte_tracks)
            draw_tlwh_tracks(byte_frame, byte_tracks, "B")

            cv2.putText(deep_frame, f"frame {frame_idx + 1}/{limit} det={len(detections)} deep={deep_count}",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(byte_frame, f"frame {frame_idx + 1}/{limit} det={len(detections)} byte={byte_count}",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            deep_writer.write(deep_frame)
            byte_writer.write(byte_frame)
            writer.writerow([frame_idx + 1, len(detections), high_count, deep_count, byte_count])

            if (frame_idx + 1) % 50 == 0:
                print(f"processed {frame_idx + 1}/{limit} frames in {time.time() - start:.1f}s")

    cap.release()
    deep_writer.release()
    byte_writer.release()
    print(f"saved deepsort video: {deep_video}")
    print(f"saved bytetrack video: {byte_video}")
    print(f"saved counts: {counts_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
