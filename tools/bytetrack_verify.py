#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import lap
import numpy as np
import torch

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.data.augment import LetterBox
from ultralytics.yolo.utils import ops
from ultralytics.yolo.utils.checks import check_imgsz


PERSON_CLASS_ID = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick YOLO + ByteTrack-style verification for person tracking.")
    parser.add_argument("--source", default="video/video.mp4", help="Input video path.")
    parser.add_argument("--model", default="yolov8m.pt", help="YOLO weights path.")
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu or 0.")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO input image size.")
    parser.add_argument("--high-conf", type=float, default=0.25, help="High-score detection threshold for first association.")
    parser.add_argument("--low-conf", type=float, default=0.03, help="Low-score detection threshold for second association.")
    parser.add_argument("--track-thresh", type=float, default=0.50, help="Minimum score to initialize a new track.")
    parser.add_argument("--iou", type=float, default=0.85, help="YOLO NMS IoU threshold.")
    parser.add_argument("--match-thresh", type=float, default=0.80, help="IoU threshold for track association.")
    parser.add_argument("--max-age", type=int, default=60, help="Frames to keep unmatched tracks alive.")
    parser.add_argument("--max-frames", type=int, default=300, help="Frames to process; 0 means full video.")
    parser.add_argument("--output-dir", default="runs/track_compare/bytetrack_verify", help="Output directory.")
    parser.add_argument("--show", action="store_true", help="Display result window.")
    return parser.parse_args()


def xyxy_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-6, None)


def linear_assign(cost: np.ndarray, thresh: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    _, x, y = lap.lapjv(cost, extend_cost=True, cost_limit=thresh)
    matches = [(i, int(j)) for i, j in enumerate(x) if j >= 0]
    unmatched_a = [i for i, j in enumerate(x) if j < 0]
    matched_b = {j for _, j in matches}
    unmatched_b = [j for j in range(cost.shape[1]) if j not in matched_b]
    return matches, unmatched_a, unmatched_b


@dataclass
class Track:
    track_id: int
    box: np.ndarray
    score: float
    age: int = 0
    hits: int = 1
    history: list[tuple[int, int]] = field(default_factory=list)

    def update(self, box: np.ndarray, score: float) -> None:
        self.box = box
        self.score = float(score)
        self.age = 0
        self.hits += 1

    def mark_missed(self) -> None:
        self.age += 1

    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return int((x1 + x2) / 2), int((y1 + y2) / 2)


class ByteTrackLite:
    def __init__(self, match_thresh: float, track_thresh: float, max_age: int):
        self.match_thresh = match_thresh
        self.track_thresh = track_thresh
        self.max_age = max_age
        self.tracks: list[Track] = []
        self.next_id = 1

    def update(self, detections: np.ndarray) -> list[Track]:
        high = detections[detections[:, 4] >= self.track_thresh] if len(detections) else detections
        low = detections[(detections[:, 4] < self.track_thresh) & (detections[:, 4] > 0)] if len(detections) else detections

        active_boxes = np.array([t.box for t in self.tracks], dtype=np.float32)
        matches, unmatched_tracks, unmatched_high = self._match(active_boxes, high[:, :4] if len(high) else np.empty((0, 4)))
        for ti, di in matches:
            self.tracks[ti].update(high[di, :4], high[di, 4])

        remaining_track_indices = unmatched_tracks
        remaining_boxes = np.array([self.tracks[i].box for i in remaining_track_indices], dtype=np.float32)
        low_matches, still_unmatched_rel, _ = self._match(remaining_boxes, low[:, :4] if len(low) else np.empty((0, 4)))
        matched_remaining = set()
        for rel_ti, di in low_matches:
            ti = remaining_track_indices[rel_ti]
            self.tracks[ti].update(low[di, :4], low[di, 4])
            matched_remaining.add(ti)

        still_unmatched = {remaining_track_indices[i] for i in still_unmatched_rel}
        for ti in still_unmatched:
            if ti not in matched_remaining:
                self.tracks[ti].mark_missed()

        for di in unmatched_high:
            if high[di, 4] >= self.track_thresh:
                self.tracks.append(Track(self.next_id, high[di, :4].copy(), float(high[di, 4])))
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        for track in self.tracks:
            track.history.append(track.center())
            if len(track.history) > 64:
                track.history = track.history[-64:]
        return [t for t in self.tracks if t.age == 0]

    def _match(self, tracks: np.ndarray, detections: np.ndarray) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        ious = xyxy_iou(tracks, detections)
        cost = 1.0 - ious
        return linear_assign(cost, 1.0 - self.match_thresh)


class YoloPersonDetector:
    def __init__(self, weights: str, device: str, imgsz: int, low_conf: float, iou: float):
        self.device = torch.device("cuda:0" if device != "cpu" and torch.cuda.is_available() else "cpu")
        self.model = AutoBackend(weights, device=self.device, fp16=False)
        stride = getattr(self.model, "stride", 32)
        self.stride = int(max(stride)) if isinstance(stride, (list, tuple)) else int(stride)
        self.imgsz = check_imgsz(imgsz, stride=self.stride, min_dim=2)
        self.low_conf = low_conf
        self.iou = iou
        self.letterbox = LetterBox(self.imgsz, auto=True, stride=self.stride)

    @torch.inference_mode()
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        image = self.letterbox(image=frame)
        tensor = torch.from_numpy(image).to(self.device)
        tensor = tensor.permute(2, 0, 1).contiguous().float() / 255.0
        tensor = tensor[None]
        preds = self.model(tensor)
        det = ops.non_max_suppression(
            preds,
            conf_thres=self.low_conf,
            iou_thres=self.iou,
            classes=[PERSON_CLASS_ID],
            max_det=1000,
        )[0]
        if len(det) == 0:
            return np.empty((0, 5), dtype=np.float32)
        det[:, :4] = ops.scale_boxes(tensor.shape[2:], det[:, :4], frame.shape).round()
        return det[:, :5].cpu().numpy().astype(np.float32)


def draw_tracks(frame: np.ndarray, tracks: list[Track]) -> None:
    for track in tracks:
        x1, y1, x2, y2 = track.box.astype(int)
        color = ((track.track_id * 37) % 255, (track.track_id * 17) % 255, (track.track_id * 97) % 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {track.track_id} person {track.score:.2f}", (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        for p1, p2 in zip(track.history, track.history[1:]):
            cv2.line(frame, p1, p2, color, 2)


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(source)
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

    out_path = output_dir / f"{source.stem}_bytetrack_lite.mp4"
    csv_path = output_dir / f"{source.stem}_bytetrack_lite_counts.csv"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    detector = YoloPersonDetector(args.model, args.device, args.imgsz, args.low_conf, args.iou)
    tracker = ByteTrackLite(args.match_thresh, args.track_thresh, args.max_age)

    start = time.time()
    with csv_path.open("w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["frame", "detections", "high_detections", "active_tracks", "frame_time_ms", "fps"])
        for frame_idx in range(limit):
            frame_start = time.time()
            ok, frame = cap.read()
            if not ok:
                break
            detections = detector(frame)
            high_count = int((detections[:, 4] >= args.track_thresh).sum()) if len(detections) else 0
            tracks = tracker.update(detections)
            draw_tracks(frame, tracks)
            cv2.putText(frame, f"frame {frame_idx + 1}/{limit} det={len(detections)} tracks={len(tracks)}",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            writer.write(frame)
            frame_time_ms = (time.time() - frame_start) * 1000.0
            current_fps = 1000.0 / frame_time_ms if frame_time_ms > 0 else 0.0
            log.writerow([frame_idx + 1, len(detections), high_count, len(tracks), f"{frame_time_ms:.3f}", f"{current_fps:.3f}"])
            if args.show:
                cv2.imshow("bytetrack-lite", frame)
                if cv2.waitKey(1) == 27:
                    break
            if (frame_idx + 1) % 50 == 0:
                elapsed = time.time() - start
                print(f"processed {frame_idx + 1}/{limit} frames in {elapsed:.1f}s")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"saved video: {out_path}")
    print(f"saved counts: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
