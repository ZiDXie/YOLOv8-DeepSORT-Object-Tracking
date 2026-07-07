# 课程设计综合性能评估

## 视频与环境

| 指标 | 数值 |
|---|---:|
| 视频帧数 | 908 |
| 分辨率 | 1280 x 720 |
| 原视频FPS | 30.00 |
| CUDA可用 | True |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |

## 检测指标

| 指标 | 数值 |
|---|---:|
| frames | 908.0 |
| avg_detections | 10.43 |
| min_detections | 2.00 |
| max_detections | 19.00 |
| zero_detection_ratio | 0.00 |
| avg_conf | 0.409 |
| conf_p10 | 0.270 |
| conf_p50 | 0.376 |
| conf_p90 | 0.594 |
| avg_box_area_ratio | 0.002 |
| small_box_ratio | 1.00 |
| medium_box_ratio | 0.00 |
| large_box_ratio | 0.00 |
| avg_latency_ms | 71.98 |
| p50_latency_ms | 68.17 |
| p95_latency_ms | 119.1 |
| p99_latency_ms | 167.7 |
| avg_pipeline_fps | 16.35 |
| peak_gpu_memory_mb | 166.0 |

## 跟踪器指标

| 跟踪器 | 平均轨迹 | 零轨迹帧率 | 跳变均值 | 大跳变帧率 | 唯一ID数 | 平均寿命 | 短轨迹比例 | 新ID/帧 | 平均FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| deepsort | 12.00 | 0.001 | 1.60 | 0.041 | 328.0 | 33.23 | 0.024 | 2.41 | 16.35 |
| bytetrack_lite | 10.43 | 0.00 | 2.07 | 0.082 | 7399.0 | 1.28 | 0.986 | 9.23 | 16.35 |

## 说明

- 本脚本在无人工标注条件下生成代理指标，不能替代 MOTA、IDF1、Precision、Recall。
- 轨迹跳变、大跳变帧率、短轨迹比例和新ID/帧越低，通常说明轨迹更稳定。
- 小目标比例可用于支撑远距离行人更难检测的场景分析。
