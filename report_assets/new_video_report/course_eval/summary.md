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
| avg_detections | 87.28 |
| min_detections | 32.00 |
| max_detections | 131.0 |
| zero_detection_ratio | 0.00 |
| avg_conf | 0.236 |
| conf_p10 | 0.061 |
| conf_p50 | 0.148 |
| conf_p90 | 0.581 |
| avg_box_area_ratio | 0.001 |
| small_box_ratio | 1.00 |
| medium_box_ratio | 0.00 |
| large_box_ratio | 0.00 |
| avg_latency_ms | 2822.0 |
| p50_latency_ms | 2836.9 |
| p95_latency_ms | 5045.0 |
| p99_latency_ms | 5719.4 |
| avg_pipeline_fps | 0.630 |
| peak_gpu_memory_mb | 716.0 |

## 跟踪器指标

| 跟踪器 | 平均轨迹 | 零轨迹帧率 | 跳变均值 | 大跳变帧率 | 唯一ID数 | 平均寿命 | 短轨迹比例 | 新ID/帧 | 平均FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| deepsort | 101.2 | 0.001 | 7.29 | 0.010 | 2386.0 | 38.53 | 0.026 | 25.13 | 0.630 |
| bytetrack_lite | 87.28 | 0.00 | 9.85 | 0.082 | 60105.0 | 1.32 | 0.983 | 79.13 | 0.630 |

## 说明

- 本脚本在无人工标注条件下生成代理指标，不能替代 MOTA、IDF1、Precision、Recall。
- 轨迹跳变、大跳变帧率、短轨迹比例和新ID/帧越低，通常说明轨迹更稳定。
- 小目标比例可用于支撑远距离行人更难检测的场景分析。
