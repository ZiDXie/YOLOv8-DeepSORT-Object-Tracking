# 课程设计综合性能评估

## 视频与环境

| 指标 | 数值 |
|---|---:|
| 视频帧数 | 6030 |
| 分辨率 | 1280 x 720 |
| 原视频FPS | 30.00 |
| CUDA可用 | True |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |

## 检测指标

| 指标 | 数值 |
|---|---:|
| frames | 6030.0 |
| avg_detections | 118.6 |
| min_detections | 32.00 |
| max_detections | 198.0 |
| zero_detection_ratio | 0.00 |
| avg_conf | 0.183 |
| conf_p10 | 0.057 |
| conf_p50 | 0.114 |
| conf_p90 | 0.433 |
| avg_box_area_ratio | 0.001 |
| small_box_ratio | 1.000 |
| medium_box_ratio | 0.000 |
| large_box_ratio | 0.00 |
| avg_latency_ms | 4109.8 |
| p50_latency_ms | 4077.3 |
| p95_latency_ms | 6600.0 |
| p99_latency_ms | 7714.6 |
| avg_pipeline_fps | 0.328 |
| peak_gpu_memory_mb | 990.2 |

## 跟踪器指标

| 跟踪器 | 平均轨迹 | 零轨迹帧率 | 跳变均值 | 大跳变帧率 | 唯一ID数 | 平均寿命 | 短轨迹比例 | 新ID/帧 | 平均FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| deepsort | 149.0 | 0.000 | 9.28 | 0.004 | 13067.0 | 68.75 | 0.013 | 33.34 | 0.328 |
| bytetrack_lite | 118.6 | 0.00 | 12.52 | 0.065 | 481893.0 | 1.48 | 0.957 | 99.31 | 0.328 |

## 说明

- 本脚本在无人工标注条件下生成代理指标，不能替代 MOTA、IDF1、Precision、Recall。
- 轨迹跳变、大跳变帧率、短轨迹比例和新ID/帧越低，通常说明轨迹更稳定。
- 小目标比例可用于支撑远距离行人更难检测的场景分析。
