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
| avg_detections | 138.6 |
| min_detections | 68.00 |
| max_detections | 205.0 |
| zero_detection_ratio | 0.00 |
| avg_conf | 0.177 |
| conf_p10 | 0.058 |
| conf_p50 | 0.117 |
| conf_p90 | 0.392 |
| avg_box_area_ratio | 0.001 |
| small_box_ratio | 1.000 |
| medium_box_ratio | 0.000 |
| large_box_ratio | 0.00 |
| avg_latency_ms | 8963.8 |
| p50_latency_ms | 6530.3 |
| p95_latency_ms | 10397.2 |
| p99_latency_ms | 12492.3 |
| avg_pipeline_fps | 0.292 |
| peak_gpu_memory_mb | 927.4 |

## 跟踪器指标

| 跟踪器 | 平均轨迹 | 零轨迹帧率 | 跳变均值 | 大跳变帧率 | 唯一ID数 | 平均寿命 | 短轨迹比例 | 新ID/帧 | 平均FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| deepsort | 169.7 | 0.001 | 10.99 | 0.002 | 4096.0 | 37.62 | 0.022 | 43.56 | 0.292 |
| bytetrack_lite | 138.6 | 0.00 | 14.30 | 0.051 | 91985.0 | 1.37 | 0.976 | 127.4 | 0.292 |

## 说明

- 本脚本在无人工标注条件下生成代理指标，不能替代 MOTA、IDF1、Precision、Recall。
- 轨迹跳变、大跳变帧率、短轨迹比例和新ID/帧越低，通常说明轨迹更稳定。
- 小目标比例可用于支撑远距离行人更难检测的场景分析。
