# DeepSORT 与 ByteTrackLite 默认/调整参数对比实验

## 视频信息

| 指标 | 数值 |
|---|---:|
| 视频路径 | `/home/xie/Project/YOLOv8-DeepSORT-Object-Tracking/video/video.mp4` |
| 分辨率 | 1280 x 720 |
| FPS | 30.00 |
| 总帧数 | 908 |
| 时长 | 30.27 s |

## 固定检测参数

| 参数 | 数值 |
|---|---:|
| model | `yolov8m.pt` |
| imgsz | 1280 |
| conf / low-conf | 0.05 |
| iou | 0.85 |
| device | `0` |

## 跟踪器参数分组

| 实验 | 跟踪器 | 关键参数 |
|---|---|---|
| deepsort_default | DeepSORT | min_conf=0.30, max_age=70, n_init=3 |
| deepsort_tuned | DeepSORT | min_conf=0.05, max_age=150, n_init=1 |
| bytetrack_default | ByteTrackLite | track_thresh=0.50, max_age=60 |
| bytetrack_tuned | ByteTrackLite | track_thresh=0.05, max_age=150 |

## 输出位置

- 跟踪视频和逐帧 CSV：`/home/xie/Project/YOLOv8-DeepSORT-Object-Tracking/runs/new_video_report/tracker_param_compare`
- 汇总表和图：`/home/xie/Project/YOLOv8-DeepSORT-Object-Tracking/report_assets/new_video_report/tracker_param_compare`
- 核心汇总表：`/home/xie/Project/YOLOv8-DeepSORT-Object-Tracking/report_assets/new_video_report/tracker_param_compare/performance_summary.md`
- 检测/轨迹数量对比图：`/home/xie/Project/YOLOv8-DeepSORT-Object-Tracking/report_assets/new_video_report/tracker_param_compare/count_comparison.png`
- 稳定性对比图：`/home/xie/Project/YOLOv8-DeepSORT-Object-Tracking/report_assets/new_video_report/tracker_param_compare/stability_comparison.png`
- 轨迹数时间序列：`/home/xie/Project/YOLOv8-DeepSORT-Object-Tracking/report_assets/new_video_report/tracker_param_compare/timeseries_overview.png`

## 报告使用建议

- 该实验固定 YOLO 检测参数，只改变跟踪器参数，因此适合写入报告的“跟踪器参数消融”部分。
- 对 DeepSORT，重点比较零轨迹帧率、轨迹跳变均值、大跳变帧率和平均轨迹数。
- 对 ByteTrackLite，重点比较 `track_thresh` 降低后是否带来更多轨迹、更低零轨迹帧率，以及是否增加轨迹跳变。
- 本实验仍是无人工标注代理指标，不能替代 MOTA、IDF1、Precision、Recall。
