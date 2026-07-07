# 新视频课程设计报告数据总输出

## 输入视频

| 指标 | 数值 |
|---|---:|
| 路径 | `/home/xie/Project/YOLOv8-DeepSORT-Object-Tracking/video/video.mp4` |
| 分辨率 | 1280 x 720 |
| FPS | 30.00 |
| 总帧数 | 908 |
| 时长 | 30.27 s |

## 关键输出

- 视频基础信息与首帧图：`/home/xie/Project/YOLOv8-DeepSORT-Object-Tracking/report_assets/new_video_report/base`
- 单独运行统计与可视化视频：`/home/xie/Project/YOLOv8-DeepSORT-Object-Tracking/runs/new_video_report`
- 单独运行性能评估：`/home/xie/Project/YOLOv8-DeepSORT-Object-Tracking/report_assets/new_video_report/performance_eval`
- DeepSORT/ByteTrack 公平综合评估：`/home/xie/Project/YOLOv8-DeepSORT-Object-Tracking/report_assets/new_video_report/course_eval`
- 4 组消融实验：`/home/xie/Project/YOLOv8-DeepSORT-Object-Tracking/report_assets/new_video_report/ablation`
- 消融汇总表：`/home/xie/Project/YOLOv8-DeepSORT-Object-Tracking/report_assets/new_video_report/ablation/ablation_summary.md`

## 运行参数

- `source`: `video/video.mp4`
- `device`: `0`
- `model`: `yolov8m.pt`
- `max_frames`: `0`
- `skip_visual_videos`: `False`
- `skip_ablation`: `False`

## 报告填写建议

- 第 1 章视频信息和场景图使用 `report_assets/new_video_report/base/metrics_summary.md` 与 `scene_frame.jpg`。
- 第 6 章 DeepSORT/ByteTrack 单独运行对比使用 `performance_eval/performance_summary.md` 和对应 PNG 图。
- 第 6.11 综合评估使用 `course_eval/summary.md`、`tracker_summary.csv` 和各类曲线图。
- YOLOv8n/YOLOv8m 默认/调整参数对比使用 `ablation/ablation_summary.md`。
- 若需要把新结果替换报告旧图，可将本目录中的 PNG 路径更新到 `课程设计报告.md`。
