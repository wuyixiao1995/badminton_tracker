# badminton_tracker

基于计算机视觉的羽毛球单打比赛自动分析系统。

**核心原则**: 系统不识别羽毛球或击球动作，仅通过球员位置变化和比赛规则推断比分。

## 项目结构

```
badminton_tracker/
├── main.py                  # 主入口，两阶段流水线（追踪→计分）
├── calibrator.py            # 球场检测 + 透视变换
├── tracker.py               # YOLOv8-pose 球员检测与脚部追踪
├── detector.py              # 就绪状态检测（滑动窗口 + 冷却机制）
├── scorer.py                # 21分制比分状态机（位置推断得分）
├── overlay.py               # 标注视频渲染（迷你球场 + 轨迹 + 比分）
├── models.py                # 数据类定义
├── config.py                # 集中管理所有可调参数
├── court_viewer.py          # 俯视图球场位置回放工具
├── debug_frames.py          # 调试帧生成（脚部位置 + 球场边界标记）
├── tracknetv3_infer.py      # TrackNetV3 羽毛球轨迹推理
├── tracknet_infer.py        # TrackNet 网球轨迹推理（对比）
├── visualize_tracknetv3.py  # 轨迹标注可视化
├── test_shuttle_detect.py   # YOLOv8 羽毛球检测测试
├── ckpts/                   # TrackNetV3 预训练权重 (Git LFS)
│   ├── TrackNet_best.pt     #   130 MB 主追踪模型
│   └── InpaintNet_best.pt   #   6 MB 轨迹修复模型
├── requirements.txt         # Python 依赖
├── PROJECT_SPEC.md          # 完整项目规格书
└── README.md                # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

依赖项: `opencv-python`, `ultralytics` (YOLOv8), `matplotlib`, `numpy`

### 2. 运行完整流水线

```bash
# 自动模式（推荐先试）
python main.py --video match.mp4 --first-server A --output-video annotated.mp4 --output-scores scores.json

# 手动校准模式（自动检测失败时使用）
python main.py --video match.mp4 --manual-calibration
```

### 3. 两阶段模式（推荐开发调试）

**Phase 1: 追踪（保存轨迹数据，耗时）**
```bash
python main.py --video match.mp4 --save-tracks tracks.jsonl --max-frames 3600 --frame-skip 3
```

**Phase 2: 计分（从轨迹回放，秒级迭代）**
```bash
python main.py --load-tracks tracks.jsonl --output-scores scores.json --first-server A
```

Phase 2 不需要重新跑 YOLO，可以反复调整计分参数快速验证。

### 4. 俯视图回放器（肉眼验证坐标映射）
```bash
python court_viewer.py tracks.jsonl --speed 2 --show-trails
```

### 5. 生成调试帧
```bash
python debug_frames.py match.mp4 --frames 0,300,600,900
```
在 `debug_frames/` 目录下查看脚部检测位置和球场边界。

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--video` | (必需) | 输入视频路径 |
| `--first-server` | A | 首局发球方 A/B |
| `--output-video` | annotated.mp4 | 输出视频路径 |
| `--output-scores` | scores.json | 比分 JSON 路径 |
| `--max-dim` | 960 | YOLO 处理分辨率（超此尺寸自动下采样） |
| `--frame-skip` | 3 | YOLO 检测间隔（每N帧检测一次） |
| `--max-frames` | 0 | 最大处理帧数（0=全部） |
| `--output-resolution` | None | 输出视频分辨率 WxH（如 1280x720） |
| `--save-tracks` | None | 保存追踪数据到 JSONL 文件 |
| `--load-tracks` | None | 从 JSONL 加载追踪数据（跳过 YOLO） |
| `--manual-calibration` | - | 手动点击4个角点校准球场 |

## 工作原理

### 流程图

```
视频输入
  │
  ├─→ Calibrator: 球场检测（白线提取 + 透视变换）
  │      └─→ 标准俯视图坐标系（13.4m × 5.18/6.10m）
  │
  └─→ Tracker: 逐帧 YOLOv8-pose 检测
         │
         ├─ 脚部定位: 双踝中点 > 单踝 > bbox底部
         ├─ 透视变换: 像素 → 球场坐标系（米）
         ├─ 身份保持: 最近邻追踪 + 3帧平滑
         │
         ├─→ Detector: 就绪状态判断
         │      ├─ 位置检查: 球员在发球区？
         │      ├─ 运动检查: 速度 < 阈值？
         │      └─ 滑动窗口: 持续1.5s → ReadyEvent
         │
         ├─→ Scorer: 比分状态机
         │      ├─ 发球方得分 → 发球权保持，比分+1
         │      ├─ 接发方得分 → 交换发球权
         │      ├─ 偶分数右区/奇分数左区
         │      └─ 11分换边 / 20:20需净胜2分 / 29:29封顶
         │
         └─→ Overlay: 可视化渲染
                ├─ 浮动面板（比分 + 迷你球场）
                ├─ 移动轨迹线（A=橙色, B=青色）
                └─ 输出 MP4 + JSON + 比分曲线图
```

### 计分推断逻辑

系统**不看羽毛球**，只看球员位置变化来推断得分：

| 场景 | 判定依据 | 动作 |
|------|---------|------|
| 两人在预期发球区 | 正常比赛回合 | 不发分 |
| 发球方左右区互换 | 发球方得分，保持发球权 | score+1 |
| 发球方球员变了 | 接发方得分，交换发球权 | score+1, 换发 |

## 配置参数

编辑 `config.py` 调整系统行为：

```python
# 追踪
PLAYER_MIN_CONFIDENCE = 0.5   # 球员检测最低置信度
SMOOTHING_WINDOW = 3          # 坐标平滑窗口（帧）
TRACK_MAX_AGE = 30            # 追踪丢失最大帧数

# 就绪检测
VELOCITY_THRESHOLD = 0.5      # 静止判定速度阈值 (m/s)
READY_DURATION = 1.5          # 就绪持续时间 (秒)
COOLDOWN_DURATION = 3.0       # 事件冷却时间 (秒)
ZONE_TOLERANCE = 1.0          # 发球区容差 (米)

# 渲染
TRAIL_LENGTH = 20             # 轨迹显示长度（帧）
OVERLAY_OPACITY = 0.6         # 覆盖层透明度
```

## 输出文件

| 文件 | 格式 | 内容 |
|------|------|------|
| `annotated.mp4` | MP4 | 带覆盖层的标注视频 |
| `scores.json` | JSON | 每分时间戳 + 比分 + 发球方 |
| `score_curve.png` | PNG | 时间-比分曲线图 |
| `tracks.jsonl` | JSONL | 逐帧球员位置数据 |

## 羽毛球轨迹追踪 (TrackNetV3)

使用 TrackNetV3 深度学习模型直接检测羽毛球轨迹（热力图方式）。

> 论文: [TrackNetV3: Enhancing ShuttleCock Tracking with Augmentations and Trajectory Rectification](https://dl.acm.org/doi/10.1145/3595916.3626370)
> 性能: **97.5% Accuracy**, 98.6% F1 on Shuttlecock Trajectory Dataset

### 预训练模型

模型权重已通过 Git LFS 上传到 `ckpts/` 目录：

| 文件 | 大小 | 说明 |
|------|------|------|
| `ckpts/TrackNet_best.pt` | 130 MB | 主追踪模型（U-Net 热力图） |
| `ckpts/InpaintNet_best.pt` | 6 MB | 轨迹修复模型（1D CNN） |

如未自动下载 LFS 文件：
```bash
git lfs pull
```

### 运行推理

```bash
# TrackNetV3 羽毛球专用推理（含时间集成，精度最高）
python tracknetv3_infer.py --video test_match.mp4 --max-frames 600 --ensemble

# 生成带置信度标注的可视化视频
python visualize_tracknetv3.py --csv tracknetv3_output_ball.csv --video test_match.mp4

# 网球 TrackNet 对比测试（效果差，仅供参考）
python tracknet_infer.py --video test_match.mp4 --max-frames 600
```

### 效果预览

![标注视频](tracknetv3_annotated.mp4)

标注含义：
- 🟡 **黄色圆圈** = 检测位置（越大置信度越高）
- 🟡 **虚线** = 连续帧轨迹
- 🔴 **红色/大圈** = 高速移动帧（很可能是真球）

### 对比测试结果

| 模型 | 训练数据 | 检测率 | 中位速度 | 假阳性率 | 结论 |
|------|---------|--------|---------|---------|------|
| TrackNet (网球) | 网球赛事 | 81.7% | 4.5 px/f | ~75% | ❌ 基本是假阳性 |
| **TrackNetV3** | 羽毛球赛事 | 67.7% | 3.0 px/f | ~75% | ⚠️ 约2-3段疑似真球 |

> 两个模型都训练于专业赛事转播画面，与业余视频差异大。需要后处理过滤（速度阈值）提高可用性。

## 视频要求

- 固定机位拍摄（不移动/变焦）
- 包含完整单打/双打球场
- 球场线清晰可见，光照均匀
- 分辨率 ≥ 720p，帧率 ≥ 25fps

## 已知问题

- [ ] 球场自动检测准确率需提升（白线提取策略待优化）
- [ ] 球员脚部在底线外时坐标外推过大
- [ ] 需要更好的就绪状态判定（中场区域检测）

## License

MIT
