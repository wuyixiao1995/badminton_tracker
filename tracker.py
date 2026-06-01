"""球员追踪模块 — 基于 YOLOv8-pose 的羽毛球球员检测与追踪

负责将视频帧中的球员检测出来、区分 A/B 身份、
帧间追踪匹配、像素坐标→球场坐标变换，
以及输出每帧追踪结果（PlayerFrame）。
"""

from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    BIRDS_EYE_HEIGHT,
    BIRDS_EYE_WIDTH,
    COCO_LEFT_ANKLE,
    COCO_RIGHT_ANKLE,
    COURT_LENGTH,
    COURT_WIDTH,
    PLAYER_MIN_CONFIDENCE,
    POSE_MODEL,
    SMOOTHING_WINDOW,
    TRACK_MAX_AGE,
)
from models import Point, PlayerFrame


class PlayerTracker:
    """基于 YOLOv8-pose 的羽毛球球员追踪器。

    功能:
      - 利用 YOLOv8-pose 模型检测人体关键点，提取脚部位置
      - 首次帧根据画面上下位置区分球员 A（近端/画面下方）和 B（远端/画面上方）
      - 逐帧最近邻匹配追踪，丢失超过阈值后重置
      - 3 帧移动平均平滑
      - 透视变换 + 球场尺度变换（像素 → 米）
    """

    def __init__(self, model_name: str = POSE_MODEL):
        """初始化追踪器。

        Args:
            model_name: YOLOv8-pose 模型路径，默认使用配置中的 POSE_MODEL
        """
        self.model = YOLO(model_name)

        # 位置历史（用于 3 帧移动平均平滑）
        self.player_a_positions: deque = deque(maxlen=SMOOTHING_WINDOW)
        self.player_b_positions: deque = deque(maxlen=SMOOTHING_WINDOW)

        # 上一帧追踪位置（用于帧间最近邻匹配）
        self.prev_positions: dict[str, tuple | None] = {"A": None, "B": None}

        # 球员追踪丢失帧计数（超过 TRACK_MAX_AGE 后位置置 None）
        self.lost_frames: dict[str, int] = {"A": 0, "B": 0}

        # 最近一次匹配成功时的检测置信度
        self._confidences: dict[str, float] = {"A": 0.0, "B": 0.0}

        # 上一帧球场坐标 + 时间戳（用于速度计算）
        self._prev_court_positions: dict[str, Point | None] = {"A": None, "B": None}
        self._prev_timestamp: float | None = None

        # 帧尺寸（由 process_frame 设置，供首帧身份识别使用）
        self._frame_shape: tuple | None = None

        # 是否已完成 A/B 身份识别
        self.identified: bool = False

    # ------------------------------------------------------------------
    # 检测
    # ------------------------------------------------------------------

    def detect_players(self, frame: np.ndarray) -> list[dict]:
        """在单帧图像上运行 YOLOv8-pose 检测，提取球员脚部位置。

        检测逻辑:
          1. 对每个检测到的人体，提取 COCO 关键点中左右脚踝（index 15, 16）
          2. 优先使用双脚踝中点 → 单脚踝 → bbox 底部中心

        Args:
            frame: 输入图像 (H, W, 3) BGR

        Returns:
            list[dict]: 每个检测到的球员信息:
              - foot_pos: (x, y) 像素坐标脚的近似位置
              - confidence: 该位置对应的检测置信度
              - bbox: (x1, y1, x2, y2) 或 None
        """
        results = self.model(frame, verbose=False)
        detections: list[dict] = []

        if not results or len(results) == 0:
            return detections

        result = results[0]

        # ----------------------------------------------------------
        # 关键点数据: (N, 17, 3) — [x, y, confidence]
        # ----------------------------------------------------------
        if result.keypoints is None or result.keypoints.data is None:
            return detections

        keypoints_data = result.keypoints.data.cpu().numpy()
        boxes = result.boxes
        has_boxes = boxes is not None

        for i, kps in enumerate(keypoints_data):
            left_ankle = kps[COCO_LEFT_ANKLE]    # [x, y, conf]
            right_ankle = kps[COCO_RIGHT_ANKLE]   # [x, y, conf]

            # bbox 信息（备用）
            bbox = None
            if has_boxes and i < len(boxes):
                bbox_arr = boxes[i].xyxy[0].cpu().numpy()
                bbox = (float(bbox_arr[0]), float(bbox_arr[1]),
                        float(bbox_arr[2]), float(bbox_arr[3]))

            # 优先使用脚踝关键点
            foot_pos = self.foot_position(kps)

            if foot_pos is not None:
                # 关键点置信度的最大值
                confidence = float(max(left_ankle[2], right_ankle[2]))
            elif bbox is not None:
                # 降级：使用 bbox 底部中心作为脚的位置
                foot_pos = ((bbox[0] + bbox[2]) / 2.0, bbox[3])
                if has_boxes and i < len(boxes) and boxes[i].conf is not None:
                    confidence = float(boxes[i].conf.item())
                else:
                    confidence = 0.0
            else:
                # 无关键点也无 bbox，跳过
                continue

            detections.append({
                "foot_pos": foot_pos,
                "confidence": confidence,
                "bbox": bbox,
            })

        return detections

    # ------------------------------------------------------------------
    # 脚部位置提取
    # ------------------------------------------------------------------

    def foot_position(self, keypoints: np.ndarray) -> tuple | None:
        """从单人的关键点数组中提取脚部位置。

        优先级:
          1. 双脚踝均可见（conf > PLAYER_MIN_CONFIDENCE）→ 返回中点
          2. 仅单脚踝可见 → 返回该脚踝
          3. 双脚踝均不可见 → 返回 None（由调用方降级到 bbox）

        Args:
            keypoints: (17, 3) 数组，每行为 [x, y, confidence]

        Returns:
            (x, y) 像素坐标，或 None
        """
        left = keypoints[COCO_LEFT_ANKLE]
        right = keypoints[COCO_RIGHT_ANKLE]

        left_vis = left[2] > PLAYER_MIN_CONFIDENCE
        right_vis = right[2] > PLAYER_MIN_CONFIDENCE

        if left_vis and right_vis:
            return ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)
        if left_vis:
            return (float(left[0]), float(left[1]))
        if right_vis:
            return (float(right[0]), float(right[1]))
        return None

    # ------------------------------------------------------------------
    # 球员身份识别
    # ------------------------------------------------------------------

    def identify_players(
        self, detections: list[dict], frame_shape: tuple
    ) -> None:
        """首帧身份识别：通过球员在画面中的上下位置区分 A/B。

        规则:
          - 画面靠下（y 值大）的球员 = A（近端/前场）
          - 画面靠上（y 值小）的球员 = B（远端/后场）

        Args:
            detections: detect_players 的输出
            frame_shape: frame.shape (H, W, C)，仅用于预留扩展
        """
        # 过滤出有有效 foot_pos 的检测
        valid = [d for d in detections if d["foot_pos"] is not None]
        if len(valid) < 2:
            return

        # 按 y 降序排列（靠下的在前）
        valid.sort(key=lambda d: d["foot_pos"][1], reverse=True)

        self.prev_positions["A"] = valid[0]["foot_pos"]
        self.prev_positions["B"] = valid[1]["foot_pos"]
        self._confidences["A"] = valid[0]["confidence"]
        self._confidences["B"] = valid[1]["confidence"]

        # 初始化平滑缓存
        self.player_a_positions.append(self.prev_positions["A"])
        self.player_b_positions.append(self.prev_positions["B"])

        self.identified = True

    # ------------------------------------------------------------------
    # 帧间追踪
    # ------------------------------------------------------------------

    def track_players(self, detections: list[dict]) -> dict[str, Point | None]:
        """帧间追踪：最近邻匹配 + 丢失处理 + 3 帧移动平均平滑。

        工作流:
          1. 如果未完成身份识别 → 调用 identify_players
          2. 对每个已追踪球员，在检测结果中寻找最近的 foot_pos
          3. 同一检测结果不会被两个球员同时匹配
          4. 未匹配到 → lost_frames++，超过阈值则位置置 None
          5. 匹配到 → 更新 prev_positions + 平滑 deque
          6. 返回平滑后的位置（deque 均值）

        Args:
            detections: detect_players 的输出列表

        Returns:
            {"A": Point(x, y) | None, "B": Point(x, y) | None}
            其中 x, y 为像素坐标（浮点）
        """
        # -- 首帧身份识别 ------------------------------------------------
        if not self.identified:
            self.identify_players(detections, self._frame_shape)
            if not self.identified:
                return {"A": None, "B": None}
            # 首帧识别后直接返回初始位置
            return self._smoothed_positions()

        # -- 常规追踪：最近邻匹配 ----------------------------------------
        remaining = list(detections)            # 尚未被匹配的检测
        matched: dict[str, tuple | None] = {"A": None, "B": None}
        matched_conf: dict[str, float] = {"A": 0.0, "B": 0.0}

        for player_id in ("A", "B"):
            prev = self.prev_positions[player_id]
            if prev is None:
                continue

            best_idx = -1
            best_dist = float("inf")

            for i, det in enumerate(remaining):
                fp = det["foot_pos"]
                if fp is None:
                    continue
                dist = np.hypot(prev[0] - fp[0], prev[1] - fp[1])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_idx >= 0:
                matched[player_id] = remaining[best_idx]["foot_pos"]
                matched_conf[player_id] = remaining[best_idx]["confidence"]
                self.lost_frames[player_id] = 0
                remaining.pop(best_idx)          # 防止双匹配
            else:
                self.lost_frames[player_id] += 1
                if self.lost_frames[player_id] > TRACK_MAX_AGE:
                    self.prev_positions[player_id] = None
                    (self.player_a_positions if player_id == "A"
                     else self.player_b_positions).clear()

        # -- 更新内部状态 ------------------------------------------------
        for player_id in ("A", "B"):
            pos = matched[player_id]
            if pos is not None:
                self.prev_positions[player_id] = pos
                self._confidences[player_id] = matched_conf[player_id]
                deq = (self.player_a_positions if player_id == "A"
                       else self.player_b_positions)
                deq.append(pos)

        return self._smoothed_positions()

    # ------------------------------------------------------------------
    # 坐标平滑辅助
    # ------------------------------------------------------------------

    def _smoothed_positions(self) -> dict[str, Point | None]:
        """计算平滑后的位置（deque 中所有帧的均值）。

        Returns:
            与 track_players 返回格式相同
        """
        result: dict[str, Point | None] = {}
        for player_id, deq in (
            ("A", self.player_a_positions),
            ("B", self.player_b_positions),
        ):
            if deq:
                xs = [p[0] for p in deq]
                ys = [p[1] for p in deq]
                result[player_id] = Point(sum(xs) / len(xs), sum(ys) / len(ys))
            else:
                result[player_id] = None
        return result

    # ------------------------------------------------------------------
    # 坐标变换（像素 → 球场米数）
    # ------------------------------------------------------------------

    def transform_positions(
        self, pixel_positions: dict[str, Point | None], M: np.ndarray,
        pixel_scale: float = 1.0
    ) -> dict[str, Point | None]:
        """将像素坐标通过透视变换 + 尺度变换转为球场坐标（米）。

        变换流程:
          1. 若 pixel_scale != 1.0, 先将像素坐标放大回原始分辨率
          2. cv2.perspectiveTransform(pixel, M) → 鸟瞰视角像素坐标
          3. x_court = x_birds / BIRDS_EYE_WIDTH  * COURT_LENGTH
          4. y_court = y_birds / BIRDS_EYE_HEIGHT * COURT_WIDTH

        Args:
            pixel_positions: {"A": Point(x,y)像素, "B": ...}
            M: 3×3 透视变换矩阵 (基于原始分辨率计算)
            pixel_scale: 像素缩放因子 (>1.0 表示原始分辨率/处理分辨率)

        Returns:
            {"A": Point(x米, y米) | None, "B": ...}
        """
        result: dict[str, Point | None] = {"A": None, "B": None}

        inv_scale = 1.0 / pixel_scale if pixel_scale > 0 else 1.0

        for player_id in ("A", "B"):
            pos = pixel_positions[player_id]
            if pos is None:
                continue

            # 缩放到原始分辨率
            px = pos.x * inv_scale
            py = pos.y * inv_scale

            # 透视变换需要 (N, 1, 2) 形状
            src = np.array([[[px, py]]], dtype=np.float32)
            dst = cv2.perspectiveTransform(src, M)
            x_bird, y_bird = dst[0][0]

            # 鸟瞰像素 → 球场米数 (y 居中: 0→-half_width, H→+half_width)
            # 不做截断 — 球员在球场外的坐标外推有物理意义
            x_court = x_bird / BIRDS_EYE_WIDTH * COURT_LENGTH
            y_court = (y_bird / BIRDS_EYE_HEIGHT - 0.5) * COURT_WIDTH

            result[player_id] = Point(x_court, y_court)

        return result

    # ------------------------------------------------------------------
    # 主流程（单帧处理）
    # ------------------------------------------------------------------

    def process_frame(
        self, frame: np.ndarray, M: np.ndarray, timestamp: float,
        pixel_scale: float = 1.0
    ) -> list[PlayerFrame]:
        """单帧全流程处理。

        步骤:
          1. detect_players  — YOLO 检测，提取脚部位置
          2. track_players   — 身份识别 + 最近邻追踪 + 平滑
          3. transform_positions — 坐标变换到球场坐标系
          4. 速度计算、组装 PlayerFrame

        Args:
            frame: 视频帧 (H, W, 3) BGR
            M: 3×3 透视变换矩阵（像素 → 鸟瞰, 基于原始分辨率）
            timestamp: 当前帧时间戳（秒）
            pixel_scale: 处理帧/原始帧的缩放比 (如 0.5=处理分辨率是原始的一半)

        Returns:
            [PlayerFrame(A), PlayerFrame(B)]  始终返回 2 个元素
            未检测到的球员 position=(0,0), confidence=0, velocity=0, zone=None
        """
        self._frame_shape = frame.shape

        # ---- 1. 检测 ---------------------------------------------------
        detections = self.detect_players(frame)

        # ---- 2. 追踪 ---------------------------------------------------
        pixel_positions = self.track_players(detections)

        # ---- 3. 坐标变换 (传入 pixel_scale 以正确回算原始分辨率坐标) -----
        court_positions = self.transform_positions(pixel_positions, M, pixel_scale)

        # ---- 4. 组装输出 -----------------------------------------------
        player_frames: list[PlayerFrame] = []
        for player_id in ("A", "B"):
            court_pos = court_positions[player_id]
            confidence = self._confidences[player_id]

            if court_pos is None:
                # 球员未检测到
                player_frames.append(PlayerFrame(
                    player_id=player_id,
                    position=Point(0.0, 0.0),
                    confidence=0.0,
                    timestamp=timestamp,
                    velocity=0.0,
                    zone=None,
                ))
                continue

            # 速度：上一帧到当前帧的位移 / 时间差
            velocity = 0.0
            prev_pos = self._prev_court_positions[player_id]
            if (self._prev_timestamp is not None
                    and prev_pos is not None
                    and timestamp > self._prev_timestamp):
                dt = timestamp - self._prev_timestamp
                dx = court_pos.x - prev_pos.x
                dy = court_pos.y - prev_pos.y
                velocity = float(np.hypot(dx, dy) / dt)

            player_frames.append(PlayerFrame(
                player_id=player_id,
                position=court_pos,
                confidence=confidence,
                timestamp=timestamp,
                velocity=velocity,
                zone=None,          # 区域判断委托给 detector 模块
            ))

        # 保存状态供下一帧速度计算
        self._prev_timestamp = timestamp
        self._prev_court_positions = court_positions

        return player_frames
