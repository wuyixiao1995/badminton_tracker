"""比赛分析覆盖层渲染 — 比分、迷你球场、球员轨迹、视频+数据导出"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque
from typing import Optional
import json
import os

from models import Point, PlayerFrame, ScoreUpdate, MatchState
from config import (
    COLOR_PLAYER_A,
    COLOR_PLAYER_B,
    COLOR_TRAIL_A,
    COLOR_TRAIL_B,
    COLOR_COURT_BG,
    COLOR_COURT_LINE,
    COLOR_TEXT,
    COLOR_TEXT_BG,
    COLOR_OVERLAY_BG,
    COLOR_ZONE_ACTIVE,
    OVERLAY_OPACITY,
    OVERLAY_WIDTH,
    OVERLAY_HEIGHT,
    TRAIL_LENGTH,
    BIRDS_EYE_WIDTH,
    BIRDS_EYE_HEIGHT,
    COURT_LENGTH,
    COURT_WIDTH,
    OUTPUT_VIDEO_CODEC,
    SHORT_SERVICE_LINE,
)


class OverlayRenderer:
    """覆盖层渲染器 — 在视频帧上绘制比分面板、迷你球场、球员轨迹"""

    def __init__(
        self,
        output_video_path: str = "annotated.mp4",
        output_scores_path: str = "scores.json",
    ):
        self.output_video_path = output_video_path
        self.output_scores_path = output_scores_path

        # 轨迹历史 (像素坐标)
        self.trail_a: deque = deque(maxlen=TRAIL_LENGTH)
        self.trail_b: deque = deque(maxlen=TRAIL_LENGTH)

        # 比分更新记录
        self.score_updates: list[ScoreUpdate] = []

        # 视频写入器
        self.video_writer: Optional[cv2.VideoWriter] = None

    # ------------------------------------------------------------------
    # 文字辅助 — 黑色描边 1px 四方向偏移
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_text_with_outline(
        img: np.ndarray,
        text: str,
        pos: tuple,
        font: int,
        scale: float,
        color: tuple,
        thickness: int = 2,
    ) -> None:
        x, y = pos
        for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            cv2.putText(img, text, (x + dx, y + dy), font, scale, COLOR_TEXT_BG, thickness)
        cv2.putText(img, text, pos, font, scale, color, thickness)

    # ------------------------------------------------------------------
    # 坐标转换: 球场坐标系 (米) → 迷你球场像素坐标
    # ------------------------------------------------------------------

    @staticmethod
    def _court_to_mini(
        point: Optional[Point],
        court_left: int,
        court_top: int,
        court_w: int,
        court_h: int,
    ) -> Optional[tuple]:
        if point is None:
            return None
        mx = court_left + (point.x / COURT_LENGTH) * court_w
        my = court_top + ((point.y + COURT_WIDTH / 2.0) / COURT_WIDTH) * court_h
        return (int(mx), int(my))

    # ------------------------------------------------------------------
    # 2 — draw_floating_panel
    # ------------------------------------------------------------------

    def draw_floating_panel(
        self,
        frame: np.ndarray,
        match_state,
        player_a: PlayerFrame,
        player_b: PlayerFrame,
    ) -> np.ndarray:
        """创建右上角半透明覆盖面板并返回"""
        h, w = frame.shape[:2]
        x1 = w - OVERLAY_WIDTH - 10
        y1 = 10
        x2 = w - 10
        y2 = OVERLAY_HEIGHT + 10

        panel = frame[y1:y2, x1:x2].copy()
        overlay_bg = np.full_like(panel, COLOR_OVERLAY_BG, dtype=np.uint8)
        panel = cv2.addWeighted(panel, 1.0 - OVERLAY_OPACITY, overlay_bg, OVERLAY_OPACITY, 0.0)
        return panel

    # ------------------------------------------------------------------
    # 3 — draw_score
    # ------------------------------------------------------------------

    def draw_score(
        self,
        panel: np.ndarray,
        match_state,
        y_offset: int = 20,
    ) -> int:
        """绘制比分文字 (如 [发]A 12 : 8 B)"""
        if match_state is None:
            return y_offset

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.8
        thickness = 2

        if match_state.server == "A":
            score_text = f"[发]A {match_state.score_a} : {match_state.score_b} B"
        else:
            score_text = f"A {match_state.score_a} : {match_state.score_b} [发]B"

        text_size = cv2.getTextSize(score_text, font, scale, thickness)[0]
        text_x = (panel.shape[1] - text_size[0]) // 2
        text_y = y_offset + text_size[1]

        self._draw_text_with_outline(panel, score_text, (text_x, text_y), font, scale, COLOR_TEXT, thickness)

        return y_offset + 50

    # ------------------------------------------------------------------
    # 4 — draw_mini_court
    # ------------------------------------------------------------------

    def draw_mini_court(
        self,
        panel: np.ndarray,
        player_a_pos: Point,
        player_b_pos: Point,
        match_state,
        y_offset: int,
    ) -> int:
        """绘制迷你俯视球场 + 发球区高亮 + 球员位置"""
        margin = 25
        c_left = margin
        c_top = y_offset
        c_w = panel.shape[1] - 2 * margin
        c_h = int(c_w * COURT_WIDTH / COURT_LENGTH)

        # ---------- 背景 ----------
        cv2.rectangle(panel, (c_left, c_top), (c_left + c_w, c_top + c_h), COLOR_COURT_BG, -1)

        # ---------- 发球区高亮 ----------
        half_w = c_w // 2
        half_h = c_h // 2

        if match_state is not None:
            side = match_state.server_should_be_in()  # "right" | "left"
            if side == "right":
                # Q2 (近端-右侧) + Q3 (远端-左侧)
                zone_pts = [
                    np.array([[c_left + half_w, c_top],
                              [c_left + c_w, c_top],
                              [c_left + c_w, c_top + half_h],
                              [c_left + half_w, c_top + half_h]], np.int32),
                    np.array([[c_left, c_top + half_h],
                              [c_left + half_w, c_top + half_h],
                              [c_left + half_w, c_top + c_h],
                              [c_left, c_top + c_h]], np.int32),
                ]
            else:
                # Q1 (近端-左侧) + Q4 (远端-右侧)
                zone_pts = [
                    np.array([[c_left, c_top],
                              [c_left + half_w, c_top],
                              [c_left + half_w, c_top + half_h],
                              [c_left, c_top + half_h]], np.int32),
                    np.array([[c_left + half_w, c_top + half_h],
                              [c_left + c_w, c_top + half_h],
                              [c_left + c_w, c_top + c_h],
                              [c_left + half_w, c_top + c_h]], np.int32),
                ]

            for z in zone_pts:
                overlay = panel.copy()
                cv2.fillPoly(overlay, [z], COLOR_ZONE_ACTIVE)
                cv2.addWeighted(overlay, 0.3, panel, 0.7, 0, panel)

        # ---------- 球场线 ----------
        # 外边框
        cv2.rectangle(panel, (c_left, c_top), (c_left + c_w, c_top + c_h), COLOR_COURT_LINE, 1)

        # 球网 (垂直中线 — 球场长度方向平分)
        net_x = c_left + c_w // 2
        cv2.line(panel, (net_x, c_top), (net_x, c_top + c_h), COLOR_COURT_LINE, 2)

        # 中心线 (水平中线 — 球场宽度方向平分, y=0)
        center_y = c_top + c_h // 2
        cv2.line(panel, (c_left, center_y), (c_left + c_w, center_y), COLOR_COURT_LINE, 1)

        # 前发球线 (短服务线, 距网 2m)
        ssl_near_x = c_left + int(((COURT_LENGTH / 2.0) - SHORT_SERVICE_LINE) / COURT_LENGTH * c_w)
        ssl_far_x = c_left + int(((COURT_LENGTH / 2.0) + SHORT_SERVICE_LINE) / COURT_LENGTH * c_w)
        cv2.line(panel, (ssl_near_x, c_top), (ssl_near_x, c_top + c_h), COLOR_COURT_LINE, 1)
        cv2.line(panel, (ssl_far_x, c_top), (ssl_far_x, c_top + c_h), COLOR_COURT_LINE, 1)

        # ---------- 球员位置 ----------
        for pos, color in [(player_a_pos, COLOR_PLAYER_A), (player_b_pos, COLOR_PLAYER_B)]:
            mini_pt = self._court_to_mini(pos, c_left, c_top, c_w, c_h)
            if mini_pt is not None:
                cv2.circle(panel, mini_pt, 5, color, -1)
                cv2.circle(panel, mini_pt, 5, COLOR_COURT_LINE, 1)

        return c_top + c_h + 10

    # ------------------------------------------------------------------
    # 5 — draw_player_trails
    # ------------------------------------------------------------------

    def draw_player_trails(self, frame: np.ndarray) -> None:
        """在画面上绘制球员移动轨迹 (最近 TRAIL_LENGTH 帧, 渐变线宽)"""
        for trail, color in [(self.trail_a, COLOR_TRAIL_A), (self.trail_b, COLOR_TRAIL_B)]:
            n = len(trail)
            if n < 2:
                continue
            pts = list(trail)
            for i in range(n - 1):
                if pts[i] is None or pts[i+1] is None:
                    continue
                thickness = max(1, int(1 + (i / max(n - 1, 1)) * 2))  # 1 -> 3 px
                pt1 = (int(pts[i][0]), int(pts[i][1]))
                pt2 = (int(pts[i+1][0]), int(pts[i+1][1]))
                cv2.line(frame, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)

    # ------------------------------------------------------------------
    # 6 — add_trail_point
    # ------------------------------------------------------------------

    def add_trail_point(self, player_id: str, pixel_pos: tuple) -> None:
        """向对应球员轨迹队列添加一个像素坐标点"""
        if pixel_pos is None:
            return
        pt = (int(pixel_pos[0]), int(pixel_pos[1]))
        if player_id == "A":
            self.trail_a.append(pt)
        elif player_id == "B":
            self.trail_b.append(pt)

    # ------------------------------------------------------------------
    # 7 — render_frame (每帧主入口)
    # ------------------------------------------------------------------

    def render_frame(
        self,
        frame: np.ndarray,
        player_a: PlayerFrame,
        player_b: PlayerFrame,
        match_state,
        pixel_a: tuple,
        pixel_b: tuple,
        timestamp: float,
    ) -> np.ndarray:
        """单帧渲染: 轨迹 → 面板(比分+迷你球场) → 时间戳"""
        # 1. 轨迹
        self.add_trail_point("A", pixel_a)
        self.add_trail_point("B", pixel_b)
        self.draw_player_trails(frame)

        # 2. 面板
        h, w = frame.shape[:2]
        px1 = w - OVERLAY_WIDTH - 10
        py1 = 10
        px2 = w - 10
        py2 = OVERLAY_HEIGHT + 10

        panel = self.draw_floating_panel(frame, match_state, player_a, player_b)

        # 3. 比分
        y_off = self.draw_score(panel, match_state)

        # 4. 迷你球场 (含发球区高亮 + 球员位置)
        a_pos = player_a.position if player_a is not None else None
        b_pos = player_b.position if player_b is not None else None
        y_off = self.draw_mini_court(panel, a_pos, b_pos, match_state, y_off)

        # 5. 面板合成回画面
        frame[py1:py2, px1:px2] = panel

        # 6. 底部时间戳
        mins = int(timestamp // 60)
        secs = int(timestamp % 60)
        ts_text = f"{mins:02d}:{secs:02d}"
        self._draw_text_with_outline(
            frame, ts_text, (10, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2,
        )

        return frame

    # ------------------------------------------------------------------
    # 8 — init_video_writer
    # ------------------------------------------------------------------

    def init_video_writer(self, frame_width: int, frame_height: int, fps: float) -> None:
        fourcc = cv2.VideoWriter_fourcc(*OUTPUT_VIDEO_CODEC)
        self.video_writer = cv2.VideoWriter(
            self.output_video_path, fourcc, fps, (frame_width, frame_height)
        )

    # ------------------------------------------------------------------
    # 9 — write_frame
    # ------------------------------------------------------------------

    def write_frame(self, frame: np.ndarray) -> None:
        if self.video_writer is not None:
            self.video_writer.write(frame)

    # ------------------------------------------------------------------
    # 10 — add_score_update
    # ------------------------------------------------------------------

    def add_score_update(self, update: ScoreUpdate) -> None:
        self.score_updates.append(update)

    # ------------------------------------------------------------------
    # 11 — export_score_curve
    # ------------------------------------------------------------------

    def export_score_curve(self, filepath: str = "score_curve.png") -> None:
        if not self.score_updates:
            return

        updates = sorted(self.score_updates, key=lambda u: u.timestamp)
        times = [u.timestamp for u in updates]
        scores_a = [u.score_a for u in updates]
        scores_b = [u.score_b for u in updates]
        servers = [u.server for u in updates]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(times, scores_a, color="red", linewidth=2, label="Player A", marker="o", markersize=3)
        ax.plot(times, scores_b, color="blue", linewidth=2, label="Player B", marker="o", markersize=3)

        # 发球权变化标记 (竖虚线)
        for i in range(1, len(servers)):
            if servers[i] != servers[i - 1]:
                ax.axvline(x=times[i], color="gray", linestyle="--", alpha=0.6, linewidth=1)

        ax.set_xlabel("Match Time (seconds)")
        ax.set_ylabel("Score")
        ax.set_title("Score Progression")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(filepath, dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    # 12 — export_scores_json
    # ------------------------------------------------------------------

    def export_scores_json(self, filepath: str = None) -> None:
        if filepath is None:
            filepath = self.output_scores_path

        if not self.score_updates:
            data = {
                "match_info": {"total_updates": 0},
                "score_updates": [],
                "final_score": "0:0",
            }
        else:
            last = self.score_updates[-1]
            data = {
                "match_info": {
                    "duration_sec": last.timestamp,
                    "total_updates": len(self.score_updates),
                },
                "score_updates": [
                    {
                        "timestamp": u.timestamp,
                        "score_a": u.score_a,
                        "score_b": u.score_b,
                        "server": u.server,
                        "confidence": u.confidence,
                        "description": u.description,
                    }
                    for u in self.score_updates
                ],
                "final_score": f"{last.score_a}:{last.score_b}",
            }

        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 13 — release
    # ------------------------------------------------------------------

    def release(self) -> None:
        """释放视频写入器 & 导出比分曲线 + JSON"""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        self.export_score_curve()
        self.export_scores_json()
