"""
court_viewer.py — 俯视图球场位置回放工具

从 tracks JSONL 读取球员位置, 在标准羽毛球场上实时动画显示。
用于肉眼验证坐标映射是否正确。

用法:
  python court_viewer.py tracks_60s.jsonl
  python court_viewer.py tracks_60s.jsonl --speed 2     # 2x倍速
  python court_viewer.py tracks_60s.jsonl --show-trails  # 显示轨迹
"""

import json
import sys
import argparse
import time
from collections import deque

import numpy as np
import cv2

# 球场尺寸常量 (像素)
COURT_LENGTH_PX = 900    # 13.4m
COURT_WIDTH_PX = 410     # 6.1m (doubles max)
MARGIN = 60
CANVAS_W = COURT_LENGTH_PX + 2 * MARGIN
CANVAS_H = COURT_WIDTH_PX + 2 * MARGIN

# 球场在原图中的位置
COURT_LEFT = MARGIN
COURT_TOP = MARGIN
NET_X = 200  # 网距底线13.4m → 200px from left (near side) after scale

# 比例尺
SCALE_X = COURT_LENGTH_PX / 13.4   # px/m
SCALE_Y = COURT_WIDTH_PX / 6.1     # px/m

SINGLES_WIDTH_PX = int(5.18 * SCALE_Y)
DOUBLES_WIDTH_PX = int(6.10 * SCALE_Y)

COLORS = {
    "bg": (34, 139, 34),           # 绿色背景
    "line": (255, 255, 255),       # 白线
    "net": (200, 200, 200),        # 灰色网
    "player_a": (0, 0, 255),      # A=红色
    "player_b": (255, 0, 0),      # B=蓝色
    "trail_a": (0, 165, 255),     # A轨迹=橙色
    "trail_b": (255, 255, 0),     # B轨迹=青色
    "text": (255, 255, 255),      # 白色文字
    "zone_active": (0, 255, 255), # 活动区域=黄色
    "service_line": (200, 200, 200),
}


def court_to_canvas(x_m, y_m):
    """球场坐标(米) → 画布像素"""
    px = COURT_LEFT + int(x_m * SCALE_X)
    py = CANVAS_H // 2 - int(y_m * SCALE_Y)  # y=0 在画布中央
    return px, py


def draw_court(canvas, court_type="doubles"):
    """绘制羽毛球场"""
    h, w = canvas.shape[:2]
    court_w = DOUBLES_WIDTH_PX if court_type == "doubles" else SINGLES_WIDTH_PX
    half_w = court_w // 2
    cy = CANVAS_H // 2

    # 绿色球场背景
    cv2.rectangle(canvas,
                  (COURT_LEFT, cy - half_w),
                  (COURT_LEFT + COURT_LENGTH_PX, cy + half_w),
                  COLORS["bg"], -1)

    # 外边界 (白色)
    cv2.rectangle(canvas,
                  (COURT_LEFT, cy - half_w),
                  (COURT_LEFT + COURT_LENGTH_PX, cy + half_w),
                  COLORS["line"], 2)

    # 中线
    cv2.line(canvas,
             (COURT_LEFT, cy),
             (COURT_LEFT + COURT_LENGTH_PX, cy),
             COLORS["line"], 1)

    # 网 (x=0处, 即近侧底线往远端 COURT_LENGTH_PX)
    net_x = COURT_LEFT
    cv2.line(canvas, (net_x, cy - half_w), (net_x, cy + half_w), COLORS["net"], 3)

    # 前发球线 (距网2m)
    short_svc_x = COURT_LEFT + int(2.0 * SCALE_X)
    cv2.line(canvas, (short_svc_x, cy - half_w), (short_svc_x, cy + half_w),
             COLORS["service_line"], 1, lineType=cv2.LINE_AA)

    # 后发球线 (双打11.4m, 单打13.4m=底线)
    if court_type == "doubles":
        long_svc_x = COURT_LEFT + int(11.4 * SCALE_X)
        cv2.line(canvas, (long_svc_x, cy - half_w), (long_svc_x, cy + half_w),
                 COLORS["service_line"], 1, lineType=cv2.LINE_AA)

    # 远侧底线
    far_x = COURT_LEFT + COURT_LENGTH_PX
    cv2.line(canvas, (far_x, cy - half_w), (far_x, cy + half_w), COLORS["line"], 2)

    # 近侧底线
    cv2.line(canvas, (COURT_LEFT, cy - half_w), (COURT_LEFT, cy + half_w), COLORS["line"], 2)

    # 单打边线 (如果双打场地)
    if court_type == "doubles":
        singles_hw = SINGLES_WIDTH_PX // 2
        cv2.line(canvas, (COURT_LEFT, cy - singles_hw),
                 (COURT_LEFT + COURT_LENGTH_PX, cy - singles_hw),
                 COLORS["service_line"], 1, lineType=cv2.LINE_AA)
        cv2.line(canvas, (COURT_LEFT, cy + singles_hw),
                 (COURT_LEFT + COURT_LENGTH_PX, cy + singles_hw),
                 COLORS["service_line"], 1, lineType=cv2.LINE_AA)

    # 发球区标注 (半透明)
    overlay = canvas.copy()
    # 近侧右发球区
    near_right_x1 = COURT_LEFT
    near_right_x2 = short_svc_x
    alpha = 0.15
    cv2.rectangle(overlay, (near_right_x1, cy), (near_right_x2, cy + half_w),
                  COLORS["zone_active"], -1)
    cv2.rectangle(overlay, (near_right_x1, cy - half_w), (near_right_x2, cy),
                  (200, 200, 0), -1)  # 不同颜色区分左右

    # 远侧发球区
    far_svc_x = COURT_LEFT + int(11.4 * SCALE_X) if court_type == "doubles" else COURT_LEFT + COURT_LENGTH_PX
    far_x = COURT_LEFT + COURT_LENGTH_PX
    cv2.rectangle(overlay, (far_svc_x, cy), (far_x, cy + half_w),
                  COLORS["zone_active"], -1)
    cv2.rectangle(overlay, (far_svc_x, cy - half_w), (far_svc_x, cy),
                  (200, 200, 0), -1)

    canvas[:] = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)

    # 网标注
    cv2.putText(canvas, "NET", (COURT_LEFT + 5, cy - half_w - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["text"], 1)

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Badminton Court Bird's-Eye Viewer")
    parser.add_argument("tracks_file", help="Path to tracks JSONL file")
    parser.add_argument("--speed", type=float, default=2.0, help="Playback speed multiplier")
    parser.add_argument("--show-trails", action="store_true", help="Show movement trails")
    parser.add_argument("--trail-length", type=int, default=30, help="Trail length in frames")
    args = parser.parse_args()

    # 读取数据
    with open(args.tracks_file, "r", encoding="utf-8") as f:
        header = json.loads(f.readline())
        records = [json.loads(line) for line in f]

    fps = header.get("fps", 60)
    court_type = header.get("court_type", "singles")
    print(f"Loaded {len(records)} frames, {court_type} court, {fps} fps")
    print(f"Speed: {args.speed}x, Trails: {args.show_trails}")
    print(f"Controls: SPACE=pause, Q=quit, LEFT/RIGHT=seek")

    # 轨迹缓存
    trail_a = deque(maxlen=args.trail_length)
    trail_b = deque(maxlen=args.trail_length)

    cv2.namedWindow("Court View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Court View", CANVAS_W, CANVAS_H)

    paused = False
    idx = 0
    delay_ms = int(1000 / fps / args.speed)

    while 0 <= idx < len(records):
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        canvas[:] = (50, 50, 50)  # 深灰背景

        draw_court(canvas, court_type)

        rec = records[idx]

        # 转换坐标到画布
        pa_x, pa_y = court_to_canvas(rec["a_x"], rec["a_y"])
        pb_x, pb_y = court_to_canvas(rec["b_x"], rec["b_y"])

        # 轨迹
        if args.show_trails:
            trail_a.append((pa_x, pa_y))
            trail_b.append((pb_x, pb_y))

            for i in range(len(trail_a) - 1):
                if trail_a[i] and trail_a[i+1]:
                    thick = max(1, int(1 + i / len(trail_a) * 3))
                    cv2.line(canvas, trail_a[i], trail_a[i+1], COLORS["trail_a"], thick, cv2.LINE_AA)

            for i in range(len(trail_b) - 1):
                if trail_b[i] and trail_b[i+1]:
                    thick = max(1, int(1 + i / len(trail_b) * 3))
                    cv2.line(canvas, trail_b[i], trail_b[i+1], COLORS["trail_b"], thick, cv2.LINE_AA)

        # 球员位置 (大圆点)
        cv2.circle(canvas, (pa_x, pa_y), 8, COLORS["player_a"], -1)
        cv2.circle(canvas, (pa_x, pa_y), 10, (255, 255, 255), 2)  # 白边
        cv2.circle(canvas, (pb_x, pb_y), 8, COLORS["player_b"], -1)
        cv2.circle(canvas, (pb_x, pb_y), 10, (255, 255, 255), 2)

        # 标签
        cv2.putText(canvas, "A", (pa_x + 15, pa_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["player_a"], 2)
        cv2.putText(canvas, "B", (pb_x + 15, pb_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["player_b"], 2)

        # 信息栏
        ts = rec["timestamp"]
        cv2.putText(canvas, f"Time: {ts:.1f}s  Frame: {rec['frame']}  Speed: {args.speed}x",
                    (10, CANVAS_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text"], 1)
        cv2.putText(canvas, f"A({rec['a_x']:.1f},{rec['a_y']:.1f})  B({rec['b_x']:.1f},{rec['b_y']:.1f})",
                    (10, CANVAS_H - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text"], 1)

        if paused:
            cv2.putText(canvas, "PAUSED", (CANVAS_W // 2 - 40, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Court View", canvas)

        key = cv2.waitKey(delay_ms if not paused else 10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Space = pause
            paused = not paused
        elif key == 81:  # Left arrow
            idx = max(0, idx - int(fps))
            trail_a.clear()
            trail_b.clear()
        elif key == 83:  # Right arrow
            idx = min(len(records) - 1, idx + int(fps))
            trail_a.clear()
            trail_b.clear()

        if not paused:
            idx += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
