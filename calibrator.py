"""球场校准器 v2 — 基于颜色分割 + 轮廓检测的球场识别

改进:
  1. 先做白色区域提取 (羽毛球场线是白色的)
  2. 再在白色mask上做边缘检测
  3. 找最大四边形作为球场
  4. 支持手动校准回退
"""

import numpy as np
import cv2
from typing import Optional, List, Tuple

from models import Point, PixelPoint, CourtCorners, CourtZones
from config import (
    COURT_LENGTH, BIRDS_EYE_WIDTH, BIRDS_EYE_HEIGHT,
    ASPECT_RATIO_SINGLES, ASPECT_RATIO_DOUBLES, ASPECT_RATIO_THRESHOLD,
    CANNY_LOW, CANNY_HIGH,
    LINE_ANGLE_TOLERANCE, MIN_EDGES_REQUIRED,
)


class Calibrator:
    """球场校准器 v2"""

    def __init__(self):
        self.M = None
        self.M_inv = None
        self.zones = None
        self.corners = None
        self.court_type = "singles"
        self.court_width = 5.18
        self._click_points = []

    # ==================================================================
    #  V2: 基于白色mask的球场检测
    # ==================================================================

    def detect_court(self, frame: np.ndarray) -> Optional[CourtCorners]:
        """检测球场 — 先提取白色线条区域, 再找最大四边形"""
        if frame is None or frame.size == 0:
            return None

        try:
            h, w = frame.shape[:2]

            # ---- 1. 提取白色/亮色区域 (球场线) ----
            white_mask = self._extract_white_lines(frame)

            # ---- 2. 在白色mask上做边缘检测 ----
            edges = cv2.Canny(white_mask, CANNY_LOW, CANNY_HIGH)

            # ---- 3. HoughLinesP 检测线段 ----
            lines = cv2.HoughLinesP(
                edges, rho=1, theta=np.pi / 180.0, threshold=60,
                minLineLength=max(50, int(min(h, w) * 0.08)),
                maxLineGap=30,
            )

            if lines is None or len(lines) < 4:
                # 回退: 在原始灰度图上重试
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
                edges2 = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
                lines = cv2.HoughLinesP(
                    edges2, rho=1, theta=np.pi / 180.0, threshold=80,
                    minLineLength=max(80, int(min(h, w) * 0.1)),
                    maxLineGap=50,
                )

            if lines is None or len(lines) < 4:
                return None

            # ---- 4. 分类线段 + 聚类 ----
            h_lines, v_lines = self._classify_lines(lines)

            if len(h_lines) < 2 or len(v_lines) < 2:
                return None

            # ---- 5. 找最外围的4条边界线 ----
            # 水平线: 取y最小(max 2条) 和 y最大(max 2条)
            # 竖直线: 取x最小(max 2条) 和 x最大(max 2条)
            h_sorted = sorted(h_lines, key=lambda s: s[4])  # sort by mid_y
            v_sorted = sorted(v_lines, key=lambda s: s[4])  # sort by mid_x

            top_lines = h_sorted[:3]      # y最小的几条
            bottom_lines = h_sorted[-3:]   # y最大的几条
            left_lines = v_sorted[:3]      # x最小的几条
            right_lines = v_sorted[-3:]    # x最大的几条

            # ---- 6. 尝试所有组合, 找最佳四边形 ----
            best_corners = None
            best_score = 0

            for top_i in range(min(2, len(top_lines))):
                for bot_i in range(min(2, len(bottom_lines))):
                    for left_i in range(min(2, len(left_lines))):
                        for right_i in range(min(2, len(right_lines))):
                            corners = self._lines_to_corners(
                                top_lines[top_i], bottom_lines[bot_i],
                                left_lines[left_i], right_lines[right_i],
                                w, h
                            )
                            if corners is None:
                                continue
                            score = self._score_corners(corners, w, h)
                            if score > best_score:
                                best_score = score
                                best_corners = corners

            if best_corners is None:
                return None

            # ---- 7. 判定场地类型 ----
            self._detect_court_type(best_corners)
            self.corners = best_corners
            return best_corners

        except Exception:
            return None

    def _extract_white_lines(self, frame: np.ndarray) -> np.ndarray:
        """提取白色/亮色线条区域

        方法: HSV颜色空间 + 亮度阈值 + 形态学操作
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 白色: 低饱和度 + 高亮度
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 40, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # 形态学: 闭运算连接断线, 开运算去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 膨胀使线条更明显
        mask = cv2.dilate(mask, kernel, iterations=1)

        return mask

    def _classify_lines(self, lines) -> Tuple[list, list]:
        """分类线段: 水平(0-30度) vs 倾斜(30-90度)"""
        h_lines = []
        v_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx * dx + dy * dy)
            if length < 30:
                continue

            angle = abs(np.arctan2(dy, dx) * 180.0 / np.pi)

            if angle < LINE_ANGLE_TOLERANCE or angle > (180 - LINE_ANGLE_TOLERANCE):
                mid_y = (y1 + y2) / 2.0
                h_lines.append((x1, y1, x2, y2, mid_y, length))
            else:
                mid_x = (x1 + x2) / 2.0
                v_lines.append((x1, y1, x2, y2, mid_x, length))

        return h_lines, v_lines

    def _lines_to_corners(self, top, bottom, left, right, img_w, img_h
                          ) -> Optional[CourtCorners]:
        """4条线求4个交点"""
        def line_from_seg(seg):
            x1, y1, x2, y2 = seg[0], seg[1], seg[2], seg[3]
            if abs(x2 - x1) < 1e-6:
                return (1.0, 0.0, -x1)  # 垂直线
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            return (m, -1.0, b)  # y = mx + b → mx - y + b = 0

        def intersect(l1, l2):
            a1, b1, c1 = l1
            a2, b2, c2 = l2
            det = a1 * b2 - a2 * b1
            if abs(det) < 1e-8:
                return None
            x = (b1 * c2 - b2 * c1) / det
            y = (a2 * c1 - a1 * c2) / det
            return (x, y)

        l_top = line_from_seg(top)
        l_bot = line_from_seg(bottom)
        l_left = line_from_seg(left)
        l_right = line_from_seg(right)

        tl = intersect(l_top, l_left)
        tr = intersect(l_top, l_right)
        br = intersect(l_bot, l_right)
        bl = intersect(l_bot, l_left)

        if any(p is None for p in [tl, tr, br, bl]):
            return None

        # 验证在画面内 (允许边距)
        margin = 300
        for px, py in [tl, tr, br, bl]:
            if px < -margin or px > img_w + margin or py < -margin or py > img_h + margin:
                return None

        return CourtCorners(
            tl=PixelPoint(int(tl[0]), int(tl[1])),
            tr=PixelPoint(int(tr[0]), int(tr[1])),
            br=PixelPoint(int(br[0]), int(br[1])),
            bl=PixelPoint(int(bl[0]), int(bl[1])),
        )

    def _score_corners(self, corners: CourtCorners, img_w: int, img_h: int) -> float:
        """评分四边形: 面积 + 矩形度 + 凸性"""
        pts = np.array([
            [corners.tl.x, corners.tl.y],
            [corners.tr.x, corners.tr.y],
            [corners.br.x, corners.br.y],
            [corners.bl.x, corners.bl.y],
        ], dtype=np.float64)

        # 面积 (Shoelace)
        area = 0.5 * abs(
            pts[0,0]*pts[1,1] + pts[1,0]*pts[2,1] +
            pts[2,0]*pts[3,1] + pts[3,0]*pts[0,1] -
            pts[1,0]*pts[0,1] - pts[2,0]*pts[1,1] -
            pts[3,0]*pts[2,1] - pts[0,0]*pts[3,1]
        )
        img_area = img_w * img_h
        area_ratio = area / img_area

        if area_ratio < 0.03 or area_ratio > 0.7:
            return 0

        # 面积分
        area_score = 1.0 if 0.05 <= area_ratio <= 0.5 else 0.5

        # 矩形度
        top_len = np.linalg.norm(pts[0] - pts[1])
        bot_len = np.linalg.norm(pts[3] - pts[2])
        left_len = np.linalg.norm(pts[0] - pts[3])
        right_len = np.linalg.norm(pts[1] - pts[2])
        h_ratio = min(top_len, bot_len) / max(top_len, bot_len, 1)
        v_ratio = min(left_len, right_len) / max(left_len, right_len, 1)
        rect_score = (h_ratio + v_ratio) / 2

        # 对角线
        d1 = np.linalg.norm(pts[0] - pts[2])
        d2 = np.linalg.norm(pts[1] - pts[3])
        diag_score = min(d1, d2) / max(d1, d2, 1)

        return 0.4 * area_score + 0.35 * rect_score + 0.25 * diag_score

    # ==================================================================
    #  场地类型判定
    # ==================================================================

    def _detect_court_type(self, corners: CourtCorners) -> str:
        pts = np.array([
            [corners.tl.x, corners.tl.y],
            [corners.tr.x, corners.tr.y],
            [corners.br.x, corners.br.y],
            [corners.bl.x, corners.bl.y],
        ], dtype=np.float64)

        top_len = np.linalg.norm(pts[0] - pts[1])
        bottom_len = np.linalg.norm(pts[3] - pts[2])
        left_len = np.linalg.norm(pts[0] - pts[3])
        right_len = np.linalg.norm(pts[1] - pts[2])

        avg_width = (top_len + bottom_len) / 2.0
        avg_length = (left_len + right_len) / 2.0

        if avg_length > 0:
            aspect = avg_width / avg_length
        else:
            aspect = 0.0

        if aspect > ASPECT_RATIO_THRESHOLD:
            self.court_type = "doubles"
            self.court_width = 6.10
        else:
            self.court_type = "singles"
            self.court_width = 5.18

        return self.court_type

    # ==================================================================
    #  透视变换
    # ==================================================================

    def compute_transform(self, corners: CourtCorners) -> np.ndarray:
        try:
            src = corners.as_array()
            dst = np.array([
                [0, 0],
                [BIRDS_EYE_WIDTH, 0],
                [BIRDS_EYE_WIDTH, BIRDS_EYE_HEIGHT],
                [0, BIRDS_EYE_HEIGHT],
            ], dtype=np.float32)
            self.M = cv2.getPerspectiveTransform(src, dst)
            self.M_inv = cv2.getPerspectiveTransform(dst, src)
            return self.M
        except Exception:
            self.M = None
            self.M_inv = None
            return None

    def define_zones(self) -> CourtZones:
        self.zones = CourtZones(court_type=self.court_type)
        return self.zones

    def calibrate(self, frame: np.ndarray) -> bool:
        try:
            corners = self.detect_court(frame)
            if corners is None:
                return False
            if self.compute_transform(corners) is None:
                return False
            self.define_zones()
            return True
        except Exception:
            return False

    # ==================================================================
    #  手动校准
    # ==================================================================

    def manual_calibrate(self, frame: np.ndarray) -> bool:
        if frame is None or frame.size == 0:
            return False
        self._click_points = []

        def _mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self._click_points.append((x, y))
                display_img = param[0]
                cv2.circle(display_img, (x, y), 6, (0, 255, 0), -1)
                idx = len(self._click_points)
                cv2.putText(display_img, str(idx), (x + 12, y - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow(window_name, display_img)

        try:
            window_name = "Click: TL(远左) -> TR(远右) -> BR(近右) -> BL(近左)"
            display = frame.copy()
            h, w = display.shape[:2]

            instructions = [
                "Click 4 corners in order:",
                "  1 - TL (far left,   远端左角)",
                "  2 - TR (far right,  远端右角)",
                "  3 - BR (near right, 近端右角)",
                "  4 - BL (near left,  近端左角)",
                "Press 'r' to reset | 'q' to quit",
            ]
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (w, 180), (0, 0, 0), -1)
            display = cv2.addWeighted(overlay, 0.6, display, 0.4, 0)
            for i, text in enumerate(instructions):
                cv2.putText(display, text, (20, 30 + i * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            cv2.imshow(window_name, display)
            cv2.setMouseCallback(window_name, _mouse_callback, [display])

            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self._click_points = []
                    cv2.destroyWindow(window_name)
                    return False
                if key == ord('r'):
                    self._click_points = []
                    display = frame.copy()
                    overlay_d = display.copy()
                    cv2.rectangle(overlay_d, (0, 0), (w, 180), (0, 0, 0), -1)
                    display = cv2.addWeighted(overlay_d, 0.6, display, 0.4, 0)
                    for i, text in enumerate(instructions):
                        cv2.putText(display, text, (20, 30 + i * 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                    cv2.imshow(window_name, display)
                    cv2.setMouseCallback(window_name, _mouse_callback, [display])
                if len(self._click_points) >= 4:
                    cv2.waitKey(300)
                    break

            cv2.destroyWindow(window_name)
            pts = self._click_points[:4]
            self._click_points = []

            corners = CourtCorners(
                tl=PixelPoint(pts[0][0], pts[0][1]),
                tr=PixelPoint(pts[1][0], pts[1][1]),
                br=PixelPoint(pts[2][0], pts[2][1]),
                bl=PixelPoint(pts[3][0], pts[3][1]),
            )

            if self.compute_transform(corners) is None:
                return False
            self._detect_court_type(corners)
            self.define_zones()
            self.corners = corners
            return True

        except Exception:
            self._click_points = []
            cv2.destroyAllWindows()
            return False

    # ==================================================================
    #  坐标转换
    # ==================================================================

    def pixel_to_court(self, px: tuple) -> Point:
        if self.M is None:
            raise RuntimeError("Not calibrated. Call calibrate() first.")

        try:
            pts = np.array([[[float(px[0]), float(px[1])]]], dtype=np.float32)
            birds_eye = cv2.perspectiveTransform(pts, self.M)
            bx, by = birds_eye[0][0]
            court_x = bx / BIRDS_EYE_WIDTH * COURT_LENGTH
            court_y = (by / BIRDS_EYE_HEIGHT - 0.5) * self.court_width
            return Point(x=float(court_x), y=float(court_y))
        except Exception:
            return Point(x=0.0, y=0.0)

    def draw_debug(self, frame: np.ndarray) -> np.ndarray:
        if frame is None:
            return frame
        vis = frame.copy()
        if self.corners is not None:
            pts = np.array([
                [self.corners.tl.x, self.corners.tl.y],
                [self.corners.tr.x, self.corners.tr.y],
                [self.corners.br.x, self.corners.br.y],
                [self.corners.bl.x, self.corners.bl.y],
            ], dtype=np.int32)
            cv2.polylines(vis, [pts], True, (0, 255, 0), 3)
            for i, (px, py) in enumerate(pts):
                cv2.circle(vis, (px, py), 8, (0, 0, 255), -1)
                cv2.putText(vis, ["TL", "TR", "BR", "BL"][i],
                            (px + 15, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(vis, f"Court: {self.court_type} ({self.court_width}m)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        return vis
