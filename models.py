"""数据模型定义 — 羽毛球比赛分析系统的所有数据结构"""

from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class Point:
    """二维坐标点 (球场坐标系, 单位: 米)"""
    x: float
    y: float

    def distance_to(self, other: "Point") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def __add__(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point") -> "Point":
        return Point(self.x - other.x, self.y - other.y)

    def __truediv__(self, scalar: float) -> "Point":
        return Point(self.x / scalar, self.y / scalar)

    def __mul__(self, scalar: float) -> "Point":
        return Point(self.x * scalar, self.y * scalar)

    def as_tuple(self) -> tuple:
        return (self.x, self.y)


@dataclass
class PixelPoint:
    """像素坐标点"""
    x: int
    y: int

    def as_tuple(self) -> tuple:
        return (self.x, self.y)


@dataclass
class CourtCorners:
    """球场四个角点 (像素坐标)"""
    tl: PixelPoint  # 左上 (top-left)
    tr: PixelPoint  # 右上 (top-right)
    br: PixelPoint  # 右下 (bottom-right)
    bl: PixelPoint  # 左下 (bottom-left)

    def as_array(self):
        """返回 4x2 numpy 数组, 用于透视变换"""
        import numpy as np
        return np.array([
            [self.tl.x, self.tl.y],
            [self.tr.x, self.tr.y],
            [self.br.x, self.br.y],
            [self.bl.x, self.bl.y],
        ], dtype=np.float32)


@dataclass
class CourtZones:
    """发球区定义 (球场坐标系, 单位: 米)

    单打场地: 13.4m × 5.18m, 半场宽 2.59m
    双打场地: 13.4m × 6.10m, 半场宽 3.05m
    发球区: 前发球线距网2m
    """

    court_type: str = "singles"  # "singles" 或 "doubles"

    def __post_init__(self):
        if self.court_type == "doubles":
            half_width = 6.10 / 2  # 3.05m
            # 双打发球区: 宽 = 3.96 - 1.98 = 1.98m (双打发球区比单打窄!)
            # 双打后发球线 = 距网11.4m
            service_right_boundary = 1.98   # 双打发球区外边线
            service_left_boundary = 1.98    # 双打发球区外边线 (对称)
            service_back = 11.4             # 双打后发球线
        else:
            half_width = 5.18 / 2  # 2.59m
            service_right_boundary = half_width   # 单打发球区 = 半场宽
            service_left_boundary = half_width
            service_back = 13.4              # 单打后发球线 = 底线

        # 右发球区(球员视角右侧), 近网侧
        self.right_near = {
            "x_range": (0.0, 2.0),
            "y_range": (0.0, service_right_boundary),
            "name": "right_near"
        }
        # 左发球区, 近网侧
        self.left_near = {
            "x_range": (0.0, 2.0),
            "y_range": (-service_left_boundary, 0.0),
            "name": "left_near"
        }
        # 右发球区, 远网侧
        self.right_far = {
            "x_range": (service_back, 13.4),
            "y_range": (0.0, service_right_boundary),
            "name": "right_far"
        }
        # 左发球区, 远网侧
        self.left_far = {
            "x_range": (service_back, 13.4),
            "y_range": (-service_left_boundary, 0.0),
            "name": "left_far"
        }
        self._half_width = half_width
        self._service_back = service_back

    def get_all_zones(self) -> list:
        return [self.right_near, self.left_near, self.right_far, self.left_far]

    def get_zone_for_point(self, point: Point, tolerance: float = 0.3) -> Optional[str]:
        """判断点位于哪个发球区"""
        for zone in self.get_all_zones():
            x_min, x_max = zone["x_range"]
            y_min, y_max = zone["y_range"]
            if (x_min - tolerance <= point.x <= x_max + tolerance and
                y_min - tolerance <= point.y <= y_max + tolerance):
                return zone["name"]
        return None


@dataclass
class PlayerFrame:
    """单帧球员追踪结果"""
    player_id: str           # "A" 或 "B"
    position: Point          # 球场坐标系位置
    confidence: float        # 检测置信度
    timestamp: float         # 视频时间戳 (秒)
    velocity: float = 0.0    # 当前速度 (m/s)
    zone: Optional[str] = None  # 所在发球区


@dataclass
class ReadyEvent:
    """就绪事件 — 两名球员都在正确发球区并静止"""
    timestamp: float
    player_a_zone: str
    player_b_zone: str
    frame_number: int


@dataclass
class ScoreUpdate:
    """比分更新记录"""
    timestamp: float
    score_a: int
    score_b: int
    server: str             # 当前发球方 "A" 或 "B"
    confidence: float       # 这次判决的置信度
    description: str = ""   # 得分描述


@dataclass
class MatchState:
    """比赛状态"""
    score_a: int = 0
    score_b: int = 0
    server: str = "A"          # 当前发球方
    rally_count: int = 0       # 本局回合数
    last_update: float = 0.0   # 上次更新时间戳
    game_over: bool = False
    winner: Optional[str] = None

    @property
    def total_score(self) -> int:
        return self.score_a + self.score_b

    @property
    def server_score(self) -> int:
        return self.score_a if self.server == "A" else self.score_b

    @property
    def receiver_score(self) -> int:
        return self.score_b if self.server == "A" else self.score_a

    def server_should_be_in(self) -> str:
        """发球方当前应在哪个发球区 (偶=右, 奇=左)"""
        if self.server_score % 2 == 0:
            return "right"  # 右发球区
        return "left"       # 左发球区

    def receiver_should_be_in(self) -> str:
        """接发方应在对角发球区"""
        return "left" if self.server_should_be_in() == "right" else "right"
