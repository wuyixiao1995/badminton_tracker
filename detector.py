"""就绪检测模块 — 基于滑动窗口的球员就绪状态检测

使用双条件（位置 + 静止）判断两名球员是否已准备好发球：
1. 位置条件: 两名球员均位于正确的发球区
2. 静止条件: 两名球员的速度均低于阈值（平滑窗口平均）
3. 持续条件: 以上两个条件连续满足 READY_DURATION 秒
"""

from collections import deque
from typing import Optional, Tuple

from models import CourtZones, PlayerFrame, ReadyEvent, MatchState
from config import VELOCITY_THRESHOLD, READY_DURATION, COOLDOWN_DURATION, ZONE_TOLERANCE


class ReadyDetector:
    """就绪状态检测器

    使用滑动窗口机制检测"就绪"状态：两名球员都在正确的发球位置
    并保持静止足够长的时间。输出 ReadyEvent 供系统后续使用。
    """

    # 用于速度平滑的窗口大小（帧数）
    _VELOCITY_WINDOW = 5

    def __init__(self, court_zones: CourtZones) -> None:
        """初始化就绪检测器

        Args:
            court_zones: 球场发球区定义，用于位置判定
        """
        self.court_zones = court_zones

        # 就绪条件首次满足的时间戳（滑动窗口起始），None 表示未进入就绪状态
        self.ready_start_time: Optional[float] = None

        # 上次触发 ReadyEvent 的时间戳，用于冷却判断
        self.last_event_time: float = 0.0

        # 是否处于冷却期（防止重复触发）
        self.is_in_cooldown: bool = False

        # 速度历史记录（双端队列，固定窗口大小）
        # 结构: {"A": deque[float], "B": deque[float]}
        self.velocity_history: dict = {
            "A": deque(maxlen=self._VELOCITY_WINDOW),
            "B": deque(maxlen=self._VELOCITY_WINDOW),
        }

        # 内部帧计数器，用于 ReadyEvent 的 frame_number 字段
        self._frame_number: int = 0

    # ------------------------------------------------------------------
    # 位置检测
    # ------------------------------------------------------------------

    def check_positions(
        self,
        player_a: PlayerFrame,
        player_b: PlayerFrame,
        expected_a_zone: str,
        expected_b_zone: str,
    ) -> bool:
        """检查两名球员是否位于各自的预期发球区

        使用 court_zones.get_zone_for_point() 判断球员实际所在区域，
        并更新 player.zone 字段。ZONE_TOLERANCE 提供模糊边界匹配。

        Args:
            player_a: 球员 A 的当前帧数据
            player_b: 球员 B 的当前帧数据
            expected_a_zone: 球员 A 的预期发球区名称
            expected_b_zone: 球员 B 的预期发球区名称

        Returns:
            两名球员都在预期发球区内返回 True，否则返回 False

        Edge cases:
            - 球员 position 无效（zone 检测返回 None）时视为不满足条件
            - 球员 A/B 角色在 match_state 中通过 server 字段区分，
              本方法仅做位置匹配，不关心谁是谁
        """
        # 检测实际所在区域（带容差）
        zone_a = self.court_zones.get_zone_for_point(player_a.position, ZONE_TOLERANCE)
        zone_b = self.court_zones.get_zone_for_point(player_b.position, ZONE_TOLERANCE)

        # 更新球员帧的 zone 字段
        player_a.zone = zone_a
        player_b.zone = zone_b

        # 任何一个球员不在任何已知发球区 -> 不满足条件
        if zone_a is None or zone_b is None:
            return False

        # 检查是否都在预期区域
        return zone_a == expected_a_zone and zone_b == expected_b_zone

    # ------------------------------------------------------------------
    # 静止检测
    # ------------------------------------------------------------------

    def check_motion(self, player_a: PlayerFrame, player_b: PlayerFrame) -> bool:
        """检查两名球员是否处于静止状态

        使用滑动窗口平均速度判断静止性。将当前帧速度加入历史队列，
        计算最近 N 帧的平均速度，若两名球员的平均速度均低于阈值
        则判断为静止。

        Args:
            player_a: 球员 A 的当前帧数据
            player_b: 球员 B 的当前帧数据

        Returns:
            两名球员均静止返回 True，否则返回 False

        Notes:
            - 窗口大小由 _VELOCITY_WINDOW 控制（默认 5 帧）
            - 历史帧不足时使用已有帧计算平均，不会产生错误
            - 速度嘈杂场景下缓存窗口能有效抑制误报
        """
        # 将当前帧速度加入历史
        self.velocity_history["A"].append(player_a.velocity)
        self.velocity_history["B"].append(player_b.velocity)

        # 计算窗口平均速度
        avg_velocity_a = sum(self.velocity_history["A"]) / len(self.velocity_history["A"])
        avg_velocity_b = sum(self.velocity_history["B"]) / len(self.velocity_history["B"])

        # 两者均低于阈值才算静止
        return avg_velocity_a < VELOCITY_THRESHOLD and avg_velocity_b < VELOCITY_THRESHOLD

    # ------------------------------------------------------------------
    # 核心就绪检测
    # ------------------------------------------------------------------

    def detect_ready(
        self,
        player_a: PlayerFrame,
        player_b: PlayerFrame,
        expected_a_zone: str,
        expected_b_zone: str,
        timestamp: float,
    ) -> Optional[ReadyEvent]:
        """核心就绪检测逻辑（滑动窗口）

        完整流程：
        1. 冷却检查 —— 冷却期内不触发新事件
        2. 位置检查 —— 调用 check_positions()
        3. 静止检查 —— 调用 check_motion()
        4. 持续计时 —— 位置+静止条件持续满足 READY_DURATION 秒后
           触发 ReadyEvent；任一条件中断则重置计时器

        Args:
            player_a: 球员 A 的当前帧数据
            player_b: 球员 B 的当前帧数据
            expected_a_zone: 球员 A 的预期发球区
            expected_b_zone: 球员 B 的预期发球区
            timestamp: 当前帧的时间戳（秒）

        Returns:
            触发 ReadyEvent 时返回事件对象；否则返回 None

        Notes:
            - 冷却逻辑防止 READY_DURATION 窗口内重复触发
            - ready_start_time 在条件中断时复位，实现"窗口重置"
        """
        # 1. 冷却检查
        if timestamp - self.last_event_time < COOLDOWN_DURATION:
            self.is_in_cooldown = True
            return None
        self.is_in_cooldown = False

        # 2. 位置检查
        positions_ok = self.check_positions(
            player_a, player_b, expected_a_zone, expected_b_zone
        )

        # 3. 静止检查
        motion_ok = self.check_motion(player_a, player_b)

        # 4. 滑动窗口逻辑
        if positions_ok and motion_ok:
            # 条件首次满足 -> 启动计时器
            if self.ready_start_time is None:
                self.ready_start_time = timestamp
            # 持续时间达到阈值 -> 触发就绪事件
            elif timestamp - self.ready_start_time >= READY_DURATION:
                self._frame_number += 1
                event = ReadyEvent(
                    timestamp=self.ready_start_time,
                    player_a_zone=player_a.zone or expected_a_zone,
                    player_b_zone=player_b.zone or expected_b_zone,
                    frame_number=self._frame_number,
                )
                self.last_event_time = timestamp
                self.ready_start_time = None
                return event
        else:
            # 任一条件不满足 -> 重置计时器
            self.ready_start_time = None

        return None

    # ------------------------------------------------------------------
    # 预期发球区计算
    # ------------------------------------------------------------------

    def get_expected_zones(self, match_state: MatchState) -> Tuple[str, str]:
        """根据比赛状态计算两名球员的预期发球区

        发球规则：
        - 发球方在自己的比分对应的一侧发球（偶数=右，奇数=左）
        - 接发方在对角侧接发
        - 球员 A 始终在近网侧，球员 B 在远网侧

        区域命名（近网侧 = 球员 A 所在端，远网侧 = 球员 B 所在端）：
        - right_near / left_near  — 近网侧右/左发球区
        - right_far  / left_far   — 远网侧右/左发球区

        Args:
            match_state: 当前比赛状态（含比分、发球方等信息）

        Returns:
            (expected_server_zone, expected_receiver_zone) 字符串元组
            例如: ("right_near", "left_far")
        """
        # 根据发球方比分奇偶确定左右侧
        if match_state.server_score % 2 == 0:
            server_side = "right"
            receiver_side = "left"
        else:
            server_side = "left"
            receiver_side = "right"

        # 根据发球方确定近网/远网
        if match_state.server == "A":
            # 发球方在近网侧
            expected_server_zone = f"{server_side}_near"
            expected_receiver_zone = f"{receiver_side}_far"
        else:
            # 发球方在远网侧
            expected_server_zone = f"{server_side}_far"
            expected_receiver_zone = f"{receiver_side}_near"

        return expected_server_zone, expected_receiver_zone

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def update(
        self,
        player_a: PlayerFrame,
        player_b: PlayerFrame,
        match_state: MatchState,
        timestamp: float,
    ) -> Optional[ReadyEvent]:
        """每帧调用的主入口方法

        职责：
        1. 根据比赛状态计算预期发球区
        2. 将预期区域分配给球员 A/B
        3. 调用 detect_ready() 执行完整就绪检测

        Args:
            player_a: 球员 A 的当前帧追踪数据
            player_b: 球员 B 的当前帧追踪数据
            match_state: 当前比赛状态（比分、发球方等）
            timestamp: 当前帧的时间戳（秒）

        Returns:
            检测到就绪事件时返回 ReadyEvent；否则返回 None

        Usage:
            ```python
            event = detector.update(player_a, player_b, match_state, ts)
            if event:
                print(f"就绪! 时间: {event.timestamp}")
            ```
        """
        # 1. 计算预期发球区
        server_zone, receiver_zone = self.get_expected_zones(match_state)

        # 2. 根据实际发球方将预期区域映射到球员 A/B
        if match_state.server == "A":
            expected_a_zone = server_zone
            expected_b_zone = receiver_zone
        else:
            expected_a_zone = receiver_zone
            expected_b_zone = server_zone

        # 3. 执行就绪检测
        return self.detect_ready(
            player_a,
            player_b,
            expected_a_zone,
            expected_b_zone,
            timestamp,
        )
