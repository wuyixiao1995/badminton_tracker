"""比分计算器 — 基于羽毛球规则的状态机

核心思路: 系统不识别羽毛球/击球动作, 仅通过球员位置变化推断比分。

位置推断逻辑:
  每次 ReadyEvent 表示两名球员在发球/接发区就绪。

  情况A (正常比赛): 两人都在预期的发球/接发区
    → 不发分, 这是正常回合的开始站位

  情况B (发球方得分): 发球方球员从右侧发球区换到左侧 (或反之)
    → 发球方保持发球权, 比分+1
    → 判定依据: 同一球员仍在发球方位置, 但左右区互换

  情况C (接发方得分): 原先的接发方球员现在站在发球方位置
    → 交换发球权, 接发方比分+1
    → 判定依据: 发球方换了人

规则 (21分制):
  - 发球方得分 → 发球方保持发球权
  - 接发方得分 → 交换发球权
  - 偶分数 → 右发球区发球, 奇分数 → 左发球区
  - 11分换边
  - 20:20 需净胜2分
  - 29:29 下一分获胜 (封顶30分)

依赖: models.py (MatchState, ScoreUpdate, ReadyEvent)
      config.py (MAX_SCORE, MIN_SCORE_DIFF, WIN_SCORE)
"""

import json
import logging
from typing import Optional
from datetime import datetime

from models import Point, ReadyEvent, ScoreUpdate, MatchState
from config import MAX_SCORE, MIN_SCORE_DIFF, WIN_SCORE

logger = logging.getLogger(__name__)


class Scorer:
    """羽毛球比分状态机

    通过分析 ReadyEvent 序列推断比分变化。
    不依赖对羽毛球/击球的视觉识别。

    Attributes:
        state: 当前比赛状态 (MatchState)
        score_history: 比分变化历史记录
        prev_ready_event: 上一次处理的就绪事件
        prev_server_zone: 上一次发球方所在区域
        sides_swapped: 是否已换边 (11分后)
    """

    def __init__(self, first_server: str = "A"):
        """初始化比分状态机

        Args:
            first_server: 首位发球方, "A" 或 "B"
        """
        if first_server not in ("A", "B"):
            raise ValueError(f"first_server must be 'A' or 'B', got '{first_server}'")

        self.state = MatchState(server=first_server)
        self.score_history: list[ScoreUpdate] = []
        self.prev_ready_event: Optional[ReadyEvent] = None
        self.prev_server_zone: Optional[str] = None     # 上一次发球方所在区
        self.prev_receiver_zone: Optional[str] = None   # 上一次接发方所在区
        self.sides_swapped = False                       # 11分换边标记
        self._first_event_processed = False              # 第一个事件仅记录, 不判分

    # ==================================================================
    #  发球区计算
    # ==================================================================

    def _server_side(self) -> str:
        """发球方当前在哪一侧 (near 或 far)

        球员A始终在近侧 (near), 球员B始终在远侧 (far)
        换边后 (sides_swapped=True): A→far, B→near
        """
        if not self.sides_swapped:
            return "near" if self.state.server == "A" else "far"
        else:
            return "far" if self.state.server == "A" else "near"

    def _server_left_right(self) -> str:
        """发球方应根据自己分数站在左区还是右区

        偶分数 → 右区 (right), 奇分数 → 左区 (left)
        """
        return "right" if self.state.server_score % 2 == 0 else "left"

    def expected_server_zone(self) -> str:
        """计算发球方应该在哪个发球区

        发球区命名: {left/right}_{near/far}
        例: right_near = 近侧右发球区

        Returns:
            发球区全名, 如 "right_near"
        """
        side = self._server_side()
        lr = self._server_left_right()
        return f"{lr}_{side}"

    def expected_receiver_zone(self) -> str:
        """计算接发方应该在哪个发球区 (发球区的对角)

        对角线关系:
          right_near ↔ left_far
          left_near  ↔ right_far

        Returns:
            接发方预期发球区全名
        """
        server_zone = self.expected_server_zone()
        # 对角线映射
        diagonal_map = {
            "right_near": "left_far",
            "left_near": "right_far",
            "right_far": "left_near",
            "left_far": "right_near",
        }
        return diagonal_map.get(server_zone, "")

    # ==================================================================
    #  比分推断核心
    # ==================================================================

    def process_ready_event(self, event: ReadyEvent) -> Optional[ScoreUpdate]:
        """处理就绪事件, 推断是否发生了得分

        这是核心比分逻辑。每个 ReadyEvent 表示两名球员在各自发球区
        站定准备发球/接发。

        判定方法:
        1. 第一个事件: 仅记录球员位置, 不发分
        2. 后续事件: 比较球员位置变化

        判定A (正常比赛,不发分):
          - 发球方在预期的发球区
          - 接发方在预期的接发区 (对角)
          - 与上一次事件相比没有异常变化

        判定B (发球方得分):
          - 发球方球员仍在发球方位置
          - 但发球区从右换到左 (或左换到右)
          - 说明发球方得了一分
          - → server_score += 1, 发球方不变

        判定C (接发方得分 = 换发):
          - 发球方球员变了 (另一球员出现在发球方位置)
          - → receiver_score += 1, server 交换

        Args:
            event: 就绪事件 (包含两名球员所在区域)

        Returns:
            ScoreUpdate 或 None (正常比赛不发分)
        """
        try:
            # ---- 第一步: 确定当前事件中谁在什么区域 -----------------
            server_player = self.state.server  # "A" 或 "B"
            receiver_player = "B" if server_player == "A" else "A"

            if server_player == "A":
                server_actual_zone = event.player_a_zone
                receiver_actual_zone = event.player_b_zone
            else:
                server_actual_zone = event.player_b_zone
                receiver_actual_zone = event.player_a_zone

            expected_s = self.expected_server_zone()
            expected_r = self.expected_receiver_zone()

            # ---- 第二步: 第一个事件仅记录 ---------------------------
            if not self._first_event_processed:
                self.prev_ready_event = event
                self.prev_server_zone = server_actual_zone
                self.prev_receiver_zone = receiver_actual_zone
                self._first_event_processed = True
                logger.debug(
                    f"First ReadyEvent at {event.timestamp:.1f}s: "
                    f"Server({server_player}) in {server_actual_zone}, "
                    f"Receiver({receiver_player}) in {receiver_actual_zone}, "
                    f"Expected: S={expected_s}, R={expected_r}"
                )
                return None

            # ---- 第三步: 比较变化 -----------------------------------

            # 情况A: 都在预期位置 → 正常比赛
            if (server_actual_zone == expected_s and
                receiver_actual_zone == expected_r):
                # 更新记录
                self.prev_ready_event = event
                self.prev_server_zone = server_actual_zone
                self.prev_receiver_zone = receiver_actual_zone
                return None

            # 情况B: 发球方在发球位置但区左右互换 → 发球方得分
            # 发球方应该在 near/far 侧 (根据球员身份)
            # 左/右互换意味着分数从偶变奇或奇变偶
            server_side_actual = self._extract_side(server_actual_zone)
            server_side_expected = self._server_side()
            expected_lr = self._server_left_right()
            actual_lr = self._extract_left_right(server_actual_zone)

            if (server_side_actual == server_side_expected and
                actual_lr and actual_lr != expected_lr and
                server_actual_zone != self.prev_server_zone):
                # 发球方在同一侧但左右互换 → 发球方得分
                return self._award_point_to_server(event)

            # 情况C: 发球方变了 → 接发方得分 (换发)
            # 检查: 前接发方现在在预期发球位置
            if server_player == "A":
                other_in_server_pos = self._is_in_server_position(
                    event.player_b_zone, "B"
                )
            else:
                other_in_server_pos = self._is_in_server_position(
                    event.player_a_zone, "A"
                )

            if other_in_server_pos and server_actual_zone != expected_s:
                return self._award_point_to_receiver(event)

            # 模糊情况: 尝试用左右区变化判断
            # 如果 prev_server_zone 记录存在且不同, 发球方可能得分
            if (self.prev_server_zone and
                server_actual_zone != self.prev_server_zone and
                server_side_actual == server_side_expected):
                # 发球方区域变了但还在自己这边 → 发球方得分
                return self._award_point_to_server(event)

            # 如果无法确定, 不发分但更新记录
            logger.debug(
                f"Ambiguous event at {event.timestamp:.1f}s: "
                f"Server({server_player}) in {server_actual_zone} "
                f"(expected {expected_s}), "
                f"Receiver in {receiver_actual_zone} (expected {expected_r})"
            )

            self.prev_ready_event = event
            self.prev_server_zone = server_actual_zone
            self.prev_receiver_zone = receiver_actual_zone
            return None

        except Exception as e:
            logger.error(f"Error processing ready event: {e}")
            return None

    def _award_point_to_server(self, event: ReadyEvent) -> ScoreUpdate:
        """发球方得分"""
        server = self.state.server
        if server == "A":
            self.state.score_a += 1
        else:
            self.state.score_b += 1

        self.state.rally_count += 1
        self.state.last_update = event.timestamp

        # 检查换边
        self._check_changeover()

        description = f"Server ({server}) scored. "
        description += f"Score: {self.state.score_a}:{self.state.score_b}"

        update = ScoreUpdate(
            timestamp=event.timestamp,
            score_a=self.state.score_a,
            score_b=self.state.score_b,
            server=server,
            confidence=0.90,
            description=description,
        )

        self._check_game_over()
        self.score_history.append(update)

        # 更新记录
        server_player = self.state.server
        if server_player == "A":
            self.prev_server_zone = event.player_a_zone
            self.prev_receiver_zone = event.player_b_zone
        else:
            self.prev_server_zone = event.player_b_zone
            self.prev_receiver_zone = event.player_a_zone
        self.prev_ready_event = event

        logger.info(f"[{event.timestamp:.1f}s] {description}")
        return update

    def _award_point_to_receiver(self, event: ReadyEvent) -> ScoreUpdate:
        """接发方得分 (换发)"""
        old_server = self.state.server
        receiver = "B" if old_server == "A" else "A"

        if receiver == "A":
            self.state.score_a += 1
        else:
            self.state.score_b += 1

        # 交换发球权
        self.state.server = receiver

        self.state.rally_count += 1
        self.state.last_update = event.timestamp

        # 检查换边
        self._check_changeover()

        description = (
            f"Receiver ({receiver}) scored, service over. "
            f"Score: {self.state.score_a}:{self.state.score_b}, "
            f"Now serving: {receiver}"
        )

        update = ScoreUpdate(
            timestamp=event.timestamp,
            score_a=self.state.score_a,
            score_b=self.state.score_b,
            server=receiver,
            confidence=0.85,
            description=description,
        )

        self._check_game_over()
        self.score_history.append(update)

        # 更新记录 (发球方已变)
        new_server = self.state.server
        if new_server == "A":
            self.prev_server_zone = event.player_a_zone
            self.prev_receiver_zone = event.player_b_zone
        else:
            self.prev_server_zone = event.player_b_zone
            self.prev_receiver_zone = event.player_a_zone
        self.prev_ready_event = event

        logger.info(f"[{event.timestamp:.1f}s] {description}")
        return update

    # ==================================================================
    #  辅助判断
    # ==================================================================

    def _extract_side(self, zone_name: str) -> Optional[str]:
        """从发球区名提取侧 (near/far)"""
        if zone_name is None:
            return None
        if "near" in zone_name:
            return "near"
        if "far" in zone_name:
            return "far"
        return None

    def _extract_left_right(self, zone_name: str) -> Optional[str]:
        """从发球区名提取左右 (left/right)"""
        if zone_name is None:
            return None
        if "left" in zone_name:
            return "left"
        if "right" in zone_name:
            return "right"
        return None

    def _is_in_server_position(self, zone: str, player_id: str) -> bool:
        """判断球员是否在发球方位置

        根据球员身份 (A=near侧, B=far侧) 判断区域是否匹配
        """
        if zone is None:
            return False

        if not self.sides_swapped:
            expected_side = "near" if player_id == "A" else "far"
        else:
            expected_side = "far" if player_id == "A" else "near"

        actual_side = self._extract_side(zone)
        return actual_side == expected_side

    def _check_changeover(self) -> None:
        """检查是否需要换边 (总分达到11分)"""
        total = self.state.score_a + self.state.score_b
        if total == 11 and not self.sides_swapped:
            self.sides_swapped = True
            logger.info("11-point changeover! Sides swapped.")

    def _check_game_over(self) -> None:
        """检查比赛是否结束

        结束条件:
        - 达到21分且领先2分, 或
        - 达到30分 (29:29后的封顶)
        """
        a, b = self.state.score_a, self.state.score_b
        diff = abs(a - b)
        max_score = max(a, b)

        if max_score >= WIN_SCORE and diff >= MIN_SCORE_DIFF:
            self.state.game_over = True
            self.state.winner = "A" if a > b else "B"
            logger.info(f"Game over! Winner: {self.state.winner}. Final: {a}:{b}")
        elif max_score >= MAX_SCORE:
            self.state.game_over = True
            self.state.winner = "A" if a > b else "B"
            logger.info(f"Game over (max score)! Winner: {self.state.winner}. Final: {a}:{b}")

    # ==================================================================
    #  公开接口
    # ==================================================================

    def process_event(self, event: ReadyEvent) -> Optional[ScoreUpdate]:
        """处理就绪事件的外部接口

        对 process_ready_event 的封装, 附加状态更新

        Args:
            event: 就绪事件

        Returns:
            ScoreUpdate 或 None
        """
        result = self.process_ready_event(event)

        if result is not None:
            self.state.last_update = event.timestamp

        return result

    def get_score_summary(self) -> dict:
        """获取当前比分摘要

        Returns:
            dict with keys: score_a, score_b, server, game_over, winner,
                           total_rallies, sides_swapped, court_type_note
        """
        return {
            "score_a": self.state.score_a,
            "score_b": self.state.score_b,
            "server": self.state.server,
            "game_over": self.state.game_over,
            "winner": self.state.winner,
            "total_rallies": self.state.rally_count,
            "sides_swapped": self.sides_swapped,
            "total_score_events": len(self.score_history),
        }

    def get_final_score_string(self) -> str:
        """获取最终比分的字符串表示"""
        return f"{self.state.score_a}:{self.state.score_b}"

    def export_score_json(self, filepath: str) -> None:
        """导出比分记录到 JSON 文件

        Args:
            filepath: JSON 文件输出路径
        """
        # Determine first server from history or current state
        if self.score_history:
            first_server = self.score_history[0].server
            # The server at first scoring event served first
        else:
            first_server = self.state.server

        output = {
            "match_info": {
                "first_server": first_server,
                "total_points": len(self.score_history),
                "sides_swapped": self.sides_swapped,
                "exported_at": datetime.now().isoformat(),
            },
            "score_updates": [
                {
                    "timestamp": round(s.timestamp, 2),
                    "score_a": s.score_a,
                    "score_b": s.score_b,
                    "server": s.server,
                    "confidence": round(s.confidence, 3),
                    "description": s.description,
                }
                for s in self.score_history
            ],
            "final_score": self.get_final_score_string(),
            "winner": self.state.winner,
            "game_over": self.state.game_over,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(f"Scores exported to {filepath}")
