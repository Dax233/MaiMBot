import enum
import time
import random
from typing import List, Tuple, Optional
from src.common.logger import get_module_logger, LogConfig, MAI_STATE_CONFIG
from src.plugins.moods.moods import MoodManager

mai_state_config = LogConfig(
    # 使用海马体专用样式
    console_format=MAI_STATE_CONFIG["console_format"],
    file_format=MAI_STATE_CONFIG["file_format"],
)
logger = get_module_logger("mai_state_manager", config=mai_state_config)


class MaiState(enum.Enum):
    """
    聊天状态:
    OFFLINE: 不在线：回复概率极低，不会进行任何聊天
    PEEKING: 看一眼手机：回复概率较低，会进行一些普通聊天
    NORMAL_CHAT: 正常聊天：回复概率较高，会进行一些普通聊天和少量的专注聊天
    FOCUSED_CHAT: 专注聊天：回复概率极高，会进行专注聊天和少量的普通聊天
    """

    OFFLINE = "不在线"
    PEEKING = "看一眼手机"
    NORMAL_CHAT = "正常聊天"
    FOCUSED_CHAT = "专注聊天"

    def get_normal_chat_max_num(self):
        if self == MaiState.OFFLINE:
            return 0
        elif self == MaiState.PEEKING:
            return 1
        elif self == MaiState.NORMAL_CHAT:
            return 3
        elif self == MaiState.FOCUSED_CHAT:
            return 2

    def get_focused_chat_max_num(self):
        if self == MaiState.OFFLINE:
            return 0
        elif self == MaiState.PEEKING:
            return 0
        elif self == MaiState.NORMAL_CHAT:
            return 1
        elif self == MaiState.FOCUSED_CHAT:
            return 2


class MaiStateInfo:
    def __init__(self):
        self.mai_status: MaiState = MaiState.OFFLINE
        self.mai_status_history: List[Tuple[MaiState, float]] = []  # 历史状态，包含 状态，时间戳
        self.last_status_change_time: float = time.time()  # 状态最后改变时间
        self.last_min_check_time: float = time.time()  # 上次1分钟规则检查时间

        # Mood management is now part of MaiStateInfo
        self.mood_manager = MoodManager.get_instance()  # Use singleton instance

    def update_mai_status(self, new_status: MaiState) -> bool:
        """
        更新聊天状态。

        Args:
            new_status: 新的 MaiState 状态。

        Returns:
            bool: 如果状态实际发生了改变则返回 True，否则返回 False。
        """
        if new_status != self.mai_status:
            self.mai_status = new_status
            current_time = time.time()
            self.last_status_change_time = current_time
            self.last_min_check_time = current_time  # Reset 1-min check on any state change
            self.mai_status_history.append((new_status, current_time))
            logger.info(f"麦麦状态更新为: {self.mai_status.value}")
            return True
        else:
            return False

    def reset_state_timer(self):
        """
        重置状态持续时间计时器和一分钟规则检查计时器。
        通常在状态保持不变但需要重新开始计时的情况下调用（例如，保持 OFFLINE）。
        """
        current_time = time.time()
        self.last_status_change_time = current_time
        self.last_min_check_time = current_time  # Also reset the 1-min check timer
        logger.debug("MaiStateInfo 状态计时器已重置。")

    def get_mood_prompt(self) -> str:
        """获取当前的心情提示词"""
        # Delegate to the internal mood manager
        return self.mood_manager.get_prompt()

    def get_current_state(self) -> MaiState:
        """获取当前的 MaiState"""
        return self.mai_status


class MaiStateManager:
    """管理 Mai 的整体状态转换逻辑"""

    def __init__(self):
        # MaiStateManager doesn't hold the state itself, it operates on a MaiStateInfo instance.
        pass

    def check_and_decide_next_state(self, current_state_info: MaiStateInfo) -> Optional[MaiState]:
        """
        根据当前状态和规则检查是否需要转换状态，并决定下一个状态。

        Args:
            current_state_info: 当前的 MaiStateInfo 实例。

        Returns:
            Optional[MaiState]: 如果需要转换，返回目标 MaiState；否则返回 None。
        """
        current_time = time.time()
        current_status = current_state_info.mai_status
        time_in_current_status = current_time - current_state_info.last_status_change_time
        time_since_last_min_check = current_time - current_state_info.last_min_check_time
        next_state: Optional[MaiState] = None

        if current_status == MaiState.OFFLINE:
            logger.info("当前[离线]，没看手机，思考要不要上线看看......")
        elif current_status == MaiState.PEEKING:
            logger.info("当前[在窥屏]，思考要不要继续聊下去......")
        elif current_status == MaiState.NORMAL_CHAT:
            logger.info("当前在[闲聊]思考要不要继续聊下去......")
        elif current_status == MaiState.FOCUSED_CHAT:
            logger.info("当前在[激情聊天]思考要不要继续聊下去......")

        # 1. 麦麦每分钟都有概率离线
        if time_since_last_min_check >= 60:
            if current_status != MaiState.OFFLINE:
                if random.random() < 0.03:  # 3% 概率切换到 OFFLINE，20分钟有50%的概率还在线
                    logger.debug(f"突然不想聊了，从 {current_status.value} 切换到 离线")
                    next_state = MaiState.OFFLINE

        # 2. 状态持续时间规则 (如果没有自行下线)
        if next_state is None:
            if current_status == MaiState.OFFLINE:
                # OFFLINE 最多保持一分钟
                # 目前是一个调试值，可以修改
                if time_in_current_status >= 60:
                    weights = [30, 30, 20, 20]
                    choices_list = [MaiState.PEEKING, MaiState.NORMAL_CHAT, MaiState.FOCUSED_CHAT, MaiState.OFFLINE]
                    next_state_candidate = random.choices(choices_list, weights=weights, k=1)[0]
                    if next_state_candidate != MaiState.OFFLINE:
                        next_state = next_state_candidate
                        logger.debug(f"上线！开始 {next_state.name}")
                    else:
                        # 继续离线状态
                        next_state = MaiState.OFFLINE

            elif current_status == MaiState.PEEKING:
                if time_in_current_status >= 600:  # PEEKING 最多持续 600 秒
                    weights = [70, 20, 10]
                    choices_list = [MaiState.OFFLINE, MaiState.NORMAL_CHAT, MaiState.FOCUSED_CHAT]
                    next_state = random.choices(choices_list, weights=weights, k=1)[0]
                    logger.debug(f"手机看完了，接下来 {next_state.name}")

            elif current_status == MaiState.NORMAL_CHAT:
                if time_in_current_status >= 300:  # NORMAL_CHAT 最多持续 300 秒
                    weights = [50, 50]
                    choices_list = [MaiState.OFFLINE, MaiState.FOCUSED_CHAT]
                    next_state = random.choices(choices_list, weights=weights, k=1)[0]
                    if next_state == MaiState.FOCUSED_CHAT:
                        logger.debug(f"继续深入聊天， {next_state.name}")
                    else:
                        logger.debug(f"聊完了，接下来 {next_state.name}")

            elif current_status == MaiState.FOCUSED_CHAT:
                if time_in_current_status >= 600:  # FOCUSED_CHAT 最多持续 600 秒
                    weights = [80, 20]
                    choices_list = [MaiState.OFFLINE, MaiState.NORMAL_CHAT]
                    next_state = random.choices(choices_list, weights=weights, k=1)[0]
                    logger.debug(f"深入聊天结束，接下来 {next_state.name}")

        # 如果决定了下一个状态，且这个状态与当前状态不同，则返回下一个状态
        if next_state is not None and next_state != current_status:
            return next_state
        # 如果决定保持 OFFLINE (next_state == MaiState.OFFLINE) 且当前也是 OFFLINE，
        # 并且是由于持续时间规则触发的，返回 OFFLINE 以便调用者可以重置计时器
        elif next_state == MaiState.OFFLINE and current_status == MaiState.OFFLINE and time_in_current_status >= 60:
            logger.debug("决定保持 OFFLINE (持续时间规则)，返回 OFFLINE 以提示重置计时器。")
            return MaiState.OFFLINE  # Return OFFLINE to signal caller that timer reset might be needed
        else:
            return None  # 没有状态转换发生或无需重置计时器
