# Programmable Friendly Conversationalist
# Prefrontal cortex
from typing import List, Optional, Dict, Any, Set
from ..message.message_base import UserInfo
import time
from dataclasses import dataclass, field
from src.common.logger import get_module_logger
from .chat_observer import ChatObserver
from .chat_states import NotificationHandler, NotificationType

logger = get_module_logger("observation_info")


class ObservationInfoHandler(NotificationHandler):
    """ObservationInfo的通知处理器"""

    def __init__(self, observation_info: "ObservationInfo"):
        """初始化处理器

        Args:
            observation_info: 要更新的ObservationInfo实例
        """
        self.observation_info = observation_info

    async def handle_notification(self, notification):
        # 获取通知类型和数据
        notification_type = notification.type
        data = notification.data

        if notification_type == NotificationType.NEW_MESSAGE:
            # 处理新消息通知
            logger.debug(f"收到新消息通知data: {data}")
            message_id = data.get("message_id")
            processed_plain_text = data.get("processed_plain_text")
            detailed_plain_text = data.get("detailed_plain_text")
            user_info = data.get("user_info")
            time_value = data.get("time")

            message = {
                "message_id": message_id,
                "processed_plain_text": processed_plain_text,
                "detailed_plain_text": detailed_plain_text,
                "user_info": user_info,
                "time": time_value,
            }

            self.observation_info.update_from_message(message)

        elif notification_type == NotificationType.COLD_CHAT:
            # 处理冷场通知
            is_cold = data.get("is_cold", False)
            self.observation_info.update_cold_chat_status(is_cold, time.time())

        elif notification_type == NotificationType.ACTIVE_CHAT:
            # 处理活跃通知
            is_active = data.get("is_active", False)
            self.observation_info.is_cold = not is_active

        elif notification_type == NotificationType.BOT_SPEAKING:
            # 处理机器人说话通知
            self.observation_info.is_typing = False
            self.observation_info.last_bot_speak_time = time.time()

        elif notification_type == NotificationType.USER_SPEAKING:
            # 处理用户说话通知
            self.observation_info.is_typing = False
            self.observation_info.last_user_speak_time = time.time()

        elif notification_type == NotificationType.MESSAGE_DELETED:
            # 处理消息删除通知
            message_id = data.get("message_id")
            self.observation_info.unprocessed_messages = [
                msg for msg in self.observation_info.unprocessed_messages if msg.get("message_id") != message_id
            ]

        elif notification_type == NotificationType.USER_JOINED:
            # 处理用户加入通知
            user_id = data.get("user_id")
            if user_id:
                self.observation_info.active_users.add(user_id)

        elif notification_type == NotificationType.USER_LEFT:
            # 处理用户离开通知
            user_id = data.get("user_id")
            if user_id:
                self.observation_info.active_users.discard(user_id)

        elif notification_type == NotificationType.ERROR:
            # 处理错误通知
            error_msg = data.get("error", "")
            logger.error(f"收到错误通知: {error_msg}")


@dataclass
class ObservationInfo:
    """决策信息类，用于收集和管理来自chat_observer的通知信息"""

    # data_list
    chat_history: List[str] = field(default_factory=list)
    unprocessed_messages: List[Dict[str, Any]] = field(default_factory=list)
    active_users: Set[str] = field(default_factory=set)

    # data
    last_bot_speak_time: Optional[float] = None
    last_user_speak_time: Optional[float] = None
    last_message_time: Optional[float] = None
    last_message_content: str = ""
    last_message_sender: Optional[str] = None
    bot_id: Optional[str] = None
    chat_history_count: int = 0
    new_messages_count: int = 0
    cold_chat_duration: float = 0.0

    # state
    is_typing: bool = False
    has_unread_messages: bool = False
    is_cold_chat: bool = False
    changed: bool = False

    # #spec
    # meta_plan_trigger: bool = False

    def __post_init__(self):
        """初始化后创建handler"""
        self.chat_observer = None
        self.handler = ObservationInfoHandler(self)

    def bind_to_chat_observer(self, chat_observer: ChatObserver):
        """绑定到指定的chat_observer

        Args:
            stream_id: 聊天流ID
        """
        self.chat_observer = chat_observer
        self.chat_observer.notification_manager.register_handler(
            target="observation_info", notification_type=NotificationType.NEW_MESSAGE, handler=self.handler
        )
        self.chat_observer.notification_manager.register_handler(
            target="observation_info", notification_type=NotificationType.COLD_CHAT, handler=self.handler
        )
        print("1919810------------------------绑定-----------------------------")

    def unbind_from_chat_observer(self):
        """解除与chat_observer的绑定"""
        if self.chat_observer:
            self.chat_observer.notification_manager.unregister_handler(
                target="observation_info", notification_type=NotificationType.NEW_MESSAGE, handler=self.handler
            )
            self.chat_observer.notification_manager.unregister_handler(
                target="observation_info", notification_type=NotificationType.COLD_CHAT, handler=self.handler
            )
            self.chat_observer = None

    def update_from_message(self, message: Dict[str, Any]):
        """从消息更新信息

        Args:
            message: 消息数据
        """
        # print("1919810-----------------------------------------------------")
        # logger.debug(f"更新信息from_message: {message}")
        self.last_message_time = message["time"]
        self.last_message_id = message["message_id"]

        self.last_message_content = message.get("processed_plain_text", "")

        user_info = UserInfo.from_dict(message.get("user_info", {}))
        self.last_message_sender = user_info.user_id

        if user_info.user_id == self.bot_id:
            self.last_bot_speak_time = message["time"]
        else:
            self.last_user_speak_time = message["time"]
            self.active_users.add(user_info.user_id)

        self.new_messages_count += 1
        self.unprocessed_messages.append(message)

        self.update_changed()

    def update_changed(self):
        """更新changed状态"""
        self.changed = True

    def update_cold_chat_status(self, is_cold: bool, current_time: float):
        """更新冷场状态

        Args:
            is_cold: 是否冷场
            current_time: 当前时间
        """
        self.is_cold_chat = is_cold
        if is_cold and self.last_message_time:
            self.cold_chat_duration = current_time - self.last_message_time

    def get_active_duration(self) -> float:
        """获取当前活跃时长

        Returns:
            float: 最后一条消息到现在的时长（秒）
        """
        if not self.last_message_time:
            return 0.0
        return time.time() - self.last_message_time

    def get_user_response_time(self) -> Optional[float]:
        """获取用户响应时间

        Returns:
            Optional[float]: 用户最后发言到现在的时长（秒），如果没有用户发言则返回None
        """
        if not self.last_user_speak_time:
            return None
        return time.time() - self.last_user_speak_time

    def get_bot_response_time(self) -> Optional[float]:
        """获取机器人响应时间

        Returns:
            Optional[float]: 机器人最后发言到现在的时长（秒），如果没有机器人发言则返回None
        """
        if not self.last_bot_speak_time:
            return None
        return time.time() - self.last_bot_speak_time

    def clear_unprocessed_messages(self):
        """清空未处理消息列表"""
        # 将未处理消息添加到历史记录中
        for message in self.unprocessed_messages:
            self.chat_history.append(message)
        # 清空未处理消息列表
        self.has_unread_messages = False
        self.unprocessed_messages.clear()
        self.chat_history_count = len(self.chat_history)
        self.new_messages_count = 0
