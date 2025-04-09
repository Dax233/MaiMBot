from typing import TYPE_CHECKING
from src.common.logger import get_module_logger
from .chat_states import NotificationHandler, Notification, NotificationType

if TYPE_CHECKING:
    from .conversation import Conversation

logger = get_module_logger("notification_handler")

class PFCNotificationHandler(NotificationHandler):
    """PFC通知处理器"""
    
    def __init__(self, conversation: 'Conversation'):
        """初始化PFC通知处理器
        
        Args:
            conversation: 对话实例
        """
        self.conversation = conversation
        
    async def handle_notification(self, notification: Notification):
        """处理通知
        
        Args:
            notification: 通知对象
        """
        logger.debug(f"收到通知: {notification.type.name}, 数据: {notification.data}")
        
        # 根据通知类型执行不同的处理
        if notification.type == NotificationType.NEW_MESSAGE:
            # 新消息通知
            await self._handle_new_message(notification)
        elif notification.type == NotificationType.COLD_CHAT:
            # 冷聊天通知
            await self._handle_cold_chat(notification)
        elif notification.type == NotificationType.COMMAND:
            # 命令通知
            await self._handle_command(notification)
        else:
            logger.warning(f"未知的通知类型: {notification.type.name}")
            
    async def _handle_new_message(self, notification: Notification):
        """处理新消息通知
        
        Args:
            notification: 通知对象
        """
        
        # 更新决策信息
        observation_info = self.conversation.observation_info
        observation_info.last_message_time = notification.data.get("time", 0)
        observation_info.add_unprocessed_message(notification.data)
        
        # 手动触发观察器更新
        self.conversation.chat_observer.trigger_update()
        
    async def _handle_cold_chat(self, notification: Notification):
        """处理冷聊天通知
        
        Args:
            notification: 通知对象
        """
        # 获取冷聊天信息
        cold_duration = notification.data.get("duration", 0)
        
        # 更新决策信息
        observation_info = self.conversation.observation_info
        observation_info.conversation_cold_duration = cold_duration
        
        logger.info(f"对话已冷: {cold_duration}秒")
        