from src.common.logger import get_module_logger
from .chat_observer import ChatObserver

logger = get_module_logger("waiter")

class Waiter:
    """等待器，用于等待对话流中的事件"""
    
    def __init__(self, stream_id: str):
        self.stream_id = stream_id
        self.chat_observer = ChatObserver.get_instance(stream_id)
    
    async def wait(self, timeout: float = 20.0) -> bool:
        """等待用户回复或超时
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            bool: 如果因为超时返回则为True，否则为False
        """
        try:
            message_before = self.chat_observer.get_last_message()
            
            # 等待新消息
            logger.debug(f"等待新消息，超时时间: {timeout}秒")
            
            is_timeout = await self.chat_observer.wait_for_update(timeout=timeout)
            if is_timeout:
                logger.debug("等待超时，没有收到新消息")
                return True
                
            # 检查是否是新消息
            message_after = self.chat_observer.get_last_message()
            if message_before and message_after and message_before.get("message_id") == message_after.get("message_id"):
                # 如果消息ID相同，说明没有新消息
                logger.debug("没有收到新消息")
                return True
                
            logger.debug("收到新消息")
            return False
            
        except Exception as e:
            logger.error(f"等待时出错: {str(e)}")
            return True 