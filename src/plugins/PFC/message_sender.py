from typing import Optional
from src.common.logger import get_module_logger
from ..chat.chat_stream import ChatStream
from ..chat.message import Message
from ..message.message_base import Seg
from src.plugins.chat.message import MessageSending, MessageSet
from src.plugins.chat.message_sender import message_manager

logger = get_module_logger("message_sender")


class DirectMessageSender:
    """直接消息发送器"""

    def __init__(self):
        pass

    async def send_message(
        self,
        chat_stream: ChatStream,
        content: str,
        reply_to_message: Optional[Message] = None,
    ) -> None:
        """发送消息到聊天流

        Args:
            chat_stream: 聊天流
            content: 消息内容
            reply_to_message: 要回复的消息（可选）
        """
        try:
            # 创建消息内容
            segments = [Seg(type="text", data={"text": content})]

            # 检查是否需要引用回复
            if reply_to_message:
                reply_id = reply_to_message.message_id
                message_sending = MessageSending(segments=segments, reply_to_id=reply_id)
            else:
                message_sending = MessageSending(segments=segments)

            # 发送消息
            message_set = MessageSet(chat_stream, message_sending.message_id)
            message_set.add_message(message_sending)
            message_manager.add_message(message_set)
            logger.info(f"PFC消息已发送: {content}")

        except Exception as e:
            logger.error(f"PFC消息发送失败: {str(e)}")
            raise
