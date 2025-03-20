from typing import Optional, Union

from ...common.database import db
from .message import MessageSending, MessageRecv
from .chat_stream import ChatStream
from src.common.logger import get_module_logger

logger = get_module_logger("message_storage")


class MessageStorage:
    async def store_message(
        self, message: Union[MessageSending, MessageRecv], chat_stream: ChatStream, topic: Optional[str] = None
    ) -> None:
        """存储消息到数据库"""
        try:
            message_data = {
                "message_id": message.message_info.message_id,
                "time": message.message_info.time,
                "chat_id": chat_stream.stream_id,
                "chat_info": chat_stream.to_dict(),
                "user_info": message.message_info.user_info.to_dict(),
                "processed_plain_text": message.processed_plain_text,
                "detailed_plain_text": message.detailed_plain_text,
                "topic": topic,
                "memorized_times": message.memorized_times,
            }
            db.messages.insert_one(message_data)
        except Exception:
            logger.exception("存储消息失败")

    async def store_recalled_message(self, message_id: str, time: str, chat_stream: ChatStream) -> None:
        """存储撤回消息到数据库"""
        if "recalled_messages" not in db.list_collection_names():
            db.create_collection("recalled_messages")
        else:
            try:
                message_data = {
                    "message_id": message_id,
                    "time": time,
                    "stream_id": chat_stream.stream_id,
                }
                db.recalled_messages.insert_one(message_data)
            except Exception:
                logger.exception("存储撤回消息失败")

    async def remove_recalled_message(self, time: str) -> None:
        """删除撤回消息"""
        try:
            db.recalled_messages.delete_many({"time": {"$lt": time - 300}})
        except Exception:
            logger.exception("删除撤回消息失败")


# 如果需要其他存储相关的函数，可以在这里添加
