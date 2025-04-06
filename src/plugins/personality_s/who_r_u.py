import random
import os
import sys
from pathlib import Path
import datetime
from typing import List, Dict, Optional

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent
env_path = project_root / ".env"

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(root_path)

from src.common.database import db  # noqa: E402


class MessageAnalyzer:
    def __init__(self):
        self.messages_collection = db["messages"]

    def get_message_context(self, message_id: int, context_length: int = 5) -> Optional[List[Dict]]:
        """
        获取指定消息ID的上下文消息列表

        Args:
            message_id (int): 消息ID
            context_length (int): 上下文长度（单侧，总长度为 2*context_length + 1）

        Returns:
            Optional[List[Dict]]: 消息列表，如果未找到则返回None
        """
        # 从数据库获取指定消息
        target_message = self.messages_collection.find_one({"message_id": message_id})
        if not target_message:
            return None

        # 获取该消息的stream_id
        stream_id = target_message.get("chat_info", {}).get("stream_id")
        if not stream_id:
            return None

        # 获取同一stream_id的所有消息
        stream_messages = list(self.messages_collection.find({"chat_info.stream_id": stream_id}).sort("time", 1))

        # 找到目标消息在列表中的位置
        target_index = None
        for i, msg in enumerate(stream_messages):
            if msg["message_id"] == message_id:
                target_index = i
                break

        if target_index is None:
            return None

        # 获取目标消息前后的消息
        start_index = max(0, target_index - context_length)
        end_index = min(len(stream_messages), target_index + context_length + 1)

        return stream_messages[start_index:end_index]

    def format_messages(self, messages: List[Dict], target_message_id: Optional[int] = None) -> str:
        """
        格式化消息列表为可读字符串

        Args:
            messages (List[Dict]): 消息列表
            target_message_id (Optional[int]): 目标消息ID，用于标记

        Returns:
            str: 格式化的消息字符串
        """
        if not messages:
            return "没有消息记录"

        reply = ""
        for msg in messages:
            # 消息时间
            msg_time = datetime.datetime.fromtimestamp(int(msg["time"])).strftime("%Y-%m-%d %H:%M:%S")

            # 获取消息内容
            message_text = msg.get("processed_plain_text", msg.get("detailed_plain_text", "无消息内容"))
            nickname = msg.get("user_info", {}).get("user_nickname", "未知用户")

            # 标记当前消息
            is_target = "→ " if target_message_id and msg["message_id"] == target_message_id else "  "

            reply += f"{is_target}[{msg_time}] {nickname}: {message_text}\n"

            if target_message_id and msg["message_id"] == target_message_id:
                reply += "  " + "-" * 50 + "\n"

        return reply

    def get_user_random_contexts(
        self, qq_id: str, num_messages: int = 10, context_length: int = 5
    ) -> tuple[List[str], str]:  # noqa: E501
        """
        获取用户的随机消息及其上下文

        Args:
            qq_id (str): QQ号
            num_messages (int): 要获取的随机消息数量
            context_length (int): 每条消息的上下文长度（单侧）

        Returns:
            tuple[List[str], str]: (每个消息上下文的格式化字符串列表, 用户昵称)
        """
        if not qq_id:
            return [], ""

        # 获取用户所有消息
        all_messages = list(self.messages_collection.find({"user_info.user_id": int(qq_id)}))
        if not all_messages:
            return [], ""

        # 获取用户昵称
        user_nickname = all_messages[0].get("chat_info", {}).get("user_info", {}).get("user_nickname", "未知用户")

        # 随机选择指定数量的消息
        selected_messages = random.sample(all_messages, min(num_messages, len(all_messages)))
        # 按时间排序
        selected_messages.sort(key=lambda x: int(x["time"]))

        # 存储所有上下文消息
        context_list = []

        # 获取每条消息的上下文
        for msg in selected_messages:
            message_id = msg["message_id"]

            # 获取消息上下文
            context_messages = self.get_message_context(message_id, context_length)
            if context_messages:
                formatted_context = self.format_messages(context_messages, message_id)
                context_list.append(formatted_context)

        return context_list, user_nickname


if __name__ == "__main__":
    # 测试代码
    analyzer = MessageAnalyzer()
    test_qq = "1026294844"  # 替换为要测试的QQ号
    print(f"测试QQ号: {test_qq}")
    print("-" * 50)
    # 获取5条消息，每条消息前后各3条上下文
    contexts, nickname = analyzer.get_user_random_contexts(test_qq, num_messages=5, context_length=3)

    print(f"用户昵称: {nickname}\n")
    # 打印每个上下文
    for i, context in enumerate(contexts, 1):
        print(f"\n随机消息 {i}/{len(contexts)}:")
        print("-" * 30)
        print(context)
        print("=" * 50)
