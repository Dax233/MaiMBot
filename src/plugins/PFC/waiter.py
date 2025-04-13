from src.common.logger import get_module_logger
from .chat_observer import ChatObserver
from .conversation_info import ConversationInfo
from src.individuality.individuality import Individuality
from ..config.config import global_config
import time
import asyncio

logger = get_module_logger("waiter")


class Waiter:
    """快 速 等 待"""

    def __init__(self, stream_id: str):
        self.chat_observer = ChatObserver.get_instance(stream_id)
        self.personality_info = Individuality.get_instance().get_prompt(type="personality", x_person=2, level=2)
        self.name = global_config.BOT_NICKNAME

        self.wait_accumulated_time = 0

    async def wait(self, conversation_info: ConversationInfo) -> bool:
        """等待

        Returns:
            bool: 是否超时（True表示超时）
        """
        # 使用当前时间作为等待开始时间
        wait_start_time = time.time()
        self.chat_observer.waiting_start_time = wait_start_time  # 设置等待开始时间

        while True:
            # 检查是否有新消息
            if self.chat_observer.new_message_after(wait_start_time):
                logger.info("等待结束，收到新消息")
                return False

            # 检查是否超时
            if time.time() - wait_start_time > 300:
                self.wait_accumulated_time += 300

                logger.info("等待超过300秒，结束对话")
                wait_goal = {
                    "goal": f"你等待了{self.wait_accumulated_time / 60}分钟，思考接下来要做什么",
                    "reason": "对方很久没有回复你的消息了",
                }
                conversation_info.goal_list.append(wait_goal)
                print(f"添加目标: {wait_goal}")

                return True

            await asyncio.sleep(1)
            logger.info("等待中...")

    async def wait_listening(self, conversation_info: ConversationInfo) -> bool:
        """等待倾听

        Returns:
            bool: 是否超时（True表示超时）
        """
        # 使用当前时间作为等待开始时间
        wait_start_time = time.time()
        self.chat_observer.waiting_start_time = wait_start_time  # 设置等待开始时间

        while True:
            # 检查是否有新消息
            if self.chat_observer.new_message_after(wait_start_time):
                logger.info("等待结束，收到新消息")
                return False

            # 检查是否超时
            if time.time() - wait_start_time > 300:
                self.wait_accumulated_time += 300
                logger.info("等待超过300秒，结束对话")
                wait_goal = {
                    "goal": f"你等待了{self.wait_accumulated_time / 60}分钟，思考接下来要做什么",
                    "reason": "对方话说一半消失了，很久没有回复",
                }
                conversation_info.goal_list.append(wait_goal)
                print(f"添加目标: {wait_goal}")

                return True

            await asyncio.sleep(1)
            logger.info("等待中...")
