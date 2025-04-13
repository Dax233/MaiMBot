import asyncio
import datetime
from typing import Dict, Any
from ..chat.message import Message
from .pfc_types import ConversationState
from .pfc import ChatObserver, GoalAnalyzer, DirectMessageSender
from src.common.logger import get_module_logger
from .action_planner import ActionPlanner
from .observation_info import ObservationInfo
from .conversation_info import ConversationInfo
from .reply_generator import ReplyGenerator
from ..chat.chat_stream import ChatStream
from ..message.message_base import UserInfo
from src.plugins.chat.chat_stream import chat_manager
from .pfc_KnowledgeFetcher import KnowledgeFetcher
from .waiter import Waiter

import traceback

logger = get_module_logger("pfc_conversation")


class Conversation:
    """对话类，负责管理单个对话的状态和行为"""

    def __init__(self, stream_id: str):
        """初始化对话实例

        Args:
            stream_id: 聊天流ID
        """
        self.stream_id = stream_id
        self.state = ConversationState.INIT
        self.should_continue = False

        # 回复相关
        self.generated_reply = ""

    async def _initialize(self):
        """初始化实例，注册所有组件"""

        try:
            self.action_planner = ActionPlanner(self.stream_id)
            self.goal_analyzer = GoalAnalyzer(self.stream_id)
            self.reply_generator = ReplyGenerator(self.stream_id)
            self.knowledge_fetcher = KnowledgeFetcher()
            self.waiter = Waiter(self.stream_id)
            self.direct_sender = DirectMessageSender()

            # 获取聊天流信息
            self.chat_stream = chat_manager.get_stream(self.stream_id)

            self.stop_action_planner = False
        except Exception as e:
            logger.error(f"初始化对话实例：注册运行组件失败: {e}")
            logger.error(traceback.format_exc())
            raise

        try:
            # 决策所需要的信息，包括自身自信和观察信息两部分
            # 注册观察器和观测信息
            self.chat_observer = ChatObserver.get_instance(self.stream_id)
            self.chat_observer.start()
            self.observation_info = ObservationInfo()
            self.observation_info.bind_to_chat_observer(self.chat_observer)
            # print(self.chat_observer.get_cached_messages(limit=)

            self.conversation_info = ConversationInfo()
        except Exception as e:
            logger.error(f"初始化对话实例：注册信息组件失败: {e}")
            logger.error(traceback.format_exc())
            raise

        # 组件准备完成，启动该论对话
        self.should_continue = True
        asyncio.create_task(self.start())

    async def start(self):
        """开始对话流程"""
        try:
            logger.info("对话系统启动中...")
            asyncio.create_task(self._plan_and_action_loop())
        except Exception as e:
            logger.error(f"启动对话系统失败: {e}")
            raise

    async def _plan_and_action_loop(self):
        """思考步，PFC核心循环模块"""
        # 获取最近的消息历史
        while self.should_continue:
            # 使用决策信息来辅助行动规划
            action, reason = await self.action_planner.plan(self.observation_info, self.conversation_info)
            if self._check_new_messages_after_planning():
                continue

            # 执行行动
            await self._handle_action(action, reason, self.observation_info, self.conversation_info)

            for goal in self.conversation_info.goal_list:
                # 检查goal是否为元组类型，如果是元组则使用索引访问，如果是字典则使用get方法
                if isinstance(goal, tuple):
                    # 假设元组的第一个元素是目标内容
                    print(f"goal: {goal}")
                    if goal[0] == "结束对话":
                        self.should_continue = False
                        break

    def _check_new_messages_after_planning(self):
        """检查在规划后是否有新消息"""
        if self.observation_info.new_messages_count > 0:
            logger.info(f"发现{self.observation_info.new_messages_count}条新消息，可能需要重新考虑行动")
            # 如果需要，可以在这里添加逻辑来根据新消息重新决定行动
            return True
        return False

    def _convert_to_message(self, msg_dict: Dict[str, Any]) -> Message:
        """将消息字典转换为Message对象"""
        try:
            chat_info = msg_dict.get("chat_info", {})
            chat_stream = ChatStream.from_dict(chat_info)
            user_info = UserInfo.from_dict(msg_dict.get("user_info", {}))

            return Message(
                message_id=msg_dict["message_id"],
                chat_stream=chat_stream,
                time=msg_dict["time"],
                user_info=user_info,
                processed_plain_text=msg_dict.get("processed_plain_text", ""),
                detailed_plain_text=msg_dict.get("detailed_plain_text", ""),
            )
        except Exception as e:
            logger.warning(f"转换消息时出错: {e}")
            raise

    async def _handle_action(
        self, action: str, reason: str, observation_info: ObservationInfo, conversation_info: ConversationInfo
    ):
        """处理规划的行动"""
        logger.info(f"执行行动: {action}, 原因: {reason}")

        # 记录action历史，先设置为stop，完成后再设置为done
        conversation_info.done_action.append(
            {
                "action": action,
                "reason": reason,
                "status": "start",
                "time": datetime.datetime.now().strftime("%H:%M:%S"),
            }
        )

        if action == "direct_reply":
            self.waiter.wait_accumulated_time = 0

            self.state = ConversationState.GENERATING
            self.generated_reply = await self.reply_generator.generate(observation_info, conversation_info)
            print(f"生成回复: {self.generated_reply}")

            # # 检查回复是否合适
            # is_suitable, reason, need_replan = await self.reply_generator.check_reply(
            #     self.generated_reply,
            #     self.current_goal
            # )

            if self._check_new_messages_after_planning():
                logger.info("333333发现新消息，重新考虑行动")
                conversation_info.done_action[-1].update(
                    {
                        "status": "recall",
                        "time": datetime.datetime.now().strftime("%H:%M:%S"),
                    }
                )
                return None

            await self._send_reply()

            conversation_info.done_action[-1].update(
                {
                    "status": "done",
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                }
            )

        elif action == "fetch_knowledge":
            self.waiter.wait_accumulated_time = 0

            self.state = ConversationState.FETCHING
            knowledge = "TODO:知识"
            topic = "TODO:关键词"

            logger.info(f"假装获取到知识{knowledge}，关键词是: {topic}")

            if knowledge:
                if topic not in self.conversation_info.knowledge_list:
                    self.conversation_info.knowledge_list.append({"topic": topic, "knowledge": knowledge})
                else:
                    self.conversation_info.knowledge_list[topic] += knowledge

        elif action == "rethink_goal":
            self.waiter.wait_accumulated_time = 0

            self.state = ConversationState.RETHINKING
            await self.goal_analyzer.analyze_goal(conversation_info, observation_info)

        elif action == "listening":
            self.state = ConversationState.LISTENING
            logger.info("倾听对方发言...")
            await self.waiter.wait_listening(conversation_info)

        elif action == "end_conversation":
            self.should_continue = False
            logger.info("决定结束对话...")

        else:  # wait
            self.state = ConversationState.WAITING
            logger.info("等待更多信息...")
            await self.waiter.wait(self.conversation_info)

    async def _send_timeout_message(self):
        """发送超时结束消息"""
        try:
            messages = self.chat_observer.get_cached_messages(limit=1)
            if not messages:
                return

            latest_message = self._convert_to_message(messages[0])
            await self.direct_sender.send_message(
                chat_stream=self.chat_stream, content="TODO:超时消息", reply_to_message=latest_message
            )
        except Exception as e:
            logger.error(f"发送超时消息失败: {str(e)}")

    async def _send_reply(self):
        """发送回复"""
        if not self.generated_reply:
            logger.warning("没有生成回复")
            return

        try:
            await self.direct_sender.send_message(chat_stream=self.chat_stream, content=self.generated_reply)
            self.chat_observer.trigger_update()  # 触发立即更新
            if not await self.chat_observer.wait_for_update():
                logger.warning("等待消息更新超时")

            self.state = ConversationState.ANALYZING
        except Exception as e:
            logger.error(f"发送消息失败: {str(e)}")
            self.state = ConversationState.ANALYZING
