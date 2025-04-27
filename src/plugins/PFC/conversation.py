import time
import asyncio
import datetime

# from .message_storage import MongoDBMessageStorage
from src.plugins.utils.chat_message_builder import build_readable_messages, get_raw_msg_before_timestamp_with_chat

# from ...config.config import global_config
from typing import Dict, Any, Optional
from ..chat.message import Message
from .pfc_types import ConversationState
from .pfc import ChatObserver, GoalAnalyzer, DirectMessageSender
from src.common.logger import get_module_logger
from .action_planner import ActionPlanner
from .observation_info import ObservationInfo
from .conversation_info import ConversationInfo
from .reply_generator import ReplyGenerator
from ..chat.chat_stream import ChatStream
from maim_message import UserInfo
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
        self.ignore_until_timestamp: Optional[float] = None

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
        try:
            logger.info(f"为 {self.stream_id} 加载初始聊天记录...")
            initial_messages = get_raw_msg_before_timestamp_with_chat(  #
                chat_id=self.stream_id,
                timestamp=time.time(),
                limit=30,  # 加载最近30条作为初始上下文，可以调整
            )
            chat_talking_prompt = await build_readable_messages(
                initial_messages,
                replace_bot_name=True,
                merge_messages=False,
                timestamp_mode="relative",
                read_mark=0.0,
            )
            if initial_messages:
                # 将加载的消息填充到 ObservationInfo 的 chat_history
                self.observation_info.chat_history = initial_messages
                self.observation_info.chat_history_str = chat_talking_prompt + "\n"
                self.observation_info.chat_history_count = len(initial_messages)

                # 更新 ObservationInfo 中的时间戳等信息
                last_msg = initial_messages[-1]
                self.observation_info.last_message_time = last_msg.get("time")
                last_user_info = UserInfo.from_dict(last_msg.get("user_info", {}))
                self.observation_info.last_message_sender = last_user_info.user_id
                self.observation_info.last_message_content = last_msg.get("processed_plain_text", "")

                # （可选）可以遍历 initial_messages 来设置 last_bot_speak_time 和 last_user_speak_time
                # 这里为了简化，只用了最后一条消息的时间，如果需要精确的发言者时间需要遍历

                logger.info(
                    f"成功加载 {len(initial_messages)} 条初始聊天记录。最后一条消息时间: {self.observation_info.last_message_time}"
                )

                # 让 ChatObserver 从加载的最后一条消息之后开始同步
                self.chat_observer.last_message_time = self.observation_info.last_message_time
                self.chat_observer.last_message_read = last_msg  # 更新 observer 的最后读取记录
            else:
                logger.info("没有找到初始聊天记录。")

        except Exception as load_err:
            logger.error(f"加载初始聊天记录时出错: {load_err}")
            # 出错也要继续，只是没有历史记录而已
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
        while self.should_continue:
            if self.ignore_until_timestamp and time.time() < self.ignore_until_timestamp:
                # 仍在忽略期间，等待下次检查
                await asyncio.sleep(30)  # 每 30 秒检查一次
                continue  # 跳过本轮循环的剩余部分
            elif self.ignore_until_timestamp and time.time() >= self.ignore_until_timestamp:
                # 忽略期结束，现在正常地结束对话
                logger.info(f"忽略时间已到 {self.stream_id}，准备结束对话。")
                self.ignore_until_timestamp = None  # 清除时间戳
                self.should_continue = False  # 现在停止循环
                # （可选）在这里记录一个 'end_conversation' 动作
                # 或者确保管理器会基于 should_continue 为 False 来清理它
                continue  # 跳过本轮循环的剩余部分，让它终止
            try:
                # --- 在规划前记录当前新消息数量 ---
                initial_new_message_count = 0
                if hasattr(self.observation_info, "new_messages_count"):
                    initial_new_message_count = self.observation_info.new_messages_count
                else:
                    logger.warning("ObservationInfo missing 'new_messages_count' before planning.")

                # 使用决策信息来辅助行动规划
                action, reason = await self.action_planner.plan(
                    self.observation_info, self.conversation_info
                )  # 注意：plan 函数内部现在不应再调用 clear_unprocessed_messages

                # --- 规划后检查是否有 *更多* 新消息到达 ---
                current_new_message_count = 0
                if hasattr(self.observation_info, "new_messages_count"):
                    current_new_message_count = self.observation_info.new_messages_count
                else:
                    logger.warning("ObservationInfo missing 'new_messages_count' after planning.")

                if current_new_message_count > initial_new_message_count:
                    # 只有当规划期间消息数量 *增加* 了，才认为需要重新规划
                    logger.info(
                        f"规划期间发现新增消息 ({initial_new_message_count} -> {current_new_message_count})，跳过本次行动，重新规划"
                    )
                    await asyncio.sleep(0.1)  # 短暂延时
                    continue  # 跳过本次行动，重新规划

                # --- 如果没有在规划期间收到更多新消息，则准备执行行动 ---

                # --- 清理未处理消息：移到这里，在执行动作前 ---
                # 只有当确实有新消息被 planner 看到，并且 action 是要处理它们的时候才清理
                if initial_new_message_count > 0 and action == "direct_reply":
                    if hasattr(self.observation_info, "clear_unprocessed_messages"):
                        # 确保 clear_unprocessed_messages 方法存在
                        logger.debug(f"准备执行 direct_reply，清理 {initial_new_message_count} 条规划时已知的新消息。")
                        await self.observation_info.clear_unprocessed_messages()
                        # 手动重置计数器，确保状态一致性（理想情况下 clear 方法会做这个）
                        if hasattr(self.observation_info, "new_messages_count"):
                            self.observation_info.new_messages_count = 0
                    else:
                        logger.error("无法清理未处理消息: ObservationInfo 缺少 clear_unprocessed_messages 方法！")
                        # 这里可能需要考虑是否继续执行 action，或者抛出错误

                # --- 执行行动 ---
                await self._handle_action(action, reason, self.observation_info, self.conversation_info)

                goal_ended = False
                if hasattr(self.conversation_info, "goal_list") and self.conversation_info.goal_list:
                    for goal in self.conversation_info.goal_list:
                        if isinstance(goal, tuple) and len(goal) > 0 and goal[0] == "结束对话":
                            goal_ended = True
                            break
                        elif isinstance(goal, dict) and goal.get("goal") == "结束对话":
                            goal_ended = True
                            break

                if goal_ended:
                    self.should_continue = False
                    logger.info("检测到'结束对话'目标，停止循环。")
                    # break # 可以选择在这里直接跳出循环

            except Exception as loop_err:
                logger.error(f"PFC主循环出错: {loop_err}")
                logger.error(traceback.format_exc())
                # 发生严重错误时可以考虑停止，或者至少等待一下再继续
                await asyncio.sleep(1)  # 发生错误时等待1秒
            # 添加短暂的异步睡眠
            if self.should_continue:  # 只有在还需要继续循环时才 sleep
                await asyncio.sleep(0.1)  # 等待 0.1 秒，给其他任务执行时间

        logger.info(f"PFC 循环结束 for stream_id: {self.stream_id}")  # 添加日志表明循环正常结束

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

        # 记录action历史，先设置为start，完成后再设置为done (这个 update 移到后面执行成功后再做)
        current_action_record = {
            "action": action,
            "plan_reason": reason,  # 使用 plan_reason 存储规划原因
            "status": "start",  # 初始状态为 start
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "final_reason": None,
        }
        conversation_info.done_action.append(current_action_record)
        # 获取刚刚添加记录的索引，方便后面更新状态
        action_index = len(conversation_info.done_action) - 1

        # --- 根据不同的 action 执行 ---
        if action == "direct_reply":
            max_reply_attempts = 3  # 设置最大尝试次数（与 reply_checker.py 中的 max_retries 保持一致或稍大）
            reply_attempt_count = 0
            is_suitable = False
            need_replan = False
            check_reason = "未进行尝试"
            final_reply_to_send = ""

            while reply_attempt_count < max_reply_attempts and not is_suitable:
                reply_attempt_count += 1
                logger.info(f"尝试生成回复 (第 {reply_attempt_count}/{max_reply_attempts} 次)...")
                self.state = ConversationState.GENERATING

                # 1. 生成回复
                self.generated_reply = await self.reply_generator.generate(observation_info, conversation_info)
                logger.info(f"第 {reply_attempt_count} 次生成的回复: {self.generated_reply}")

                # 2. 检查回复
                self.state = ConversationState.CHECKING
                try:
                    current_goal_str = conversation_info.goal_list[0][0] if conversation_info.goal_list else ""
                    # 注意：这里传递的是 reply_attempt_count - 1 作为 retry_count 给 checker
                    is_suitable, check_reason, need_replan = await self.reply_generator.check_reply(
                        reply=self.generated_reply,
                        goal=current_goal_str,
                        chat_history=observation_info.chat_history,
                        chat_history_str=observation_info.chat_history_str,
                        retry_count=reply_attempt_count - 1,  # 传递当前尝试次数（从0开始计数）
                    )
                    logger.info(
                        f"第 {reply_attempt_count} 次检查结果: 合适={is_suitable}, 原因='{check_reason}', 需重新规划={need_replan}"
                    )

                    if is_suitable:
                        final_reply_to_send = self.generated_reply  # 保存合适的回复
                        break  # 回复合适，跳出循环

                    elif need_replan:
                        logger.warning(f"第 {reply_attempt_count} 次检查建议重新规划，停止尝试。原因: {check_reason}")
                        break  # 如果检查器建议重新规划，也停止尝试

                    # 如果不合适但不需要重新规划，循环会继续进行下一次尝试
                except Exception as check_err:
                    logger.error(f"第 {reply_attempt_count} 次调用 ReplyChecker 时出错: {check_err}")
                    check_reason = f"第 {reply_attempt_count} 次检查过程出错: {check_err}"
                    # 如果检查本身出错，可以选择跳出循环或继续尝试
                    # 这里选择跳出循环，避免无限循环在检查错误上
                    break

            # 循环结束，处理最终结果
            if is_suitable:
                # 回复合适且已保存在 final_reply_to_send 中
                # 检查是否有新消息进来 (在所有尝试结束后再检查一次)
                if self._check_new_messages_after_planning():
                    logger.info("生成回复期间收到新消息，取消发送，重新规划行动")
                    conversation_info.done_action[action_index].update(
                        {
                            "status": "recall",
                            "final_reason": f"有新消息，取消发送: {final_reply_to_send}",
                            "time": datetime.datetime.now().strftime("%H:%M:%S"),
                        }
                    )
                    # 这里直接返回，不执行后续发送和wait
                    return

                # 发送合适的回复
                self.generated_reply = final_reply_to_send  # 确保 self.generated_reply 是最终要发送的内容
                await self._send_reply()

                # 更新 action 历史状态为 done
                conversation_info.done_action[action_index].update(
                    {
                        "status": "done",
                        "time": datetime.datetime.now().strftime("%H:%M:%S"),
                    }
                )

            else:
                # 循环结束但没有找到合适的回复（达到最大次数或检查出错/建议重规划）
                logger.warning(f"经过 {reply_attempt_count} 次尝试，未能生成合适的回复。最终原因: {check_reason}")
                conversation_info.done_action[action_index].update(
                    {
                        "status": "recall",  # 标记为 recall 因为没有成功发送
                        "final_reason": f"尝试{reply_attempt_count}次后失败: {check_reason}",
                        "time": datetime.datetime.now().strftime("%H:%M:%S"),
                    }
                )

                # 执行 Wait 操作
                logger.info("由于无法生成合适回复，执行 'wait' 操作...")
                self.state = ConversationState.WAITING
                # 直接调用 wait 方法
                await self.waiter.wait(self.conversation_info)
                # 可以选择添加一条新的 action 记录来表示这个 wait
                wait_action_record = {
                    "action": "wait",
                    "plan_reason": "因 direct_reply 多次尝试失败而执行的后备等待",
                    "status": "done",  # wait 完成后可以认为是 done
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                    "final_reason": None,
                }
                conversation_info.done_action.append(wait_action_record)

        elif action == "fetch_knowledge":
            self.waiter.wait_accumulated_time = 0
            self.state = ConversationState.FETCHING
            knowledge = "TODO:知识"
            topic = "TODO:关键词"
            logger.info(f"假装获取到知识{knowledge}，关键词是: {topic}")
            if knowledge:
                pass  # 简单处理
            # 标记 action 为 done
            conversation_info.done_action[action_index].update(
                {
                    "status": "done",
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                }
            )

        elif action == "rethink_goal":
            self.waiter.wait_accumulated_time = 0
            self.state = ConversationState.RETHINKING
            await self.goal_analyzer.analyze_goal(conversation_info, observation_info)
            # 标记 action 为 done
            conversation_info.done_action[action_index].update(
                {
                    "status": "done",
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                }
            )

        elif action == "listening":
            self.state = ConversationState.LISTENING
            logger.info("倾听对方发言...")
            await self.waiter.wait_listening(conversation_info)
            # listening 和 wait 通常在完成后不需要标记为 done，因为它们是持续状态，
            # 但如果需要记录，可以在 waiter 返回后标记。目前逻辑是 waiter 返回后主循环继续。
            # 为了统一，可以暂时在这里也标记一下（或者都不标记）
            conversation_info.done_action[action_index].update(
                {
                    "status": "done",  # 或 "completed"
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                }
            )

        elif action == "end_conversation":
            self.should_continue = False  # 设置循环停止标志
            logger.info("决定结束对话...")
            # 标记 action 为 done
            conversation_info.done_action[action_index].update(
                {
                    "status": "done",
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                }
            )
            # 这里不需要 return，主循环会在下一轮检查 should_continue

        elif action == "block_and_ignore":
            logger.info("不想再理你了...")
            # 1. 标记对话为暂时忽略
            ignore_duration_seconds = 10 * 60  # 10 分钟
            self.ignore_until_timestamp = time.time() + ignore_duration_seconds
            logger.info(f"将忽略此对话直到: {datetime.datetime.fromtimestamp(self.ignore_until_timestamp)}")
            conversation_info.done_action[action_index].update(
                {
                    "status": "done",  # 或者一个自定义状态，比如 "ignored"
                    "final_reason": "Detected potential harassment, ignoring temporarily.",  # 检测到潜在骚扰，暂时忽略
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                }
            )
            self.state = ConversationState.IGNORED

        else:  # 对应 'wait' 动作
            self.state = ConversationState.WAITING
            logger.info("等待更多信息...")
            await self.waiter.wait(self.conversation_info)
            # 同 listening，可以考虑是否标记状态
            conversation_info.done_action[action_index].update(
                {
                    "status": "done",  # 或 "completed"
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                }
            )

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
            # 外层 try: 捕获发送消息和后续处理中的主要错误
            _current_time = time.time()  # 获取当前时间戳
            reply_content = self.generated_reply  # 获取要发送的内容

            # 发送消息
            await self.direct_sender.send_message(chat_stream=self.chat_stream, content=reply_content)

            # 原有的触发更新和等待代码
            self.chat_observer.trigger_update()
            if not await self.chat_observer.wait_for_update():
                logger.warning("等待 ChatObserver 更新完成超时")

            self.state = ConversationState.ANALYZING  # 更新对话状态

        except Exception as e:
            # 这是外层 try 对应的 except
            logger.error(f"发送消息或更新状态时失败: {str(e)}")
            self.state = ConversationState.ANALYZING  # 出错也要尝试恢复状态
