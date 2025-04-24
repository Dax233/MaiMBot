import time
import asyncio
import datetime
from .message_storage import MongoDBMessageStorage
from ...config.config import global_config
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
        try:
            logger.info(f"为 {self.stream_id} 加载初始聊天记录...")
            storage = MongoDBMessageStorage() # 创建存储实例
            # 获取当前时间点之前最多 N 条消息 (比如 30 条)
            # get_messages_before 返回的是按时间正序排列的列表
            initial_messages = await storage.get_messages_before(
                chat_id=self.stream_id, 
                time_point=time.time(), 
                limit=30 # 加载最近20条作为初始上下文，可以调整
            )
            if initial_messages:
                # 将加载的消息填充到 ObservationInfo 的 chat_history
                self.observation_info.chat_history = initial_messages
                self.observation_info.chat_history_count = len(initial_messages)
                
                # 更新 ObservationInfo 中的时间戳等信息
                last_msg = initial_messages[-1]
                self.observation_info.last_message_time = last_msg.get('time')
                last_user_info = UserInfo.from_dict(last_msg.get("user_info", {}))
                self.observation_info.last_message_sender = last_user_info.user_id
                self.observation_info.last_message_content = last_msg.get("processed_plain_text", "")
                
                # （可选）可以遍历 initial_messages 来设置 last_bot_speak_time 和 last_user_speak_time
                # 这里为了简化，只用了最后一条消息的时间，如果需要精确的发言者时间需要遍历
                
                logger.info(f"成功加载 {len(initial_messages)} 条初始聊天记录。最后一条消息时间: {self.observation_info.last_message_time}")
                
                # 让 ChatObserver 从加载的最后一条消息之后开始同步
                self.chat_observer.last_message_time = self.observation_info.last_message_time
                self.chat_observer.last_message_read = last_msg # 更新 observer 的最后读取记录
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
            try:
                # --- 在规划前记录当前新消息数量 ---
                initial_new_message_count = 0
                if hasattr(self.observation_info, 'new_messages_count'):
                    initial_new_message_count = self.observation_info.new_messages_count
                else:
                     logger.warning("ObservationInfo missing 'new_messages_count' before planning.")

                # 使用决策信息来辅助行动规划
                action, reason = await self.action_planner.plan(self.observation_info, self.conversation_info) # 注意：plan 函数内部现在不应再调用 clear_unprocessed_messages

                # --- 规划后检查是否有 *更多* 新消息到达 ---
                current_new_message_count = 0
                if hasattr(self.observation_info, 'new_messages_count'):
                     current_new_message_count = self.observation_info.new_messages_count
                else:
                     logger.warning("ObservationInfo missing 'new_messages_count' after planning.")

                if current_new_message_count > initial_new_message_count:
                    # 只有当规划期间消息数量 *增加* 了，才认为需要重新规划
                    logger.info(f"规划期间发现新增消息 ({initial_new_message_count} -> {current_new_message_count})，跳过本次行动，重新规划")
                    await asyncio.sleep(0.1) # 短暂延时
                    continue # 跳过本次行动，重新规划

                # --- 如果没有在规划期间收到更多新消息，则准备执行行动 ---

                # --- 清理未处理消息：移到这里，在执行动作前 ---
                # 只有当确实有新消息被 planner 看到，并且 action 是要处理它们的时候才清理
                if initial_new_message_count > 0 and action == "direct_reply":
                    if hasattr(self.observation_info, 'clear_unprocessed_messages'):
                         # 确保 clear_unprocessed_messages 方法存在
                         logger.debug(f"准备执行 direct_reply，清理 {initial_new_message_count} 条规划时已知的新消息。")
                         self.observation_info.clear_unprocessed_messages()
                         # 手动重置计数器，确保状态一致性（理想情况下 clear 方法会做这个）
                         if hasattr(self.observation_info, 'new_messages_count'):
                             self.observation_info.new_messages_count = 0
                    else:
                         logger.error("无法清理未处理消息: ObservationInfo 缺少 clear_unprocessed_messages 方法！")
                         # 这里可能需要考虑是否继续执行 action，或者抛出错误


                # --- 执行行动 ---
                await self._handle_action(action, reason, self.observation_info, self.conversation_info)

                goal_ended = False
                if hasattr(self.conversation_info, 'goal_list') and self.conversation_info.goal_list:
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
                 await asyncio.sleep(1) # 发生错误时等待1秒
            #添加短暂的异步睡眠
            if self.should_continue: # 只有在还需要继续循环时才 sleep
                await asyncio.sleep(0.1) # 等待 0.1 秒，给其他任务执行时间

        logger.info(f"PFC 循环结束 for stream_id: {self.stream_id}") # 添加日志表明循环正常结束    

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
            "plan_reason": reason, #使用 plan_reason 存储规划原因
            "status": "start", # 初始状态为 start
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "final_reason": None 
        }
        conversation_info.done_action.append(current_action_record)
        # 获取刚刚添加记录的索引，方便后面更新状态
        action_index = len(conversation_info.done_action) - 1

        # --- 根据不同的 action 执行 ---
        if action == "direct_reply":
            # --- 这个 if 块内部的所有代码都需要正确缩进 ---
            self.waiter.wait_accumulated_time = 0 # 重置等待时间

            self.state = ConversationState.GENERATING
            # 生成回复
            self.generated_reply = await self.reply_generator.generate(observation_info, conversation_info)
            logger.info(f"生成回复: {self.generated_reply}") # 使用 logger

            # --- 调用 ReplyChecker 检查回复 ---
            is_suitable = False # 先假定不合适，检查通过再改为 True
            check_reason = "检查未执行" # 用不同的变量名存储检查原因
            need_replan = False
            try:
                # 尝试获取当前主要目标
                current_goal_str = conversation_info.goal_list[0][0] if conversation_info.goal_list else ""
                
                # 调用检查器
                is_suitable, check_reason, need_replan = await self.reply_generator.check_reply(
                    reply=self.generated_reply,
                    goal=current_goal_str,
                    chat_history=observation_info.chat_history, # 传入最新的历史记录！
                    retry_count=0 
                )
                logger.info(f"回复检查结果: 合适={is_suitable}, 原因='{check_reason}', 需重新规划={need_replan}")

            except Exception as check_err:
                logger.error(f"调用 ReplyChecker 时出错: {check_err}")
                check_reason = f"检查过程出错: {check_err}" # 记录错误原因
                # is_suitable 保持 False

            # --- 处理检查结果 ---
            if is_suitable:
                # 回复合适，继续执行
                # 检查是否有新消息进来
                if self._check_new_messages_after_planning():
                    logger.info("检查到新消息，取消发送已生成的回复，重新规划行动")
                    # 更新 action 状态为 recall
                    conversation_info.done_action[action_index].update({ 
                        "status": "recall",
                        "reason": f"有新消息，取消发送: {self.generated_reply}", # 更新原因
                        "time": datetime.datetime.now().strftime("%H:%M:%S"),
                    })
                    return None # 退出 _handle_action

                # 发送回复
                await self._send_reply() # 这个函数内部会处理自己的错误

                # 更新 action 历史状态为 done
                conversation_info.done_action[action_index].update({
                    "status": "done",
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                })

            else:
                # 回复不合适
                logger.warning(f"生成的回复被 ReplyChecker 拒绝: '{self.generated_reply}'. 原因: {check_reason}")
                # 更新 action 状态为 recall (因为没执行发送)
                conversation_info.done_action[action_index].update({ 
                    "status": "recall",
                    "final_reason": check_reason,
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                })

                # 如果检查器建议重新规划
                if need_replan:
                    logger.info("ReplyChecker 建议重新规划目标。")
                    # 可选：在此处清空目标列表以强制重新规划
                    # conversation_info.goal_list = [] 
                
                # 注意：不发送消息，也不执行后面的代码

            # --- 之前重复的代码块已被删除 ---

        elif action == "fetch_knowledge":
            self.waiter.wait_accumulated_time = 0
            self.state = ConversationState.FETCHING
            knowledge = "TODO:知识"
            topic = "TODO:关键词"
            logger.info(f"假装获取到知识{knowledge}，关键词是: {topic}")
            if knowledge:
                pass # 简单处理
            # 标记 action 为 done
            conversation_info.done_action[action_index].update({
                "status": "done",
                "time": datetime.datetime.now().strftime("%H:%M:%S"),
            })

        elif action == "rethink_goal":
            self.waiter.wait_accumulated_time = 0
            self.state = ConversationState.RETHINKING
            await self.goal_analyzer.analyze_goal(conversation_info, observation_info)
            # 标记 action 为 done
            conversation_info.done_action[action_index].update({
                "status": "done",
                "time": datetime.datetime.now().strftime("%H:%M:%S"),
            })

        elif action == "listening":
            self.state = ConversationState.LISTENING
            logger.info("倾听对方发言...")
            await self.waiter.wait_listening(conversation_info)
            # listening 和 wait 通常在完成后不需要标记为 done，因为它们是持续状态，
            # 但如果需要记录，可以在 waiter 返回后标记。目前逻辑是 waiter 返回后主循环继续。
            # 为了统一，可以暂时在这里也标记一下（或者都不标记）
            conversation_info.done_action[action_index].update({
                "status": "done", # 或 "completed"
                "time": datetime.datetime.now().strftime("%H:%M:%S"),
            })


        elif action == "end_conversation":
            self.should_continue = False # 设置循环停止标志
            logger.info("决定结束对话...")
            # 标记 action 为 done
            conversation_info.done_action[action_index].update({
                "status": "done",
                "time": datetime.datetime.now().strftime("%H:%M:%S"),
            })
            # 这里不需要 return，主循环会在下一轮检查 should_continue

        else: # 对应 'wait' 动作
            self.state = ConversationState.WAITING
            logger.info("等待更多信息...")
            await self.waiter.wait(self.conversation_info)
            # 同 listening，可以考虑是否标记状态
            conversation_info.done_action[action_index].update({
                "status": "done", # 或 "completed"
                "time": datetime.datetime.now().strftime("%H:%M:%S"),
            })

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
            current_time = time.time() # 获取当前时间戳
            reply_content = self.generated_reply # 获取要发送的内容

            # 发送消息
            await self.direct_sender.send_message(chat_stream=self.chat_stream, content=reply_content)
            logger.info(f"消息已发送: {reply_content}") # 可以在发送后加个日志确认

            # --- 添加的立即更新状态逻辑开始 ---
            try:
                # 内层 try: 专门捕获手动更新状态时可能出现的错误
                # 创建一个代表刚刚发送的消息的字典
                bot_message_info = {
                    "message_id": f"bot_sent_{current_time}", # 创建一个简单的唯一ID
                    "time": current_time,
                    "user_info": UserInfo( # 使用 UserInfo 类构建用户信息
                                user_id=str(global_config.BOT_QQ),
                                user_nickname=global_config.BOT_NICKNAME,
                                platform=self.chat_stream.platform # 从 chat_stream 获取平台信息
                            ).to_dict(), # 转换为字典格式存储
                    "processed_plain_text": reply_content, # 使用发送的内容
                    "detailed_plain_text": f"{int(current_time)},{global_config.BOT_NICKNAME}:{reply_content}", # 构造一个简单的详细文本, 时间戳取整
                    # 可以根据需要添加其他字段，保持与 observation_info.chat_history 中其他消息结构一致
                }

                # 直接更新 ObservationInfo 实例
                if self.observation_info:
                    self.observation_info.chat_history.append(bot_message_info) # 将消息添加到历史记录末尾
                    self.observation_info.last_bot_speak_time = current_time # 更新 Bot 最后发言时间
                    self.observation_info.last_message_time = current_time # 更新最后消息时间
                    logger.debug("已手动将Bot发送的消息添加到 ObservationInfo")
                else:
                    logger.warning("无法手动更新 ObservationInfo：实例不存在")

            except Exception as update_err:
                logger.error(f"手动更新 ObservationInfo 时出错: {update_err}")
            # --- 添加的立即更新状态逻辑结束 ---


            # 原有的触发更新和等待代码
            self.chat_observer.trigger_update()
            if not await self.chat_observer.wait_for_update():
                logger.warning("等待 ChatObserver 更新完成超时")

            self.state = ConversationState.ANALYZING # 更新对话状态

        except Exception as e:
            # 这是外层 try 对应的 except
            logger.error(f"发送消息或更新状态时失败: {str(e)}")
            self.state = ConversationState.ANALYZING # 出错也要尝试恢复状态