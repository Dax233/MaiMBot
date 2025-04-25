import asyncio
import time
import traceback
from typing import List, Optional, Dict, Any
from src.plugins.chat.message import MessageRecv, BaseMessageInfo, MessageThinking, MessageSending
from src.plugins.chat.message import MessageSet, Seg  # Local import needed after move
from src.plugins.chat.chat_stream import ChatStream
from src.plugins.chat.message import UserInfo
from src.plugins.chat.chat_stream import chat_manager
from src.common.logger import get_module_logger, LogConfig, PFC_STYLE_CONFIG  # 引入 DEFAULT_CONFIG
from src.plugins.models.utils_model import LLMRequest
from src.config.config import global_config
from src.plugins.chat.utils_image import image_path_to_base64  # Local import needed after move
from src.plugins.utils.timer_calculater import Timer  # <--- Import Timer
from src.plugins.heartFC_chat.heartFC_generator import HeartFCGenerator
from src.do_tool.tool_use import ToolUser
from ..chat.message_sender import message_manager  # <-- Import the global manager
from src.plugins.emoji_system.emoji_manager import emoji_manager
from src.plugins.utils.json_utils import process_llm_tool_response  # 导入新的JSON工具
from src.heart_flow.sub_mind import SubMind
from src.heart_flow.observation import Observation
from src.plugins.heartFC_chat.heartflow_prompt_builder import global_prompt_manager
# --- End import ---


INITIAL_DURATION = 60.0


# 定义日志配置 (使用 loguru 格式)
interest_log_config = LogConfig(
    console_format=PFC_STYLE_CONFIG["console_format"],  # 使用默认控制台格式
    file_format=PFC_STYLE_CONFIG["file_format"],  # 使用默认文件格式
)
logger = get_module_logger("HeartFCLoop", config=interest_log_config)  # Logger Name Changed


PLANNER_TOOL_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "decide_reply_action",
            "description": "根据当前聊天内容和上下文，决定机器人是否应该回复以及如何回复。",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["no_reply", "text_reply", "emoji_reply"],
                        "description": "决定采取的行动：'no_reply'(不回复), 'text_reply'(文本回复, 可选附带表情) 或 'emoji_reply'(仅表情回复)。",
                    },
                    "reasoning": {"type": "string", "description": "做出此决定的简要理由。"},
                    "emoji_query": {
                        "type": "string",
                        "description": "如果行动是'emoji_reply'，指定表情的主题或概念。如果行动是'text_reply'且希望在文本后追加表情，也在此指定表情主题。",
                    },
                },
                "required": ["action", "reasoning"],
            },
        },
    }
]


class HeartFChatting:
    """
    管理一个连续的Plan-Replier-Sender循环
    用于在特定聊天流中生成回复。
    其生命周期现在由其关联的 SubHeartflow 的 FOCUSED 状态控制。
    """

    def __init__(self, chat_id: str, sub_mind: SubMind, observations: Observation):
        """
        HeartFChatting 初始化函数

        参数:
            chat_id: 聊天流唯一标识符(如stream_id)
        """
        # 基础属性
        self.stream_id: str = chat_id  # 聊天流ID
        self.chat_stream: Optional[ChatStream] = None  # 关联的聊天流
        self.sub_mind: SubMind = sub_mind  # 关联的子思维
        self.observations: Observation = observations  # 关联的观察

        # 初始化状态控制
        self._initialized = False  # 是否已初始化标志
        self._processing_lock = asyncio.Lock()  # 处理锁(确保单次Plan-Replier-Sender周期)

        # 依赖注入存储
        self.gpt_instance = HeartFCGenerator()  # 文本回复生成器
        self.tool_user = ToolUser()  # 工具使用实例

        # LLM规划器配置
        self.planner_llm = LLMRequest(
            model=global_config.llm_normal,
            temperature=global_config.llm_normal["temp"],
            max_tokens=1000,
            request_type="action_planning",  # 用于动作规划
        )

        # 循环控制内部状态
        self._loop_active: bool = False  # 循环是否正在运行
        self._loop_task: Optional[asyncio.Task] = None  # 主循环任务

    def _get_log_prefix(self) -> str:
        """获取日志前缀，包含可读的流名称"""
        stream_name = chat_manager.get_stream_name(self.stream_id) or self.stream_id
        return f"[{stream_name}]"

    async def _initialize(self) -> bool:
        """
        懒初始化以使用提供的标识符解析chat_stream和sub_hf。
        确保实例已准备好处理触发器。
        """
        if self._initialized:
            return True
        log_prefix = self._get_log_prefix()  # 获取前缀
        try:
            self.chat_stream = chat_manager.get_stream(self.stream_id)
            if not self.chat_stream:
                logger.error(f"{log_prefix} 获取ChatStream失败。")
                return False

            self._initialized = True
            logger.info(f"枫感觉到了，激发了HeartFChatting{log_prefix} 初始化成功。")
            return True
        except Exception as e:
            logger.error(f"{log_prefix} 初始化失败: {e}")
            logger.error(traceback.format_exc())
            return False

    async def start(self):
        """
        显式尝试启动 HeartFChatting 的主循环。
        如果循环未激活，则启动循环。
        """
        log_prefix = self._get_log_prefix()
        if not self._initialized:
            if not await self._initialize():
                logger.error(f"{log_prefix} 无法启动循环: 初始化失败。")
                return
        logger.info(f"{log_prefix} 尝试显式启动循环...")
        await self._start_loop_if_needed()

    async def _start_loop_if_needed(self):
        """检查是否需要启动主循环，如果未激活则启动。"""
        log_prefix = self._get_log_prefix()
        should_start_loop = False
        # 直接检查是否激活，无需检查计时器
        if not self._loop_active:
            should_start_loop = True
            self._loop_active = True  # 标记为活动，防止重复启动

        if should_start_loop:
            # 检查是否已有任务在运行（理论上不应该，因为 _loop_active=False）
            if self._loop_task and not self._loop_task.done():
                logger.warning(f"{log_prefix} 发现之前的循环任务仍在运行（不符合预期）。取消旧任务。")
                self._loop_task.cancel()
                try:
                    # 等待旧任务确实被取消
                    await asyncio.wait_for(self._loop_task, timeout=0.5)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass  # 忽略取消或超时错误
                self._loop_task = None  # 清理旧任务引用

            logger.info(f"{log_prefix} 循环未激活，启动主循环...")
            # 创建新的循环任务
            self._loop_task = asyncio.create_task(self._run_pf_loop())
            # 添加完成回调
            self._loop_task.add_done_callback(self._handle_loop_completion)
        # else:
        # logger.trace(f"{log_prefix} 不需要启动循环（已激活）") # 可以取消注释以进行调试

    def _handle_loop_completion(self, task: asyncio.Task):
        """当 _run_pf_loop 任务完成时执行的回调。"""
        log_prefix = self._get_log_prefix()
        try:
            exception = task.exception()
            if exception:
                logger.error(f"{log_prefix} HeartFChatting: 枫脱离了聊天(异常): {exception}")
                logger.error(traceback.format_exc())  # Log full traceback for exceptions
            else:
                # Loop completing normally now means it was cancelled/shutdown externally
                logger.info(f"{log_prefix} HeartFChatting: 枫脱离了聊天 (外部停止)")
        except asyncio.CancelledError:
            logger.info(f"{log_prefix} HeartFChatting: 枫脱离了聊天(任务取消)")
        finally:
            self._loop_active = False
            self._loop_task = None
            if self._processing_lock.locked():
                logger.warning(f"{log_prefix} HeartFChatting: 处理锁在循环结束时仍被锁定，强制释放。")
                self._processing_lock.release()

    async def _run_pf_loop(self):
        """
        主循环，持续进行计划并可能回复消息，直到被外部取消。
        管理每个循环周期的处理锁。
        """
        log_prefix = self._get_log_prefix()
        logger.info(f"{log_prefix} HeartFChatting: 枫打算好好聊聊 (进入专注模式)")
        try:
            thinking_id = ""
            while True:  # Loop indefinitely until cancelled
                cycle_timers = {}  # <--- Initialize timers dict for this cycle

                # Access MessageManager directly
                if message_manager.check_if_sending_message_exist(self.stream_id, thinking_id):
                    # logger.info(f"{log_prefix} HeartFChatting: 麦麦还在发消息，等会再规划")
                    await asyncio.sleep(1)
                    continue
                else:
                    # logger.info(f"{log_prefix} HeartFChatting: 麦麦不发消息了，开始规划")
                    pass

                # 记录循环周期开始时间，用于计时和休眠计算
                loop_cycle_start_time = time.monotonic()
                action_taken_this_cycle = False
                acquired_lock = False
                planner_start_db_time = 0.0  # 初始化

                try:
                    with Timer("Total Cycle", cycle_timers) as _total_timer:  # <--- Start total cycle timer
                        # Use try_acquire pattern or timeout?
                        await self._processing_lock.acquire()
                        acquired_lock = True
                        # logger.debug(f"{log_prefix} HeartFChatting: 循环获取到处理锁")

                        # 在规划前记录数据库时间戳
                        planner_start_db_time = time.time()

                        # --- Planner --- #
                        planner_result = {}
                        with Timer("Planner", cycle_timers):  # <--- Start Planner timer
                            planner_result = await self._planner()
                        action = planner_result.get("action", "error")
                        reasoning = planner_result.get("reasoning", "Planner did not provide reasoning.")
                        emoji_query = planner_result.get("emoji_query", "")
                        llm_error = planner_result.get("llm_error", False)

                        if llm_error:
                            logger.error(f"{log_prefix} Planner LLM 失败，跳过本周期回复尝试。理由: {reasoning}")
                            # Optionally add a longer sleep?
                            action_taken_this_cycle = False  # Ensure no action is counted
                            # Continue to sleep logic

                        elif action == "text_reply":
                            logger.debug(f"{log_prefix} HeartFChatting: 枫决定回复文本. 理由: {reasoning}")
                            action_taken_this_cycle = True
                            anchor_message = await self._get_anchor_message()
                            if not anchor_message:
                                logger.error(f"{log_prefix} 循环: 无法获取锚点消息用于回复. 跳过周期.")
                            else:
                                # --- Create Thinking Message (Moved) ---
                                thinking_id = await self._create_thinking_message(anchor_message)
                                if not thinking_id:
                                    logger.error(f"{log_prefix} 循环: 无法创建思考ID. 跳过周期.")
                                else:
                                    replier_result = None
                                    try:
                                        # --- Replier Work --- #
                                        with Timer("Replier", cycle_timers):  # <--- Start Replier timer
                                            replier_result = await self._replier_work(
                                                anchor_message=anchor_message,
                                                thinking_id=thinking_id,
                                                reason=reasoning,
                                            )
                                    except Exception as e_replier:
                                        logger.error(f"{log_prefix} 循环: 回复器工作失败: {e_replier}")
                                        # self._cleanup_thinking_message(thinking_id) <-- Remove cleanup call

                                    if replier_result:
                                        # --- Sender Work --- #
                                        try:
                                            with Timer("Sender", cycle_timers):  # <--- Start Sender timer
                                                await self._sender(
                                                    thinking_id=thinking_id,
                                                    anchor_message=anchor_message,
                                                    response_set=replier_result,
                                                    send_emoji=emoji_query,
                                                )
                                            # logger.info(f"{log_prefix} 循环: 发送器完成成功.")
                                        except Exception as e_sender:
                                            logger.error(f"{log_prefix} 循环: 发送器失败: {e_sender}")
                                            # _sender should handle cleanup, but double check
                                            # self._cleanup_thinking_message(thinking_id) <-- Remove cleanup call
                                    else:
                                        logger.warning(f"{log_prefix} 循环: 回复器未产生结果. 跳过发送.")
                                        # self._cleanup_thinking_message(thinking_id) <-- Remove cleanup call
                        elif action == "emoji_reply":
                            logger.info(
                                f"{log_prefix} HeartFChatting: 枫决定回复表情 ('{emoji_query}'). 理由: {reasoning}"
                            )
                            action_taken_this_cycle = True
                            anchor = await self._get_anchor_message()
                            if anchor:
                                try:
                                    # --- Handle Emoji (Moved) --- #
                                    with Timer("Emoji Handler", cycle_timers):  # <--- Start Emoji timer
                                        await self._handle_emoji(anchor, [], emoji_query)
                                except Exception as e_emoji:
                                    logger.error(f"{log_prefix} 循环: 发送表情失败: {e_emoji}")
                            else:
                                logger.warning(f"{log_prefix} 循环: 无法发送表情, 无法获取锚点.")
                            action_taken_this_cycle = True  # 即使发送失败，Planner 也决策了动作

                        elif action == "no_reply":
                            logger.info(f"{log_prefix} HeartFChatting: 枫决定不回复. 原因: {reasoning}")
                            action_taken_this_cycle = False  # 标记为未执行动作
                            # --- 新增：等待新消息 ---
                            logger.debug(f"{log_prefix} HeartFChatting: 开始等待新消息 (自 {planner_start_db_time})...")
                            observation = None

                            observation = self.observations[0]

                            if observation:
                                with Timer("Wait New Msg", cycle_timers):  # <--- Start Wait timer
                                    wait_start_time = time.monotonic()
                                    while True:
                                        # 检查是否有新消息
                                        has_new = await observation.has_new_messages_since(planner_start_db_time)
                                        if has_new:
                                            logger.info(f"{log_prefix} HeartFChatting: 检测到新消息，结束等待。")
                                            break  # 收到新消息，退出等待

                                        # 检查等待是否超时（例如，防止无限等待）
                                        if time.monotonic() - wait_start_time > 60:  # 等待60秒示例
                                            logger.warning(f"{log_prefix} HeartFChatting: 等待新消息超时（60秒）。")
                                            break  # 超时退出

                                        # 等待一段时间再检查
                                        try:
                                            await asyncio.sleep(1.5)  # 检查间隔
                                        except asyncio.CancelledError:
                                            logger.info(f"{log_prefix} 等待新消息的 sleep 被中断。")
                                            raise  # 重新抛出取消错误，以便外层循环处理
                            else:
                                logger.warning(
                                    f"{log_prefix} HeartFChatting: 无法获取 Observation 实例，无法等待新消息。"
                                )
                            # --- 等待结束 ---

                        elif action == "error":  # Action specifically set to error by planner
                            logger.error(f"{log_prefix} HeartFChatting: Planner返回错误状态. 原因: {reasoning}")
                            action_taken_this_cycle = False

                        else:  # Unknown action from planner
                            logger.warning(
                                f"{log_prefix} HeartFChatting: Planner返回未知动作 '{action}'. 原因: {reasoning}"
                            )
                            action_taken_this_cycle = False

                    # --- Print Timer Results --- #
                    if cycle_timers:  # 先检查cycle_timers是否非空
                        timer_strings = []
                        for name, elapsed in cycle_timers.items():
                            # 直接格式化存储在字典中的浮点数 elapsed
                            formatted_time = f"{elapsed * 1000:.2f}毫秒" if elapsed < 1 else f"{elapsed:.2f}秒"
                            timer_strings.append(f"{name}: {formatted_time}")

                        if timer_strings:  # 如果有有效计时器数据才打印
                            logger.debug(f"{log_prefix} 该次决策耗时: {'; '.join(timer_strings)}")

                    # --- Timer Decrement Removed --- #
                    cycle_duration = time.monotonic() - loop_cycle_start_time

                except Exception as e_cycle:
                    logger.error(f"{log_prefix} 循环周期执行时发生错误: {e_cycle}")
                    logger.error(traceback.format_exc())
                    if acquired_lock and self._processing_lock.locked():
                        self._processing_lock.release()
                        acquired_lock = False
                        logger.warning(f"{log_prefix} 由于循环周期中的错误释放了处理锁.")

                finally:
                    if acquired_lock:
                        self._processing_lock.release()
                        # logger.trace(f"{log_prefix} 循环释放了处理锁.") # Reduce noise

                if cycle_duration > 0.1:
                    logger.debug(f"{log_prefix} HeartFChatting: 周期耗时 {cycle_duration:.2f}s.")

                # --- Delay --- #
                try:
                    sleep_duration = 0.0
                    if not action_taken_this_cycle and cycle_duration < 1.5:
                        sleep_duration = 1.5 - cycle_duration
                    elif cycle_duration < 0.2:  # Keep minimal sleep even after action
                        sleep_duration = 0.2

                    if sleep_duration > 0:
                        # logger.debug(f"{log_prefix} Sleeping for {sleep_duration:.2f}s")
                        await asyncio.sleep(sleep_duration)

                except asyncio.CancelledError:
                    logger.info(f"{log_prefix} Sleep interrupted, loop likely cancelling.")
                    break  # Exit loop immediately on cancellation

        except asyncio.CancelledError:
            logger.info(f"{log_prefix} HeartFChatting: 枫的聊天主循环被取消了")
        except Exception as e_loop_outer:
            logger.error(f"{log_prefix} HeartFChatting: 枫的聊天主循环意外出错: {e_loop_outer}")
            logger.error(traceback.format_exc())
        finally:
            # State reset is primarily handled by _handle_loop_completion callback
            logger.info(f"{log_prefix} HeartFChatting: 枫的聊天主循环结束。")

    async def _planner(self) -> Dict[str, Any]:
        """
        规划器 (Planner): 使用LLM根据上下文决定是否和如何回复。
        """
        log_prefix = self._get_log_prefix()
        observed_messages: List[dict] = []

        current_mind: Optional[str] = None
        llm_error = False

        try:
            observation = self.observations[0]
            await observation.observe()
            observed_messages = observation.talking_message
            observed_messages_str = observation.talking_message_str
        except Exception as e:
            logger.error(f"{log_prefix}[Planner] 获取观察信息时出错: {e}")

        try:
            current_mind, _past_mind = await self.sub_mind.do_thinking_before_reply()
        except Exception as e_subhf:
            logger.error(f"{log_prefix}[Planner] SubHeartflow 思考失败: {e_subhf}")
            current_mind = "[思考时出错]"

        # --- 使用 LLM 进行决策 --- #
        action = "no_reply"  # 默认动作
        emoji_query = ""  # 默认表情查询
        reasoning = "默认决策或获取决策失败"
        llm_error = False  # LLM错误标志

        try:
            prompt = await self._build_planner_prompt(
                observed_messages_str, current_mind, self.sub_mind.structured_info
            )
            payload = {
                "model": self.planner_llm.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "tools": PLANNER_TOOL_DEFINITION,
                "tool_choice": {"type": "function", "function": {"name": "decide_reply_action"}},
            }

            # 执行LLM请求
            try:
                response = await self.planner_llm._execute_request(
                    endpoint="/chat/completions", payload=payload, prompt=prompt
                )
            except Exception as req_e:
                logger.error(f"{log_prefix}[Planner] LLM请求执行失败: {req_e}")
                return {
                    "action": "error",
                    "reasoning": f"LLM请求执行失败: {req_e}",
                    "emoji_query": "",
                    "current_mind": current_mind,
                    "observed_messages": observed_messages,
                    "llm_error": True,
                }

            # 使用辅助函数处理工具调用响应
            success, arguments, error_msg = process_llm_tool_response(
                response, expected_tool_name="decide_reply_action", log_prefix=f"{log_prefix}[Planner] "
            )

            if success:
                # 提取决策参数
                action = arguments.get("action", "no_reply")
                reasoning = arguments.get("reasoning", "未提供理由")
                emoji_query = arguments.get("emoji_query", "")

                # 记录决策结果
                logger.debug(f"{log_prefix}[Planner] 决策结果: {action}, 理由: {reasoning}, 表情查询: '{emoji_query}'")
            else:
                # 处理工具调用失败
                logger.warning(f"{log_prefix}[Planner] {error_msg}")
                action = "error"
                reasoning = error_msg
                llm_error = True

        except Exception as llm_e:
            logger.error(f"{log_prefix}[Planner] Planner LLM处理过程中出错: {llm_e}")
            logger.error(traceback.format_exc())  # 记录完整堆栈以便调试
            action = "error"
            reasoning = f"LLM处理失败: {llm_e}"
            llm_error = True
        # --- 结束 LLM 决策 --- #

        return {
            "action": action,
            "reasoning": reasoning,
            "emoji_query": emoji_query,
            "current_mind": current_mind,
            "observed_messages": observed_messages,
            "llm_error": llm_error,
        }

    async def _get_anchor_message(self) -> Optional[MessageRecv]:
        """
        重构观察到的最后一条消息作为回复的锚点，
        如果重构失败或观察为空，则创建一个占位符。
        """

        try:
            placeholder_id = f"mid_pf_{int(time.time() * 1000)}"
            placeholder_user = UserInfo(
                user_id="system_trigger", user_nickname="System Trigger", platform=self.chat_stream.platform
            )
            placeholder_msg_info = BaseMessageInfo(
                message_id=placeholder_id,
                platform=self.chat_stream.platform,
                group_info=self.chat_stream.group_info,
                user_info=placeholder_user,
                time=time.time(),
            )
            placeholder_msg_dict = {
                "message_info": placeholder_msg_info.to_dict(),
                "processed_plain_text": "[System Trigger Context]",
                "raw_message": "",
                "time": placeholder_msg_info.time,
            }
            anchor_message = MessageRecv(placeholder_msg_dict)
            anchor_message.update_chat_stream(self.chat_stream)
            logger.info(
                f"{self._get_log_prefix()} Created placeholder anchor message: ID={anchor_message.message_info.message_id}"
            )
            return anchor_message

        except Exception as e:
            logger.error(f"{self._get_log_prefix()} Error getting/creating anchor message: {e}")
            logger.error(traceback.format_exc())
            return None

    # --- 发送器 (Sender) --- #
    async def _sender(
        self,
        thinking_id: str,
        anchor_message: MessageRecv,
        response_set: List[str],
        send_emoji: str,  # Emoji query decided by planner or tools
    ):
        """
        发送器 (Sender): 使用本类的方法发送生成的回复。
        处理相关的操作，如发送表情和更新关系。
        """
        log_prefix = self._get_log_prefix()

        first_bot_msg: Optional[MessageSending] = None
        # 尝试发送回复消息
        first_bot_msg = await self._send_response_messages(anchor_message, response_set, thinking_id)
        if first_bot_msg:
            # --- 处理关联表情(如果指定) --- #
            if send_emoji:
                logger.info(f"{log_prefix}[Sender-{thinking_id}] 正在发送关联表情: '{send_emoji}'")
                # 优先使用first_bot_msg作为锚点，否则回退到原始锚点
                emoji_anchor = first_bot_msg if first_bot_msg else anchor_message
                await self._handle_emoji(emoji_anchor, response_set, send_emoji)

        else:
            # logger.warning(f"{log_prefix}[Sender-{thinking_id}] 发送回复失败(_send_response_messages返回None)。思考消息{thinking_id}可能已被移除。")
            # 无需清理，因为_send_response_messages返回None意味着已处理/已删除
            raise RuntimeError("发送回复失败，_send_response_messages返回None")

    async def shutdown(self):
        """优雅关闭HeartFChatting实例，取消活动循环任务"""
        log_prefix = self._get_log_prefix()
        logger.info(f"{log_prefix} 正在关闭HeartFChatting...")

        # 取消循环任务
        if self._loop_task and not self._loop_task.done():
            logger.info(f"{log_prefix} 正在取消HeartFChatting循环任务")
            self._loop_task.cancel()
            try:
                await asyncio.wait_for(self._loop_task, timeout=1.0)
                logger.info(f"{log_prefix} HeartFChatting循环任务已取消")
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception as e:
                logger.error(f"{log_prefix} 取消循环任务出错: {e}")
        else:
            logger.info(f"{log_prefix} 没有活动的HeartFChatting循环任务")

        # 清理状态
        self._loop_active = False
        self._loop_task = None
        if self._processing_lock.locked():
            self._processing_lock.release()
            logger.warning(f"{log_prefix} 已释放处理锁")

        logger.info(f"{log_prefix} HeartFChatting关闭完成")

    async def _build_planner_prompt(
        self, observed_messages_str: str, current_mind: Optional[str], structured_info: Dict[str, Any]
    ) -> str:
        """构建 Planner LLM 的提示词"""

        # 准备结构化信息块
        structured_info_block = ""
        if structured_info:
            structured_info_block = f"以下是一些额外的信息：\n{structured_info}\n"

        # 准备聊天内容块
        chat_content_block = ""
        if observed_messages_str:
            chat_content_block = "观察到的最新聊天内容如下 (最近的消息在最后)：\n---\n"
            chat_content_block += observed_messages_str
            chat_content_block += "\n---"
        else:
            chat_content_block = "当前没有观察到新的聊天内容。\n"

        # 准备当前思维块
        current_mind_block = ""
        if current_mind:
            current_mind_block = f"\n---\n{current_mind}\n---\n\n"
        else:
            current_mind_block = " [没有特别的想法] \n\n"

        # 获取提示词模板并填充数据
        prompt = (await global_prompt_manager.get_prompt_async("planner_prompt")).format(
            bot_name=global_config.BOT_NICKNAME,
            structured_info_block=structured_info_block,
            chat_content_block=chat_content_block,
            current_mind_block=current_mind_block,
        )

        return prompt

    # --- 回复器 (Replier) 的定义 --- #
    async def _replier_work(
        self,
        reason: str,
        anchor_message: MessageRecv,
        thinking_id: str,
    ) -> Optional[List[str]]:
        """
        回复器 (Replier): 核心逻辑用于生成回复。
        """
        log_prefix = self._get_log_prefix()
        response_set: Optional[List[str]] = None
        try:
            response_set = await self.gpt_instance.generate_response(
                structured_info=self.sub_mind.structured_info,
                current_mind_info=self.sub_mind.current_mind,
                reason=reason,
                message=anchor_message,  # Pass anchor_message positionally (matches 'message' parameter)
                thinking_id=thinking_id,  # Pass thinking_id positionally
            )

            if not response_set:
                logger.warning(f"{log_prefix}[Replier-{thinking_id}] LLM生成了一个空回复集。")
                return None

            return response_set

        except Exception as e:
            logger.error(f"{log_prefix}[Replier-{thinking_id}] Unexpected error in replier_work: {e}")
            logger.error(traceback.format_exc())
            return None

    # --- Methods moved from HeartFCController start ---
    async def _create_thinking_message(self, anchor_message: Optional[MessageRecv]) -> Optional[str]:
        """创建思考消息 (尝试锚定到 anchor_message)"""
        if not anchor_message or not anchor_message.chat_stream:
            logger.error(f"{self._get_log_prefix()} 无法创建思考消息，缺少有效的锚点消息或聊天流。")
            return None

        chat = anchor_message.chat_stream
        messageinfo = anchor_message.message_info
        bot_user_info = UserInfo(
            user_id=global_config.BOT_QQ,
            user_nickname=global_config.BOT_NICKNAME,
            platform=messageinfo.platform,
        )

        thinking_time_point = round(time.time(), 2)
        thinking_id = "mt" + str(thinking_time_point)
        thinking_message = MessageThinking(
            message_id=thinking_id,
            chat_stream=chat,
            bot_user_info=bot_user_info,
            reply=anchor_message,  # 回复的是锚点消息
            thinking_start_time=thinking_time_point,
        )
        # Access MessageManager directly
        await message_manager.add_message(thinking_message)
        return thinking_id

    async def _send_response_messages(
        self, anchor_message: Optional[MessageRecv], response_set: List[str], thinking_id: str
    ) -> Optional[MessageSending]:
        """发送回复消息 (尝试锚定到 anchor_message)"""
        if not anchor_message or not anchor_message.chat_stream:
            logger.error(f"{self._get_log_prefix()} 无法发送回复，缺少有效的锚点消息或聊天流。")
            return None

        chat = anchor_message.chat_stream
        container = await message_manager.get_container(chat.stream_id)
        thinking_message = None

        # 移除思考消息
        for msg in container.messages[:]:  # Iterate over a copy
            if isinstance(msg, MessageThinking) and msg.message_info.message_id == thinking_id:
                thinking_message = msg
                container.messages.remove(msg)  # Remove the message directly here
                logger.debug(f"{self._get_log_prefix()} Removed thinking message {thinking_id} via iteration.")
                break

        if not thinking_message:
            stream_name = chat_manager.get_stream_name(chat.stream_id) or chat.stream_id  # 获取流名称
            logger.warning(f"[{stream_name}] {thinking_id}，思考太久了，超时被移除")
            return None

        thinking_start_time = thinking_message.thinking_start_time
        message_set = MessageSet(chat, thinking_id)
        mark_head = False
        first_bot_msg = None
        bot_user_info = UserInfo(
            user_id=global_config.BOT_QQ,
            user_nickname=global_config.BOT_NICKNAME,
            platform=anchor_message.message_info.platform,
        )
        for msg_text in response_set:
            message_segment = Seg(type="text", data=msg_text)
            bot_message = MessageSending(
                message_id=thinking_id,  # 使用 thinking_id 作为批次标识
                chat_stream=chat,
                bot_user_info=bot_user_info,
                sender_info=anchor_message.message_info.user_info,  # 发送给锚点消息的用户
                message_segment=message_segment,
                reply=anchor_message,  # 回复锚点消息
                is_head=not mark_head,
                is_emoji=False,
                thinking_start_time=thinking_start_time,
            )
            if not mark_head:
                mark_head = True
                first_bot_msg = bot_message
            message_set.add_message(bot_message)

        # Access MessageManager directly
        await message_manager.add_message(message_set)
        return first_bot_msg

    async def _handle_emoji(self, anchor_message: Optional[MessageRecv], response_set: List[str], send_emoji: str = ""):
        """处理表情包 (尝试锚定到 anchor_message)"""

        if not anchor_message or not anchor_message.chat_stream:
            logger.error(f"{self._get_log_prefix()} 无法处理表情包，缺少有效的锚点消息或聊天流。")
            return

        chat = anchor_message.chat_stream

        if send_emoji:
            emoji_raw = await emoji_manager.get_emoji_for_text(send_emoji)
        else:
            emoji_text_source = "".join(response_set) if response_set else ""
            emoji_raw = await emoji_manager.get_emoji_for_text(emoji_text_source)

        if emoji_raw:
            emoji_path, _description = emoji_raw
            emoji_cq = image_path_to_base64(emoji_path)
            thinking_time_point = round(time.time(), 2)
            message_segment = Seg(type="emoji", data=emoji_cq)
            bot_user_info = UserInfo(
                user_id=global_config.BOT_QQ,
                user_nickname=global_config.BOT_NICKNAME,
                platform=anchor_message.message_info.platform,
            )
            bot_message = MessageSending(
                message_id="me" + str(thinking_time_point),  # 使用不同的 ID 前缀?
                chat_stream=chat,
                bot_user_info=bot_user_info,
                sender_info=anchor_message.message_info.user_info,
                message_segment=message_segment,
                reply=anchor_message,  # 回复锚点消息
                is_head=False,
                is_emoji=True,
            )
            # Access MessageManager directly
            await message_manager.add_message(bot_message)
