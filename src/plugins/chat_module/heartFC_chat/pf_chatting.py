import asyncio
import time
import traceback
from typing import List, Optional, Dict, Any, TYPE_CHECKING
import json

from ....config.config import global_config
from ...chat.message import MessageRecv, BaseMessageInfo, MessageThinking, MessageSending
from ...chat.chat_stream import ChatStream
from ...message import UserInfo
from src.heart_flow.heartflow import heartflow, SubHeartflow
from src.plugins.chat.chat_stream import chat_manager
from .messagesender import MessageManager
from src.common.logger import get_module_logger, LogConfig, DEFAULT_CONFIG  # 引入 DEFAULT_CONFIG
from src.plugins.models.utils_model import LLMRequest

# 定义日志配置 (使用 loguru 格式)
interest_log_config = LogConfig(
    console_format=DEFAULT_CONFIG["console_format"],  # 使用默认控制台格式
    file_format=DEFAULT_CONFIG["file_format"],  # 使用默认文件格式
)
logger = get_module_logger("PFChattingLoop", config=interest_log_config)  # Logger Name Changed


# Forward declaration for type hinting
if TYPE_CHECKING:
    from .heartFC_chat import HeartFC_Chat

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
                        "description": "决定采取的行动：'no_reply'(不回复), 'text_reply'(文本回复) 或 'emoji_reply'(表情回复)。",
                    },
                    "reasoning": {"type": "string", "description": "做出此决定的简要理由。"},
                    "emoji_query": {
                        "type": "string",
                        "description": '如果行动是\'emoji_reply\',则指定表情的主题或概念（例如，"开心"、"困惑"）。仅在需要表情回复时提供。',
                    },
                },
                "required": ["action", "reasoning"],  # 强制要求提供行动和理由
            },
        },
    }
]


class PFChatting:
    """
    Manages a continuous Plan-Filter-Check (now Plan-Replier-Sender) loop
    for generating replies within a specific chat stream, controlled by a timer.
    The loop runs as long as the timer > 0.
    """

    def __init__(self, chat_id: str, heartfc_chat_instance: "HeartFC_Chat"):
        """
        初始化PFChatting实例。

        Args:
            chat_id: The identifier for the chat stream (e.g., stream_id).
            heartfc_chat_instance: 访问共享资源和方法的主HeartFC_Chat实例。
        """
        self.heartfc_chat = heartfc_chat_instance  # 访问logger, gpt, tool_user, _send_response_messages等。
        self.stream_id: str = chat_id
        self.chat_stream: Optional[ChatStream] = None
        self.sub_hf: Optional[SubHeartflow] = None
        self._initialized = False
        self._init_lock = asyncio.Lock()  # Ensure initialization happens only once
        self._processing_lock = asyncio.Lock()  # 确保只有一个 Plan-Replier-Sender 周期在运行
        self._timer_lock = asyncio.Lock()  # 用于安全更新计时器

        self.planner_llm = LLMRequest(
            model=global_config.llm_normal,
            temperature=global_config.llm_normal["temp"],
            max_tokens=1000,
            request_type="action_planning",
        )

        # Internal state for loop control
        self._loop_timer: float = 0.0  # Remaining time for the loop in seconds
        self._loop_active: bool = False  # Is the loop currently running?
        self._loop_task: Optional[asyncio.Task] = None  # Stores the main loop task
        self._trigger_count_this_activation: int = 0  # Counts triggers within an active period
        self._initial_duration: float = 30.0  # 首次触发增加的时间
        self._last_added_duration: float = self._initial_duration  # <--- 新增：存储上次增加的时间

        # Removed pending_replies as processing is now serial within the loop
        # self.pending_replies: Dict[str, PendingReply] = {}

    def _get_log_prefix(self) -> str:
        """获取日志前缀，包含可读的流名称"""
        stream_name = chat_manager.get_stream_name(self.stream_id) or self.stream_id
        return f"[{stream_name}]"

    async def _initialize(self) -> bool:
        """
        Lazy initialization to resolve chat_stream and sub_hf using the provided identifier.
        Ensures the instance is ready to handle triggers.
        """
        async with self._init_lock:
            if self._initialized:
                return True
            log_prefix = self._get_log_prefix()  # 获取前缀
            try:
                self.chat_stream = chat_manager.get_stream(self.stream_id)

                if not self.chat_stream:
                    logger.error(f"{log_prefix} 获取ChatStream失败。")
                    return False

                # 子心流(SubHeartflow)可能初始不存在但后续会被创建
                # 在需要它的方法中应优雅处理其可能缺失的情况
                self.sub_hf = heartflow.get_subheartflow(self.stream_id)
                if not self.sub_hf:
                    logger.warning(f"{log_prefix} 获取SubHeartflow失败。一些功能可能受限。")
                    # 决定是否继续初始化。目前允许初始化。

                self._initialized = True
                logger.info(f"麦麦感觉到了，激发了PFChatting{log_prefix} 初始化成功。")
                return True
            except Exception as e:
                logger.error(f"{log_prefix} 初始化失败: {e}")
                logger.error(traceback.format_exc())
                return False

    async def add_time(self):
        """
        Adds time to the loop timer with decay and starts the loop if it's not active.
        First trigger adds initial duration, subsequent triggers add 50% of the previous addition.
        """
        log_prefix = self._get_log_prefix()
        if not self._initialized:
            if not await self._initialize():
                logger.error(f"{log_prefix} 无法添加时间: 未初始化。")
                return

        async with self._timer_lock:
            duration_to_add: float = 0.0

            if not self._loop_active:  # First trigger for this activation cycle
                duration_to_add = self._initial_duration  # 使用初始值
                self._last_added_duration = duration_to_add  # 更新上次增加的值
                self._trigger_count_this_activation = 1  # Start counting
                logger.info(f"{log_prefix} First trigger in activation. Adding {duration_to_add:.2f}s.")
            else:  # Loop is already active, apply 50% reduction
                self._trigger_count_this_activation += 1
                duration_to_add = self._last_added_duration * 0.5
                self._last_added_duration = duration_to_add  # 更新上次增加的值
                logger.info(
                    f"{log_prefix} Trigger #{self._trigger_count_this_activation}. Adding {duration_to_add:.2f}s (50% of previous). Timer was {self._loop_timer:.1f}s."
                )

            # 添加计算出的时间
            new_timer_value = self._loop_timer + duration_to_add
            self._loop_timer = max(0, new_timer_value)
            logger.info(f"{log_prefix} Timer is now {self._loop_timer:.1f}s.")

            # Start the loop if it wasn't active and timer is positive
            if not self._loop_active and self._loop_timer > 0:
                logger.info(f"{log_prefix} Timer > 0 and loop not active. Starting PF loop.")
                self._loop_active = True
                if self._loop_task and not self._loop_task.done():
                    logger.warning(f"{log_prefix} Found existing loop task unexpectedly during start. Cancelling it.")
                    self._loop_task.cancel()

                self._loop_task = asyncio.create_task(self._run_pf_loop())
                self._loop_task.add_done_callback(self._handle_loop_completion)
            elif self._loop_active:
                logger.debug(f"{log_prefix} Loop already active. Timer extended.")

    def _handle_loop_completion(self, task: asyncio.Task):
        """当 _run_pf_loop 任务完成时执行的回调。"""
        log_prefix = self._get_log_prefix()
        try:
            # Check if the task raised an exception
            exception = task.exception()
            if exception:
                logger.error(f"{log_prefix} PFChatting: 麦麦脱离了聊天(异常)")
                logger.error(traceback.format_exc())
            else:
                logger.debug(f"{log_prefix} PFChatting: 麦麦脱离了聊天")
        except asyncio.CancelledError:
            logger.info(f"{log_prefix} PFChatting: 麦麦脱离了聊天(异常取消)")
        finally:
            # Reset state regardless of how the task finished
            self._loop_active = False
            self._loop_task = None
            self._last_added_duration = self._initial_duration  # <--- 重置下次首次触发的增加时间
            self._trigger_count_this_activation = 0  # 重置计数器
            # Ensure lock is released if the loop somehow exited while holding it
            if self._processing_lock.locked():
                logger.warning(f"{log_prefix} PFChatting: 锁没有正常释放")
                self._processing_lock.release()

    async def _run_pf_loop(self):
        """
        主循环，当计时器>0时持续进行计划并可能回复消息
        管理每个循环周期的处理锁
        """
        logger.info(f"{self._get_log_prefix()} PFChatting: 麦麦打算好好聊聊")
        try:
            while True:
                # 使用计时器锁安全地检查当前计时器值
                async with self._timer_lock:
                    current_timer = self._loop_timer
                    if current_timer <= 0:
                        logger.info(
                            f"{self._get_log_prefix()} PFChatting: 聊太久了，麦麦打算休息一下(已经聊了{current_timer:.1f}秒)，退出PFChatting"
                        )
                        break  # 退出条件：计时器到期

                # 记录循环开始时间
                loop_cycle_start_time = time.monotonic()
                # 标记本周期是否执行了操作
                action_taken_this_cycle = False

                # 获取处理锁，确保每个计划-回复-发送周期独占执行
                acquired_lock = False
                try:
                    await self._processing_lock.acquire()
                    acquired_lock = True
                    # logger.debug(f"{self._get_log_prefix()} PFChatting: 循环获取到处理锁")

                    # --- Planner ---
                    # Planner decides action, reasoning, emoji_query, etc.
                    planner_result = await self._planner()  # Modify planner to return decision dict
                    action = planner_result.get("action", "error")
                    reasoning = planner_result.get("reasoning", "Planner did not provide reasoning.")
                    emoji_query = planner_result.get("emoji_query", "")
                    current_mind = planner_result.get("current_mind", "[Mind unavailable]")
                    send_emoji_from_tools = planner_result.get("send_emoji_from_tools", "")
                    observed_messages = planner_result.get("observed_messages", [])  # Planner needs to return this

                    if action == "text_reply":
                        logger.info(f"{self._get_log_prefix()} PFChatting: 麦麦决定回复文本.")
                        action_taken_this_cycle = True
                        # --- 回复器 ---
                        anchor_message = await self._get_anchor_message(observed_messages)
                        if not anchor_message:
                            logger.error(f"{self._get_log_prefix()} 循环: 无法获取锚点消息用于回复. 跳过周期.")
                        else:
                            thinking_id = await self.heartfc_chat._create_thinking_message(anchor_message)
                            if not thinking_id:
                                logger.error(f"{self._get_log_prefix()} 循环: 无法创建思考ID. 跳过周期.")
                            else:
                                replier_result = None
                                try:
                                    # 直接 await 回复器工作
                                    replier_result = await self._replier_work(
                                        observed_messages=observed_messages,
                                        anchor_message=anchor_message,
                                        thinking_id=thinking_id,
                                        current_mind=current_mind,
                                        send_emoji=send_emoji_from_tools,
                                    )
                                except Exception as e_replier:
                                    logger.error(f"{self._get_log_prefix()} 循环: 回复器工作失败: {e_replier}")
                                    self._cleanup_thinking_message(thinking_id)  # 清理思考消息
                                    # 继续循环, 视为非操作周期

                                if replier_result:
                                    # --- Sender ---
                                    try:
                                        await self._sender(thinking_id, anchor_message, replier_result)
                                        logger.info(f"{self._get_log_prefix()} 循环: 发送器完成成功.")
                                    except Exception as e_sender:
                                        logger.error(f"{self._get_log_prefix()} 循环: 发送器失败: {e_sender}")
                                        self._cleanup_thinking_message(thinking_id)  # 确保发送失败时清理
                                        # 继续循环, 视为非操作周期
                                else:
                                    # Replier failed to produce result
                                    logger.warning(f"{self._get_log_prefix()} 循环: 回复器未产生结果. 跳过发送.")
                                    self._cleanup_thinking_message(thinking_id)  # 清理思考消息

                    elif action == "emoji_reply":
                        logger.info(f"{self._get_log_prefix()} PFChatting: 麦麦决定回复表情 ('{emoji_query}').")
                        action_taken_this_cycle = True
                        anchor = await self._get_anchor_message(observed_messages)
                        if anchor:
                            try:
                                await self.heartfc_chat._handle_emoji(anchor, [], emoji_query)
                            except Exception as e_emoji:
                                logger.error(f"{self._get_log_prefix()} 循环: 发送表情失败: {e_emoji}")
                        else:
                            logger.warning(f"{self._get_log_prefix()} 循环: 无法发送表情, 无法获取锚点.")

                    elif action == "no_reply":
                        logger.info(f"{self._get_log_prefix()} PFChatting: 麦麦决定不回复. 原因: {reasoning}")
                        # Do nothing else, action_taken_this_cycle remains False

                    elif action == "error":
                        logger.error(f"{self._get_log_prefix()} PFChatting: 麦麦回复出错. 原因: {reasoning}")
                        # 视为非操作周期

                    else:  # Unknown action
                        logger.warning(f"{self._get_log_prefix()} PFChatting: 麦麦做了奇怪的事情. 原因: {reasoning}")
                        # 视为非操作周期

                except Exception as e_cycle:
                    # Catch errors occurring within the locked section (e.g., planner crash)
                    logger.error(f"{self._get_log_prefix()} 循环周期执行时发生错误: {e_cycle}")
                    logger.error(traceback.format_exc())
                    # Ensure lock is released if an error occurs before the finally block
                    if acquired_lock and self._processing_lock.locked():
                        self._processing_lock.release()
                        acquired_lock = False  # 防止在 finally 块中重复释放
                        logger.warning(f"{self._get_log_prefix()} 由于循环周期中的错误释放了处理锁.")

                finally:
                    # Ensure the lock is always released after a cycle
                    if acquired_lock:
                        self._processing_lock.release()
                        logger.debug(f"{self._get_log_prefix()} 循环释放了处理锁.")

                # --- Timer Decrement ---
                cycle_duration = time.monotonic() - loop_cycle_start_time
                async with self._timer_lock:
                    self._loop_timer -= cycle_duration
                    logger.debug(
                        f"{self._get_log_prefix()} PFChatting: 麦麦聊了{cycle_duration:.2f}秒. 还能聊: {self._loop_timer:.1f}s."
                    )

                # --- Delay ---
                # Add a small delay, especially if no action was taken, to prevent busy-waiting
                try:
                    if not action_taken_this_cycle and cycle_duration < 1.5:
                        # If nothing happened and cycle was fast, wait a bit longer
                        await asyncio.sleep(1.5 - cycle_duration)
                    elif cycle_duration < 0.2:  # Minimum delay even if action was taken
                        await asyncio.sleep(0.2)
                except asyncio.CancelledError:
                    logger.info(f"{self._get_log_prefix()} Sleep interrupted, likely loop cancellation.")
                    break  # Exit loop if cancelled during sleep

        except asyncio.CancelledError:
            logger.info(f"{self._get_log_prefix()} PFChatting: 麦麦的聊天被取消了")
        except Exception as e_loop_outer:
            # Catch errors outside the main cycle lock (should be rare)
            logger.error(f"{self._get_log_prefix()} PFChatting: 麦麦的聊天出错了: {e_loop_outer}")
            logger.error(traceback.format_exc())
        finally:
            # Reset trigger count when loop finishes
            async with self._timer_lock:
                self._trigger_count_this_activation = 0
                logger.debug(f"{self._get_log_prefix()} Trigger count reset to 0 as loop finishes.")
            logger.info(f"{self._get_log_prefix()} PFChatting: 麦麦的聊天结束了")
            # State reset (_loop_active, _loop_task) is handled by _handle_loop_completion callback

    async def _planner(self) -> Dict[str, Any]:
        """
        规划器 (Planner): 使用LLM根据上下文决定是否和如何回复。
        Returns a dictionary containing the decision and context.
        {'action': str, 'reasoning': str, 'emoji_query': str, 'current_mind': str,
         'send_emoji_from_tools': str, 'observed_messages': List[dict]}
        """
        log_prefix = self._get_log_prefix()
        observed_messages: List[dict] = []
        tool_result_info = {}
        get_mid_memory_id = []
        send_emoji_from_tools = ""  # Renamed for clarity
        current_mind: Optional[str] = None

        # --- 获取最新的观察信息 ---
        try:
            if self.sub_hf and self.sub_hf._get_primary_observation():
                observation = self.sub_hf._get_primary_observation()
                logger.debug(f"{log_prefix}[Planner] 调用 observation.observe()...")
                await observation.observe()  # 主动观察以获取最新消息
                observed_messages = observation.talking_message  # 获取更新后的消息列表
                logger.debug(f"{log_prefix}[Planner] 获取到 {len(observed_messages)} 条观察消息。")
            else:
                logger.warning(f"{log_prefix}[Planner] 无法获取 SubHeartflow 或 Observation 来获取消息。")
        except Exception as e:
            logger.error(f"{log_prefix}[Planner] 获取观察信息时出错: {e}")
            logger.error(traceback.format_exc())
        # --- 结束获取观察信息 ---

        # --- (Moved from _replier_work) 1. 思考前使用工具 ---
        try:
            observation_context_text = ""
            if observed_messages:
                context_texts = [
                    msg.get("detailed_plain_text", "") for msg in observed_messages if msg.get("detailed_plain_text")
                ]
                observation_context_text = "\n".join(context_texts)
                logger.debug(f"{log_prefix}[Planner] Context for tools: {observation_context_text[:100]}...")

            if observation_context_text and self.sub_hf:
                # Ensure SubHeartflow exists for tool use context
                tool_result = await self.heartfc_chat.tool_user.use_tool(
                    message_txt=observation_context_text, chat_stream=self.chat_stream, sub_heartflow=self.sub_hf
                )
                if tool_result.get("used_tools", False):
                    tool_result_info = tool_result.get("structured_info", {})
                    logger.debug(f"{log_prefix}[Planner] Tool results: {tool_result_info}")
                    if "mid_chat_mem" in tool_result_info:
                        get_mid_memory_id = [
                            mem["content"] for mem in tool_result_info["mid_chat_mem"] if "content" in mem
                        ]
                    if "send_emoji" in tool_result_info and tool_result_info["send_emoji"]:
                        send_emoji_from_tools = tool_result_info["send_emoji"][0].get("content", "")  # Use renamed var
            elif not self.sub_hf:
                logger.warning(f"{log_prefix}[Planner] Skipping tool use because SubHeartflow is not available.")

        except Exception as e_tool:
            logger.error(f"{log_prefix}[Planner] Tool use failed: {e_tool}")
            # Continue even if tool use fails
        # --- 结束工具使用 ---

        # 心流思考，然后plan
        try:
            if self.sub_hf:
                # Ensure arguments match the current do_thinking_before_reply signature
                current_mind, past_mind = await self.sub_hf.do_thinking_before_reply(
                    chat_stream=self.chat_stream,
                    extra_info=tool_result_info,
                    obs_id=get_mid_memory_id,
                )
                logger.info(f"{log_prefix}[Planner] SubHeartflow thought: {current_mind}")
            else:
                logger.warning(f"{log_prefix}[Planner] Skipping SubHeartflow thinking because it is not available.")
                current_mind = "[心流思考不可用]"  # Set a default/indicator value

        except Exception as e_shf:
            logger.error(f"{log_prefix}[Planner] SubHeartflow thinking failed: {e_shf}")
            logger.error(traceback.format_exc())
            current_mind = "[心流思考出错]"

        # --- 使用 LLM 进行决策 ---
        action = "no_reply"  # Default action
        emoji_query = ""
        reasoning = "默认决策或获取决策失败"
        llm_error = False  # Flag for LLM failure

        try:
            # 构建提示 (Now includes current_mind)
            prompt = self._build_planner_prompt(observed_messages, current_mind)
            logger.debug(f"{log_prefix}[Planner] Prompt: {prompt}")

            # 准备 LLM 请求 Payload
            payload = {
                "model": self.planner_llm.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "tools": PLANNER_TOOL_DEFINITION,
                "tool_choice": {"type": "function", "function": {"name": "decide_reply_action"}},  # 强制调用此工具
            }

            logger.debug(f"{log_prefix}[Planner] 发送 Planner LLM 请求...")
            # 调用 LLM
            response = await self.planner_llm._execute_request(
                endpoint="/chat/completions", payload=payload, prompt=prompt
            )

            # 解析 LLM 响应
            if len(response) == 3:  # 期望返回 content, reasoning_content, tool_calls
                _, _, tool_calls = response
                if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
                    # 通常强制调用后只会有一个 tool_call
                    tool_call = tool_calls[0]
                    if (
                        tool_call.get("type") == "function"
                        and tool_call.get("function", {}).get("name") == "decide_reply_action"
                    ):
                        try:
                            arguments = json.loads(tool_call["function"]["arguments"])
                            action = arguments.get("action", "no_reply")
                            reasoning = arguments.get("reasoning", "未提供理由")
                            if action == "emoji_reply":
                                # Planner's decision overrides tool's emoji if action is emoji_reply
                                emoji_query = arguments.get(
                                    "emoji_query", send_emoji_from_tools
                                )  # Use tool emoji as default if planner asks for emoji
                            logger.info(
                                f"{log_prefix}[Planner] LLM 决策: {action}, 理由: {reasoning}, EmojiQuery: '{emoji_query}'"
                            )
                        except json.JSONDecodeError as json_e:
                            logger.error(
                                f"{log_prefix}[Planner] 解析工具参数失败: {json_e}. Arguments: {tool_call['function'].get('arguments')}"
                            )
                            action = "error"
                            reasoning = "工具参数解析失败"
                            llm_error = True
                        except Exception as parse_e:
                            logger.error(f"{log_prefix}[Planner] 处理工具参数时出错: {parse_e}")
                            action = "error"
                            reasoning = "处理工具参数时出错"
                            llm_error = True
                    else:
                        logger.warning(
                            f"{log_prefix}[Planner] LLM 未按预期调用 'decide_reply_action' 工具。Tool calls: {tool_calls}"
                        )
                        action = "error"
                        reasoning = "LLM未调用预期工具"
                        llm_error = True
                else:
                    logger.warning(f"{log_prefix}[Planner] LLM 响应中未包含有效的工具调用。Tool calls: {tool_calls}")
                    action = "error"
                    reasoning = "LLM响应无工具调用"
                    llm_error = True
            else:
                logger.warning(f"{log_prefix}[Planner] LLM 未返回预期的工具调用响应。Response parts: {len(response)}")
                action = "error"
                reasoning = "LLM响应格式错误"
                llm_error = True

        except Exception as llm_e:
            logger.error(f"{log_prefix}[Planner] Planner LLM 调用失败: {llm_e}")
            logger.error(traceback.format_exc())
            action = "error"
            reasoning = f"LLM 调用失败: {llm_e}"
            llm_error = True

        # --- 返回决策结果 ---
        # Note: Lock release is handled by the loop now
        return {
            "action": action,
            "reasoning": reasoning,
            "emoji_query": emoji_query,  # Specific query if action is emoji_reply
            "current_mind": current_mind,
            "send_emoji_from_tools": send_emoji_from_tools,  # Emoji suggested by pre-thinking tools
            "observed_messages": observed_messages,
            "llm_error": llm_error,  # Indicate if LLM decision process failed
        }

    async def _get_anchor_message(self, observed_messages: List[dict]) -> Optional[MessageRecv]:
        """
        重构观察到的最后一条消息作为回复的锚点，
        如果重构失败或观察为空，则创建一个占位符。
        """
        if not self.chat_stream:
            logger.error(f"{self._get_log_prefix()} 无法获取锚点消息: ChatStream 不可用.")
            return None

        try:
            last_msg_dict = None
            if observed_messages:
                last_msg_dict = observed_messages[-1]

            if last_msg_dict:
                try:
                    # Attempt reconstruction from the last observed message dictionary
                    anchor_message = MessageRecv(last_msg_dict, chat_stream=self.chat_stream)
                    # Basic validation
                    if not (
                        anchor_message
                        and anchor_message.message_info
                        and anchor_message.message_info.message_id
                        and anchor_message.message_info.user_info
                    ):
                        raise ValueError("重构的 MessageRecv 缺少必要信息.")
                    logger.debug(
                        f"{self._get_log_prefix()} 重构的锚点消息: ID={anchor_message.message_info.message_id}"
                    )
                    return anchor_message
                except Exception as e_reconstruct:
                    logger.warning(
                        f"{self._get_log_prefix()} 从观察到的消息重构 MessageRecv 失败: {e_reconstruct}. 创建占位符."
                    )
            else:
                logger.warning(f"{self._get_log_prefix()} observed_messages 为空. 创建占位符锚点消息.")

            # --- Create Placeholder ---
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
                "processed_plain_text": "[System Trigger Context]",  # Placeholder text
                "raw_message": "",
                "time": placeholder_msg_info.time,
            }
            anchor_message = MessageRecv(placeholder_msg_dict)
            anchor_message.update_chat_stream(self.chat_stream)  # Associate with the stream
            logger.info(
                f"{self._get_log_prefix()} Created placeholder anchor message: ID={anchor_message.message_info.message_id}"
            )
            return anchor_message

        except Exception as e:
            logger.error(f"{self._get_log_prefix()} Error getting/creating anchor message: {e}")
            logger.error(traceback.format_exc())
            return None

    def _cleanup_thinking_message(self, thinking_id: str):
        """Safely removes the thinking message."""
        try:
            container = MessageManager().get_container(self.stream_id)
            container.remove_message(thinking_id, msg_type=MessageThinking)
            logger.debug(f"{self._get_log_prefix()} Cleaned up thinking message {thinking_id}.")
        except Exception as e:
            logger.error(f"{self._get_log_prefix()} Error cleaning up thinking message {thinking_id}: {e}")

    async def _sender(self, thinking_id: str, anchor_message: MessageRecv, replier_result: Dict[str, Any]):
        """
        发送器 (Sender): 使用HeartFC_Chat的方法发送生成的回复。
        被 _run_pf_loop 直接调用和 await。
        也处理相关的操作，如发送表情和更新关系。
        Raises exception on failure to signal the loop.
        """
        # replier_result should contain 'response_set' and 'send_emoji'
        response_set = replier_result.get("response_set")
        send_emoji = replier_result.get("send_emoji", "")  # Emoji determined by tools, passed via replier

        if not response_set:
            logger.error(f"{self._get_log_prefix()}[Sender-{thinking_id}] Called with empty response_set.")
            # Clean up thinking message before raising error
            self._cleanup_thinking_message(thinking_id)
            raise ValueError("Sender called with no response_set")  # Signal failure to loop

        first_bot_msg: Optional[MessageSending] = None
        send_success = False
        try:
            # --- Send the main text response ---
            logger.debug(f"{self._get_log_prefix()}[Sender-{thinking_id}] Sending response messages...")
            # This call implicitly handles replacing the MessageThinking with MessageSending/MessageSet
            first_bot_msg = await self.heartfc_chat._send_response_messages(anchor_message, response_set, thinking_id)

            if first_bot_msg:
                send_success = True  # Mark success
                logger.info(f"{self._get_log_prefix()}[Sender-{thinking_id}] Successfully sent reply.")

                # --- Handle associated emoji (if determined by tools) ---
                if send_emoji:
                    logger.info(
                        f"{self._get_log_prefix()}[Sender-{thinking_id}] Sending associated emoji: {send_emoji}"
                    )
                    try:
                        # Use first_bot_msg as anchor if available, otherwise fallback to original anchor
                        emoji_anchor = first_bot_msg if first_bot_msg else anchor_message
                        await self.heartfc_chat._handle_emoji(emoji_anchor, response_set, send_emoji)
                    except Exception as e_emoji:
                        logger.error(
                            f"{self._get_log_prefix()}[Sender-{thinking_id}] Failed to send associated emoji: {e_emoji}"
                        )
                        # Log error but don't fail the whole send process for emoji failure

                # --- Update relationship ---
                try:
                    await self.heartfc_chat._update_relationship(anchor_message, response_set)
                    logger.debug(f"{self._get_log_prefix()}[Sender-{thinking_id}] Updated relationship.")
                except Exception as e_rel:
                    logger.error(
                        f"{self._get_log_prefix()}[Sender-{thinking_id}] Failed to update relationship: {e_rel}"
                    )
                    # Log error but don't fail the whole send process for relationship update failure

            else:
                # Sending failed (e.g., _send_response_messages found thinking message already gone)
                send_success = False
                logger.warning(
                    f"{self._get_log_prefix()}[Sender-{thinking_id}] Failed to send reply (maybe thinking message expired or was removed?)."
                )
                # No need to clean up thinking message here, _send_response_messages implies it's gone or handled
                raise RuntimeError("Sending reply failed, _send_response_messages returned None.")  # Signal failure

        except Exception as e:
            # Catch potential errors during sending or post-send actions
            logger.error(f"{self._get_log_prefix()}[Sender-{thinking_id}] Error during sending process: {e}")
            logger.error(traceback.format_exc())
            # Ensure thinking message is cleaned up if send failed mid-way and wasn't handled
            if not send_success:
                self._cleanup_thinking_message(thinking_id)
            raise  # Re-raise the exception to signal failure to the loop

        # No finally block needed for lock management

    async def shutdown(self):
        """
        Gracefully shuts down the PFChatting instance by cancelling the active loop task.
        """
        logger.info(f"{self._get_log_prefix()} Shutting down PFChatting...")
        if self._loop_task and not self._loop_task.done():
            logger.info(f"{self._get_log_prefix()} Cancelling active PF loop task.")
            self._loop_task.cancel()
            try:
                # Wait briefly for the task to acknowledge cancellation
                await asyncio.wait_for(self._loop_task, timeout=5.0)
            except asyncio.CancelledError:
                logger.info(f"{self._get_log_prefix()} PF loop task cancelled successfully.")
            except asyncio.TimeoutError:
                logger.warning(f"{self._get_log_prefix()} Timeout waiting for PF loop task cancellation.")
            except Exception as e:
                logger.error(f"{self._get_log_prefix()} Error during loop task cancellation: {e}")
        else:
            logger.info(f"{self._get_log_prefix()} No active PF loop task found to cancel.")

        # Ensure loop state is reset even if task wasn't running or cancellation failed
        self._loop_active = False
        self._loop_task = None

        # Double-check lock state (should be released by loop completion/cancellation handler)
        if self._processing_lock.locked():
            logger.warning(f"{self._get_log_prefix()} Releasing processing lock during shutdown.")
            self._processing_lock.release()

        logger.info(f"{self._get_log_prefix()} PFChatting shutdown complete.")

    def _build_planner_prompt(self, observed_messages: List[dict], current_mind: Optional[str]) -> str:
        """构建 Planner LLM 的提示词 (现在包含 current_mind)"""
        prompt = f"你的名字是 {global_config.BOT_NICKNAME}。是一个女大二学生，正在QQ聊天，正在决定是否以及如何回应当前的聊天。\n"

        # Add current mind state if available
        if current_mind:
            prompt += f"\n你当前的内部想法是：\n---\n{current_mind}\n---\n\n"
        else:
            prompt += "\n你当前没有特别的内部想法。\n"

        if observed_messages:
            context_text = "\n".join(
                [msg.get("detailed_plain_text", "") for msg in observed_messages if msg.get("detailed_plain_text")]
            )
            prompt += "观察到的最新聊天内容如下：\n---\n"
            prompt += context_text[:1500]  # Limit context length
            prompt += "\n---\n"
        else:
            prompt += "当前没有观察到新的聊天内容。\n"

        prompt += (
            "\n请结合你的内部想法和观察到的聊天内容，分析情况并使用 'decide_reply_action' 工具来决定你的最终行动。\n"
        )
        prompt += "决策依据：\n"
        prompt += "1. 如果聊天内容无聊、与你无关、或者你的内部想法认为不适合回复，选择 'no_reply'。\n"
        prompt += "2. 如果聊天内容值得回应，且适合用文字表达（参考你的内部想法），选择 'text_reply'。\n"
        prompt += (
            "3. 如果聊天内容或你的内部想法适合用一个表情来回应，选择 'emoji_reply' 并提供表情主题 'emoji_query'。\n"
        )
        prompt += "4. 如果你已经回复过消息，也没有人又回复你，选择'no_reply'。"
        prompt += "必须调用 'decide_reply_action' 工具并提供 'action' 和 'reasoning'。"

        return prompt

    # --- 回复器 (Replier) 的定义 --- #
    async def _replier_work(
        self,
        observed_messages: List[dict],
        anchor_message: MessageRecv,
        thinking_id: str,
        current_mind: Optional[str],
        send_emoji: str,
    ) -> Optional[Dict[str, Any]]:
        """
        回复器 (Replier): 核心逻辑用于生成回复。
        被 _run_pf_loop 直接调用和 await。
        Returns dict with 'response_set' and 'send_emoji' or None on failure.
        """
        log_prefix = self._get_log_prefix()
        response_set: Optional[List[str]] = None
        try:
            # --- Tool Use and SubHF Thinking are now in _planner ---

            # --- Generate Response with LLM ---
            logger.debug(f"{log_prefix}[Replier-{thinking_id}] Calling LLM to generate response...")
            # 注意：实际的生成调用是在 self.heartfc_chat.gpt.generate_response 中
            response_set = await self.heartfc_chat.gpt.generate_response(
                anchor_message,
                thinking_id,
                # current_mind 不再直接传递给 gpt.generate_response，
                # 因为 generate_response 内部会通过 thinking_id 或其他方式获取所需上下文
            )

            if not response_set:
                logger.warning(f"{log_prefix}[Replier-{thinking_id}] LLM生成了一个空回复集。")
                return None  # Indicate failure

            # --- 准备并返回结果 ---
            logger.info(f"{log_prefix}[Replier-{thinking_id}] 成功生成了回复集: {' '.join(response_set)[:50]}...")
            return {
                "response_set": response_set,
                "send_emoji": send_emoji,  # Pass through the emoji determined earlier (usually by tools)
            }

        except Exception as e:
            logger.error(f"{log_prefix}[Replier-{thinking_id}] Unexpected error in replier_work: {e}")
            logger.error(traceback.format_exc())
            return None  # Indicate failure
