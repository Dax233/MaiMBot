import asyncio
import time
import traceback
import random  # <-- 添加导入
from typing import List, Optional, Dict, Any, Deque
from collections import deque
from src.plugins.chat.message import MessageRecv, BaseMessageInfo, MessageThinking, MessageSending
from src.plugins.chat.message import Seg  # Local import needed after move
from src.plugins.chat.chat_stream import ChatStream
from src.plugins.chat.message import UserInfo
from src.plugins.chat.chat_stream import chat_manager
from src.common.logger import get_module_logger, LogConfig, PFC_STYLE_CONFIG  # 引入 DEFAULT_CONFIG
from src.plugins.models.utils_model import LLMRequest
from src.config.config import global_config
from src.plugins.chat.utils_image import image_path_to_base64  # Local import needed after move
from src.plugins.utils.timer_calculator import Timer  # <--- Import Timer
from src.plugins.heartFC_chat.heartFC_generator import HeartFCGenerator
from src.do_tool.tool_use import ToolUser
from src.plugins.emoji_system.emoji_manager import emoji_manager
from src.plugins.utils.json_utils import process_llm_tool_response  # 导入新的JSON工具
from src.heart_flow.sub_mind import SubMind
from src.heart_flow.observation import Observation
from src.plugins.heartFC_chat.heartflow_prompt_builder import global_prompt_manager
import contextlib
from src.plugins.utils.chat_message_builder import num_new_messages_since
from src.plugins.heartFC_chat.heartFC_Cycleinfo import CycleInfo
from .heartFC_sender import HeartFCSender
# --- End import ---


INITIAL_DURATION = 60.0


# 定义日志配置 (使用 loguru 格式)
interest_log_config = LogConfig(
    console_format=PFC_STYLE_CONFIG["console_format"],  # 使用默认控制台格式
    file_format=PFC_STYLE_CONFIG["file_format"],  # 使用默认文件格式
)
logger = get_module_logger("HeartFCLoop", config=interest_log_config)  # Logger Name Changed


# 默认动作定义
DEFAULT_ACTIONS = {"no_reply": "不回复", "text_reply": "文本回复, 可选附带表情", "emoji_reply": "仅表情回复"}


class ActionManager:
    """动作管理器：控制每次决策可以使用的动作"""

    def __init__(self):
        # 初始化为默认动作集
        self._available_actions: Dict[str, str] = DEFAULT_ACTIONS.copy()

    def get_available_actions(self) -> Dict[str, str]:
        """获取当前可用的动作集"""
        return self._available_actions

    def add_action(self, action_name: str, description: str) -> bool:
        """
        添加新的动作

        参数:
            action_name: 动作名称
            description: 动作描述

        返回:
            bool: 是否添加成功
        """
        if action_name in self._available_actions:
            return False
        self._available_actions[action_name] = description
        return True

    def remove_action(self, action_name: str) -> bool:
        """
        移除指定动作

        参数:
            action_name: 动作名称

        返回:
            bool: 是否移除成功
        """
        if action_name not in self._available_actions:
            return False
        del self._available_actions[action_name]
        return True

    def clear_actions(self):
        """清空所有动作"""
        self._available_actions.clear()

    def reset_to_default(self):
        """重置为默认动作集"""
        self._available_actions = DEFAULT_ACTIONS.copy()

    def get_planner_tool_definition(self) -> List[Dict[str, Any]]:
        """获取当前动作集对应的规划器工具定义"""
        return [
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
                                "enum": list(self._available_actions.keys()),
                                "description": "决定采取的行动："
                                + ", ".join([f"'{k}'({v})" for k, v in self._available_actions.items()]),
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


# 在文件开头添加自定义异常类
class HeartFCError(Exception):
    """麦麦聊天系统基础异常类"""

    pass


class PlannerError(HeartFCError):
    """规划器异常"""

    pass


class ReplierError(HeartFCError):
    """回复器异常"""

    pass


class SenderError(HeartFCError):
    """发送器异常"""

    pass


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
        self.observations: List[Observation] = observations  # 关联的观察列表，用于监控聊天流状态

        # 日志前缀
        self.log_prefix: str = f"[{chat_manager.get_stream_name(chat_id) or chat_id}]"

        # 动作管理器
        self.action_manager = ActionManager()

        # 初始化状态控制
        self._initialized = False  # 是否已初始化标志
        self._processing_lock = asyncio.Lock()  # 处理锁(确保单次Plan-Replier-Sender周期)

        # 依赖注入存储
        self.gpt_instance = HeartFCGenerator()  # 文本回复生成器
        self.tool_user = ToolUser()  # 工具使用实例
        self.heart_fc_sender = HeartFCSender()

        # LLM规划器配置
        self.planner_llm = LLMRequest(
            model=global_config.llm_plan,
            max_tokens=1000,
            request_type="action_planning",  # 用于动作规划
        )

        # 循环控制内部状态
        self._loop_active: bool = False  # 循环是否正在运行
        self._loop_task: Optional[asyncio.Task] = None  # 主循环任务

        # 添加循环信息管理相关的属性
        self._cycle_counter = 0
        self._cycle_history: Deque[CycleInfo] = deque(maxlen=10)  # 保留最近10个循环的信息
        self._current_cycle: Optional[CycleInfo] = None

    async def _initialize(self) -> bool:
        """
        懒初始化以使用提供的标识符解析chat_stream。
        确保实例已准备好处理触发器。
        """
        if self._initialized:
            return True

        self.chat_stream = chat_manager.get_stream(self.stream_id)
        if not self.chat_stream:
            logger.error(f"{self.log_prefix} 获取ChatStream失败。")
            return False

        # 更新日志前缀（以防流名称发生变化）
        self.log_prefix = f"[{chat_manager.get_stream_name(self.stream_id) or self.stream_id}]"

        self._initialized = True
        logger.info(f"感觉到了，可以开始激情水群{self.log_prefix} ")
        return True

    async def start(self):
        """
        启动 HeartFChatting 的主循环。
        注意：调用此方法前必须确保已经成功初始化。
        """
        logger.info(f"{self.log_prefix} 开始激情水群(HFC)...")
        await self._start_loop_if_needed()

    async def _start_loop_if_needed(self):
        """检查是否需要启动主循环，如果未激活则启动。"""
        # 如果循环已经激活，直接返回
        if self._loop_active:
            return

        # 标记为活动状态，防止重复启动
        self._loop_active = True

        # 检查是否已有任务在运行（理论上不应该，因为 _loop_active=False）
        if self._loop_task and not self._loop_task.done():
            logger.warning(f"{self.log_prefix} 发现之前的循环任务仍在运行（不符合预期）。取消旧任务。")
            self._loop_task.cancel()
            try:
                # 等待旧任务确实被取消
                await asyncio.wait_for(self._loop_task, timeout=0.5)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass  # 忽略取消或超时错误
            self._loop_task = None  # 清理旧任务引用

        logger.info(f"{self.log_prefix} 启动激情水群(HFC)主循环...")
        # 创建新的循环任务
        self._loop_task = asyncio.create_task(self._hfc_loop())
        # 添加完成回调
        self._loop_task.add_done_callback(self._handle_loop_completion)

    def _handle_loop_completion(self, task: asyncio.Task):
        """当 _hfc_loop 任务完成时执行的回调。"""
        try:
            exception = task.exception()
            if exception:
                logger.error(f"{self.log_prefix} HeartFChatting: 脱离了聊天(异常): {exception}")
                logger.error(traceback.format_exc())  # Log full traceback for exceptions
            else:
                # Loop completing normally now means it was cancelled/shutdown externally
                logger.info(f"{self.log_prefix} HeartFChatting: 脱离了聊天 (外部停止)")
        except asyncio.CancelledError:
            logger.info(f"{self.log_prefix} HeartFChatting: 脱离了聊天(任务取消)")
        finally:
            self._loop_active = False
            self._loop_task = None
            if self._processing_lock.locked():
                logger.warning(f"{self.log_prefix} HeartFChatting: 处理锁在循环结束时仍被锁定，强制释放。")
                self._processing_lock.release()

    async def _hfc_loop(self):
        """主循环，持续进行计划并可能回复消息，直到被外部取消。"""
        try:
            while True:  # 主循环
                # 创建新的循环信息
                self._cycle_counter += 1
                self._current_cycle = CycleInfo(self._cycle_counter)

                # 初始化周期状态
                cycle_timers = {}
                loop_cycle_start_time = time.monotonic()

                # 执行规划和处理阶段
                async with self._get_cycle_context() as acquired_lock:
                    if not acquired_lock:
                        continue

                    # 记录规划开始时间点
                    planner_start_db_time = time.time()

                    # 主循环：思考->决策->执行
                    action_taken, thinking_id = await self._think_plan_execute_loop(cycle_timers, planner_start_db_time)

                    # 更新循环信息
                    self._current_cycle.set_thinking_id(thinking_id)
                    self._current_cycle.timers = cycle_timers

                    # 防止循环过快消耗资源
                    await self._handle_cycle_delay(action_taken, loop_cycle_start_time, self.log_prefix)

                # 完成当前循环并保存历史
                self._current_cycle.complete_cycle()
                self._cycle_history.append(self._current_cycle)

                # 记录循环信息和计时器结果
                timer_strings = []
                for name, elapsed in cycle_timers.items():
                    formatted_time = f"{elapsed * 1000:.2f}毫秒" if elapsed < 1 else f"{elapsed:.2f}秒"
                    timer_strings.append(f"{name}: {formatted_time}")

                logger.debug(
                    f"{self.log_prefix}  第 #{self._current_cycle.cycle_id}次思考完成,"
                    f"耗时: {self._current_cycle.end_time - self._current_cycle.start_time:.2f}秒, "
                    f"动作: {self._current_cycle.action_type}"
                    + (f"\n计时器详情: {'; '.join(timer_strings)}" if timer_strings else "")
                )

        except asyncio.CancelledError:
            logger.info(f"{self.log_prefix} HeartFChatting: 麦麦的激情水群(HFC)被取消了")
        except Exception as e:
            logger.error(f"{self.log_prefix} HeartFChatting: 意外错误: {e}")
            logger.error(traceback.format_exc())

    @contextlib.asynccontextmanager
    async def _get_cycle_context(self):
        """
        循环周期的上下文管理器

        用于确保资源的正确获取和释放：
        1. 获取处理锁
        2. 执行操作
        3. 释放锁
        """
        acquired = False
        try:
            await self._processing_lock.acquire()
            acquired = True
            yield acquired
        finally:
            if acquired and self._processing_lock.locked():
                self._processing_lock.release()

    async def _check_new_messages(self, start_time: float) -> bool:
        """
        检查从指定时间点后是否有新消息

        参数:
            start_time: 开始检查的时间点

        返回:
            bool: 是否有新消息
        """
        try:
            new_msg_count = num_new_messages_since(self.stream_id, start_time)
            if new_msg_count > 0:
                logger.info(f"{self.log_prefix} 检测到{new_msg_count}条新消息")
                return True
            return False
        except Exception as e:
            logger.error(f"{self.log_prefix} 检查新消息时出错: {e}")
            return False

    async def _think_plan_execute_loop(self, cycle_timers: dict, planner_start_db_time: float) -> tuple[bool, str]:
        """执行规划阶段"""
        try:
            # think:思考
            current_mind = await self._get_submind_thinking(cycle_timers)
            # 记录子思维思考内容
            if self._current_cycle:
                self._current_cycle.set_response_info(sub_mind_thinking=current_mind)

            # plan:决策
            with Timer("决策", cycle_timers):
                planner_result = await self._planner(current_mind, cycle_timers)

            action = planner_result.get("action", "error")
            reasoning = planner_result.get("reasoning", "未提供理由")

            self._current_cycle.set_action_info(action, reasoning, False)

            # 在获取规划结果后检查新消息
            if await self._check_new_messages(planner_start_db_time):
                if random.random() < 0.3:
                    logger.info(f"{self.log_prefix} 看到了新消息，麦麦决定重新观察和规划...")
                    # 重新规划
                    with Timer("重新决策", cycle_timers):
                        self._current_cycle.replanned = True
                        planner_result = await self._planner(current_mind, cycle_timers, is_re_planned=True)
                    logger.info(f"{self.log_prefix} 重新规划完成.")

            # 解析规划结果
            action = planner_result.get("action", "error")
            reasoning = planner_result.get("reasoning", "未提供理由")
            # 更新循环信息
            self._current_cycle.set_action_info(action, reasoning, True)

            # 处理LLM错误
            if planner_result.get("llm_error"):
                logger.error(f"{self.log_prefix} LLM失败: {reasoning}")
                return False, ""

            # execute:执行
            with Timer("执行动作", cycle_timers):
                return await self._handle_action(
                    action, reasoning, planner_result.get("emoji_query", ""), cycle_timers, planner_start_db_time
                )

        except PlannerError as e:
            logger.error(f"{self.log_prefix} 规划错误: {e}")
            # 更新循环信息
            self._current_cycle.set_action_info("error", str(e), False)
            return False, ""

    async def _handle_action(
        self, action: str, reasoning: str, emoji_query: str, cycle_timers: dict, planner_start_db_time: float
    ) -> tuple[bool, str]:
        """
        处理规划动作

        参数:
            action: 动作类型
            reasoning: 决策理由
            emoji_query: 表情查询
            cycle_timers: 计时器字典
            planner_start_db_time: 规划开始时间

        返回:
            tuple[bool, str]: (是否执行了动作, 思考消息ID)
        """
        action_handlers = {
            "text_reply": self._handle_text_reply,
            "emoji_reply": self._handle_emoji_reply,
            "no_reply": self._handle_no_reply,
        }

        handler = action_handlers.get(action)
        if not handler:
            logger.warning(f"{self.log_prefix} 未知动作: {action}, 原因: {reasoning}")
            return False, ""

        try:
            if action == "text_reply":
                return await handler(reasoning, emoji_query, cycle_timers)
            elif action == "emoji_reply":
                return await handler(reasoning, emoji_query), ""
            else:  # no_reply
                return await handler(reasoning, planner_start_db_time, cycle_timers), ""
        except HeartFCError as e:
            logger.error(f"{self.log_prefix} 处理{action}时出错: {e}")
            return False, ""

    async def _handle_text_reply(self, reasoning: str, emoji_query: str, cycle_timers: dict) -> tuple[bool, str]:
        """
        处理文本回复

        工作流程：
        1. 获取锚点消息
        2. 创建思考消息
        3. 生成回复
        4. 发送消息

        参数:
            reasoning: 回复原因
            emoji_query: 表情查询
            cycle_timers: 计时器字典

        返回:
            tuple[bool, str]: (是否回复成功, 思考消息ID)
        """

        # 获取锚点消息
        anchor_message = await self._get_anchor_message()
        if not anchor_message:
            raise PlannerError("无法获取锚点消息")

        # 创建思考消息
        thinking_id = await self._create_thinking_message(anchor_message)
        if not thinking_id:
            raise PlannerError("无法创建思考消息")

        try:
            # 生成回复
            with Timer("生成回复", cycle_timers):
                reply = await self._replier_work(
                    anchor_message=anchor_message,
                    thinking_id=thinking_id,
                    reason=reasoning,
                )

            if not reply:
                raise ReplierError("回复生成失败")

            # 发送消息

            with Timer("发送消息", cycle_timers):
                await self._sender(
                    thinking_id=thinking_id,
                    anchor_message=anchor_message,
                    response_set=reply,
                    send_emoji=emoji_query,
                )

            return True, thinking_id

        except (ReplierError, SenderError) as e:
            logger.error(f"{self.log_prefix} 回复失败: {e}")
            return True, thinking_id  # 仍然返回thinking_id以便跟踪

    async def _handle_emoji_reply(self, reasoning: str, emoji_query: str) -> bool:
        """
        处理表情回复

        工作流程：
        1. 获取锚点消息
        2. 发送表情

        参数:
            reasoning: 回复原因
            emoji_query: 表情查询

        返回:
            bool: 是否发送成功
        """
        logger.info(f"{self.log_prefix} 决定回复表情({emoji_query}): {reasoning}")

        try:
            anchor = await self._get_anchor_message()
            if not anchor:
                raise PlannerError("无法获取锚点消息")

            await self._handle_emoji(anchor, [], emoji_query)
            return True

        except Exception as e:
            logger.error(f"{self.log_prefix} 表情发送失败: {e}")
            return False

    async def _handle_no_reply(self, reasoning: str, planner_start_db_time: float, cycle_timers: dict) -> bool:
        """
        处理不回复的情况

        工作流程：
        1. 等待新消息
        2. 超时或收到新消息时返回

        参数:
            reasoning: 不回复的原因
            planner_start_db_time: 规划开始时间
            cycle_timers: 计时器字典

        返回:
            bool: 是否成功处理
        """
        logger.info(f"{self.log_prefix} 决定不回复: {reasoning}")

        observation = self.observations[0] if self.observations else None

        try:
            with Timer("Wait New Msg", cycle_timers):
                return await self._wait_for_new_message(observation, planner_start_db_time, self.log_prefix)
        except asyncio.CancelledError:
            logger.info(f"{self.log_prefix} 等待被中断")
            raise

    async def _wait_for_new_message(self, observation, planner_start_db_time: float, log_prefix: str) -> bool:
        """
        等待新消息

        参数:
            observation: 观察实例
            planner_start_db_time: 开始等待的时间
            log_prefix: 日志前缀

        返回:
            bool: 是否检测到新消息
        """
        wait_start_time = time.monotonic()
        while True:
            if await observation.has_new_messages_since(planner_start_db_time):
                logger.info(f"{log_prefix} 检测到新消息")
                return True

            if time.monotonic() - wait_start_time > 300:
                logger.warning(f"{log_prefix} 等待超时(300秒)")
                return False

            await asyncio.sleep(1.5)

    async def _log_cycle_timers(self, cycle_timers: dict, log_prefix: str):
        """记录循环周期的计时器结果"""
        if cycle_timers:
            timer_strings = []
            for name, elapsed in cycle_timers.items():
                formatted_time = f"{elapsed * 1000:.2f}毫秒" if elapsed < 1 else f"{elapsed:.2f}秒"
                timer_strings.append(f"{name}: {formatted_time}")

            if timer_strings:
                logger.debug(f"{log_prefix} 该次决策耗时: {'; '.join(timer_strings)}")

    async def _handle_cycle_delay(self, action_taken_this_cycle: bool, cycle_start_time: float, log_prefix: str):
        """处理循环延迟"""
        cycle_duration = time.monotonic() - cycle_start_time
        # if cycle_duration > 0.1:
        # logger.debug(f"{log_prefix} HeartFChatting: 周期耗时 {cycle_duration:.2f}s.")

        try:
            sleep_duration = 0.0
            if not action_taken_this_cycle and cycle_duration < 1:
                sleep_duration = 1 - cycle_duration
            elif cycle_duration < 0.2:
                sleep_duration = 0.2

            if sleep_duration > 0:
                await asyncio.sleep(sleep_duration)

        except asyncio.CancelledError:
            logger.info(f"{log_prefix} Sleep interrupted, loop likely cancelling.")
            raise

    async def _get_submind_thinking(self, cycle_timers: dict) -> str:
        """
        获取子思维的思考结果

        返回:
            str: 思考结果，如果思考失败则返回错误信息
        """
        try:
            with Timer("观察", cycle_timers):
                observation = self.observations[0]
                await observation.observe()

            # 获取上一个循环的信息
            last_cycle = self._cycle_history[-1] if self._cycle_history else None

            with Timer("思考", cycle_timers):
                # 获取上一个循环的动作
                # 传递上一个循环的信息给 do_thinking_before_reply
                current_mind, _past_mind = await self.sub_mind.do_thinking_before_reply(last_cycle=last_cycle)
                return current_mind
        except Exception as e:
            logger.error(f"{self.log_prefix}[SubMind] 思考失败: {e}")
            logger.error(traceback.format_exc())
            return "[思考时出错]"

    async def _planner(self, current_mind: str, cycle_timers: dict, is_re_planned: bool = False) -> Dict[str, Any]:
        """
        规划器 (Planner): 使用LLM根据上下文决定是否和如何回复。

        参数:
            current_mind: 子思维的当前思考结果
        """
        logger.info(f"{self.log_prefix}[Planner] 开始{'重新' if is_re_planned else ''}执行规划器")

        # 获取观察信息
        observation = self.observations[0]
        if is_re_planned:
            await observation.observe()
        observed_messages = observation.talking_message
        observed_messages_str = observation.talking_message_str

        # --- 使用 LLM 进行决策 --- #
        action = "no_reply"  # 默认动作
        emoji_query = ""  # 默认表情查询
        reasoning = "默认决策或获取决策失败"
        llm_error = False  # LLM错误标志

        try:
            # 构建提示词

            if is_re_planned:
                replan_prompt = await self._build_replan_prompt(
                    self._current_cycle.action_type, self._current_cycle.reasoning
                )
                prompt = replan_prompt
            else:
                replan_prompt = ""
            prompt = await self._build_planner_prompt(
                observed_messages_str, current_mind, self.sub_mind.structured_info, replan_prompt
            )
            payload = {
                "model": global_config.llm_plan["name"],
                "messages": [{"role": "user", "content": prompt}],
                "tools": self.action_manager.get_planner_tool_definition(),
                "tool_choice": {"type": "function", "function": {"name": "decide_reply_action"}},
            }

            # 执行LLM请求

            try:
                response = await self.planner_llm._execute_request(
                    endpoint="/chat/completions", payload=payload, prompt=prompt
                )
            except Exception as req_e:
                logger.error(f"{self.log_prefix}[Planner] LLM请求执行失败: {req_e}")
                return {
                    "action": "error",
                    "reasoning": f"LLM请求执行失败: {req_e}",
                    "emoji_query": "",
                    "current_mind": current_mind,
                    "observed_messages": observed_messages,
                    "llm_error": True,
                }

            # 处理LLM响应
            with Timer("使用工具", cycle_timers):
                # 使用辅助函数处理工具调用响应
                success, arguments, error_msg = process_llm_tool_response(
                    response, expected_tool_name="decide_reply_action", log_prefix=f"{self.log_prefix}[Planner] "
                )

                if success:
                    # 提取决策参数
                    action = arguments.get("action", "no_reply")
                    # 验证动作是否在可用动作集中
                    if action not in self.action_manager.get_available_actions():
                        logger.warning(
                            f"{self.log_prefix}[Planner] LLM返回了未授权的动作: {action}，使用默认动作no_reply"
                        )
                        action = "no_reply"
                        reasoning = f"LLM返回了未授权的动作: {action}"
                    else:
                        reasoning = arguments.get("reasoning", "未提供理由")
                        emoji_query = arguments.get("emoji_query", "")

                    # 记录决策结果
                    logger.debug(
                        f"{self.log_prefix}[要做什么]\nPrompt:\n{prompt}\n\n决策结果: {action}, 理由: {reasoning}, 表情查询: '{emoji_query}'"
                    )
                else:
                    # 处理工具调用失败
                    logger.warning(f"{self.log_prefix}[Planner] {error_msg}")
                    action = "error"
                    reasoning = error_msg
                    llm_error = True

        except Exception as llm_e:
            logger.error(f"{self.log_prefix}[Planner] Planner LLM处理过程中出错: {llm_e}")
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
                f"{self.log_prefix} Created placeholder anchor message: ID={anchor_message.message_info.message_id}"
            )
            return anchor_message

        except Exception as e:
            logger.error(f"{self.log_prefix} Error getting/creating anchor message: {e}")
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
        发送器 (Sender): 使用 HeartFCSender 实例发送生成的回复。
        处理相关的操作，如发送表情和更新关系。
        """
        logger.info(f"{self.log_prefix}开始发送回复 (使用 HeartFCSender)")

        first_bot_msg: Optional[MessageSending] = None
        try:
            # _send_response_messages 现在将使用 self.sender 内部处理注册和发送
            # 它需要负责创建 MessageThinking 和 MessageSending 对象
            # 并调用 self.sender.register_thinking 和 self.sender.type_and_send_message
            first_bot_msg = await self._send_response_messages(
                anchor_message=anchor_message, response_set=response_set, thinking_id=thinking_id
            )

            if first_bot_msg:
                # --- 处理关联表情(如果指定) --- #
                if send_emoji:
                    logger.info(f"{self.log_prefix}正在发送关联表情: '{send_emoji}'")
                    # 优先使用 first_bot_msg 作为锚点，否则回退到原始锚点
                    emoji_anchor = first_bot_msg
                    await self._handle_emoji(emoji_anchor, response_set, send_emoji)
            else:
                # 如果 _send_response_messages 返回 None，表示在发送前就失败或没有消息可发送
                logger.warning(
                    f"{self.log_prefix}[Sender-{thinking_id}] 未能发送任何回复消息 (_send_response_messages 返回 None)。"
                )
                # 这里可能不需要抛出异常，取决于 _send_response_messages 的具体实现

        except Exception as e:
            # 异常现在由 type_and_send_message 内部处理日志，这里只记录发送流程失败
            logger.error(f"{self.log_prefix}[Sender-{thinking_id}] 发送回复过程中遇到错误: {e}")
            # 思考状态应已在 type_and_send_message 的 finally 块中清理
            # 可以选择重新抛出或根据业务逻辑处理
            # raise RuntimeError(f"发送回复失败: {e}") from e

    async def shutdown(self):
        """优雅关闭HeartFChatting实例，取消活动循环任务"""
        logger.info(f"{self.log_prefix} 正在关闭HeartFChatting...")

        # 取消循环任务
        if self._loop_task and not self._loop_task.done():
            logger.info(f"{self.log_prefix} 正在取消HeartFChatting循环任务")
            self._loop_task.cancel()
            try:
                await asyncio.wait_for(self._loop_task, timeout=1.0)
                logger.info(f"{self.log_prefix} HeartFChatting循环任务已取消")
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception as e:
                logger.error(f"{self.log_prefix} 取消循环任务出错: {e}")
        else:
            logger.info(f"{self.log_prefix} 没有活动的HeartFChatting循环任务")

        # 清理状态
        self._loop_active = False
        self._loop_task = None
        if self._processing_lock.locked():
            self._processing_lock.release()
            logger.warning(f"{self.log_prefix} 已释放处理锁")

        logger.info(f"{self.log_prefix} HeartFChatting关闭完成")

    async def _build_replan_prompt(self, action: str, reasoning: str) -> str:
        """构建 Replanner LLM 的提示词"""
        prompt = (await global_prompt_manager.get_prompt_async("replan_prompt")).format(
            action=action,
            reasoning=reasoning,
        )
        return prompt

    async def _build_planner_prompt(
        self,
        observed_messages_str: str,
        current_mind: Optional[str],
        structured_info: Dict[str, Any],
        replan_prompt: str,
    ) -> str:
        """构建 Planner LLM 的提示词"""

        # 准备结构化信息块
        structured_info_block = ""
        if structured_info:
            structured_info_block = f"以下是一些额外的信息：\n{structured_info}\n"

        # 准备聊天内容块
        chat_content_block = ""
        if observed_messages_str:
            chat_content_block = "观察到的最新聊天内容如下：\n---\n"
            chat_content_block += observed_messages_str
            chat_content_block += "\n---"
        else:
            chat_content_block = "当前没有观察到新的聊天内容。\n"

        # 准备当前思维块
        current_mind_block = ""
        if current_mind:
            current_mind_block = f"{current_mind}"
        else:
            current_mind_block = "[没有特别的想法]"

        # 获取提示词模板并填充数据
        prompt = (await global_prompt_manager.get_prompt_async("planner_prompt")).format(
            bot_name=global_config.BOT_NICKNAME,
            structured_info_block=structured_info_block,
            chat_content_block=chat_content_block,
            current_mind_block=current_mind_block,
            replan=replan_prompt,
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
                logger.warning(f"{self.log_prefix}[Replier-{thinking_id}] LLM生成了一个空回复集。")
                return None

            return response_set

        except Exception as e:
            logger.error(f"{self.log_prefix}[Replier-{thinking_id}] Unexpected error in replier_work: {e}")
            logger.error(traceback.format_exc())
            return None

    # --- Methods moved from HeartFCController start ---
    async def _create_thinking_message(self, anchor_message: Optional[MessageRecv]) -> Optional[str]:
        """创建思考消息 (尝试锚定到 anchor_message)"""
        if not anchor_message or not anchor_message.chat_stream:
            logger.error(f"{self.log_prefix} 无法创建思考消息，缺少有效的锚点消息或聊天流。")
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
        await self.heart_fc_sender.register_thinking(thinking_message)
        return thinking_id

    async def _send_response_messages(
        self, anchor_message: Optional[MessageRecv], response_set: List[str], thinking_id: str
    ) -> Optional[MessageSending]:
        """发送回复消息 (尝试锚定到 anchor_message)，使用 HeartFCSender"""
        if not anchor_message or not anchor_message.chat_stream:
            logger.error(f"{self.log_prefix} 无法发送回复，缺少有效的锚点消息或聊天流。")
            return None

        chat = anchor_message.chat_stream
        chat_id = chat.stream_id
        stream_name = chat_manager.get_stream_name(chat_id) or chat_id  # 获取流名称用于日志

        # 检查思考过程是否仍在进行，并获取开始时间
        thinking_start_time = await self.heart_fc_sender.get_thinking_start_time(chat_id, thinking_id)

        if thinking_start_time is None:
            logger.warning(f"[{stream_name}] {thinking_id} 思考过程未找到或已结束，无法发送回复。")
            return None

        # 记录锚点消息ID和回复文本（在发送前记录）
        self._current_cycle.set_response_info(
            response_text=response_set, anchor_message_id=anchor_message.message_info.message_id
        )

        mark_head = False
        first_bot_msg: Optional[MessageSending] = None
        reply_message_ids = []  # 记录实际发送的消息ID
        bot_user_info = UserInfo(
            user_id=global_config.BOT_QQ,
            user_nickname=global_config.BOT_NICKNAME,
            platform=anchor_message.message_info.platform,
        )

        for i, msg_text in enumerate(response_set):
            # 为每个消息片段生成唯一ID
            part_message_id = f"{thinking_id}_{i}"
            message_segment = Seg(type="text", data=msg_text)
            bot_message = MessageSending(
                message_id=part_message_id,  # 使用片段的唯一ID
                chat_stream=chat,
                bot_user_info=bot_user_info,
                sender_info=anchor_message.message_info.user_info,
                message_segment=message_segment,
                reply=anchor_message,  # 回复原始锚点
                is_head=not mark_head,
                is_emoji=False,
                thinking_start_time=thinking_start_time,  # 传递原始思考开始时间
            )
            try:
                if not mark_head:
                    mark_head = True
                    first_bot_msg = bot_message  # 保存第一个成功发送的消息对象
                    await self.heart_fc_sender.type_and_send_message(bot_message, type=False)
                else:
                    await self.heart_fc_sender.type_and_send_message(bot_message, type=True)

                reply_message_ids.append(part_message_id)  # 记录我们生成的ID

            except Exception as e:
                logger.error(
                    f"{self.log_prefix}[Sender-{thinking_id}] 发送回复片段 {i} ({part_message_id}) 时失败: {e}"
                )
                # 这里可以选择是继续发送下一个片段还是中止

        # 在尝试发送完所有片段后，完成原始的 thinking_id 状态
        try:
            await self.heart_fc_sender.complete_thinking(chat_id, thinking_id)
        except Exception as e:
            logger.error(f"{self.log_prefix}[Sender-{thinking_id}] 完成思考状态 {thinking_id} 时出错: {e}")

        self._current_cycle.set_response_info(
            response_text=response_set,  # 保留原始文本
            anchor_message_id=anchor_message.message_info.message_id,  # 保留锚点ID
            reply_message_ids=reply_message_ids,  # 添加实际发送的ID列表
        )

        return first_bot_msg  # 返回第一个成功发送的消息对象

    async def _handle_emoji(self, anchor_message: Optional[MessageRecv], response_set: List[str], send_emoji: str = ""):
        """处理表情包 (尝试锚定到 anchor_message)，使用 HeartFCSender"""
        if not anchor_message or not anchor_message.chat_stream:
            logger.error(f"{self.log_prefix} 无法处理表情包，缺少有效的锚点消息或聊天流。")
            return

        chat = anchor_message.chat_stream

        emoji_raw = await emoji_manager.get_emoji_for_text(send_emoji)

        if emoji_raw:
            emoji_path, description = emoji_raw

            emoji_cq = image_path_to_base64(emoji_path)
            thinking_time_point = round(time.time(), 2)  # 用于唯一ID
            message_segment = Seg(type="emoji", data=emoji_cq)
            bot_user_info = UserInfo(
                user_id=global_config.BOT_QQ,
                user_nickname=global_config.BOT_NICKNAME,
                platform=anchor_message.message_info.platform,
            )
            bot_message = MessageSending(
                message_id="me" + str(thinking_time_point),  # 表情消息的唯一ID
                chat_stream=chat,
                bot_user_info=bot_user_info,
                sender_info=anchor_message.message_info.user_info,
                message_segment=message_segment,
                reply=anchor_message,  # 回复原始锚点
                is_head=False,  # 表情通常不是头部消息
                is_emoji=True,
                # 不需要 thinking_start_time
            )

            try:
                await self.heart_fc_sender.send_and_store(bot_message)
            except Exception as e:
                logger.error(f"{self.log_prefix} 发送表情包 {bot_message.message_info.message_id} 时失败: {e}")

    def get_cycle_history(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取循环历史记录

        参数:
            last_n: 获取最近n个循环的信息，如果为None则获取所有历史记录

        返回:
            List[Dict[str, Any]]: 循环历史记录列表
        """
        history = list(self._cycle_history)
        if last_n is not None:
            history = history[-last_n:]
        return [cycle.to_dict() for cycle in history]

    def get_last_cycle_info(self) -> Optional[Dict[str, Any]]:
        """获取最近一个循环的信息"""
        if self._cycle_history:
            return self._cycle_history[-1].to_dict()
        return None
