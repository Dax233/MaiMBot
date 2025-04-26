import asyncio
import time
import random
from typing import Dict, Any, Optional, List
import json  # 导入 json 模块

# 导入日志模块
from src.common.logger import get_module_logger, LogConfig, SUBHEARTFLOW_MANAGER_STYLE_CONFIG

# 导入聊天流管理模块
from src.plugins.chat.chat_stream import chat_manager

# 导入心流相关类
from src.heart_flow.sub_heartflow import SubHeartflow, ChatState
from src.heart_flow.mai_state_manager import MaiStateInfo
from .observation import ChattingObservation

# 导入LLM请求工具
from src.plugins.models.utils_model import LLMRequest
from src.config.config import global_config

import traceback

# 初始化日志记录器

subheartflow_manager_log_config = LogConfig(
    console_format=SUBHEARTFLOW_MANAGER_STYLE_CONFIG["console_format"],
    file_format=SUBHEARTFLOW_MANAGER_STYLE_CONFIG["file_format"],
)
logger = get_module_logger("subheartflow_manager", config=subheartflow_manager_log_config)

# 子心流管理相关常量
INACTIVE_THRESHOLD_SECONDS = 3600  # 子心流不活跃超时时间(秒)


class SubHeartflowManager:
    """管理所有活跃的 SubHeartflow 实例。"""

    def __init__(self, mai_state_info: MaiStateInfo):
        self.subheartflows: Dict[Any, "SubHeartflow"] = {}
        self._lock = asyncio.Lock()  # 用于保护 self.subheartflows 的访问
        self.mai_state_info: MaiStateInfo = mai_state_info  # 存储传入的 MaiStateInfo 实例

        # 为 LLM 状态评估创建一个 LLMRequest 实例
        # 使用与 Heartflow 相同的模型和参数
        self.llm_state_evaluator = LLMRequest(
            model=global_config.llm_heartflow,  # 与 Heartflow 一致
            temperature=0.6,  # 与 Heartflow 一致
            max_tokens=1000,  # 与 Heartflow 一致 (虽然可能不需要这么多)
            request_type="subheartflow_state_eval",  # 保留特定的请求类型
        )

    def get_all_subheartflows(self) -> List["SubHeartflow"]:
        """获取所有当前管理的 SubHeartflow 实例列表 (快照)。"""
        return list(self.subheartflows.values())

    async def get_or_create_subheartflow(self, subheartflow_id: Any) -> Optional["SubHeartflow"]:
        """获取或创建指定ID的子心流实例

        Args:
            subheartflow_id: 子心流唯一标识符
            # mai_states 参数已被移除，使用 self.mai_state_info

        Returns:
            成功返回SubHeartflow实例，失败返回None
        """
        async with self._lock:
            # 检查是否已存在该子心流
            if subheartflow_id in self.subheartflows:
                subflow = self.subheartflows[subheartflow_id]
                if subflow.should_stop:
                    logger.warning(f"尝试获取已停止的子心流 {subheartflow_id}，正在重新激活")
                    subflow.should_stop = False  # 重置停止标志

                subflow.last_active_time = time.time()  # 更新活跃时间
                # logger.debug(f"获取到已存在的子心流: {subheartflow_id}")
                return subflow

            try:
                # 初始化子心流, 传入存储的 mai_state_info
                new_subflow = SubHeartflow(subheartflow_id, self.mai_state_info)

                # 异步初始化
                await new_subflow.initialize()

                # 添加聊天观察者
                observation = ChattingObservation(chat_id=subheartflow_id)
                new_subflow.add_observation(observation)

                # 注册子心流
                self.subheartflows[subheartflow_id] = new_subflow
                heartflow_name = chat_manager.get_stream_name(subheartflow_id) or subheartflow_id
                logger.info(f"[{heartflow_name}] 开始接收消息")

                # 启动后台任务
                asyncio.create_task(new_subflow.subheartflow_start_working())

                return new_subflow
            except Exception as e:
                logger.error(f"创建子心流 {subheartflow_id} 失败: {e}", exc_info=True)
                return None

    async def sleep_subheartflow(self, subheartflow_id: Any, reason: str) -> bool:
        """停止指定的子心流并清理资源"""
        subheartflow = self.subheartflows.get(subheartflow_id)
        if not subheartflow:
            return False

        stream_name = chat_manager.get_stream_name(subheartflow_id) or subheartflow_id
        logger.info(f"[子心流管理] 正在停止 {stream_name}, 原因: {reason}")

        try:
            # 设置状态为ABSENT释放资源
            if subheartflow.chat_state.chat_status != ChatState.ABSENT:
                logger.debug(f"[子心流管理] 设置 {stream_name} 状态为ABSENT")
                await subheartflow.change_chat_state(ChatState.ABSENT)
            else:
                logger.debug(f"[子心流管理] {stream_name} 已是ABSENT状态")
        except Exception as e:
            logger.error(f"[子心流管理] 设置ABSENT状态失败: {e}")

    def get_inactive_subheartflows(self, max_age_seconds=INACTIVE_THRESHOLD_SECONDS):
        """识别并返回需要清理的不活跃(处于ABSENT状态超过一小时)子心流(id, 原因)"""
        current_time = time.time()
        flows_to_stop = []

        for subheartflow_id, subheartflow in list(self.subheartflows.items()):
            state = subheartflow.chat_state.chat_status
            if state != ChatState.ABSENT:
                continue
            subheartflow.update_last_chat_state_time()
            absent_last_time = subheartflow.chat_state_last_time
            if max_age_seconds and (current_time - absent_last_time) > max_age_seconds:
                flows_to_stop.append(subheartflow_id)

        return flows_to_stop

    async def enforce_subheartflow_limits(self):
        """根据主状态限制停止超额子心流(优先停不活跃的)"""
        # 使用 self.mai_state_info 获取当前状态和限制
        current_mai_state = self.mai_state_info.get_current_state()
        normal_limit = current_mai_state.get_normal_chat_max_num()
        focused_limit = current_mai_state.get_focused_chat_max_num()
        logger.debug(f"[限制] 状态:{current_mai_state.value}, 普通限:{normal_limit}, 专注限:{focused_limit}")

        # 分类统计当前子心流
        normal_flows = []
        focused_flows = []
        for flow_id, flow in list(self.subheartflows.items()):
            if flow.chat_state.chat_status == ChatState.CHAT:
                normal_flows.append((flow_id, getattr(flow, "last_active_time", 0)))
            elif flow.chat_state.chat_status == ChatState.FOCUSED:
                focused_flows.append((flow_id, getattr(flow, "last_active_time", 0)))

        logger.debug(f"[限制] 当前数量 - 普通:{len(normal_flows)}, 专注:{len(focused_flows)}")
        stopped = 0

        # 处理普通聊天超额
        if len(normal_flows) > normal_limit:
            excess = len(normal_flows) - normal_limit
            logger.info(f"[限制] 普通聊天超额({len(normal_flows)}>{normal_limit}), 停止{excess}个")
            normal_flows.sort(key=lambda x: x[1])
            for flow_id, _ in normal_flows[:excess]:
                if await self.sleep_subheartflow(flow_id, f"普通聊天超额(限{normal_limit})"):
                    stopped += 1

        # 处理专注聊天超额(需重新统计)
        focused_flows = [
            (fid, t)
            for fid, f in list(self.subheartflows.items())
            if (t := getattr(f, "last_active_time", 0)) and f.chat_state.chat_status == ChatState.FOCUSED
        ]
        if len(focused_flows) > focused_limit:
            excess = len(focused_flows) - focused_limit
            logger.info(f"[限制] 专注聊天超额({len(focused_flows)}>{focused_limit}), 停止{excess}个")
            focused_flows.sort(key=lambda x: x[1])
            for flow_id, _ in focused_flows[:excess]:
                if await self.sleep_subheartflow(flow_id, f"专注聊天超额(限{focused_limit})"):
                    stopped += 1

        if stopped:
            logger.info(f"[限制] 已停止{stopped}个子心流, 剩余:{len(self.subheartflows)}")
        else:
            logger.debug(f"[限制] 无需停止, 当前总数:{len(self.subheartflows)}")

    async def deactivate_all_subflows(self):
        """将所有子心流的状态更改为 ABSENT (例如主状态变为OFFLINE时调用)"""
        # logger.info("[停用] 开始将所有子心流状态设置为 ABSENT")
        # 使用 list() 创建一个当前值的快照，防止在迭代时修改字典
        flows_to_update = list(self.subheartflows.values())

        if not flows_to_update:
            logger.debug("[停用] 无活跃子心流，无需操作")
            return

        changed_count = 0
        for subflow in flows_to_update:
            flow_id = subflow.subheartflow_id
            stream_name = chat_manager.get_stream_name(flow_id) or flow_id
            # 再次检查子心流是否仍然存在于管理器中，以防万一在迭代过程中被移除

            if subflow.chat_state.chat_status != ChatState.ABSENT:
                logger.debug(
                    f"正在将子心流 {stream_name} 的状态从 {subflow.chat_state.chat_status.value} 更改为 ABSENT"
                )
                try:
                    # 调用 change_chat_state 将状态设置为 ABSENT
                    await subflow.change_chat_state(ChatState.ABSENT)
                    # 验证状态是否真的改变了
                    if (
                        flow_id in self.subheartflows
                        and self.subheartflows[flow_id].chat_state.chat_status == ChatState.ABSENT
                    ):
                        changed_count += 1
                    else:
                        logger.warning(
                            f"[停用] 尝试更改子心流 {stream_name} 状态后，状态仍未变为 ABSENT 或子心流已消失。"
                        )
                except Exception as e:
                    logger.error(f"[停用] 更改子心流 {stream_name} 状态为 ABSENT 时出错: {e}", exc_info=True)
            else:
                logger.debug(f"[停用] 子心流 {stream_name} 已处于 ABSENT 状态，无需更改。")

        logger.info(
            f"下限完成，共处理 {len(flows_to_update)} 个子心流，成功将 {changed_count} 个子心流的状态更改为 ABSENT。"
        )

    async def evaluate_interest_and_promote(self):
        """评估子心流兴趣度，满足条件且未达上限则提升到FOCUSED状态（基于start_hfc_probability）"""
        log_prefix = "[兴趣评估]"
        # 使用 self.mai_state_info 获取当前状态和限制
        current_state = self.mai_state_info.get_current_state()
        focused_limit = current_state.get_focused_chat_max_num()

        if int(time.time()) % 20 == 0:  # 每20秒输出一次
            logger.debug(f"{log_prefix} 当前状态 ({current_state.value}) 可以在{focused_limit}个群激情聊天")

        if focused_limit <= 0:
            # logger.debug(f"{log_prefix} 当前状态 ({current_state.value}) 不允许 FOCUSED 子心流")
            return

        current_focused_count = self.count_subflows_by_state(ChatState.FOCUSED)
        if current_focused_count >= focused_limit:
            logger.debug(f"{log_prefix} 已达专注上限 ({current_focused_count}/{focused_limit})")
            return

        for sub_hf in list(self.subheartflows.values()):
            flow_id = sub_hf.subheartflow_id
            stream_name = chat_manager.get_stream_name(flow_id) or flow_id

            # 跳过非CHAT状态或已经是FOCUSED状态的子心流
            if sub_hf.chat_state.chat_status == ChatState.FOCUSED:
                continue

            from .mai_state_manager import enable_unlimited_hfc_chat

            if not enable_unlimited_hfc_chat:
                if sub_hf.chat_state.chat_status != ChatState.CHAT:
                    continue

            # 检查是否满足提升概率
            if random.random() >= sub_hf.interest_chatting.start_hfc_probability:
                continue

            # 再次检查是否达到上限
            if current_focused_count >= focused_limit:
                logger.debug(f"{log_prefix} [{stream_name}] 已达专注上限")
                break

            # 获取最新状态并执行提升
            current_subflow = self.subheartflows.get(flow_id)
            if not current_subflow:
                continue

            logger.info(
                f"{log_prefix} [{stream_name}] 触发 认真水群 (概率={current_subflow.interest_chatting.start_hfc_probability:.2f})"
            )

            # 执行状态提升
            await current_subflow.change_chat_state(ChatState.FOCUSED)

            # 验证提升结果
            if (
                final_subflow := self.subheartflows.get(flow_id)
            ) and final_subflow.chat_state.chat_status == ChatState.FOCUSED:
                current_focused_count += 1

    async def randomly_deactivate_subflows(self, deactivation_probability: float = 0.1):
        """以一定概率将 FOCUSED 或 CHAT 状态的子心流回退到 ABSENT 状态。"""
        log_prefix_manager = "[子心流管理器-随机停用]"
        logger.debug(f"{log_prefix_manager} 开始随机停用检查... (概率: {deactivation_probability:.0%})")

        # 使用快照安全遍历
        subflows_snapshot = list(self.subheartflows.values())
        deactivated_count = 0

        try:
            for sub_hf in subflows_snapshot:
                flow_id = sub_hf.subheartflow_id
                stream_name = chat_manager.get_stream_name(flow_id) or flow_id
                log_prefix_flow = f"[{stream_name}]"
                current_state = sub_hf.chat_state.chat_status

                # 只处理 FOCUSED 或 CHAT 状态
                if current_state not in [ChatState.FOCUSED, ChatState.CHAT]:
                    continue

                # 检查随机概率
                if random.random() < deactivation_probability:
                    logger.info(
                        f"{log_prefix_manager} {log_prefix_flow} 随机触发停用 (从 {current_state.value}) -> ABSENT"
                    )

                    # 获取当前实例以检查最新状态
                    current_subflow = self.subheartflows.get(flow_id)
                    if not current_subflow or current_subflow.chat_state.chat_status != current_state:
                        logger.warning(f"{log_prefix_manager} {log_prefix_flow} 尝试停用时状态已改变或实例消失，跳过。")
                        continue

                    # --- 状态设置 --- #
                    # 注意：这里传递的状态数量是 *停用前* 的状态数量
                    await current_subflow.change_chat_state(ChatState.ABSENT)

                    # --- 状态验证 (可选) ---
                    final_subflow = self.subheartflows.get(flow_id)
                    if final_subflow:
                        final_state = final_subflow.chat_state.chat_status
                        if final_state == ChatState.ABSENT:
                            logger.debug(
                                f"{log_prefix_manager} {log_prefix_flow} 成功从 {current_state.value} 停用到 ABSENT 状态"
                            )
                            deactivated_count += 1
                        else:
                            logger.warning(
                                f"{log_prefix_manager} {log_prefix_flow} 尝试停用到 ABSENT 后状态仍为 {final_state.value}"
                            )
                    else:
                        logger.warning(f"{log_prefix_manager} {log_prefix_flow} 停用后验证时子心流 {flow_id} 消失")

        except Exception as e:
            logger.error(f"{log_prefix_manager} 随机停用周期出错: {e}", exc_info=True)

        if deactivated_count > 0:
            logger.info(f"{log_prefix_manager} 随机停用周期结束, 成功停用 {deactivated_count} 个子心流。")
        else:
            logger.debug(f"{log_prefix_manager} 随机停用周期结束, 未停用任何子心流。")

    async def evaluate_and_transition_subflows_by_llm(self):
        """
        使用LLM评估每个子心流的状态，并根据LLM的判断执行状态转换（ABSENT <-> CHAT）。
        注意：此函数包含对假设的LLM函数的调用。
        """
        log_prefix = "[LLM状态评估]"
        logger.info(f"{log_prefix} 开始基于LLM评估子心流状态...")

        # 获取当前状态和限制，用于CHAT激活检查
        current_mai_state = self.mai_state_info.get_current_state()
        chat_limit = current_mai_state.get_normal_chat_max_num()

        transitioned_to_chat = 0
        transitioned_to_absent = 0

        async with self._lock:  # 在锁内获取快照并迭代
            subflows_snapshot = list(self.subheartflows.values())
            # 使用不上锁的版本，因为我们已经在锁内
            current_chat_count = self.count_subflows_by_state_nolock(ChatState.CHAT)

            if not subflows_snapshot:
                logger.info(f"{log_prefix} 当前没有子心流需要评估。")
                return

            for sub_hf in subflows_snapshot:
                flow_id = sub_hf.subheartflow_id
                stream_name = chat_manager.get_stream_name(flow_id) or flow_id
                current_subflow_state = sub_hf.chat_state.chat_status

                # --- 获取观察内容 ---
                # 从 sub_hf.observations 获取 ChattingObservation 并提取信息
                _observation_summary = "没有可用的观察信息。"  # 默认值
                try:
                    # 检查 observations 列表是否存在且不为空

                    # 假设第一个观察者是 ChattingObservation
                    first_observation = sub_hf.observations[0]
                    if isinstance(first_observation, ChattingObservation):
                        # 组合中期记忆和当前聊天内容
                        current_chat = first_observation.talking_message_str or "当前无聊天内容。"
                        combined_summary = f"当前聊天内容:\n{current_chat}"
                    else:
                        logger.warning(f"{log_prefix} [{stream_name}] 第一个观察者不是 ChattingObservation 类型。")

                except Exception as e:
                    logger.warning(f"{log_prefix} [{stream_name}] 获取观察信息失败: {e}", exc_info=True)
                    # 保留默认值或错误信息
                    combined_summary = f"获取观察信息时出错: {e}"

                # --- 获取麦麦状态 ---
                mai_state_description = f"麦麦当前状态: {current_mai_state.value}。"

                # --- 针对 ABSENT 状态 ---
                if current_subflow_state == ChatState.ABSENT:
                    # 构建Prompt
                    prompt = (
                        f"子心流 [{stream_name}] 当前处于非活跃(ABSENT)状态.\n"
                        f"{mai_state_description}\n"
                        f"最近观察到的内容摘要:\n---\n{combined_summary}\n---\n"
                        f"基于以上信息，该子心流是否表现出足够的活跃迹象或重要性，"
                        f"值得将其唤醒并进入常规聊天(CHAT)状态？\n"
                        f"请以 JSON 格式回答，包含一个键 'decision'，其值为 true 或 false.\n"
                        f'例如：{{"decision": true}}\n'
                        f"请只输出有效的 JSON 对象。"
                    )

                    # 调用LLM评估
                    should_activate = await self._llm_evaluate_state_transition(prompt)
                    if should_activate is None:  # 处理解析失败或意外情况
                        logger.warning(f"{log_prefix} [{stream_name}] LLM评估返回无效结果，跳过。")
                        continue

                    if should_activate:
                        # 检查CHAT限额
                        # 使用不上锁的版本，因为我们已经在锁内
                        current_chat_count = self.count_subflows_by_state_nolock(ChatState.CHAT)
                        if current_chat_count < chat_limit:
                            logger.info(
                                f"{log_prefix} [{stream_name}] LLM建议激活到CHAT状态，且未达上限({current_chat_count}/{chat_limit})。正在尝试转换..."
                            )
                            await sub_hf.change_chat_state(ChatState.CHAT)
                            if sub_hf.chat_state.chat_status == ChatState.CHAT:
                                transitioned_to_chat += 1
                            else:
                                logger.warning(f"{log_prefix} [{stream_name}] 尝试激活到CHAT失败。")
                        else:
                            logger.info(
                                f"{log_prefix} [{stream_name}] LLM建议激活到CHAT状态，但已达到上限({current_chat_count}/{chat_limit})。跳过转换。"
                            )

                # --- 针对 CHAT 状态 ---
                elif current_subflow_state == ChatState.CHAT:
                    # 构建Prompt
                    prompt = (
                        f"子心流 [{stream_name}] 当前处于常规聊天(CHAT)状态.\n"
                        f"{mai_state_description}\n"
                        f"最近观察到的内容摘要:\n---\n{combined_summary}\n---\n"
                        f"基于以上信息，该子心流是否表现出不活跃、对话结束或不再需要关注的迹象，"
                        f"应该让其进入休眠(ABSENT)状态？\n"
                        f"请以 JSON 格式回答，包含一个键 'decision'，其值为 true (表示应休眠) 或 false (表示不应休眠).\n"
                        f'例如：{{"decision": true}}\n'
                        f"请只输出有效的 JSON 对象。"
                    )

                    # 调用LLM评估
                    should_deactivate = await self._llm_evaluate_state_transition(prompt)
                    if should_deactivate is None:  # 处理解析失败或意外情况
                        logger.warning(f"{log_prefix} [{stream_name}] LLM评估返回无效结果，跳过。")
                        continue

                    if should_deactivate:
                        logger.info(f"{log_prefix} [{stream_name}] LLM建议进入ABSENT状态。正在尝试转换...")
                        await sub_hf.change_chat_state(ChatState.ABSENT)
                        if sub_hf.chat_state.chat_status == ChatState.ABSENT:
                            transitioned_to_absent += 1

        logger.info(
            f"{log_prefix} LLM评估周期结束。"
            f" 成功转换到CHAT: {transitioned_to_chat}."
            f" 成功转换到ABSENT: {transitioned_to_absent}."
        )

    async def _llm_evaluate_state_transition(self, prompt: str) -> Optional[bool]:
        """
        使用 LLM 评估是否应进行状态转换，期望 LLM 返回 JSON 格式。

        Args:
            prompt: 提供给 LLM 的提示信息，要求返回 {"decision": true/false}。

        Returns:
            Optional[bool]: 如果成功解析 LLM 的 JSON 响应并提取了 'decision' 键的值，则返回该布尔值。
                        如果 LLM 调用失败、返回无效 JSON 或 JSON 中缺少 'decision' 键或其值不是布尔型，则返回 None。
        """
        log_prefix = "[LLM状态评估]"
        try:
            # --- 真实的 LLM 调用 ---
            response_text, _ = await self.llm_state_evaluator.generate_response_async(prompt)
            logger.debug(
                f"{log_prefix} 使用模型 {self.llm_state_evaluator.model_name} 评估，原始响应: ```{response_text}```"
            )

            # --- 解析 JSON 响应 ---
            try:
                # 尝试去除可能的Markdown代码块标记
                cleaned_response = response_text.strip().strip("`").strip()
                if cleaned_response.startswith("json"):
                    cleaned_response = cleaned_response[4:].strip()

                data = json.loads(cleaned_response)
                decision = data.get("decision")  # 使用 .get() 避免 KeyError

                if isinstance(decision, bool):
                    logger.debug(f"{log_prefix} LLM评估结果 (来自JSON): {'建议转换' if decision else '建议不转换'}")
                    return decision
                else:
                    logger.warning(
                        f"{log_prefix} LLM 返回的 JSON 中 'decision' 键的值不是布尔型: {decision}。响应: {response_text}"
                    )
                    return None  # 值类型不正确

            except json.JSONDecodeError as json_err:
                logger.warning(f"{log_prefix} LLM 返回的响应不是有效的 JSON: {json_err}。响应: {response_text}")
                # 尝试在非JSON响应中查找关键词作为后备方案 (可选)
                if "true" in response_text.lower():
                    logger.debug(f"{log_prefix} 在非JSON响应中找到 'true'，解释为建议转换")
                    return True
                if "false" in response_text.lower():
                    logger.debug(f"{log_prefix} 在非JSON响应中找到 'false'，解释为建议不转换")
                    return False
                return None  # JSON 解析失败，也未找到关键词
            except Exception as parse_err:  # 捕获其他可能的解析错误
                logger.warning(f"{log_prefix} 解析 LLM JSON 响应时发生意外错误: {parse_err}。响应: {response_text}")
                return None

        except Exception as e:
            logger.error(f"{log_prefix} 调用 LLM 或处理其响应时出错: {e}", exc_info=True)
            traceback.print_exc()
            return None  # LLM 调用或处理失败

    def count_subflows_by_state(self, state: ChatState) -> int:
        """统计指定状态的子心流数量"""
        count = 0
        # 遍历所有子心流实例
        for subheartflow in self.subheartflows.values():
            # 检查子心流状态是否匹配
            if subheartflow.chat_state.chat_status == state:
                count += 1
        return count

    def count_subflows_by_state_nolock(self, state: ChatState) -> int:
        """
        统计指定状态的子心流数量 (不上锁版本)。
        警告：仅应在已持有 self._lock 的上下文中使用此方法。
        """
        count = 0
        for subheartflow in self.subheartflows.values():
            if subheartflow.chat_state.chat_status == state:
                count += 1
        return count

    def get_active_subflow_minds(self) -> List[str]:
        """获取所有活跃(非ABSENT)子心流的当前想法"""
        minds = []
        for subheartflow in self.subheartflows.values():
            # 检查子心流是否活跃(非ABSENT状态)
            if subheartflow.chat_state.chat_status != ChatState.ABSENT:
                minds.append(subheartflow.sub_mind.current_mind)
        return minds

    def update_main_mind_in_subflows(self, main_mind: str):
        """更新所有子心流的主心流想法"""
        updated_count = sum(
            1
            for _, subheartflow in list(self.subheartflows.items())
            if subheartflow.subheartflow_id in self.subheartflows
        )
        logger.debug(f"[子心流管理器] 更新了{updated_count}个子心流的主想法")

    async def delete_subflow(self, subheartflow_id: Any):
        """删除指定的子心流。"""
        async with self._lock:
            subflow = self.subheartflows.pop(subheartflow_id, None)
            if subflow:
                logger.info(f"正在删除 SubHeartflow: {subheartflow_id}...")
                try:
                    # 调用 shutdown 方法确保资源释放
                    await subflow.shutdown()
                    logger.info(f"SubHeartflow {subheartflow_id} 已成功删除。")
                except Exception as e:
                    logger.error(f"删除 SubHeartflow {subheartflow_id} 时出错: {e}", exc_info=True)
            else:
                logger.warning(f"尝试删除不存在的 SubHeartflow: {subheartflow_id}")
