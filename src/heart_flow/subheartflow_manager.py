import asyncio
import time
import random
from typing import Dict, Any, Optional, List

# 导入日志模块
from src.common.logger import get_module_logger, LogConfig, SUBHEARTFLOW_MANAGER_STYLE_CONFIG

# 导入聊天流管理模块
from src.plugins.chat.chat_stream import chat_manager

# 导入心流相关类
from src.heart_flow.sub_heartflow import SubHeartflow, ChatState
from src.heart_flow.mai_state_manager import MaiState, MaiStateInfo
from .observation import ChattingObservation

# 初始化日志记录器

subheartflow_manager_log_config = LogConfig(
    console_format=SUBHEARTFLOW_MANAGER_STYLE_CONFIG["console_format"],
    file_format=SUBHEARTFLOW_MANAGER_STYLE_CONFIG["file_format"],
)
logger = get_module_logger("subheartflow_manager", config=subheartflow_manager_log_config)

# 子心流管理相关常量
INACTIVE_THRESHOLD_SECONDS = 1200  # 子心流不活跃超时时间(秒)


class SubHeartflowManager:
    """管理所有活跃的 SubHeartflow 实例。"""

    def __init__(self):
        self.subheartflows: Dict[Any, "SubHeartflow"] = {}
        self._lock = asyncio.Lock()  # 用于保护 self.subheartflows 的访问

    def get_all_subheartflows(self) -> List["SubHeartflow"]:
        """获取所有当前管理的 SubHeartflow 实例列表 (快照)。"""
        return list(self.subheartflows.values())

    def get_all_subheartflows_ids(self) -> List[Any]:
        """获取所有当前管理的 SubHeartflow ID 列表。"""
        return list(self.subheartflows.keys())

    def get_subheartflow(self, subheartflow_id: Any) -> Optional["SubHeartflow"]:
        """获取指定 ID 的 SubHeartflow 实例。"""
        # 注意：这里没有加锁，假设读取操作相对安全或在已知上下文中调用
        # 如果并发写操作很多，get 也应该加锁
        subflow = self.subheartflows.get(subheartflow_id)
        if subflow:
            subflow.last_active_time = time.time()  # 获取时更新活动时间
        return subflow

    async def create_or_get_subheartflow(
        self, subheartflow_id: Any, mai_states: MaiStateInfo
    ) -> Optional["SubHeartflow"]:
        """获取或创建指定ID的子心流实例

        Args:
            subheartflow_id: 子心流唯一标识符
            mai_states: 当前麦麦状态信息

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

            # 创建新的子心流实例
            logger.info(f"子心流 {subheartflow_id} 不存在，正在创建...")
            try:
                # 初始化子心流
                new_subflow = SubHeartflow(subheartflow_id, mai_states)

                # 添加聊天观察者
                observation = ChattingObservation(chat_id=subheartflow_id)
                new_subflow.add_observation(observation)

                # 注册子心流
                self.subheartflows[subheartflow_id] = new_subflow
                logger.info(f"子心流 {subheartflow_id} 创建成功")

                # 启动后台任务
                asyncio.create_task(new_subflow.subheartflow_start_working())

                return new_subflow
            except Exception as e:
                logger.error(f"创建子心流 {subheartflow_id} 失败: {e}", exc_info=True)
                return None

    async def stop_subheartflow(self, subheartflow_id: Any, reason: str) -> bool:
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
                states_num = (
                    self.count_subflows_by_state(ChatState.ABSENT),
                    self.count_subflows_by_state(ChatState.CHAT),
                    self.count_subflows_by_state(ChatState.FOCUSED),
                )
                await subheartflow.set_chat_state(ChatState.ABSENT, states_num)
            else:
                logger.debug(f"[子心流管理] {stream_name} 已是ABSENT状态")
        except Exception as e:
            logger.error(f"[子心流管理] 设置ABSENT状态失败: {e}")

        # 停止子心流内部循环
        subheartflow.should_stop = True

        # 取消后台任务
        task = subheartflow.task
        if task and not task.done():
            task.cancel()
            logger.debug(f"[子心流管理] 已取消 {stream_name} 的后台任务")

        # 从管理字典中移除
        if subheartflow_id in self.subheartflows:
            del self.subheartflows[subheartflow_id]
            logger.debug(f"[子心流管理] 已移除 {stream_name}")
            return True
        else:
            logger.warning(f"[子心流管理] {stream_name} 已被提前移除")
            return False

    def cleanup_inactive_subheartflows(self, max_age_seconds=INACTIVE_THRESHOLD_SECONDS):
        """识别并返回需要清理的不活跃子心流(id, 原因)"""
        current_time = time.time()
        flows_to_stop = []

        for subheartflow_id, subheartflow in list(self.subheartflows.items()):
            # 只检查有interest_chatting的子心流
            if hasattr(subheartflow, "interest_chatting") and subheartflow.interest_chatting:
                last_interact = subheartflow.interest_chatting.last_interaction_time
                if max_age_seconds and (current_time - last_interact) > max_age_seconds:
                    reason = f"不活跃时间({current_time - last_interact:.0f}s) > 阈值({max_age_seconds}s)"
                    name = chat_manager.get_stream_name(subheartflow_id) or subheartflow_id
                    logger.debug(f"[清理] 标记 {name} 待移除: {reason}")
                    flows_to_stop.append((subheartflow_id, reason))

        if flows_to_stop:
            logger.info(f"[清理] 发现 {len(flows_to_stop)} 个不活跃子心流")
        return flows_to_stop

    async def enforce_subheartflow_limits(self, current_mai_state: MaiState):
        """根据主状态限制停止超额子心流(优先停不活跃的)"""
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
                if await self.stop_subheartflow(flow_id, f"普通聊天超额(限{normal_limit})"):
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
                if await self.stop_subheartflow(flow_id, f"专注聊天超额(限{focused_limit})"):
                    stopped += 1

        if stopped:
            logger.info(f"[限制] 已停止{stopped}个子心流, 剩余:{len(self.subheartflows)}")
        else:
            logger.debug(f"[限制] 无需停止, 当前总数:{len(self.subheartflows)}")

    async def activate_random_subflows_to_chat(self, current_mai_state: MaiState):
        """主状态激活时，随机选择ABSENT子心流进入CHAT状态"""
        limit = current_mai_state.get_normal_chat_max_num()
        if limit <= 0:
            logger.info("[激活] 当前状态不允许CHAT子心流")
            return

        # 获取所有ABSENT状态的子心流
        absent_flows = [flow for flow in self.subheartflows.values() if flow.chat_state.chat_status == ChatState.ABSENT]

        num_to_activate = min(limit, len(absent_flows))
        if num_to_activate <= 0:
            logger.info(f"[激活] 无可用ABSENT子心流(限额:{limit}, 可用:{len(absent_flows)})")
            return

        logger.info(f"[激活] 随机选择{num_to_activate}个ABSENT子心流进入CHAT状态")
        activated_count = 0

        for flow in random.sample(absent_flows, num_to_activate):
            flow_id = flow.subheartflow_id
            stream_name = chat_manager.get_stream_name(flow_id) or flow_id

            if flow_id not in self.subheartflows:
                logger.warning(f"[激活] 跳过{stream_name}, 子心流已不存在")
                continue

            logger.debug(f"[激活] 正在激活子心流{stream_name}")

            states_num = (
                self.count_subflows_by_state(ChatState.ABSENT),
                self.count_subflows_by_state(ChatState.CHAT),
                self.count_subflows_by_state(ChatState.FOCUSED),
            )

            await flow.set_chat_state(ChatState.CHAT, states_num)

            if flow.chat_state.chat_status == ChatState.CHAT:
                activated_count += 1
            else:
                logger.warning(f"[激活] {stream_name}状态设置失败")

        logger.info(f"[激活] 完成, 成功激活{activated_count}个子心流")

    async def deactivate_all_subflows(self):
        """停用所有子心流(主状态变为OFFLINE时调用)"""
        logger.info("[停用] 开始停用所有子心流")
        flow_ids = list(self.subheartflows.keys())

        if not flow_ids:
            logger.info("[停用] 无活跃子心流")
            return

        stopped_count = 0
        for flow_id in flow_ids:
            if await self.stop_subheartflow(flow_id, "主状态离线"):
                stopped_count += 1

        logger.info(f"[停用] 完成, 尝试停止{len(flow_ids)}个, 成功{stopped_count}个")

    async def evaluate_interest_and_promote(self, current_mai_state: MaiStateInfo):
        """评估子心流兴趣度，满足条件且未达上限则提升到FOCUSED状态（基于start_hfc_probability）"""
        log_prefix_manager = "[子心流管理器-兴趣评估]"
        logger.debug(f"{log_prefix_manager} 开始周期... 当前状态: {current_mai_state.get_current_state().value}")

        # 获取 FOCUSED 状态的数量上限
        current_state_enum = current_mai_state.get_current_state()
        focused_limit = current_state_enum.get_focused_chat_max_num()
        if focused_limit <= 0:
            logger.debug(
                f"{log_prefix_manager} 当前状态 ({current_state_enum.value}) 不允许 FOCUSED 子心流, 跳过提升检查。"
            )
            return

        # 获取当前 FOCUSED 状态的数量 (初始值)
        current_focused_count = self.count_subflows_by_state(ChatState.FOCUSED)
        logger.debug(f"{log_prefix_manager} 专注上限: {focused_limit}, 当前专注数: {current_focused_count}")

        # 使用快照安全遍历
        subflows_snapshot = list(self.subheartflows.values())
        promoted_count = 0  # 记录本次提升的数量
        try:
            for sub_hf in subflows_snapshot:
                flow_id = sub_hf.subheartflow_id
                stream_name = chat_manager.get_stream_name(flow_id) or flow_id
                log_prefix_flow = f"[{stream_name}]"

                # 只处理 CHAT 状态的子心流
                # The code snippet is checking if the `chat_status` attribute of `sub_hf.chat_state` is not equal to
                # `ChatState.CHAT`. If the condition is met, the code will continue to the next iteration of the loop
                # or block of code where this snippet is located.
                # if sub_hf.chat_state.chat_status != ChatState.CHAT:
                #     continue

                # 检查是否满足提升概率
                should_hfc = random.random() < sub_hf.interest_chatting.start_hfc_probability
                if not should_hfc:
                    continue

                # --- 关键检查：检查 FOCUSED 数量是否已达上限 ---
                # 注意：在循环内部再次获取当前数量，因为之前的提升可能已经改变了计数
                # 使用已经记录并在循环中更新的 current_focused_count
                if current_focused_count >= focused_limit:
                    logger.debug(
                        f"{log_prefix_manager} {log_prefix_flow} 达到专注上限 ({current_focused_count}/{focused_limit}), 无法提升。概率={sub_hf.interest_chatting.start_hfc_probability:.2f}"
                    )
                    continue  # 跳过这个子心流，继续检查下一个

                # --- 执行提升 ---
                # 获取当前实例以检查最新状态 (防御性编程)
                current_subflow = self.subheartflows.get(flow_id)
                if not current_subflow:
                    logger.warning(f"{log_prefix_manager} {log_prefix_flow} 尝试提升时状态已改变或实例消失，跳过。")
                    continue

                logger.info(
                    f"{log_prefix_manager} {log_prefix_flow} 兴趣评估触发升级 (prob={sub_hf.interest_chatting.start_hfc_probability:.2f}, 上限:{focused_limit}, 当前:{current_focused_count}) -> FOCUSED"
                )

                states_num = (
                    self.count_subflows_by_state(ChatState.ABSENT),
                    self.count_subflows_by_state(ChatState.CHAT),  # 这个值在提升前计算
                    current_focused_count,  # 这个值在提升前计算
                )

                # --- 状态设置 ---
                original_state = current_subflow.chat_state.chat_status  # 记录原始状态
                await current_subflow.set_chat_state(ChatState.FOCUSED, states_num)

                # --- 状态验证 ---
                final_subflow = self.subheartflows.get(flow_id)
                if final_subflow:
                    final_state = final_subflow.chat_state.chat_status
                    if final_state == ChatState.FOCUSED:
                        logger.debug(
                            f"{log_prefix_manager} {log_prefix_flow} 成功从 {original_state.value} 升级到 FOCUSED 状态"
                        )
                        promoted_count += 1
                        # 提升成功后，更新当前专注计数，以便后续检查能使用最新值
                        current_focused_count += 1
                    elif final_state == original_state:  # 状态未变
                        logger.warning(
                            f"{log_prefix_manager} {log_prefix_flow} 尝试从 {original_state.value} 升级 FOCUSED 失败，状态仍为: {final_state.value} (可能被内部逻辑阻止)"
                        )
                    else:  # 状态变成其他了?
                        logger.warning(
                            f"{log_prefix_manager} {log_prefix_flow} 尝试从 {original_state.value} 升级 FOCUSED 后状态变为 {final_state.value}"
                        )
                else:  # 子心流消失了?
                    logger.warning(f"{log_prefix_manager} {log_prefix_flow} 升级后验证时子心流 {flow_id} 消失")

        except Exception as e:
            logger.error(f"{log_prefix_manager} 兴趣评估周期出错: {e}", exc_info=True)

        if promoted_count > 0:
            logger.info(f"{log_prefix_manager} 评估周期结束, 成功提升 {promoted_count} 个子心流到 FOCUSED。")
        else:
            logger.debug(f"{log_prefix_manager} 评估周期结束, 未提升任何子心流。")

    async def randomly_deactivate_subflows(self, deactivation_probability: float = 0.3):
        """以一定概率将 FOCUSED 或 CHAT 状态的子心流回退到 ABSENT 状态。"""
        log_prefix_manager = "[子心流管理器-随机停用]"
        logger.debug(f"{log_prefix_manager} 开始随机停用检查... (概率: {deactivation_probability:.0%})")

        # 使用快照安全遍历
        subflows_snapshot = list(self.subheartflows.values())
        deactivated_count = 0

        # 预先计算状态数量，因为 set_chat_state 需要
        states_num_before = (
            self.count_subflows_by_state(ChatState.ABSENT),
            self.count_subflows_by_state(ChatState.CHAT),
            self.count_subflows_by_state(ChatState.FOCUSED),
        )

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
                    await current_subflow.set_chat_state(ChatState.ABSENT, states_num_before)

                    # --- 状态验证 (可选) ---
                    final_subflow = self.subheartflows.get(flow_id)
                    if final_subflow:
                        final_state = final_subflow.chat_state.chat_status
                        if final_state == ChatState.ABSENT:
                            logger.debug(
                                f"{log_prefix_manager} {log_prefix_flow} 成功从 {current_state.value} 停用到 ABSENT 状态"
                            )
                            deactivated_count += 1
                            # 注意：停用后不需要更新 states_num_before，因为它只用于 set_chat_state 的限制检查
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

    def count_subflows_by_state(self, state: ChatState) -> int:
        """统计指定状态的子心流数量

        Args:
            state: 要统计的聊天状态枚举值

        Returns:
            int: 处于该状态的子心流数量
        """
        count = 0
        # 遍历所有子心流实例
        for subheartflow in self.subheartflows.values():
            # 检查子心流状态是否匹配
            if subheartflow.chat_state.chat_status == state:
                count += 1
        return count

    def get_active_subflow_minds(self) -> List[str]:
        """获取所有活跃(非ABSENT)子心流的当前想法

        返回:
            List[str]: 包含所有活跃子心流当前想法的列表
        """
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

    async def deactivate_subflow(self, subheartflow_id: Any):
        """停用并移除指定的子心流。"""
        async with self._lock:
            subflow = self.subheartflows.pop(subheartflow_id, None)
            if subflow:
                logger.info(f"正在停用 SubHeartflow: {subheartflow_id}...")
                try:
                    # --- 调用 shutdown 方法 ---
                    await subflow.shutdown()
                    # --- 结束调用 ---
                    logger.info(f"SubHeartflow {subheartflow_id} 已成功停用。")
                except Exception as e:
                    logger.error(f"停用 SubHeartflow {subheartflow_id} 时出错: {e}", exc_info=True)
            else:
                logger.warning(f"尝试停用不存在的 SubHeartflow: {subheartflow_id}")

    async def cleanup_inactive_subflows(self, inactive_threshold_seconds: int):
        """清理长时间不活跃的子心流。"""
        current_time = time.time()
        inactive_ids = []
        # 不加锁地迭代，识别不活跃的 ID
        for sub_id, subflow in self.subheartflows.items():
            # 检查 last_active_time 是否存在且是数值
            last_active = getattr(subflow, "last_active_time", 0)
            if isinstance(last_active, (int, float)):
                if current_time - last_active > inactive_threshold_seconds:
                    inactive_ids.append(sub_id)
                    logger.info(
                        f"发现不活跃的 SubHeartflow: {sub_id} (上次活跃: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_active))})"
                    )
            else:
                logger.warning(f"SubHeartflow {sub_id} 的 last_active_time 无效: {last_active}。跳过清理检查。")

        if inactive_ids:
            logger.info(f"准备清理 {len(inactive_ids)} 个不活跃的 SubHeartflows: {inactive_ids}")
            # 逐个停用（deactivate_subflow 会加锁）
            tasks = [self.deactivate_subflow(sub_id) for sub_id in inactive_ids]
            await asyncio.gather(*tasks)
            logger.info("不活跃的 SubHeartflows 清理完成。")
        # else:
        # logger.debug("没有发现不活跃的 SubHeartflows 需要清理。")
