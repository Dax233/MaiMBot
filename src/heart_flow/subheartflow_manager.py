import asyncio
import time
import random
from typing import Dict, Any, Optional, List

# 导入日志模块
from src.common.logger import get_module_logger

# 导入聊天流管理模块
from src.plugins.chat.chat_stream import chat_manager

# 导入心流相关类
from src.heart_flow.sub_heartflow import SubHeartflow, ChatState
from src.heart_flow.mai_state_manager import MaiState, MaiStateInfo
from .observation import ChattingObservation

# 初始化日志记录器
logger = get_module_logger("subheartflow_manager")

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

    async def evaluate_interest_and_promote(self):
        """评估CHAT状态的子心流兴趣度，满足条件则提升到FOCUSED状态"""
        logger.debug("[子心流管理器] 开始兴趣评估周期...")
        evaluated_count = 0
        promoted_count = 0

        # 使用快照安全遍历
        subflows_snapshot = list(self.subheartflows.values())

        for sub_hf in subflows_snapshot:
            flow_id = sub_hf.subheartflow_id
            if flow_id in self.subheartflows and self.subheartflows[flow_id].chat_state.chat_status == ChatState.CHAT:
                evaluated_count += 1
                stream_name = chat_manager.get_stream_name(flow_id) or flow_id
                log_prefix = f"[{stream_name}]"

                should_promote = await sub_hf.should_evaluate_reply()
                if should_promote:
                    logger.info(f"{log_prefix} 兴趣评估触发升级: CHAT -> FOCUSED")
                    states_num = (
                        self.count_subflows_by_state(ChatState.ABSENT),
                        self.count_subflows_by_state(ChatState.CHAT),
                        self.count_subflows_by_state(ChatState.FOCUSED),
                    )
                    await sub_hf.set_chat_state(ChatState.FOCUSED, states_num)
                    if (
                        self.subheartflows.get(flow_id)
                        and self.subheartflows[flow_id].chat_state.chat_status == ChatState.FOCUSED
                    ):
                        promoted_count += 1
                        logger.debug(f"{log_prefix} 成功升级到FOCUSED状态")
                    else:
                        logger.info(f"{log_prefix} 升级FOCUSED可能被限制阻止")

        if evaluated_count > 0:
            logger.debug(f"[子心流管理器] 评估完成. 评估{evaluated_count}个CHAT流, 升级{promoted_count}个到FOCUSED")

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
                minds.append(subheartflow.current_mind)
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
