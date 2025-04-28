import asyncio
import traceback
from typing import Optional, Coroutine, Callable, Any, List

from src.common.logger_manager import get_logger

# Need manager types for dependency injection
from src.heart_flow.mai_state_manager import MaiStateManager, MaiStateInfo
from src.heart_flow.subheartflow_manager import SubHeartflowManager
from src.heart_flow.interest_logger import InterestLogger


logger = get_logger("background_tasks")

# 新增随机停用间隔 (5 分钟)
RANDOM_DEACTIVATION_INTERVAL_SECONDS = 300
# 新增兴趣评估间隔
INTEREST_EVAL_INTERVAL_SECONDS = 5


class BackgroundTaskManager:
    """管理 Heartflow 的后台周期性任务。"""

    def __init__(
        self,
        mai_state_info: MaiStateInfo,  # Needs current state info
        mai_state_manager: MaiStateManager,
        subheartflow_manager: SubHeartflowManager,
        interest_logger: InterestLogger,
        update_interval: int,
        cleanup_interval: int,
        log_interval: int,
        # 新增兴趣评估间隔参数
        interest_eval_interval: int = INTEREST_EVAL_INTERVAL_SECONDS,
        # 新增随机停用间隔参数
        random_deactivation_interval: int = RANDOM_DEACTIVATION_INTERVAL_SECONDS,
    ):
        self.mai_state_info = mai_state_info
        self.mai_state_manager = mai_state_manager
        self.subheartflow_manager = subheartflow_manager
        self.interest_logger = interest_logger

        # Intervals
        self.update_interval = update_interval
        self.cleanup_interval = cleanup_interval
        self.log_interval = log_interval
        self.interest_eval_interval = interest_eval_interval  # 存储兴趣评估间隔
        self.random_deactivation_interval = random_deactivation_interval  # 存储随机停用间隔

        # Task references
        self._state_update_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._logging_task: Optional[asyncio.Task] = None
        self._interest_eval_task: Optional[asyncio.Task] = None  # 新增兴趣评估任务引用
        self._random_deactivation_task: Optional[asyncio.Task] = None  # 新增随机停用任务引用
        self._hf_judge_state_update_task: Optional[asyncio.Task] = None  # 新增状态评估任务引用
        self._tasks: List[Optional[asyncio.Task]] = []  # Keep track of all tasks

    async def start_tasks(self):
        """启动所有后台任务

        功能说明:
        - 启动核心后台任务: 状态更新、清理、日志记录、兴趣评估和随机停用
        - 每个任务启动前检查是否已在运行
        - 将任务引用保存到任务列表
        """

        # 任务配置列表: (任务变量名, 任务函数, 任务名称, 日志级别, 额外日志信息, 任务对象引用属性名)
        task_configs = [
            (
                self._state_update_task,
                lambda: self._run_state_update_cycle(self.update_interval),
                "hf_state_update",
                "debug",
                f"聊天状态更新任务已启动 间隔:{self.update_interval}s",
                "_state_update_task",
            ),
            (
                self._hf_judge_state_update_task,
                lambda: self._run_hf_judge_state_update_cycle(60),
                "hf_judge_state_update",
                "debug",
                f"状态评估任务已启动 间隔:{60}s",
                "_hf_judge_state_update_task",
            ),
            (
                self._cleanup_task,
                self._run_cleanup_cycle,
                "hf_cleanup",
                "info",
                f"清理任务已启动 间隔:{self.cleanup_interval}s",
                "_cleanup_task",
            ),
            (
                self._logging_task,
                self._run_logging_cycle,
                "hf_logging",
                "info",
                f"日志任务已启动 间隔:{self.log_interval}s",
                "_logging_task",
            ),
            # 新增兴趣评估任务配置
            (
                self._interest_eval_task,
                self._run_interest_eval_cycle,
                "hf_interest_eval",
                "debug",  # 设为debug，避免过多日志
                f"兴趣评估任务已启动 间隔:{self.interest_eval_interval}s",
                "_interest_eval_task",
            ),
        ]

        # 统一启动所有任务
        for _task_var, task_func, task_name, log_level, log_msg, task_attr_name in task_configs:
            # 检查任务变量是否存在且未完成
            current_task_var = getattr(self, task_attr_name)
            if current_task_var is None or current_task_var.done():
                new_task = asyncio.create_task(task_func(), name=task_name)
                setattr(self, task_attr_name, new_task)  # 更新任务变量
                if new_task not in self._tasks:  # 避免重复添加
                    self._tasks.append(new_task)

                # 根据配置记录不同级别的日志
                getattr(logger, log_level)(log_msg)
            else:
                logger.warning(f"{task_name}任务已在运行")

    async def stop_tasks(self):
        """停止所有后台任务。

        该方法会:
        1. 遍历所有后台任务并取消未完成的任务
        2. 等待所有取消操作完成
        3. 清空任务列表
        """
        logger.info("正在停止所有后台任务...")
        cancelled_count = 0

        # 第一步：取消所有运行中的任务
        for task in self._tasks:
            if task and not task.done():
                task.cancel()  # 发送取消请求
                cancelled_count += 1

        # 第二步：处理取消结果
        if cancelled_count > 0:
            logger.debug(f"正在等待{cancelled_count}个任务完成取消...")
            # 使用gather等待所有取消操作完成，忽略异常
            await asyncio.gather(*[t for t in self._tasks if t and t.cancelled()], return_exceptions=True)
            logger.info(f"成功取消{cancelled_count}个后台任务")
        else:
            logger.info("没有需要取消的后台任务")

        # 第三步：清空任务列表
        self._tasks = []  # 重置任务列表

    async def _run_periodic_loop(
        self, task_name: str, interval: int, task_func: Callable[..., Coroutine[Any, Any, None]], **kwargs
    ):
        """周期性任务主循环"""
        while True:
            start_time = asyncio.get_event_loop().time()
            # logger.debug(f"开始执行后台任务: {task_name}")

            try:
                await task_func(**kwargs)  # 执行实际任务
            except asyncio.CancelledError:
                logger.info(f"任务 {task_name} 已取消")
                break
            except Exception as e:
                logger.error(f"任务 {task_name} 执行出错: {e}")
                logger.error(traceback.format_exc())

            # 计算并执行间隔等待
            elapsed = asyncio.get_event_loop().time() - start_time
            sleep_time = max(0, interval - elapsed)
            # if sleep_time < 0.1:  # 任务超时处理, DEBUG 时可能干扰断点
            #     logger.warning(f"任务 {task_name} 超时执行 ({elapsed:.2f}s > {interval}s)")
            await asyncio.sleep(sleep_time)

        logger.debug(f"任务循环结束: {task_name}")  # 调整日志信息

    async def _perform_state_update_work(self):
        """执行状态更新工作"""
        previous_status = self.mai_state_info.get_current_state()
        next_state = self.mai_state_manager.check_and_decide_next_state(self.mai_state_info)

        state_changed = False

        if next_state is not None:
            state_changed = self.mai_state_info.update_mai_status(next_state)

            # 处理保持离线状态的特殊情况
            if not state_changed and next_state == previous_status == self.mai_state_info.mai_status.OFFLINE:
                self.mai_state_info.reset_state_timer()
                logger.debug("[后台任务] 保持离线状态并重置计时器")
                state_changed = True  # 触发后续处理

        if state_changed:
            current_state = self.mai_state_info.get_current_state()
            await self.subheartflow_manager.enforce_subheartflow_limits()

            # 状态转换处理

            if (
                current_state == self.mai_state_info.mai_status.OFFLINE
                and previous_status != self.mai_state_info.mai_status.OFFLINE
            ):
                logger.info("检测到离线，停用所有子心流")
                await self.subheartflow_manager.deactivate_all_subflows()

    async def _perform_hf_judge_state_update_work(self):
        """调用llm检测是否转换ABSENT-CHAT状态"""
        logger.info("[状态评估任务] 开始基于LLM评估子心流状态...")
        await self.subheartflow_manager.evaluate_and_transition_subflows_by_llm()

    async def _perform_cleanup_work(self):
        """执行子心流清理任务
        1. 获取需要清理的不活跃子心流列表
        2. 逐个停止这些子心流
        3. 记录清理结果
        """
        # 获取需要清理的子心流列表(包含ID和原因)
        flows_to_stop = self.subheartflow_manager.get_inactive_subheartflows()

        if not flows_to_stop:
            return  # 没有需要清理的子心流直接返回

        logger.info(f"准备删除 {len(flows_to_stop)} 个不活跃(1h)子心流")
        stopped_count = 0

        # 逐个停止子心流
        for flow_id in flows_to_stop:
            success = await self.subheartflow_manager.delete_subflow(flow_id)
            if success:
                stopped_count += 1
                logger.debug(f"[清理任务] 已停止子心流 {flow_id}")

        # 记录最终清理结果
        logger.info(f"[清理任务] 清理完成, 共停止 {stopped_count}/{len(flows_to_stop)} 个子心流")

    async def _perform_logging_work(self):
        """执行一轮状态日志记录。"""
        await self.interest_logger.log_all_states()

    # --- 新增兴趣评估工作函数 ---
    async def _perform_interest_eval_work(self):
        """执行一轮子心流兴趣评估与提升检查。"""
        # 直接调用 subheartflow_manager 的方法，并传递当前状态信息
        await self.subheartflow_manager.evaluate_interest_and_promote()

    # --- 结束新增 ---

    # --- 结束新增 ---

    # --- Specific Task Runners --- #
    async def _run_state_update_cycle(self, interval: int):
        await self._run_periodic_loop(
            task_name="State Update", interval=interval, task_func=self._perform_state_update_work
        )

    async def _run_hf_judge_state_update_cycle(self, interval: int):
        await self._run_periodic_loop(
            task_name="State Update", interval=interval, task_func=self._perform_hf_judge_state_update_work
        )

    async def _run_cleanup_cycle(self):
        await self._run_periodic_loop(
            task_name="Subflow Cleanup", interval=self.cleanup_interval, task_func=self._perform_cleanup_work
        )

    async def _run_logging_cycle(self):
        await self._run_periodic_loop(
            task_name="State Logging", interval=self.log_interval, task_func=self._perform_logging_work
        )

    # --- 新增兴趣评估任务运行器 ---
    async def _run_interest_eval_cycle(self):
        await self._run_periodic_loop(
            task_name="Interest Evaluation",
            interval=self.interest_eval_interval,
            task_func=self._perform_interest_eval_work,
        )
