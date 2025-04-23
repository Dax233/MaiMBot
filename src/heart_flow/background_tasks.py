import asyncio
import traceback
from typing import Optional, Coroutine, Callable, Any, List

from src.common.logger import get_module_logger

# Need manager types for dependency injection
from src.heart_flow.mai_state_manager import MaiStateManager, MaiStateInfo
from src.heart_flow.subheartflow_manager import SubHeartflowManager
from src.heart_flow.interest_logger import InterestLogger

logger = get_module_logger("background_tasks")


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
        inactive_threshold: int,
    ):
        self.mai_state_info = mai_state_info
        self.mai_state_manager = mai_state_manager
        self.subheartflow_manager = subheartflow_manager
        self.interest_logger = interest_logger

        # Intervals
        self.update_interval = update_interval
        self.cleanup_interval = cleanup_interval
        self.log_interval = log_interval
        self.inactive_threshold = inactive_threshold  # For cleanup task

        # Task references
        self._state_update_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._logging_task: Optional[asyncio.Task] = None
        self._tasks: List[Optional[asyncio.Task]] = []  # Keep track of all tasks

    async def start_tasks(self):
        """启动所有后台任务"""
        # 状态更新任务
        if self._state_update_task is None or self._state_update_task.done():
            self._state_update_task = asyncio.create_task(
                self._run_state_update_cycle(self.update_interval), name="hf_state_update"
            )
            self._tasks.append(self._state_update_task)
            logger.debug(f"聊天状态更新任务已启动 间隔:{self.update_interval}s")
        else:
            logger.warning("状态更新任务已在运行")

        # 清理任务
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._run_cleanup_cycle(), name="hf_cleanup")
            self._tasks.append(self._cleanup_task)
            logger.info(f"清理任务已启动 间隔:{self.cleanup_interval}s 阈值:{self.inactive_threshold}s")
        else:
            logger.warning("清理任务已在运行")

        # 日志任务
        if self._logging_task is None or self._logging_task.done():
            self._logging_task = asyncio.create_task(self._run_logging_cycle(), name="hf_logging")
            self._tasks.append(self._logging_task)
            logger.info(f"日志任务已启动 间隔:{self.log_interval}s")
        else:
            logger.warning("日志任务已在运行")

        # # 初始状态检查
        # initial_state = self.mai_state_info.get_current_state()
        # if initial_state != self.mai_state_info.mai_status.OFFLINE:
        #     logger.info(f"初始状态:{initial_state.value} 触发初始激活检查")
        #     asyncio.create_task(self.subheartflow_manager.activate_random_subflows_to_chat(initial_state))

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
            if sleep_time < 0.1:  # 任务超时处理
                logger.warning(f"任务 {task_name} 超时执行 ({elapsed:.2f}s > {interval}s)")
            await asyncio.sleep(sleep_time)

        # 非离线状态时评估兴趣
        if self.mai_state_info.get_current_state() != self.mai_state_info.mai_status.OFFLINE:
            await self.subheartflow_manager.evaluate_interest_and_promote()

        logger.debug(f"任务循环结束, 当前状态: {self.mai_state_info.get_current_state().value}")

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
            await self.subheartflow_manager.enforce_subheartflow_limits(current_state)

            # 状态转换处理
            if (
                previous_status == self.mai_state_info.mai_status.OFFLINE
                and current_state != self.mai_state_info.mai_status.OFFLINE
            ):
                logger.info("[后台任务] 主状态激活，触发子流激活")
                await self.subheartflow_manager.activate_random_subflows_to_chat(current_state)
            elif (
                current_state == self.mai_state_info.mai_status.OFFLINE
                and previous_status != self.mai_state_info.mai_status.OFFLINE
            ):
                logger.info("[后台任务] 主状态离线，触发子流停用")
                await self.subheartflow_manager.deactivate_all_subflows()

    async def _perform_cleanup_work(self):
        """执行一轮子心流清理操作。"""
        flows_to_stop = self.subheartflow_manager.cleanup_inactive_subheartflows(self.inactive_threshold)
        if flows_to_stop:
            logger.info(f"[Background Task Cleanup] Attempting to stop {len(flows_to_stop)} inactive flows...")
            stopped_count = 0
            for flow_id, reason in flows_to_stop:
                if await self.subheartflow_manager.stop_subheartflow(flow_id, f"定期清理: {reason}"):
                    stopped_count += 1
            logger.info(f"[Background Task Cleanup] Cleanup cycle finished. Stopped {stopped_count} inactive flows.")
        else:
            logger.debug("[Background Task Cleanup] Cleanup cycle finished. No inactive flows found.")

    async def _perform_logging_work(self):
        """执行一轮兴趣日志记录。"""
        await self.interest_logger.log_interest_states()

    # --- Specific Task Runners --- #
    async def _run_state_update_cycle(self, interval: int):
        await self._run_periodic_loop(
            task_name="State Update", interval=interval, task_func=self._perform_state_update_work
        )

    async def _run_cleanup_cycle(self):
        await self._run_periodic_loop(
            task_name="Subflow Cleanup", interval=self.cleanup_interval, task_func=self._perform_cleanup_work
        )

    async def _run_logging_cycle(self):
        await self._run_periodic_loop(
            task_name="Interest Logging", interval=self.log_interval, task_func=self._perform_logging_work
        )
