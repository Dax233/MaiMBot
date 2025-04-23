import asyncio
import time
import json
import os
import traceback
from typing import TYPE_CHECKING, Dict, List

from src.common.logger import get_module_logger

# Need chat_manager to get stream names
from src.plugins.chat.chat_stream import chat_manager

if TYPE_CHECKING:
    from src.heart_flow.subheartflow_manager import SubHeartflowManager
    from src.heart_flow.sub_heartflow import SubHeartflow  # For type hint in get_interest_states

logger = get_module_logger("interest_logger")

# Consider moving log directory/filename constants here
LOG_DIRECTORY = "logs/interest"
HISTORY_LOG_FILENAME = "interest_history.log"


class InterestLogger:
    """负责定期记录所有子心流的兴趣状态到日志文件。"""

    def __init__(self, subheartflow_manager: "SubHeartflowManager"):
        self.subheartflow_manager = subheartflow_manager
        self._history_log_file_path = os.path.join(LOG_DIRECTORY, HISTORY_LOG_FILENAME)
        self._ensure_log_directory()

    def _ensure_log_directory(self):
        """确保日志目录存在。"""
        try:
            os.makedirs(LOG_DIRECTORY, exist_ok=True)
            logger.info(f"已确保日志目录 '{LOG_DIRECTORY}' 存在")
        except OSError as e:
            logger.error(f"创建日志目录 '{LOG_DIRECTORY}' 出错: {e}")

    async def get_all_interest_states(self) -> Dict[str, Dict]:
        """并发获取所有活跃子心流的当前兴趣状态。"""
        _states = {}
        # Get snapshot from the manager
        all_flows: List["SubHeartflow"] = self.subheartflow_manager.get_all_subheartflows()
        tasks = []
        results = {}

        if not all_flows:
            logger.debug("未找到任何子心流状态")
            return results

        # logger.debug(f"正在获取 {len(all_flows)} 个子心流的兴趣状态...")
        for subheartflow in all_flows:
            if self.subheartflow_manager.get_subheartflow(subheartflow.subheartflow_id):
                tasks.append(
                    asyncio.create_task(
                        subheartflow.get_interest_state(), name=f"get_state_{subheartflow.subheartflow_id}"
                    )
                )
            else:
                logger.warning(f"子心流 {subheartflow.subheartflow_id} 在创建任务前已消失")

        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=5.0)

            if pending:
                logger.warning(f"获取兴趣状态超时，有 {len(pending)} 个任务未完成")
                for task in pending:
                    task.cancel()

            for task in done:
                try:
                    stream_id_str = task.get_name().split("get_state_")[-1]
                    stream_id = stream_id_str
                except IndexError:
                    logger.error(f"无法从任务名 {task.get_name()} 中提取 stream_id")
                    continue

                try:
                    result = task.result()
                    results[stream_id] = result
                except asyncio.CancelledError:
                    logger.warning(f"获取子心流 {stream_id} 兴趣状态的任务已取消(超时)", exc_info=False)
                except Exception as e:
                    logger.warning(f"获取子心流 {stream_id} 兴趣状态出错: {e}")

        logger.trace(f"成功获取 {len(results)} 个兴趣状态")
        return results

    async def log_interest_states(self):
        """获取所有子心流的兴趣状态并写入日志文件。"""
        # logger.debug("开始定期记录兴趣状态...")
        try:
            current_timestamp = time.time()
            all_interest_states = await self.get_all_interest_states()

            if not all_interest_states:
                logger.debug("没有获取到任何兴趣状态")
                return

            count = 0
            try:
                with open(self._history_log_file_path, "a", encoding="utf-8") as f:
                    items_snapshot = list(all_interest_states.items())
                    for stream_id, state in items_snapshot:
                        group_name = stream_id
                        try:
                            chat_stream = chat_manager.get_stream(stream_id)
                            if chat_stream and chat_stream.group_info:
                                group_name = chat_stream.group_info.group_name
                            elif chat_stream and not chat_stream.group_info:
                                group_name = (
                                    f"私聊_{chat_stream.user_info.user_nickname}"
                                    if chat_stream.user_info
                                    else stream_id
                                )
                        except Exception as e:
                            logger.trace(f"无法获取 stream_id {stream_id} 的群组名: {e}")
                            pass

                        log_entry = {
                            "timestamp": round(current_timestamp, 2),
                            "stream_id": stream_id,
                            "interest_level": state.get("interest_level", 0.0),
                            "group_name": group_name,
                            "reply_probability": state.get("current_reply_probability", 0.0),
                            "is_above_threshold": state.get("is_above_threshold", False),
                        }
                        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                        count += 1
                # logger.debug(f"成功记录 {count} 条兴趣历史到 {self._history_log_file_path}")
            except IOError as e:
                logger.error(f"写入兴趣历史日志到 {self._history_log_file_path} 出错: {e}")

        except Exception as e:
            logger.error(f"定期记录兴趣历史时发生意外错误: {e}")
            logger.error(traceback.format_exc())
