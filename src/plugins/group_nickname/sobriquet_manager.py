import asyncio
import threading
import random
import time
import json
import re
from typing import Dict, Optional, List, Any
from pymongo.errors import OperationFailure, DuplicateKeyError
from src.common.logger_manager import get_logger
from src.common.database import db
from src.config.config import global_config
from src.chat.models.utils_model import LLMRequest
from .sobriquet_db import SobriquetDB # 用于写入和确保结构
from .sobriquet_mapper import _build_mapping_prompt
from .sobriquet_utils import select_sobriquets_for_prompt, format_sobriquet_prompt_injection
from src.chat.person_info.person_info import person_info_manager
# 从新的 profile_manager 导入
from src.experimental.profile.profile_manager import profile_manager # 使用单例实例

from src.chat.message_receive.chat_stream import ChatStream
from src.chat.message_receive.message import MessageRecv
from src.chat.utils.chat_message_builder import build_readable_messages, get_raw_msg_before_timestamp_with_chat

logger = get_logger("SobriquetManager")
logger_helper = get_logger("AsyncLoopHelper")


def run_async_loop(loop: asyncio.AbstractEventLoop, coro):
    """
    运行给定的协程直到完成，并确保循环最终关闭。
    """
    # ... (此函数保持不变) ...
    try:
        logger_helper.debug(f"Running coroutine in loop {id(loop)}...")
        result = loop.run_until_complete(coro)
        logger_helper.debug(f"Coroutine completed in loop {id(loop)}.")
        return result
    except asyncio.CancelledError:
        logger_helper.info(f"Coroutine in loop {id(loop)} was cancelled.")
    except Exception as e:
        logger_helper.error(f"Error in async loop {id(loop)}: {e}", exc_info=True)
    finally:
        try:
            all_tasks = asyncio.all_tasks(loop)
            current_task = asyncio.current_task(loop)
            tasks_to_cancel = [
                task for task in all_tasks if task is not current_task
            ]
            if tasks_to_cancel:
                logger_helper.info(f"Cancelling {len(tasks_to_cancel)} outstanding tasks in loop {id(loop)}...")
                for task in tasks_to_cancel:
                    task.cancel()
                loop.run_until_complete(asyncio.gather(*tasks_to_cancel, return_exceptions=True))
                logger_helper.info(f"Outstanding tasks cancelled in loop {id(loop)}.")
            if loop.is_running():
                loop.stop()
                logger_helper.info(f"Asyncio loop {id(loop)} stopped.")
            if not loop.is_closed():
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()
                logger_helper.info(f"Asyncio loop {id(loop)} closed.")
        except Exception as close_err:
            logger_helper.error(f"Error during asyncio loop cleanup for loop {id(loop)}: {close_err}", exc_info=True)


class SobriquetManager:
    """
    管理群组绰号分析、处理、存储和使用的单例类。
    封装了 LLM 调用、后台处理线程和数据库交互 (通过 SobriquetDB 和 ProfileManager)。
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        # ... (此方法保持不变) ...
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    logger.info("正在创建 SobriquetManager 单例实例...")
                    cls._instance = super(SobriquetManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """
        初始化 SobriquetManager。
        """
        # ... (大部分初始化逻辑保持不变, 除了 db_handler 初始化) ...
        if hasattr(self, "_initialized") and self._initialized:
            return

        with self._lock:
            if hasattr(self, "_initialized") and self._initialized:
                return

            logger.info("正在初始化 SobriquetManager 组件...")
            self.is_enabled = global_config.group_sobriquet.enable_sobriquet_mapping

            # 数据库处理器 - SobriquetDB 用于写入操作
            profile_info_collection_name = global_config.group_sobriquet.get("profile_info_collection_name", "profile_info")
            profile_info_collection = getattr(db, profile_info_collection_name, None)
            
            if profile_info_collection is None:
                logger.warning(f"未能从数据库获取 '{profile_info_collection_name}' 集合。请检查数据库配置和 common.database.py。")

            self.db_handler = SobriquetDB(profile_info_collection) 
            if not self.db_handler.is_available(): # SobriquetDB 实例的可用性
                logger.error(f"SobriquetDB 数据库处理器初始化失败 (集合: {profile_info_collection_name})，SobriquetManager 功能受限。")
                self.is_enabled = False
            else:
                logger.info(f"SobriquetDB 数据库处理器 (集合: {profile_info_collection_name}) 初始化成功。")
            
            # ProfileManager 用于读取操作，它内部也会检查集合可用性
            if not profile_manager.is_available() and self.is_enabled:
                logger.error(f"ProfileManager 指向的集合 '{profile_info_collection_name}' 不可用，但 SobriquetManager 仍尝试启用。功能可能受限。")
                # 根据策略，如果 ProfileManager 不可用，也可以将 self.is_enabled 设为 False

            # LLM 映射器
            self.llm_mapper: Optional[LLMRequest] = None
            if self.is_enabled:
                try:
                    model_config = global_config.model.sobriquet_mapping
                    if model_config and model_config.get("name"):
                        self.llm_mapper = LLMRequest(
                            model=model_config,
                            temperature=model_config.get("temp", 0.5),
                            max_tokens=model_config.get("max_tokens", 256),
                            request_type="sobriquet_mapping",
                        )
                        logger.info("绰号映射 LLM 映射器初始化成功。")
                    else:
                        logger.warning("绰号映射 LLM 配置无效或缺失 'name'，功能禁用。")
                        self.is_enabled = False 
                except KeyError as ke:
                    logger.error(f"初始化绰号映射 LLM 时缺少配置项: {ke}，功能禁用。", exc_info=True)
                    self.llm_mapper = None
                    self.is_enabled = False
                except Exception as e:
                    logger.error(f"初始化绰号映射 LLM 映射器失败: {e}，功能禁用。", exc_info=True)
                    self.llm_mapper = None
                    self.is_enabled = False

            # 队列和线程
            self.queue_max_size = global_config.group_sobriquet.sobriquet_queue_max_size
            self.sobriquet_queue: asyncio.Queue = asyncio.Queue(maxsize=self.queue_max_size)
            self._stop_event = threading.Event()
            self._sobriquet_thread: Optional[threading.Thread] = None
            self.sleep_interval = global_config.group_sobriquet.sobriquet_process_sleep_interval

            self._initialized = True
            logger.info(f"SobriquetManager 初始化完成。当前启用状态: {self.is_enabled}")

    def start_processor(self):
        """启动后台处理线程（如果已启用且未运行）。"""
        if not self.is_enabled:
            logger.info("绰号处理功能已禁用，处理器未启动。")
            return
        max_sobriquets_cfg = global_config.group_sobriquet.get("max_sobriquets_in_prompt", 0)
        if not isinstance(max_sobriquets_cfg, int) or max_sobriquets_cfg <= 0:
            logger.error(f"[错误] 配置项 'max_sobriquets_in_prompt' ({max_sobriquets_cfg}) 不合适，绰号处理功能已禁用！")
            self.is_enabled = False 
            return

        if self._sobriquet_thread is None or not self._sobriquet_thread.is_alive():
            logger.info("正在启动绰号处理器线程...")
            self._stop_event.clear()
            self._sobriquet_thread = threading.Thread(
                target=self._run_processor_in_thread,
                daemon=True,
            )
            self._sobriquet_thread.start()
            logger.info(f"绰号处理器线程已启动 (ID: {self._sobriquet_thread.ident})")
        else:
            logger.warning("绰号处理器线程已在运行中。")

    def stop_processor(self):
        """停止后台处理线程。"""
        if self._sobriquet_thread and self._sobriquet_thread.is_alive():
            logger.info("正在停止绰号处理器线程...")
            self._stop_event.set()
            try:
                self._sobriquet_thread.join(timeout=10)
                if self._sobriquet_thread.is_alive():
                    logger.warning("绰号处理器线程在超时后仍未停止。")
            except Exception as e:
                logger.error(f"停止绰号处理器线程时出错: {e}", exc_info=True)
            finally:
                if self._sobriquet_thread and not self._sobriquet_thread.is_alive():
                    logger.info("绰号处理器线程已成功停止。")
                self._sobriquet_thread = None
        else:
            logger.info("绰号处理器线程未在运行或已被清理。")

    async def trigger_sobriquet_analysis(
        self,
        anchor_message: MessageRecv,
        bot_reply: List[str],
        chat_stream: Optional[ChatStream] = None,
    ):
        """
        准备数据并将其排队等待绰号分析（如果满足条件）。
        """
        if not self.is_enabled:
            return

        analysis_probability = global_config.group_sobriquet.get("sobriquet_analysis_probability", 1.0)
        if not (0 <= analysis_probability <= 1.0): 
            logger.warning(f"无效的绰号分析概率配置: {analysis_probability}，将使用默认值 1.0。")
            analysis_probability = 1.0
        
        if random.random() > analysis_probability:
            logger.debug(f"跳过绰号分析：随机概率 ({analysis_probability*100:.1f}%) 未命中。")
            return

        current_chat_stream = chat_stream or anchor_message.chat_stream
        if not current_chat_stream or not current_chat_stream.group_info:
            logger.debug("跳过绰号分析：非群聊或无效的聊天流。")
            return

        log_prefix = f"[{current_chat_stream.stream_id}]"
        try:
            history_limit = global_config.group_sobriquet.sobriquet_analysis_history_limit
            history_messages = get_raw_msg_before_timestamp_with_chat(
                chat_id=current_chat_stream.stream_id,
                timestamp=time.time(),
                limit=history_limit,
            )
            chat_history_str = await build_readable_messages(
                messages=history_messages,
                replace_bot_name=True,
                merge_messages=False,
                timestamp_mode="relative",
                read_mark=0.0,
                truncate=False,
            )
            bot_reply_str = " ".join(bot_reply) if bot_reply else ""
            group_id = str(current_chat_stream.group_info.group_id)
            platform = current_chat_stream.platform

            user_ids_in_history = {
                str(msg["user_info"]["user_id"]) for msg in history_messages if msg.get("user_info", {}).get("user_id")
            }
            user_name_map = {}
            if user_ids_in_history:
                try:
                    # 假设 relationship_manager 或 person_info_manager 提供此功能
                    names_data = await profile_manager.get_person_names_batch(platform, list(user_ids_in_history)) # MODIFIED: or relationship_manager
                except AttributeError: # 如果 profile_manager 没有这个方法，尝试 relationship_manager
                    from src.chat.person_info.relationship_manager import relationship_manager as rm
                    names_data = await rm.get_person_names_batch(platform, list(user_ids_in_history))
                except Exception as e:
                    logger.error(f"{log_prefix} 批量获取 person_name 时出错: {e}", exc_info=True)
                    names_data = {}

                for user_id_str_hist in user_ids_in_history:
                    if user_id_str_hist in names_data and names_data[user_id_str_hist]:
                        user_name_map[user_id_str_hist] = names_data[user_id_str_hist]
                    else:
                        latest_display_name = next(
                            (
                                m["user_info"].get("user_nickname") or m["user_info"].get("user_cardname") 
                                for m in reversed(history_messages)
                                if str(m["user_info"].get("user_id")) == user_id_str_hist and (m["user_info"].get("user_nickname") or m["user_info"].get("user_cardname"))
                            ),
                            None,
                        )
                        bot_user_id_str = str(global_config.bot.qq_account) 
                        user_name_map[user_id_str_hist] = (
                            latest_display_name or f"{global_config.bot.nickname}(你)" 
                            if user_id_str_hist == bot_user_id_str
                            else latest_display_name or f"用户({user_id_str_hist[-4:]})" 
                        )

            item = (chat_history_str, bot_reply_str, platform, group_id, user_name_map)
            await self._add_to_queue(item, platform, group_id)

        except Exception as e:
            logger.error(f"{log_prefix} 触发绰号分析时出错: {e}", exc_info=True)


    async def get_sobriquet_prompt_injection(self, chat_stream: ChatStream, message_list_before_now: List[Dict]) -> str:
        """
        获取并格式化用于 Prompt 注入的绰号信息字符串。
        现在调用 ProfileManager 来获取数据。
        """
        if not self.is_enabled or not chat_stream or not chat_stream.group_info:
            return ""

        # 确保 profile_manager 可用
        if not profile_manager.is_available():
            logger.warning("ProfileManager 不可用，无法获取绰号注入数据。")
            return ""

        log_prefix = f"[{chat_stream.stream_id}]"
        try:
            group_id = str(chat_stream.group_info.group_id)
            platform = chat_stream.platform
            user_ids_in_context = {
                str(msg["user_info"]["user_id"])
                for msg in message_list_before_now
                if msg.get("user_info", {}).get("user_id")
            }

            if not user_ids_in_context:
                recent_speakers_limit = global_config.group_sobriquet.get("recent_speakers_limit_for_injection", 5)
                recent_speakers = chat_stream.get_recent_speakers(limit=recent_speakers_limit)
                user_ids_in_context.update(str(speaker["user_id"]) for speaker in recent_speakers if speaker.get("user_id"))

            if not user_ids_in_context:
                logger.debug(f"{log_prefix} 未找到上下文用户用于绰号注入。")
                return ""
            
            # 调用 ProfileManager 获取格式化好的数据
            # 期望格式: {person_name: {"user_id": "uid_str", "sobriquets": [{"绰号A": 次数}, ...]}}
            all_sobriquets_data_for_prompt = await profile_manager.get_users_group_sobriquets_for_prompt_injection_data(
                platform, list(user_ids_in_context), group_id
            )
            
            if all_sobriquets_data_for_prompt:
                # select_sobriquets_for_prompt 和 format_sobriquet_prompt_injection 来自 .sobriquet_utils
                selected_sobriquets_with_uid = select_sobriquets_for_prompt(all_sobriquets_data_for_prompt)
                injection_str = format_sobriquet_prompt_injection(selected_sobriquets_with_uid)
                if injection_str:
                    logger.debug(f"{log_prefix} 生成的绰号 Prompt 注入 (部分):\n{injection_str.strip()[:200]}...")
                return injection_str
            else:
                logger.debug(f"{log_prefix} 未从 ProfileManager 获取到用于注入的绰号数据。")
                return ""

        except Exception as e:
            logger.error(f"{log_prefix} 获取绰号注入时出错: {e}", exc_info=True)
            return ""

    async def _add_to_queue(self, item: tuple, platform: str, group_id: str):
        """将项目异步添加到内部处理队列。"""
        # ... (此方法保持不变) ...
        try:
            await self.sobriquet_queue.put(item)
            logger.debug(
                f"已将项目添加到平台 '{platform}' 群组 '{group_id}' 的绰号队列。当前大小: {self.sobriquet_queue.qsize()}"
            )
        except asyncio.QueueFull:
            logger.warning(
                f"绰号队列已满 (最大={self.queue_max_size})。平台 '{platform}' 群组 '{group_id}' 的项目被丢弃。"
            )
        except Exception as e:
            logger.error(f"将项目添加到绰号队列时出错: {e}", exc_info=True)


    async def _analyze_and_update_sobriquets(self, item: tuple):
        """处理单个队列项目：调用 LLM 分析并更新数据库 (通过 SobriquetDB)。"""
        # ... (此方法中的数据库更新部分使用 self.db_handler，保持不变) ...
        if not isinstance(item, tuple) or len(item) != 5:
            logger.warning(f"从队列接收到无效项目: {type(item)} 内容: {item}")
            return

        chat_history_str, bot_reply, platform, group_id_str, user_name_map = item 
        log_prefix = f"[{platform}:{group_id_str}]"
        logger.debug(f"{log_prefix} 开始处理绰号分析任务...")

        if not self.llm_mapper: 
            logger.error(f"{log_prefix} LLM 映射器不可用，无法执行分析。")
            return
        # SobriquetDB 的可用性在初始化时已检查，但这里再次确认 self.db_handler
        if not self.db_handler or not self.db_handler.is_available():
            logger.error(f"{log_prefix} SobriquetDB 数据库处理器不可用，无法更新计数。")
            return

        analysis_result = await self._call_llm_for_analysis(chat_history_str, bot_reply, user_name_map)

        if analysis_result.get("is_exist") and analysis_result.get("data"):
            sobriquet_map_to_update = analysis_result["data"]
            logger.info(f"{log_prefix} LLM 找到绰号映射，准备更新数据库: {sobriquet_map_to_update}")

            for user_id_str, sobriquet_name in sobriquet_map_to_update.items():
                if not user_id_str or not sobriquet_name: 
                    logger.warning(f"{log_prefix} 跳过无效条目: user_id='{user_id_str}', sobriquet='{sobriquet_name}'")
                    continue
                
                try:
                    person_id = person_info_manager.get_person_id(platform, user_id_str)
                    if not person_id:
                        logger.error(
                            f"{log_prefix} 无法为 platform='{platform}', user_id='{user_id_str}' 生成 person_id，跳过此用户。"
                        )
                        continue
                    
                    # 使用 self.db_handler (SobriquetDB instance)
                    self.db_handler.ensure_profile_and_platform_user(person_id, platform, user_id_str)
                    self.db_handler.update_group_sobriquet_count(person_id, platform, group_id_str, sobriquet_name)
                    logger.debug(f"{log_prefix} 已为 person_id '{person_id}' (user_id '{user_id_str}') 更新/添加绰号 '{sobriquet_name}' 在群组 '{group_id_str}'。")

                except (OperationFailure, DuplicateKeyError) as db_err:
                    logger.exception(
                        f"{log_prefix} 数据库操作失败 ({type(db_err).__name__}): user_id {user_id_str}, 绰号 {sobriquet_name}. 错误: {db_err}"
                    )
                except Exception as e: 
                    logger.exception(f"{log_prefix} 处理用户 {user_id_str} 的绰号 '{sobriquet_name}' 时发生意外错误：{e}")
        else:
            logger.debug(f"{log_prefix} LLM 未找到可靠的绰号映射或分析失败。")


    async def _call_llm_for_analysis(
        self,
        chat_history_str: str,
        bot_reply: str,
        user_name_map: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        内部方法：调用 LLM 分析聊天记录和 Bot 回复，提取可靠的 用户ID-绰号 映射。
        """
        # ... (此方法保持不变) ...
        if not self.llm_mapper:
            logger.error("LLM 映射器未初始化，无法执行分析。")
            return {"is_exist": False}

        prompt = _build_mapping_prompt(chat_history_str, bot_reply, user_name_map)

        try:
            response_content, _, _ = await self.llm_mapper.generate_response(prompt)
            if not response_content:
                logger.warning("LLM 返回了空的绰号映射内容。")
                return {"is_exist": False}

            response_content_stripped = response_content.strip()
            json_str = ""
            markdown_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_content_stripped, re.DOTALL)
            if markdown_match:
                json_str = markdown_match.group(1).strip()
            else:
                brace_match = re.search(r"(\{.*?\})", response_content_stripped, re.DOTALL)
                if brace_match:
                    json_str = brace_match.group(1).strip()
                else:
                    logger.warning(f"LLM 响应似乎不包含有效的 JSON 对象。响应 (首200字符): {response_content_stripped[:200]}")
                    return {"is_exist": False}
            
            try:
                result = json.loads(json_str)
            except json.JSONDecodeError as json_err:
                logger.error(f"解析 LLM 响应 JSON 失败: {json_err}。"
                             f"\n尝试解析的 JSON 字符串: {json_str}"
                             f"\n原始 LLM 响应 (首500字符): {response_content_stripped[:500]}")
                return {"is_exist": False}

            if not isinstance(result, dict):
                logger.warning(f"LLM 响应 JSON 解析结果不是字典类型。解析结果: {result} (类型: {type(result)})")
                return {"is_exist": False}

            is_exist = result.get("is_exist")
            if not isinstance(is_exist, bool):
                logger.warning(f"LLM 响应中的 'is_exist' 字段不是布尔类型或缺失。值为: {is_exist}")
                return {"is_exist": False} 

            if is_exist: 
                original_data = result.get("data")
                if isinstance(original_data, dict) and original_data: 
                    filtered_data = self._filter_llm_results(original_data, user_name_map)
                    if not filtered_data:
                        logger.info("所有找到的绰号映射都被过滤掉了。")
                        return {"is_exist": False, "data": {}} 
                    else:
                        logger.info(f"过滤后的绰号映射: {filtered_data}")
                        return {"is_exist": True, "data": filtered_data}
                else: 
                    logger.warning(f"LLM 响应格式错误: is_exist=True 但 data 字段无效或为空。原始 data: {original_data}")
                    return {"is_exist": False, "data": {}} 
            else: 
                logger.info("LLM 明确指示未找到可靠的绰号映射 (is_exist=False)。")
                return {"is_exist": False, "data": {}} 

        except Exception as e: 
            logger.error(f"绰号映射 LLM 调用或结果处理过程中发生意外错误: {e}", exc_info=True)
            return {"is_exist": False, "data": {}}

    def _filter_llm_results(self, original_data: Dict[str, str], user_name_map_for_prompt: Dict[str, str]) -> Dict[str, str]:
        """
        过滤 LLM 返回的绰号映射结果。
        """
        # ... (此方法保持不变) ...
        filtered_data = {}
        bot_qq_str = str(global_config.bot.qq_account) if global_config.bot.qq_account else None

        for user_id_str, sobriquet_name in original_data.items():
            if not isinstance(user_id_str, str) or not isinstance(sobriquet_name, str): 
                logger.warning(f"过滤掉非字符串 user_id 或 sobriquet_name: ID类型 {type(user_id_str)}, 绰号类型 {type(sobriquet_name)}")
                continue
            
            if bot_qq_str and user_id_str == bot_qq_str:
                name_in_prompt = user_name_map_for_prompt.get(user_id_str, "")
                if "(你)" in name_in_prompt or name_in_prompt == global_config.bot.nickname: 
                    logger.debug(f"过滤掉机器人自身的映射 (ID: {user_id_str}, Prompt名称: {name_in_prompt})")
                    continue

            if not sobriquet_name or sobriquet_name.isspace():
                logger.debug(f"过滤掉用户 {user_id_str} 的空绰号。")
                continue
            
            min_len = global_config.group_sobriquet.get("sobriquet_min_length", 1) 
            max_len = global_config.group_sobriquet.get("sobriquet_max_length", 15)
            cleaned_sobriquet = sobriquet_name.strip()
            if not (min_len <= len(cleaned_sobriquet) <= max_len):
               logger.debug(f"过滤掉用户 {user_id_str} 的绰号 '{cleaned_sobriquet}': 长度 ({len(cleaned_sobriquet)}) 不在 [{min_len}-{max_len}] 范围内。")
               continue
            
            filtered_data[user_id_str] = cleaned_sobriquet
        return filtered_data

    def _run_processor_in_thread(self):
        """后台线程入口函数。"""
        # ... (此方法保持不变) ...
        thread_id = threading.get_ident()
        logger.info(f"绰号处理器线程启动 (线程 ID: {thread_id})...")
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            logger.info(f"(线程 ID: {thread_id}) Asyncio 事件循环已创建并设置。")
            run_async_loop(loop, self._processing_loop())
        except Exception as e: 
            logger.error(f"绰号处理器线程 (ID: {thread_id}) 发生顶层错误: {e}", exc_info=True)
        finally:
            logger.info(f"绰号处理器线程结束 (线程 ID: {thread_id}).")

    async def _processing_loop(self):
        """后台异步处理循环。"""
        # ... (此方法保持不变) ...
        logger.info("绰号异步处理循环已启动。")
        while not self._stop_event.is_set():
            try:
                item = await asyncio.wait_for(self.sobriquet_queue.get(), timeout=self.sleep_interval)
                if item: 
                    await self._analyze_and_update_sobriquets(item)
                    self.sobriquet_queue.task_done()
            except asyncio.TimeoutError:
                continue 
            except asyncio.CancelledError:
                logger.info("绰号处理循环被取消。")
                break 
            except Exception as e: 
                logger.error(f"绰号处理循环在处理项目时出错: {e}", exc_info=True)
                if not self._stop_event.is_set(): 
                    await asyncio.sleep(global_config.group_sobriquet.get("error_sleep_interval", 5)) 
        
        logger.info("绰号异步处理循环已结束。")

# 单例实例
sobriquet_manager = SobriquetManager()
