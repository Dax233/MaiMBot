import asyncio
import threading
import random
import time
import json
import re
from typing import Dict, Optional, List, Any
from pymongo.errors import OperationFailure, DuplicateKeyError # 从 sobriquet_db 中处理
from src.common.logger_manager import get_logger
from src.common.database import db
from src.config.config import global_config
from src.chat.models.utils_model import LLMRequest
from .sobriquet_db import SobriquetDB
from .sobriquet_mapper import _build_mapping_prompt
from .sobriquet_utils import select_sobriquets_for_prompt, format_sobriquet_prompt_injection
from src.chat.person_info.person_info import person_info_manager # 仍需用它获取 person_info_pid
from src.experimental.profile.profile_manager import profile_manager 

from src.chat.message_receive.chat_stream import ChatStream
from src.chat.message_receive.message import MessageRecv
from src.chat.utils.chat_message_builder import build_readable_messages, get_raw_msg_before_timestamp_with_chat

logger = get_logger("SobriquetManager")
logger_helper = get_logger("AsyncLoopHelper") # run_async_loop 的 logger

# run_async_loop 函数定义保持不变
def run_async_loop(loop: asyncio.AbstractEventLoop, coro):
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
            tasks_to_cancel = [task for task in all_tasks if task is not current_task]
            if tasks_to_cancel:
                logger_helper.info(f"Cancelling {len(tasks_to_cancel)} outstanding tasks in loop {id(loop)}...")
                for task in tasks_to_cancel: task.cancel()
                loop.run_until_complete(asyncio.gather(*tasks_to_cancel, return_exceptions=True))
                logger_helper.info(f"Outstanding tasks cancelled in loop {id(loop)}.")
            if loop.is_running(): loop.stop(); logger_helper.info(f"Asyncio loop {id(loop)} stopped.")
            if not loop.is_closed():
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()
                logger_helper.info(f"Asyncio loop {id(loop)} closed.")
        except Exception as close_err:
            logger_helper.error(f"Error during asyncio loop cleanup for loop {id(loop)}: {close_err}", exc_info=True)

class SobriquetManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    logger.info("正在创建 SobriquetManager 单例实例...")
                    cls._instance = super(SobriquetManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
        with self._lock:
            if hasattr(self, "_initialized") and self._initialized:
                return
            logger.info("正在初始化 SobriquetManager 组件...")
            self.is_enabled = global_config.profile.enable_sobriquet_mapping

            profile_info_collection_name = "profile_info"
            profile_info_collection = getattr(db, profile_info_collection_name, None)
            
            if profile_info_collection is None:
                logger.warning(f"未能从数据库获取 '{profile_info_collection_name}' 集合。")

            self.db_handler = SobriquetDB(profile_info_collection) 
            if not self.db_handler.is_available():
                logger.error(f"SobriquetDB 初始化失败 (集合: {profile_info_collection_name})，功能受限。")
                self.is_enabled = False
            else:
                logger.info(f"SobriquetDB (集合: {profile_info_collection_name}) 初始化成功。")
            
            if not profile_manager.is_available() and self.is_enabled:
                logger.error(f"ProfileManager (集合: {profile_info_collection_name}) 不可用，功能将受限。")
                self.is_enabled = False # 如果 profile_manager 依赖的集合有问题，也应该禁用

            self.llm_mapper: Optional[LLMRequest] = None
            # ... (LLM 初始化 和 队列/线程 初始化逻辑不变) ...
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
                    self.llm_mapper = None; self.is_enabled = False
                except Exception as e:
                    logger.error(f"初始化绰号映射 LLM 映射器失败: {e}，功能禁用。", exc_info=True)
                    self.llm_mapper = None; self.is_enabled = False

            self.queue_max_size = global_config.profile.sobriquet_queue_max_size
            self.sobriquet_queue: asyncio.Queue = asyncio.Queue(maxsize=self.queue_max_size)
            self._stop_event = threading.Event()
            self._sobriquet_thread: Optional[threading.Thread] = None
            self.sleep_interval = global_config.profile.sobriquet_process_sleep_interval
            self._initialized = True
            logger.info(f"SobriquetManager 初始化完成。当前启用状态: {self.is_enabled}")

    # start_processor, stop_processor, _add_to_queue, _run_processor_in_thread, _processing_loop 保持不变
    def start_processor(self):
        if not self.is_enabled: logger.info("绰号处理功能已禁用，处理器未启动。"); return
        max_sobriquets_cfg = global_config.profile.max_sobriquets_in_prompt
        if not isinstance(max_sobriquets_cfg, int) or max_sobriquets_cfg <= 0:
            logger.error(f"[错误] 配置 'max_sobriquets_in_prompt'({max_sobriquets_cfg})不合适，功能禁用！"); self.is_enabled = False; return
        if self._sobriquet_thread is None or not self._sobriquet_thread.is_alive():
            logger.info("正在启动绰号处理器线程..."); self._stop_event.clear()
            self._sobriquet_thread = threading.Thread(target=self._run_processor_in_thread, daemon=True)
            self._sobriquet_thread.start()
            logger.info(f"绰号处理器线程已启动 (ID: {self._sobriquet_thread.ident})")
        else: logger.warning("绰号处理器线程已在运行中。")

    def stop_processor(self):
        if self._sobriquet_thread and self._sobriquet_thread.is_alive():
            logger.info("正在停止绰号处理器线程..."); self._stop_event.set()
            try:
                self._sobriquet_thread.join(timeout=10)
                if self._sobriquet_thread.is_alive(): logger.warning("绰号处理器线程在超时后仍未停止。")
            except Exception as e: logger.error(f"停止绰号处理器线程时出错: {e}", exc_info=True)
            finally:
                if self._sobriquet_thread and not self._sobriquet_thread.is_alive(): logger.info("绰号处理器线程已成功停止。")
                self._sobriquet_thread = None
        else: logger.info("绰号处理器线程未在运行或已被清理。")

    async def _add_to_queue(self, item: tuple, platform: str, group_id: str):
        try:
            await self.sobriquet_queue.put(item)
            logger.debug(f"项目已添加至 {platform}-{group_id} 绰号队列。大小: {self.sobriquet_queue.qsize()}")
        except asyncio.QueueFull: logger.warning(f"绰号队列已满。{platform}-{group_id} 项目被丢弃。")
        except Exception as e: logger.error(f"添加项目到绰号队列出错: {e}", exc_info=True)


    async def trigger_sobriquet_analysis(
        self, anchor_message: MessageRecv, bot_reply: List[str], chat_stream: Optional[ChatStream] = None
    ):
        # trigger_sobriquet_analysis 中的 user_name_map 构建逻辑保持不变，它依赖 relationship_manager 来获取 person_name
        # ... (此方法上半部分，构建 user_name_map 的逻辑不变) ...
        if not self.is_enabled: return
        analysis_probability = global_config.profile.get("sobriquet_analysis_probability", 1.0)
        if not (0 <= analysis_probability <= 1.0): analysis_probability = 1.0
        if random.random() > analysis_probability: logger.debug(f"跳过绰号分析：概率未命中。"); return

        current_chat_stream = chat_stream or anchor_message.chat_stream
        if not current_chat_stream or not current_chat_stream.group_info: logger.debug("跳过绰号分析：非群聊。"); return
        
        log_prefix = f"[{current_chat_stream.stream_id}]"
        try:
            history_limit = global_config.profile.sobriquet_analysis_history_limit
            history_messages = get_raw_msg_before_timestamp_with_chat(current_chat_stream.stream_id, time.time(), history_limit)
            chat_history_str = await build_readable_messages(history_messages, True, False, "relative", 0.0, False)
            bot_reply_str = " ".join(bot_reply) if bot_reply else ""
            group_id = str(current_chat_stream.group_info.group_id)
            platform = current_chat_stream.platform
            
            user_ids_in_history = {str(msg["user_info"]["user_id"]) for msg in history_messages if msg.get("user_info", {}).get("user_id")}
            user_name_map = {} 
            if user_ids_in_history:
                try:
                    # 依然需要 person_name 来构建 prompt 中的已知用户信息，所以调用 relationship_manager
                    from src.chat.person_info.relationship_manager import relationship_manager as rm # 局部导入
                    names_data = await rm.get_person_names_batch(platform, list(user_ids_in_history))
                except Exception as e:
                    logger.error(f"{log_prefix} 批量获取 person_name 出错: {e}", exc_info=True); names_data = {}
                
                for uid_str in user_ids_in_history:
                    if uid_str in names_data and names_data[uid_str]: user_name_map[uid_str] = names_data[uid_str]
                    else:
                        latest_display_name = next((m["user_info"].get("user_nickname") or m["user_info"].get("user_cardname") 
                                                   for m in reversed(history_messages) 
                                                   if str(m["user_info"].get("user_id")) == uid_str and 
                                                      (m["user_info"].get("user_nickname") or m["user_info"].get("user_cardname"))), None)
                        bot_uid_str = str(global_config.bot.qq_account)
                        user_name_map[uid_str] = (latest_display_name or f"{global_config.bot.nickname}(你)") if uid_str == bot_uid_str \
                                               else (latest_display_name or f"用户({uid_str[-4:]})")
            
            item = (chat_history_str, bot_reply_str, platform, group_id, user_name_map)
            await self._add_to_queue(item, platform, group_id)
        except Exception as e: logger.error(f"{log_prefix} 触发绰号分析时出错: {e}", exc_info=True)

    async def get_sobriquet_prompt_injection(self, chat_stream: ChatStream, message_list_before_now: List[Dict]) -> str:
        # get_sobriquet_prompt_injection 调用 profile_manager 获取数据，逻辑不变
        # ... (此方法调用 profile_manager，保持不变) ...
        if not self.is_enabled or not chat_stream or not chat_stream.group_info: return ""
        if not profile_manager.is_available(): logger.warning("ProfileManager 不可用，无法获取绰号注入。"); return ""

        log_prefix = f"[{chat_stream.stream_id}]"
        try:
            group_id = str(chat_stream.group_info.group_id)
            platform = chat_stream.platform
            user_ids_in_context = {str(msg["user_info"]["user_id"]) for msg in message_list_before_now if msg.get("user_info", {}).get("user_id")}
            if not user_ids_in_context:
                limit = global_config.profile.get("recent_speakers_limit_for_injection", 5)
                user_ids_in_context.update(str(s["user_id"]) for s in chat_stream.get_recent_speakers(limit=limit) if s.get("user_id"))
            if not user_ids_in_context: logger.debug(f"{log_prefix} 无上下文用户用于绰号注入。"); return ""
            
            data_for_prompt = await profile_manager.get_users_group_sobriquets_for_prompt_injection_data(
                platform, list(user_ids_in_context), group_id
            )
            if data_for_prompt:
                selected = select_sobriquets_for_prompt(data_for_prompt)
                injection_str = format_sobriquet_prompt_injection(selected)
                if injection_str: logger.debug(f"{log_prefix} 生成绰号注入 (部分):\n{injection_str.strip()[:200]}...")
                return injection_str
            else: logger.debug(f"{log_prefix} 未从 ProfileManager 获取到绰号数据。"); return ""
        except Exception as e: logger.error(f"{log_prefix} 获取绰号注入时出错: {e}", exc_info=True); return ""


    async def _analyze_and_update_sobriquets(self, item: tuple):
        if not isinstance(item, tuple) or len(item) != 5:
            logger.warning(f"从队列接收到无效项目: {type(item)} 内容: {item}")
            return

        chat_history_str, bot_reply, platform, group_id_str, user_name_map = item
        log_prefix = f"[{platform}:{group_id_str}]"
        # logger.debug(f"{log_prefix} 开始处理绰号分析任务...") # 减少冗余日志

        if not self.llm_mapper: logger.error(f"{log_prefix} LLM 映射器不可用。"); return
        if not self.db_handler or not self.db_handler.is_available(): 
            logger.error(f"{log_prefix} SobriquetDB 不可用。"); return

        analysis_result = await self._call_llm_for_analysis(chat_history_str, bot_reply, user_name_map)

        if analysis_result.get("is_exist") and analysis_result.get("data"):
            sobriquet_map_to_update = analysis_result["data"]
            logger.info(f"{log_prefix} LLM找到绰号映射，准备更新: {sobriquet_map_to_update}")

            for platform_user_id_str, sobriquet_name in sobriquet_map_to_update.items():
                if not platform_user_id_str or not sobriquet_name:
                    logger.warning(f"{log_prefix} 跳过无效条目: platform_uid='{platform_user_id_str}', sobriquet='{sobriquet_name}'")
                    continue
                
                try:
                    # 1. 获取 person_info_pid (代表自然人，来自 person_info 体系)
                    person_info_pid = person_info_manager.get_person_id(platform, platform_user_id_str)
                    if not person_info_pid:
                        logger.error(f"{log_prefix} 无法为 platform='{platform}', uid='{platform_user_id_str}' 获取 person_info_pid。")
                        continue
                    
                    # 2. 基于 person_info_pid 生成 profile_info 文档的 _id
                    profile_doc_id = profile_manager.generate_profile_document_id(person_info_pid)
                    
                    # 3. 确保 profile_info 文档存在，并记录平台账户信息和 person_info_pid_ref
                    #    这个方法会使用 profile_doc_id 作为 _id
                    self.db_handler.ensure_profile_document_exists(
                        profile_doc_id, person_info_pid, platform, platform_user_id_str
                    )
                    
                    # 4. 更新绰号计数 (使用 profile_doc_id 定位文档)
                    self.db_handler.update_group_sobriquet_count(
                        profile_doc_id, platform, group_id_str, sobriquet_name
                    )
                    
                    logger.debug(f"{log_prefix} 已为 profile_doc_id '{profile_doc_id}' (uid '{platform_user_id_str}') 更新/添加绰号 '{sobriquet_name}' @ grp '{group_id_str}'。")

                except ValueError as ve: # 来自 generate_profile_document_id
                     logger.error(f"{log_prefix} 生成 profile_doc_id 失败: {ve} for uid: {platform_user_id_str}, pipid: {person_info_pid if 'person_info_pid' in locals() else 'N/A'}")
                except Exception as e: # 包括 OperationFailure, DuplicateKeyError 等数据库异常
                    logger.exception(f"{log_prefix} 处理用户 {platform_user_id_str} 绰号 '{sobriquet_name}' 时意外错误：{e}")
        else:
            logger.debug(f"{log_prefix} LLM 未找到可靠绰号映射或分析失败。")

    # _call_llm_for_analysis, _filter_llm_results, _run_processor_in_thread, _processing_loop 保持不变
    async def _call_llm_for_analysis( self, chat_history_str: str, bot_reply: str,user_name_map: Dict[str, str]) -> Dict[str, Any]:
        if not self.llm_mapper: logger.error("LLM映射器未初始化。"); return {"is_exist": False}
        prompt = _build_mapping_prompt(chat_history_str, bot_reply, user_name_map)
        try:
            response_content, _, _ = await self.llm_mapper.generate_response(prompt)
            if not response_content: logger.warning("LLM返回空绰号映射内容。"); return {"is_exist": False}
            json_str = ""; stripped_content = response_content.strip()
            m_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped_content, re.DOTALL)
            if m_match: json_str = m_match.group(1).strip()
            else:
                b_match = re.search(r"(\{.*?\})", stripped_content, re.DOTALL)
                if b_match: json_str = b_match.group(1).strip()
                else: logger.warning(f"LLM响应不含有效JSON。响应(首200): {stripped_content[:200]}"); return {"is_exist": False}
            try: result = json.loads(json_str)
            except json.JSONDecodeError as je: logger.error(f"解析LLM JSON失败: {je}\nJSON str: {json_str}\n原始响应(首500): {stripped_content[:500]}"); return {"is_exist": False}
            if not isinstance(result, dict): logger.warning(f"LLM响应非字典。类型: {type(result)}"); return {"is_exist": False}
            is_exist = result.get("is_exist")
            if not isinstance(is_exist, bool): logger.warning(f"LLM 'is_exist'字段无效: {is_exist}"); return {"is_exist": False} 
            if is_exist:
                data = result.get("data")
                if isinstance(data, dict) and data:
                    filtered = self._filter_llm_results(data, user_name_map)
                    if not filtered: logger.info("所有绰号映射均被过滤。"); return {"is_exist": False, "data": {}}
                    logger.info(f"过滤后绰号映射: {filtered}"); return {"is_exist": True, "data": filtered}
                logger.warning(f"LLM is_exist=True 但 data 无效: {data}"); return {"is_exist": False, "data": {}}
            logger.info("LLM明确未找到绰号映射 (is_exist=False)。"); return {"is_exist": False, "data": {}}
        except Exception as e: logger.error(f"LLM调用或处理中意外错误: {e}", exc_info=True); return {"is_exist": False, "data": {}}

    def _filter_llm_results(self, original_data: Dict[str, str], user_name_map_for_prompt: Dict[str, str]) -> Dict[str, str]:
        filtered = {}; bot_qq = str(global_config.bot.qq_account) if global_config.bot.qq_account else None
        min_l, max_l = global_config.profile.get("sobriquet_min_length",1), global_config.profile.get("sobriquet_max_length",15)
        for uid, s_name in original_data.items():
            if not isinstance(uid, str) or not isinstance(s_name, str): continue
            if bot_qq and uid == bot_qq and ("(你)" in user_name_map_for_prompt.get(uid,"") or user_name_map_for_prompt.get(uid,"") == global_config.bot.nickname) : continue
            if not s_name or s_name.isspace(): continue
            cleaned_s = s_name.strip()
            if not (min_l <= len(cleaned_s) <= max_l): logger.debug(f"过滤绰号'{cleaned_s}':长度({len(cleaned_s)})不符。"); continue
            filtered[uid] = cleaned_s
        return filtered

    def _run_processor_in_thread(self):
        tid = threading.get_ident(); logger.info(f"绰号处理器线程启动 (ID: {tid})...")
        try:
            loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
            logger.info(f"(ID: {tid}) Asyncio事件循环已创建设置。")
            run_async_loop(loop, self._processing_loop())
        except Exception as e: logger.error(f"绰号处理器线程(ID:{tid})顶层错误: {e}", exc_info=True)
        finally: logger.info(f"绰号处理器线程结束 (ID: {tid}).")

    async def _processing_loop(self):
        logger.info("绰号异步处理循环已启动。")
        while not self._stop_event.is_set():
            try:
                item = await asyncio.wait_for(self.sobriquet_queue.get(), timeout=self.sleep_interval)
                if item: await self._analyze_and_update_sobriquets(item); self.sobriquet_queue.task_done()
            except asyncio.TimeoutError: continue
            except asyncio.CancelledError: logger.info("绰号处理循环被取消。"); break
            except Exception as e:
                logger.error(f"绰号处理循环处理项目时出错: {e}", exc_info=True)
                if not self._stop_event.is_set(): await asyncio.sleep(global_config.profile.get("error_sleep_interval", 5))
        logger.info("绰号异步处理循环已结束。")

sobriquet_manager = SobriquetManager()
