import time
import traceback
from typing import List, Optional, Dict
import asyncio
from asyncio import Lock
from ...moods.moods import MoodManager
from ....config.config import global_config
from ...chat.emoji_manager import emoji_manager
from .heartFC__generator import ResponseGenerator
from ...chat.message import MessageSending, MessageRecv, MessageThinking, MessageSet
from .messagesender import MessageManager
from ...chat.utils_image import image_path_to_base64
from ...message import UserInfo, Seg
from src.heart_flow.heartflow import heartflow
from src.common.logger import get_module_logger, CHAT_STYLE_CONFIG, LogConfig
from ...person_info.relationship_manager import relationship_manager
from src.plugins.respon_info_catcher.info_catcher import info_catcher_manager
from ...utils.timer_calculater import Timer
from src.do_tool.tool_use import ToolUser
from .interest import InterestManager
from src.plugins.chat.chat_stream import chat_manager
from src.plugins.chat.message import BaseMessageInfo
from .pf_chatting import PFChatting

# 定义日志配置
chat_config = LogConfig(
    console_format=CHAT_STYLE_CONFIG["console_format"],
    file_format=CHAT_STYLE_CONFIG["file_format"],
)

logger = get_module_logger("heartFC_chat", config=chat_config)

# 检测群聊兴趣的间隔时间
INTEREST_MONITOR_INTERVAL_SECONDS = 1


class HeartFC_Chat:
    _instance = None  # For potential singleton access if needed by MessageManager

    def __init__(self):
        # --- Updated Init ---
        if HeartFC_Chat._instance is not None:
            # Prevent re-initialization if used as a singleton
            return
        self.gpt = ResponseGenerator()
        self.mood_manager = MoodManager.get_instance()
        self.mood_manager.start_mood_update()
        self.tool_user = ToolUser()
        self.interest_manager = InterestManager()
        self._interest_monitor_task: Optional[asyncio.Task] = None
        # --- New PFChatting Management ---
        self.pf_chatting_instances: Dict[str, PFChatting] = {}
        self._pf_chatting_lock = Lock()
        # --- End New PFChatting Management ---
        HeartFC_Chat._instance = self  # Register instance
        # --- End Updated Init ---

    # --- Added Class Method for Singleton Access ---
    @classmethod
    def get_instance(cls):
        return cls._instance

    # --- End Added Class Method ---

    async def start(self):
        """启动异步任务,如回复启动器"""
        logger.debug("HeartFC_Chat 正在启动异步任务...")
        self._initialize_monitor_task()
        logger.info("HeartFC_Chat 异步任务启动完成")

    def _initialize_monitor_task(self):
        """启动后台兴趣监控任务，可以检查兴趣是否足以开启心流对话"""
        if self._interest_monitor_task is None or self._interest_monitor_task.done():
            try:
                loop = asyncio.get_running_loop()
                self._interest_monitor_task = loop.create_task(self._interest_monitor_loop())
            except RuntimeError:
                logger.error("创建兴趣监控任务失败：没有运行中的事件循环。")
                raise
        else:
            logger.warning("跳过兴趣监控任务创建：任务已存在或正在运行。")

    # --- Added PFChatting Instance Manager ---
    async def _get_or_create_pf_chatting(self, stream_id: str) -> Optional[PFChatting]:
        """获取现有PFChatting实例或创建新实例。"""
        async with self._pf_chatting_lock:
            if stream_id not in self.pf_chatting_instances:
                logger.info(f"为流 {stream_id} 创建新的PFChatting实例")
                # 传递 self (HeartFC_Chat 实例) 进行依赖注入
                instance = PFChatting(stream_id, self)
                # 执行异步初始化
                if not await instance._initialize():
                    logger.error(f"为流 {stream_id} 初始化PFChatting失败")
                    return None
                self.pf_chatting_instances[stream_id] = instance
            return self.pf_chatting_instances[stream_id]

    # --- End Added PFChatting Instance Manager ---

    async def _interest_monitor_loop(self):
        """后台任务，定期检查兴趣度变化并触发回复"""
        logger.info("兴趣监控循环开始...")
        while True:
            await asyncio.sleep(INTEREST_MONITOR_INTERVAL_SECONDS)
            try:
                # 从心流中获取活跃流
                active_stream_ids = list(heartflow.get_all_subheartflows_streams_ids())
                for stream_id in active_stream_ids:
                    stream_name = chat_manager.get_stream_name(stream_id) or stream_id  # 获取流名称
                    sub_hf = heartflow.get_subheartflow(stream_id)
                    if not sub_hf:
                        logger.warning(f"监控循环: 无法获取活跃流 {stream_name} 的 sub_hf")
                        continue

                    should_trigger = False
                    try:
                        interest_chatting = self.interest_manager.get_interest_chatting(stream_id)
                        if interest_chatting:
                            should_trigger = interest_chatting.should_evaluate_reply()
                        else:
                            logger.trace(
                                f"[{stream_name}] 没有找到对应的 InterestChatting 实例，跳过基于兴趣的触发检查。"
                            )
                    except Exception as e:
                        logger.error(f"检查兴趣触发器时出错 流 {stream_name}: {e}")
                        logger.error(traceback.format_exc())

                    if should_trigger:
                        # 启动一次麦麦聊天
                        pf_instance = await self._get_or_create_pf_chatting(stream_id)
                        if pf_instance:
                            asyncio.create_task(pf_instance.add_time())
                        else:
                            logger.error(f"[{stream_name}] 无法获取或创建PFChatting实例。跳过触发。")

            except asyncio.CancelledError:
                logger.info("兴趣监控循环已取消。")
                break
            except Exception as e:
                logger.error(f"兴趣监控循环错误: {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(5)  # 发生错误时等待

    async def _create_thinking_message(self, anchor_message: Optional[MessageRecv]):
        """创建思考消息 (尝试锚定到 anchor_message)"""
        if not anchor_message or not anchor_message.chat_stream:
            logger.error("无法创建思考消息，缺少有效的锚点消息或聊天流。")
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

        MessageManager().add_message(thinking_message)
        return thinking_id

    async def _send_response_messages(
        self, anchor_message: Optional[MessageRecv], response_set: List[str], thinking_id
    ) -> Optional[MessageSending]:
        """发送回复消息 (尝试锚定到 anchor_message)"""
        if not anchor_message or not anchor_message.chat_stream:
            logger.error("无法发送回复，缺少有效的锚点消息或聊天流。")
            return None

        chat = anchor_message.chat_stream
        container = MessageManager().get_container(chat.stream_id)
        thinking_message = None
        for msg in container.messages:
            if isinstance(msg, MessageThinking) and msg.message_info.message_id == thinking_id:
                thinking_message = msg
                container.messages.remove(msg)
                break
        if not thinking_message:
            stream_name = chat_manager.get_stream_name(chat.stream_id) or chat.stream_id  # 获取流名称
            logger.warning(f"[{stream_name}] 未找到对应的思考消息 {thinking_id}，可能已超时被移除")
            return None

        thinking_start_time = thinking_message.thinking_start_time
        message_set = MessageSet(chat, thinking_id)
        mark_head = False
        first_bot_msg = None
        for msg_text in response_set:
            message_segment = Seg(type="text", data=msg_text)
            bot_message = MessageSending(
                message_id=thinking_id,  # 使用 thinking_id 作为批次标识
                chat_stream=chat,
                bot_user_info=UserInfo(
                    user_id=global_config.BOT_QQ,
                    user_nickname=global_config.BOT_NICKNAME,
                    platform=anchor_message.message_info.platform,
                ),
                sender_info=anchor_message.message_info.user_info,  # 发送给锚点消息的用户
                message_segment=message_segment,
                reply=anchor_message,  # 回复锚点消息
                is_head=not mark_head,
                is_emoji=False,
                thinking_start_time=thinking_start_time,
            )
            if not mark_head:
                mark_head = True
                first_bot_msg = bot_message
            message_set.add_message(bot_message)

        if message_set.messages:  # 确保有消息才添加
            MessageManager().add_message(message_set)
            return first_bot_msg
        else:
            stream_name = chat_manager.get_stream_name(chat.stream_id) or chat.stream_id  # 获取流名称
            logger.warning(f"[{stream_name}] 没有生成有效的回复消息集，无法发送。")
            return None

    async def _handle_emoji(self, anchor_message: Optional[MessageRecv], response_set, send_emoji=""):
        """处理表情包 (尝试锚定到 anchor_message)"""
        if not anchor_message or not anchor_message.chat_stream:
            logger.error("无法处理表情包，缺少有效的锚点消息或聊天流。")
            return

        chat = anchor_message.chat_stream
        if send_emoji:
            emoji_raw = await emoji_manager.get_emoji_for_text(send_emoji)
        else:
            emoji_text_source = "".join(response_set) if response_set else ""
            emoji_raw = await emoji_manager.get_emoji_for_text(emoji_text_source)

        if emoji_raw:
            emoji_path, description = emoji_raw
            emoji_cq = image_path_to_base64(emoji_path)
            # 使用当前时间戳，因为没有原始消息的时间戳
            thinking_time_point = round(time.time(), 2)
            message_segment = Seg(type="emoji", data=emoji_cq)
            bot_message = MessageSending(
                message_id="me" + str(thinking_time_point),  # 使用不同的 ID 前缀?
                chat_stream=chat,
                bot_user_info=UserInfo(
                    user_id=global_config.BOT_QQ,
                    user_nickname=global_config.BOT_NICKNAME,
                    platform=anchor_message.message_info.platform,
                ),
                sender_info=anchor_message.message_info.user_info,
                message_segment=message_segment,
                reply=anchor_message,  # 回复锚点消息
                is_head=False,
                is_emoji=True,
            )
            MessageManager().add_message(bot_message)

    async def _update_relationship(self, anchor_message: Optional[MessageRecv], response_set):
        """更新关系情绪 (尝试基于 anchor_message)"""
        if not anchor_message or not anchor_message.chat_stream:
            logger.error("无法更新关系情绪，缺少有效的锚点消息或聊天流。")
            return

        # 关系更新依赖于理解回复是针对谁的，以及原始消息的上下文
        # 这里的实现可能需要调整，取决于关系管理器如何工作
        ori_response = ",".join(response_set)
        # 注意：anchor_message.processed_plain_text 是锚点消息的文本，不一定是思考的全部上下文
        stance, emotion = await self.gpt._get_emotion_tags(ori_response, anchor_message.processed_plain_text)
        await relationship_manager.calculate_update_relationship_value(
            chat_stream=anchor_message.chat_stream,  # 使用锚点消息的流
            label=emotion,
            stance=stance,
        )
        self.mood_manager.update_mood_from_emotion(emotion, global_config.mood_intensity_factor)

    # 暂不使用
    async def trigger_reply_generation(self, stream_id: str, observed_messages: List[dict]):
        """根据 SubHeartflow 的触发信号生成回复 (基于观察)"""
        stream_name = chat_manager.get_stream_name(stream_id) or stream_id  # <--- 在开始时获取名称
        chat = None
        sub_hf = None
        anchor_message: Optional[MessageRecv] = None  # <--- 重命名，用于锚定回复的消息对象
        userinfo: Optional[UserInfo] = None
        messageinfo: Optional[BaseMessageInfo] = None

        timing_results = {}
        current_mind = None
        response_set = None
        thinking_id = None
        info_catcher = None

        try:
            # --- 1. 获取核心对象：ChatStream 和 SubHeartflow ---
            try:
                with Timer("获取聊天流和子心流", timing_results):
                    chat = chat_manager.get_stream(stream_id)
                    if not chat:
                        logger.error(f"[{stream_name}] 无法找到聊天流对象，无法生成回复。")
                        return
                    sub_hf = heartflow.get_subheartflow(stream_id)
                    if not sub_hf:
                        logger.error(f"[{stream_name}] 无法找到子心流对象，无法生成回复。")
                        return
            except Exception as e:
                logger.error(f"[{stream_name}] 获取 ChatStream 或 SubHeartflow 时出错: {e}")
                logger.error(traceback.format_exc())
                return

            # --- 2. 尝试从 observed_messages 重建最后一条消息作为锚点, 失败则创建占位符 --- #
            try:
                with Timer("获取或创建锚点消息", timing_results):
                    reconstruction_failed = False
                    if observed_messages:
                        try:
                            last_msg_dict = observed_messages[-1]
                            logger.debug(
                                f"[{stream_name}] Attempting to reconstruct MessageRecv from last observed message."
                            )
                            anchor_message = MessageRecv(last_msg_dict, chat_stream=chat)
                            if not (
                                anchor_message
                                and anchor_message.message_info
                                and anchor_message.message_info.message_id
                                and anchor_message.message_info.user_info
                            ):
                                raise ValueError("Reconstructed MessageRecv missing essential info.")
                            userinfo = anchor_message.message_info.user_info
                            messageinfo = anchor_message.message_info
                            logger.debug(
                                f"[{stream_name}] Successfully reconstructed anchor message: ID={messageinfo.message_id}, Sender={userinfo.user_nickname}"
                            )
                        except Exception as e_reconstruct:
                            logger.warning(
                                f"[{stream_name}] Reconstructing MessageRecv from observed message failed: {e_reconstruct}. Will create placeholder."
                            )
                            reconstruction_failed = True
                    else:
                        logger.warning(
                            f"[{stream_name}] observed_messages is empty. Will create placeholder anchor message."
                        )
                        reconstruction_failed = True  # Treat empty observed_messages as a failure to reconstruct

                    # 如果重建失败或 observed_messages 为空，创建占位符
                    if reconstruction_failed:
                        placeholder_id = f"mid_{int(time.time() * 1000)}"  # 使用毫秒时间戳增加唯一性
                        placeholder_user = UserInfo(user_id="system_trigger", user_nickname="系统触发")
                        placeholder_msg_info = BaseMessageInfo(
                            message_id=placeholder_id,
                            platform=chat.platform,
                            group_info=chat.group_info,
                            user_info=placeholder_user,
                            time=time.time(),
                            # 其他 BaseMessageInfo 可能需要的字段设为默认值或 None
                        )
                        # 创建 MessageRecv 实例，注意它需要消息字典结构，我们创建一个最小化的
                        placeholder_msg_dict = {
                            "message_info": placeholder_msg_info.to_dict(),
                            "processed_plain_text": "",  # 提供空文本
                            "raw_message": "",
                            "time": placeholder_msg_info.time,
                        }
                        # 先只用字典创建实例
                        anchor_message = MessageRecv(placeholder_msg_dict)
                        # 然后调用方法更新 chat_stream
                        anchor_message.update_chat_stream(chat)
                        userinfo = anchor_message.message_info.user_info
                        messageinfo = anchor_message.message_info
                        logger.info(
                            f"[{stream_name}] Created placeholder anchor message: ID={messageinfo.message_id}, Sender={userinfo.user_nickname}"
                        )

            except Exception as e:
                logger.error(f"[{stream_name}] 获取或创建锚点消息时出错: {e}")
                logger.error(traceback.format_exc())
                anchor_message = None  # 确保出错时 anchor_message 为 None

            # --- 4. 检查并发思考限制 (使用 anchor_message 简化获取) ---
            try:
                container = MessageManager().get_container(chat.stream_id)
                thinking_count = container.count_thinking_messages()
                max_thinking_messages = getattr(global_config, "max_concurrent_thinking_messages", 3)
                if thinking_count >= max_thinking_messages:
                    logger.warning(f"聊天流 {stream_name} 已有 {thinking_count} 条思考消息，取消回复。")
                    return
            except Exception as e:
                logger.error(f"[{stream_name}] 检查并发思考限制时出错: {e}")
                return

            # --- 5. 创建思考消息 (使用 anchor_message) ---
            try:
                with Timer("创建思考消息", timing_results):
                    # 注意：这里传递 anchor_message 给 _create_thinking_message
                    thinking_id = await self._create_thinking_message(anchor_message)
            except Exception as e:
                logger.error(f"[{stream_name}] 创建思考消息失败: {e}")
                return
            if not thinking_id:
                logger.error(f"[{stream_name}] 未能成功创建思考消息 ID，无法继续回复流程。")
                return

            # --- 6. 信息捕捉器 (使用 anchor_message) ---
            logger.trace(f"[{stream_name}] 创建捕捉器，thinking_id:{thinking_id}")
            info_catcher = info_catcher_manager.get_info_catcher(thinking_id)
            info_catcher.catch_decide_to_response(anchor_message)

            # --- 7. 思考前使用工具 --- #
            get_mid_memory_id = []
            tool_result_info = {}
            send_emoji = ""
            observation_context_text = ""  # 从 observation 获取上下文文本
            try:
                # --- 使用传入的 observed_messages 构建上下文文本 --- #
                if observed_messages:
                    # 可以选择转换全部消息，或只转换最后几条
                    # 这里示例转换全部消息
                    context_texts = []
                    for msg_dict in observed_messages:
                        # 假设 detailed_plain_text 字段包含所需文本
                        # 你可能需要更复杂的逻辑来格式化，例如添加发送者和时间
                        text = msg_dict.get("detailed_plain_text", "")
                        if text:
                            context_texts.append(text)
                    observation_context_text = " ".join(context_texts)
                else:
                    logger.warning(f"[{stream_name}] observed_messages 列表为空，无法为工具提供上下文。")

                if observation_context_text:
                    with Timer("思考前使用工具", timing_results):
                        tool_result = await self.tool_user.use_tool(
                            message_txt=observation_context_text,  # <--- 使用观察上下文
                            chat_stream=chat,
                            sub_heartflow=sub_hf,
                        )
                        if tool_result.get("used_tools", False):
                            if "structured_info" in tool_result:
                                tool_result_info = tool_result["structured_info"]
                                get_mid_memory_id = []
                                for tool_name, tool_data in tool_result_info.items():
                                    if tool_name == "mid_chat_mem":
                                        for mid_memory in tool_data:
                                            get_mid_memory_id.append(mid_memory["content"])
                                    if tool_name == "send_emoji":
                                        send_emoji = tool_data[0]["content"]
            except Exception as e:
                logger.error(f"[{stream_name}] 思考前工具调用失败: {e}")
                logger.error(traceback.format_exc())

            # --- 8. 调用 SubHeartflow 进行思考 (不传递具体消息文本和发送者) ---
            try:
                with Timer("生成内心想法(SubHF)", timing_results):
                    # 不再传递 message_txt 和 sender_info, SubHeartflow 应基于其内部观察
                    current_mind, past_mind = await sub_hf.do_thinking_before_reply(
                        # sender_info=userinfo,
                        chat_stream=chat,
                        extra_info=tool_result_info,
                        obs_id=get_mid_memory_id,
                    )
                    logger.info(f"[{stream_name}] SubHeartflow 思考完成: {current_mind}")
            except Exception as e:
                logger.error(f"[{stream_name}] SubHeartflow 思考失败: {e}")
                logger.error(traceback.format_exc())
                if info_catcher:
                    info_catcher.done_catch()
                return  # 思考失败则不继续
            if info_catcher:
                info_catcher.catch_afer_shf_step(timing_results.get("生成内心想法(SubHF)"), past_mind, current_mind)

            # --- 9. 调用 ResponseGenerator 生成回复 (使用 anchor_message 和 current_mind) ---
            try:
                with Timer("生成最终回复(GPT)", timing_results):
                    # response_set = await self.gpt.generate_response(anchor_message, thinking_id, current_mind=current_mind)
                    response_set = await self.gpt.generate_response(anchor_message, thinking_id)
            except Exception as e:
                logger.error(f"[{stream_name}] GPT 生成回复失败: {e}")
                logger.error(traceback.format_exc())
                if info_catcher:
                    info_catcher.done_catch()
                return
            if info_catcher:
                info_catcher.catch_after_generate_response(timing_results.get("生成最终回复(GPT)"))
            if not response_set:
                logger.info(f"[{stream_name}] 回复生成失败或为空。")
                if info_catcher:
                    info_catcher.done_catch()
                return

            # --- 10. 发送消息 (使用 anchor_message) ---
            first_bot_msg = None
            try:
                with Timer("发送消息", timing_results):
                    first_bot_msg = await self._send_response_messages(anchor_message, response_set, thinking_id)
            except Exception as e:
                logger.error(f"[{stream_name}] 发送消息失败: {e}")
                logger.error(traceback.format_exc())
            if info_catcher:
                info_catcher.catch_after_response(timing_results.get("发送消息"), response_set, first_bot_msg)
                info_catcher.done_catch()  # 完成捕捉

            # --- 11. 处理表情包 (使用 anchor_message) ---
            try:
                with Timer("处理表情包", timing_results):
                    if send_emoji:
                        logger.info(f"[{stream_name}] 决定发送表情包 {send_emoji}")
                        await self._handle_emoji(anchor_message, response_set, send_emoji)
            except Exception as e:
                logger.error(f"[{stream_name}] 处理表情包失败: {e}")
                logger.error(traceback.format_exc())

            # --- 12. 记录性能日志 --- #
            timing_str = " | ".join([f"{step}: {duration:.2f}秒" for step, duration in timing_results.items()])
            response_msg = " ".join(response_set) if response_set else "无回复"
            logger.info(
                f"[{stream_name}] 回复任务完成 (Observation Triggered): | 思维消息: {response_msg[:30]}... | 性能计时: {timing_str}"
            )

            # --- 13. 更新关系情绪 (使用 anchor_message) ---
            if first_bot_msg:  # 仅在成功发送消息后
                try:
                    with Timer("更新关系情绪", timing_results):
                        await self._update_relationship(anchor_message, response_set)
                except Exception as e:
                    logger.error(f"[{stream_name}] 更新关系情绪失败: {e}")
                    logger.error(traceback.format_exc())

        except Exception as e:
            logger.error(f"回复生成任务失败 (trigger_reply_generation V4 - Observation Triggered): {e}")
            logger.error(traceback.format_exc())

        finally:
            # 可以在这里添加清理逻辑，如果有的话
            pass
