import time
import threading  # 导入 threading
from random import random
import traceback
import asyncio
from typing import List, Dict
from ..moods.moods import MoodManager
from ...config.config import global_config
from ..chat.emoji_manager import emoji_manager
from .normal_chat_generator import ResponseGenerator
from ..chat.message import MessageSending, MessageRecv, MessageThinking, MessageSet
from ..chat.message_sender import message_manager
from ..storage.storage import MessageStorage
from ..chat.utils import is_mentioned_bot_in_message
from ..chat.utils_image import image_path_to_base64
from ..willing.willing_manager import willing_manager
from ..message import UserInfo, Seg
from src.common.logger import get_module_logger, CHAT_STYLE_CONFIG, LogConfig
from src.plugins.chat.chat_stream import ChatStream, chat_manager
from src.plugins.person_info.relationship_manager import relationship_manager
from src.plugins.respon_info_catcher.info_catcher import info_catcher_manager
from src.plugins.utils.timer_calculater import Timer
from src.heart_flow.heartflow import heartflow
from src.heart_flow.sub_heartflow import ChatState

# 定义日志配置
chat_config = LogConfig(
    console_format=CHAT_STYLE_CONFIG["console_format"],
    file_format=CHAT_STYLE_CONFIG["file_format"],
)

logger = get_module_logger("normal_chat", config=chat_config)


class NormalChat:
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                # Double-check locking
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # 防止重复初始化
        if self._initialized:
            return
        with self.__class__._lock:  # 使用类锁确保线程安全
            if self._initialized:
                return
            logger.info("正在初始化 NormalChat 单例...")  # 添加日志
            self.storage = MessageStorage()
            self.gpt = ResponseGenerator()
            self.mood_manager = MoodManager.get_instance()
            # 用于存储每个 chat stream 的兴趣监控任务
            self._interest_monitoring_tasks: Dict[str, asyncio.Task] = {}
            self._initialized = True
            logger.info("NormalChat 单例初始化完成。")  # 添加日志

    @classmethod
    def get_instance(cls):
        """获取 NormalChat 的单例实例。"""
        if cls._instance is None:
            # 如果实例还未创建（理论上应该在 main 中初始化，但作为备用）
            logger.warning("NormalChat 实例在首次 get_instance 时创建。")
            cls()  # 调用构造函数来创建实例
        return cls._instance

    @staticmethod
    async def _create_thinking_message(message, chat, userinfo, messageinfo):
        """创建思考消息"""
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
            reply=message,
            thinking_start_time=thinking_time_point,
        )

        await message_manager.add_message(thinking_message)

        return thinking_id

    @staticmethod
    async def _send_response_messages(message, chat, response_set: List[str], thinking_id) -> MessageSending:
        """发送回复消息"""
        container = await message_manager.get_container(chat.stream_id)
        thinking_message = None

        for msg in container.messages[:]:
            if isinstance(msg, MessageThinking) and msg.message_info.message_id == thinking_id:
                thinking_message = msg
                container.messages.remove(msg)
                break

        if not thinking_message:
            logger.warning(f"[{chat.stream_id}] 未找到对应的思考消息 {thinking_id}，可能已超时被移除")
            return None

        thinking_start_time = thinking_message.thinking_start_time
        message_set = MessageSet(chat, thinking_id)

        mark_head = False
        first_bot_msg = None
        for msg in response_set:
            message_segment = Seg(type="text", data=msg)
            bot_message = MessageSending(
                message_id=thinking_id,
                chat_stream=chat,
                bot_user_info=UserInfo(
                    user_id=global_config.BOT_QQ,
                    user_nickname=global_config.BOT_NICKNAME,
                    platform=message.message_info.platform,
                ),
                sender_info=message.message_info.user_info,
                message_segment=message_segment,
                reply=message,
                is_head=not mark_head,
                is_emoji=False,
                thinking_start_time=thinking_start_time,
                apply_set_reply_logic=True,
            )
            if not mark_head:
                mark_head = True
                first_bot_msg = bot_message
            message_set.add_message(bot_message)

        await message_manager.add_message(message_set)

        return first_bot_msg

    @staticmethod
    async def _handle_emoji(message, chat, response):
        """处理表情包"""
        if random() < global_config.emoji_chance:
            emoji_raw = await emoji_manager.get_emoji_for_text(response)
            if emoji_raw:
                emoji_path, description = emoji_raw
                emoji_cq = image_path_to_base64(emoji_path)

                thinking_time_point = round(message.message_info.time, 2)

                message_segment = Seg(type="emoji", data=emoji_cq)
                bot_message = MessageSending(
                    message_id="mt" + str(thinking_time_point),
                    chat_stream=chat,
                    bot_user_info=UserInfo(
                        user_id=global_config.BOT_QQ,
                        user_nickname=global_config.BOT_NICKNAME,
                        platform=message.message_info.platform,
                    ),
                    sender_info=message.message_info.user_info,
                    message_segment=message_segment,
                    reply=message,
                    is_head=False,
                    is_emoji=True,
                    apply_set_reply_logic=True,
                )
                await message_manager.add_message(bot_message)

    async def _update_relationship(self, message: MessageRecv, response_set):
        """更新关系情绪"""
        ori_response = ",".join(response_set)
        stance, emotion = await self.gpt._get_emotion_tags(ori_response, message.processed_plain_text)
        await relationship_manager.calculate_update_relationship_value(
            chat_stream=message.chat_stream, label=emotion, stance=stance
        )
        self.mood_manager.update_mood_from_emotion(emotion, global_config.mood_intensity_factor)

    async def _find_interested_message(self, chat: ChatStream) -> None:
        # 此函数设计为后台任务，轮询指定 chat 的兴趣消息。
        # 它通常由外部代码在 chat 流活跃时启动。
        stream_id = chat.stream_id  # 获取 stream_id

        # logger.info(f"[{stream_id}] 兴趣消息监控任务启动。") # 减少日志
        while True:
            await asyncio.sleep(1)  # 每秒检查一次

            subheartflow = heartflow.get_subheartflow(stream_id)

            if not subheartflow or subheartflow.should_stop:
                # logger.info(f"[{stream_id}] SubHeartflow 不存在或已停止，兴趣消息监控任务退出。") # 减少日志
                break

            interest_dict = subheartflow.get_interest_dict()
            items_to_process = list(interest_dict.items())

            if not items_to_process:
                continue

            for msg_id, (message, interest_value, is_mentioned) in items_to_process:
                # --- 在处理前检查 SubHeartflow 的状态 --- #
                current_chat_state = subheartflow.chat_state.chat_status
                stream_name = chat_manager.get_stream_name(stream_id) or stream_id

                if current_chat_state != ChatState.CHAT:
                    # 如果不是闲聊状态 (可能是 ABSENT 或 FOCUSED)，则跳过推理聊天
                    # logger.debug(f"[{stream_name}] 跳过处理兴趣消息 {msg_id}，因为当前状态为 {current_chat_state.value}")
                    # 移除消息并继续下一个
                    removed_item = interest_dict.pop(msg_id, None)
                    if removed_item:
                        # logger.debug(f"[{stream_name}] 已从兴趣字典中移除消息 {msg_id} (因状态跳过)") # 减少日志
                        pass
                    continue  # 处理下一条消息
                # --- 结束状态检查 --- #

                # --- 检查 HeartFChatting 是否活跃 (改为检查 SubHeartflow 状态) --- #
                is_focused = subheartflow.chat_state.chat_status == ChatState.FOCUSED

                if is_focused:  # New check: If the subflow is focused, NormalChat shouldn't process
                    removed_item = interest_dict.pop(msg_id, None)
                    if removed_item:
                        # logger.debug(f"[{stream_name}] SubHeartflow 处于 FOCUSED 状态，已跳过并移除 NormalChat 兴趣消息 {msg_id}") # Reduce noise
                        pass
                    continue
                # --- 结束检查 --- #

                # 只有当状态为 CHAT 且 HeartFChatting 不活跃 (即 Subflow 不是 FOCUSED) 时才执行以下处理逻辑
                try:
                    await self.normal_normal_chat(
                        message=message,
                        chat=chat,
                        is_mentioned=is_mentioned,
                        interested_rate=interest_value,
                    )
                except Exception as e:
                    logger.error(f"[{stream_name}] 处理兴趣消息 {msg_id} 时出错: {e}\n{traceback.format_exc()}")
                finally:
                    removed_item = interest_dict.pop(msg_id, None)
                    if removed_item:
                        # logger.debug(f"[{stream_name}] 已从兴趣字典中移除消息 {msg_id}") # 减少日志
                        pass

    async def normal_normal_chat(
        self, message: MessageRecv, chat: ChatStream, is_mentioned: bool, interested_rate: float
    ) -> None:
        timing_results = {}
        userinfo = message.message_info.user_info
        messageinfo = message.message_info
        stream_id = chat.stream_id
        stream_name = chat_manager.get_stream_name(stream_id) or stream_id

        # --- 在开始时检查 SubHeartflow 状态 --- #
        sub_hf = heartflow.get_subheartflow(stream_id)
        if not sub_hf:
            logger.warning(f"[{stream_name}] 无法获取 SubHeartflow，无法执行 normal_normal_chat。")
            return

        current_chat_state = sub_hf.chat_state.chat_status
        if current_chat_state != ChatState.CHAT:
            logger.debug(
                f"[{stream_name}] 跳过 normal_normal_chat，因为 SubHeartflow 状态为 {current_chat_state.value} (需要 CHAT)。"
            )
            # 可以在这里添加 not_reply_handle 逻辑吗？ 如果不回复，也需要清理意愿。
            # 注意：willing_manager.setup 尚未调用
            willing_manager.setup(message, chat, is_mentioned, interested_rate)  # 先 setup
            await willing_manager.not_reply_handle(message.message_info.message_id)
            willing_manager.delete(message.message_info.message_id)
            return
        # --- 结束状态检查 --- #

        # --- 接下来的逻辑只在 ChatState.CHAT 状态下执行 --- #

        is_mentioned, reply_probability = is_mentioned_bot_in_message(message)
        # 意愿管理器：设置当前message信息
        willing_manager.setup(message, chat, is_mentioned, interested_rate)

        # 获取回复概率
        is_willing = False
        if reply_probability != 1:
            is_willing = True
            reply_probability = await willing_manager.get_reply_probability(message.message_info.message_id)

            if message.message_info.additional_config:
                if "maimcore_reply_probability_gain" in message.message_info.additional_config.keys():
                    reply_probability += message.message_info.additional_config["maimcore_reply_probability_gain"]

        # 打印消息信息
        mes_name = chat.group_info.group_name if chat.group_info else "私聊"
        current_time = time.strftime("%H:%M:%S", time.localtime(message.message_info.time))
        willing_log = f"[回复意愿:{await willing_manager.get_willing(chat.stream_id):.2f}]" if is_willing else ""
        logger.info(
            f"[{current_time}][{mes_name}]"
            f"{message.message_info.user_info.user_nickname}:"
            f"{message.processed_plain_text}{willing_log}[概率:{reply_probability * 100:.1f}%]"
        )
        do_reply = False
        if random() < reply_probability:
            do_reply = True

            # 回复前处理
            await willing_manager.before_generate_reply_handle(message.message_info.message_id)

            # 创建思考消息
            with Timer("创建思考消息", timing_results):
                thinking_id = await self._create_thinking_message(message, chat, userinfo, messageinfo)

            logger.debug(f"创建捕捉器，thinking_id:{thinking_id}")

            info_catcher = info_catcher_manager.get_info_catcher(thinking_id)
            info_catcher.catch_decide_to_response(message)

            # 生成回复
            sub_hf = heartflow.get_subheartflow(stream_id)

            try:
                with Timer("生成回复", timing_results):
                    response_set = await self.gpt.generate_response(
                        sub_hf=sub_hf,
                        message=message,
                        thinking_id=thinking_id,
                    )

                info_catcher.catch_after_generate_response(timing_results["生成回复"])
            except Exception as e:
                logger.error(f"回复生成出现错误：{str(e)} {traceback.format_exc()}")
                response_set = None

            if not response_set:
                logger.info(f"[{chat.stream_id}] 模型未生成回复内容")
                # 如果模型未生成回复，移除思考消息
                container = await message_manager.get_container(chat.stream_id)
                # thinking_message = None
                for msg in container.messages[:]:  # Iterate over a copy
                    if isinstance(msg, MessageThinking) and msg.message_info.message_id == thinking_id:
                        # thinking_message = msg
                        container.messages.remove(msg)
                        # container.remove_message(msg) # 直接移除
                        logger.debug(f"[{chat.stream_id}] 已移除未产生回复的思考消息 {thinking_id}")
                        break
                return  # 不发送回复

            logger.info(f"[{chat.stream_id}] 回复内容: {response_set}")

            # 发送回复
            with Timer("消息发送", timing_results):
                first_bot_msg = await self._send_response_messages(message, chat, response_set, thinking_id)

            info_catcher.catch_after_response(timing_results["消息发送"], response_set, first_bot_msg)

            info_catcher.done_catch()

            # 处理表情包
            with Timer("处理表情包", timing_results):
                await self._handle_emoji(message, chat, response_set[0])

            # 更新关系情绪
            with Timer("关系更新", timing_results):
                await self._update_relationship(message, response_set)

            # 回复后处理
            await willing_manager.after_generate_reply_handle(message.message_info.message_id)

        # 输出性能计时结果
        if do_reply:
            timing_str = " | ".join([f"{step}: {duration:.2f}秒" for step, duration in timing_results.items()])
            trigger_msg = message.processed_plain_text
            response_msg = " ".join(response_set) if response_set else "无回复"
            logger.info(f"触发消息: {trigger_msg[:20]}... | 推理消息: {response_msg[:20]}... | 性能计时: {timing_str}")
        else:
            # 不回复处理
            await willing_manager.not_reply_handle(message.message_info.message_id)

        # 意愿管理器：注销当前message信息
        willing_manager.delete(message.message_info.message_id)

    @staticmethod
    def _check_ban_words(text: str, chat, userinfo) -> bool:
        """检查消息中是否包含过滤词"""
        for word in global_config.ban_words:
            if word in text:
                logger.info(
                    f"[{chat.group_info.group_name if chat.group_info else '私聊'}]{userinfo.user_nickname}:{text}"
                )
                logger.info(f"[过滤词识别]消息中含有{word}，filtered")
                return True
        return False

    @staticmethod
    def _check_ban_regex(text: str, chat, userinfo) -> bool:
        """检查消息是否匹配过滤正则表达式"""
        for pattern in global_config.ban_msgs_regex:
            if pattern.search(text):
                logger.info(
                    f"[{chat.group_info.group_name if chat.group_info else '私聊'}]{userinfo.user_nickname}:{text}"
                )
                logger.info(f"[正则表达式过滤]消息匹配到{pattern}，filtered")
                return True
        return False

    async def start_monitoring_interest(self, chat: ChatStream):
        """为指定的 ChatStream 启动兴趣消息监控任务（如果尚未运行）。"""
        stream_id = chat.stream_id
        if stream_id not in self._interest_monitoring_tasks or self._interest_monitoring_tasks[stream_id].done():
            logger.info(f"为聊天流 {stream_id} 启动兴趣消息监控任务...")
            # 创建新任务
            task = asyncio.create_task(self._find_interested_message(chat))
            # 添加完成回调
            task.add_done_callback(lambda t: self._handle_task_completion(stream_id, t))
            self._interest_monitoring_tasks[stream_id] = task
        # else:
        # logger.debug(f"聊天流 {stream_id} 的兴趣消息监控任务已在运行。")

    def _handle_task_completion(self, stream_id: str, task: asyncio.Task):
        """兴趣监控任务完成时的回调函数。"""
        try:
            # 检查任务是否因异常而结束
            exception = task.exception()
            if exception:
                logger.error(f"聊天流 {stream_id} 的兴趣监控任务因异常结束: {exception}")
                logger.error(traceback.format_exc())  # 记录完整的 traceback
            else:
                logger.info(f"聊天流 {stream_id} 的兴趣监控任务正常结束。")
        except asyncio.CancelledError:
            logger.info(f"聊天流 {stream_id} 的兴趣监控任务被取消。")
        except Exception as e:
            logger.error(f"处理聊天流 {stream_id} 任务完成回调时出错: {e}")
        finally:
            # 从字典中移除已完成或取消的任务
            if stream_id in self._interest_monitoring_tasks:
                del self._interest_monitoring_tasks[stream_id]
                logger.debug(f"已从监控任务字典中移除 {stream_id}")

    async def stop_monitoring_interest(self, stream_id: str):
        """停止指定聊天流的兴趣监控任务。"""
        if stream_id in self._interest_monitoring_tasks:
            task = self._interest_monitoring_tasks[stream_id]
            if task and not task.done():
                task.cancel()  # 尝试取消任务
                logger.info(f"尝试取消聊天流 {stream_id} 的兴趣监控任务。")
                try:
                    await task  # 等待任务响应取消
                except asyncio.CancelledError:
                    logger.info(f"聊天流 {stream_id} 的兴趣监控任务已成功取消。")
                except Exception as e:
                    logger.error(f"等待聊天流 {stream_id} 监控任务取消时出现异常: {e}")
            # 在回调函数 _handle_task_completion 中移除任务
        # else:
        # logger.debug(f"聊天流 {stream_id} 没有正在运行的兴趣监控任务可停止。")
