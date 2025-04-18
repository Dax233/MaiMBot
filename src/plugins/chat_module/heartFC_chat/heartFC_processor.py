import time
import traceback
from ...memory_system.Hippocampus import HippocampusManager
from ....config.config import global_config
from ...chat.message import MessageRecv
from ...storage.storage import MessageStorage
from ...chat.utils import is_mentioned_bot_in_message
from ...message import Seg
from src.heart_flow.heartflow import heartflow
from src.common.logger import get_module_logger, CHAT_STYLE_CONFIG, LogConfig
from ...chat.chat_stream import chat_manager
from ...chat.message_buffer import message_buffer
from ...utils.timer_calculater import Timer
from .interest import InterestManager
from .heartFC_chat import HeartFC_Chat  # 导入 HeartFC_Chat 以调用回复生成

# 定义日志配置
processor_config = LogConfig(
    console_format=CHAT_STYLE_CONFIG["console_format"],
    file_format=CHAT_STYLE_CONFIG["file_format"],
)
logger = get_module_logger("heartFC_processor", config=processor_config)

# # 定义兴趣度增加触发回复的阈值 (移至 InterestManager)
# INTEREST_INCREASE_THRESHOLD = 0.5


class HeartFC_Processor:
    def __init__(self, chat_instance: HeartFC_Chat):
        self.storage = MessageStorage()
        self.interest_manager = (
            InterestManager()
        )  # TODO: 可能需要传递 chat_instance 给 InterestManager 或修改其方法签名
        self.chat_instance = chat_instance  # 持有 HeartFC_Chat 实例

    async def process_message(self, message_data: str) -> None:
        """处理接收到的消息，更新状态，并将回复决策委托给 InterestManager"""
        timing_results = {}  # 初始化 timing_results
        message = None
        try:
            message = MessageRecv(message_data)
            groupinfo = message.message_info.group_info
            userinfo = message.message_info.user_info
            messageinfo = message.message_info

            # 消息加入缓冲池
            await message_buffer.start_caching_messages(message)

            # 创建聊天流
            chat = await chat_manager.get_or_create_stream(
                platform=messageinfo.platform,
                user_info=userinfo,
                group_info=groupinfo,
            )
            if not chat:
                logger.error(
                    f"无法为消息创建或获取聊天流: user {userinfo.user_id}, group {groupinfo.group_id if groupinfo else 'None'}"
                )
                return

            message.update_chat_stream(chat)

            # 创建心流与chat的观察 (在接收消息时创建，以便后续观察和思考)
            heartflow.create_subheartflow(chat.stream_id)

            await message.process()
            logger.trace(f"消息处理成功: {message.processed_plain_text}")

            # 过滤词/正则表达式过滤
            if self._check_ban_words(message.processed_plain_text, chat, userinfo) or self._check_ban_regex(
                message.raw_message, chat, userinfo
            ):
                return
            logger.trace(f"过滤词/正则表达式过滤成功: {message.processed_plain_text}")

            # 查询缓冲器结果
            buffer_result = await message_buffer.query_buffer_result(message)

            # 处理缓冲器结果 (Bombing logic)
            if not buffer_result:
                F_type = "seglist"
                if message.message_segment.type != "seglist":
                    F_type = message.message_segment.type
                else:
                    if (
                        isinstance(message.message_segment.data, list)
                        and all(isinstance(x, Seg) for x in message.message_segment.data)
                        and len(message.message_segment.data) == 1
                    ):
                        F_type = message.message_segment.data[0].type
                if F_type == "text":
                    logger.debug(f"触发缓冲，消息：{message.processed_plain_text}")
                elif F_type == "image":
                    logger.debug("触发缓冲，表情包/图片等待中")
                elif F_type == "seglist":
                    logger.debug("触发缓冲，消息列表等待中")
                return  # 被缓冲器拦截，不生成回复

            # ---- 只有通过缓冲的消息才进行存储和后续处理 ----

            # 存储消息 (使用可能被缓冲器更新过的 message)
            try:
                await self.storage.store_message(message, chat)
                logger.trace(f"存储成功 (通过缓冲后): {message.processed_plain_text}")
            except Exception as e:
                logger.error(f"存储消息失败: {e}")
                logger.error(traceback.format_exc())
                # 存储失败可能仍需考虑是否继续，暂时返回
                return

            # 激活度计算 (使用可能被缓冲器更新过的 message.processed_plain_text)
            is_mentioned, _ = is_mentioned_bot_in_message(message)
            interested_rate = 0.0  # 默认值
            try:
                with Timer("记忆激活", timing_results):
                    interested_rate = await HippocampusManager.get_instance().get_activate_from_text(
                        message.processed_plain_text,
                        fast_retrieval=True,  # 使用更新后的文本
                    )
                logger.trace(f"记忆激活率 (通过缓冲后): {interested_rate:.2f}")
            except Exception as e:
                logger.error(f"计算记忆激活率失败: {e}")
                logger.error(traceback.format_exc())

            if is_mentioned:
                interested_rate += 0.8

            # 更新兴趣度
            try:
                self.interest_manager.increase_interest(chat.stream_id, value=interested_rate)
                current_interest = self.interest_manager.get_interest(chat.stream_id)  # 获取更新后的值用于日志
                logger.trace(
                    f"使用激活率 {interested_rate:.2f} 更新后 (通过缓冲后)，当前兴趣度: {current_interest:.2f}"
                )

            except Exception as e:
                logger.error(f"更新兴趣度失败: {e}")  # 调整日志消息
                logger.error(traceback.format_exc())
            # ---- 兴趣度计算和更新结束 ----

            # 打印消息接收和处理信息
            mes_name = chat.group_info.group_name if chat.group_info else "私聊"
            current_time = time.strftime("%H:%M:%S", time.localtime(message.message_info.time))
            logger.info(
                f"[{current_time}][{mes_name}]"
                f"{chat.user_info.user_nickname}:"
                f"{message.processed_plain_text}"
                f"兴趣度: {current_interest:.2f}"
            )

            # 回复触发逻辑已移至 HeartFC_Chat 的监控任务

        except Exception as e:
            logger.error(f"消息处理失败 (process_message V3): {e}")
            logger.error(traceback.format_exc())
            if message:  # 记录失败的消息内容
                logger.error(f"失败消息原始内容: {message.raw_message}")

    def _check_ban_words(self, text: str, chat, userinfo) -> bool:
        """检查消息中是否包含过滤词"""
        for word in global_config.ban_words:
            if word in text:
                logger.info(
                    f"[{chat.group_info.group_name if chat.group_info else '私聊'}]{userinfo.user_nickname}:{text}"
                )
                logger.info(f"[过滤词识别]消息中含有{word}，filtered")
                return True
        return False

    def _check_ban_regex(self, text: str, chat, userinfo) -> bool:
        """检查消息是否匹配过滤正则表达式"""
        for pattern in global_config.ban_msgs_regex:
            if pattern.search(text):
                logger.info(
                    f"[{chat.group_info.group_name if chat.group_info else '私聊'}]{userinfo.user_nickname}:{text}"
                )
                logger.info(f"[正则表达式过滤]消息匹配到{pattern}，filtered")
                return True
        return False
