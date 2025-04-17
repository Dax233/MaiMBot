import time
from random import random
import traceback
from typing import List
from ...memory_system.Hippocampus import HippocampusManager
from ...moods.moods import MoodManager
from ....config.config import global_config
from ...chat.emoji_manager import emoji_manager
from .think_flow_generator import ResponseGenerator
from ...chat.message import MessageSending, MessageRecv, MessageThinking, MessageSet
from ...chat.messagesender import message_manager
from ...storage.storage import MessageStorage
from ...chat.utils import is_mentioned_bot_in_message, get_recent_group_detailed_plain_text
from ...chat.utils_image import image_path_to_base64
from ...willing.willing_manager import willing_manager
from ...message import UserInfo, Seg
from src.heart_flow.heartflow import heartflow
from src.common.logger import get_module_logger, CHAT_STYLE_CONFIG, LogConfig
from ...chat.chat_stream import chat_manager
from ...person_info.relationship_manager import relationship_manager
from ...chat.message_buffer import message_buffer
from src.plugins.respon_info_catcher.info_catcher import info_catcher_manager
from ...utils.timer_calculater import Timer
from src.do_tool.tool_use import ToolUser

# 定义日志配置
chat_config = LogConfig(
    console_format=CHAT_STYLE_CONFIG["console_format"],
    file_format=CHAT_STYLE_CONFIG["file_format"],
)

logger = get_module_logger("think_flow_chat", config=chat_config)


class ThinkFlowChat:
    def __init__(self):
        self.storage = MessageStorage()
        self.gpt = ResponseGenerator()
        self.mood_manager = MoodManager.get_instance()
        self.mood_manager.start_mood_update()
        self.tool_user = ToolUser()

    async def _create_thinking_message(self, message, chat, userinfo, messageinfo):
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

        message_manager.add_message(thinking_message)

        return thinking_id

    async def _send_response_messages(self, message, chat, response_set: List[str], thinking_id) -> MessageSending:
        """发送回复消息"""
        container = message_manager.get_container(chat.stream_id)
        thinking_message = None

        for msg in container.messages:
            if isinstance(msg, MessageThinking) and msg.message_info.message_id == thinking_id:
                thinking_message = msg
                container.messages.remove(msg)
                break

        if not thinking_message:
            logger.warning("未找到对应的思考消息，可能已超时被移除")
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
            )
            if not mark_head:
                mark_head = True
                first_bot_msg = bot_message

            # print(f"thinking_start_time:{bot_message.thinking_start_time}")
            message_set.add_message(bot_message)
        message_manager.add_message(message_set)
        return first_bot_msg

    async def _handle_emoji(self, message, chat, response, send_emoji=""):
        """处理表情包"""
        if send_emoji:
            emoji_raw = await emoji_manager.get_emoji_for_text(send_emoji)
        else:
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
            )

            message_manager.add_message(bot_message)

    async def _update_relationship(self, message: MessageRecv, response_set):
        """更新关系情绪"""
        ori_response = ",".join(response_set)
        stance, emotion = await self.gpt._get_emotion_tags(ori_response, message.processed_plain_text)
        await relationship_manager.calculate_update_relationship_value(
            chat_stream=message.chat_stream, label=emotion, stance=stance
        )
        self.mood_manager.update_mood_from_emotion(emotion, global_config.mood_intensity_factor)

    async def process_message(self, message_data: str) -> None:
        """处理消息并生成回复"""
        timing_results = {}
        response_set = None

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
        message.update_chat_stream(chat)

        # 创建心流与chat的观察
        heartflow.create_subheartflow(chat.stream_id)

        await message.process()
        logger.trace(f"消息处理成功{message.processed_plain_text}")

        # 过滤词/正则表达式过滤
        if self._check_ban_words(message.processed_plain_text, chat, userinfo) or self._check_ban_regex(
            message.raw_message, chat, userinfo
        ):
            return
        logger.trace(f"过滤词/正则表达式过滤成功{message.processed_plain_text}")

        await self.storage.store_message(message, chat)
        logger.trace(f"存储成功{message.processed_plain_text}")

        # 记忆激活
        with Timer("记忆激活", timing_results):
            interested_rate = await HippocampusManager.get_instance().get_activate_from_text(
                message.processed_plain_text, fast_retrieval=True
            )
        logger.trace(f"记忆激活: {interested_rate}")

        # 查询缓冲器结果，会整合前面跳过的消息，改变processed_plain_text
        buffer_result = await message_buffer.query_buffer_result(message)

        # 处理提及
        is_mentioned, reply_probability = is_mentioned_bot_in_message(message)

        # 意愿管理器：设置当前message信息
        willing_manager.setup(message, chat, is_mentioned, interested_rate)

        # 处理缓冲器结果
        if not buffer_result:
            await willing_manager.bombing_buffer_message_handle(message.message_info.message_id)
            willing_manager.delete(message.message_info.message_id)
            F_type = "seglist"
            if message.message_segment.type != "seglist":
                F_type =message.message_segment.type
            else:
                if (isinstance(message.message_segment.data, list) 
                and all(isinstance(x, Seg) for x in message.message_segment.data)
                and len(message.message_segment.data) == 1):
                    F_type = message.message_segment.data[0].type
            if F_type == "text":
                logger.info(f"触发缓冲，已炸飞消息：{message.processed_plain_text}")
            elif F_type == "image":
                logger.info("触发缓冲，已炸飞表情包/图片")
            elif F_type == "seglist":
                logger.info("触发缓冲，已炸飞消息列")
            return

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
            f"{chat.user_info.user_nickname}:"
            f"{message.processed_plain_text}{willing_log}[概率:{reply_probability * 100:.1f}%]"
        )

        do_reply = False
        if random() < reply_probability:
            try:
                do_reply = True

                # 回复前处理
                await willing_manager.before_generate_reply_handle(message.message_info.message_id)

                # 创建思考消息
                try:
                    with Timer("创建思考消息", timing_results):
                        thinking_id = await self._create_thinking_message(message, chat, userinfo, messageinfo)
                except Exception as e:
                    logger.error(f"心流创建思考消息失败: {e}")

                logger.trace(f"创建捕捉器，thinking_id:{thinking_id}")

                info_catcher = info_catcher_manager.get_info_catcher(thinking_id)
                info_catcher.catch_decide_to_response(message)

                # 观察
                try:
                    with Timer("观察", timing_results):
                        await heartflow.get_subheartflow(chat.stream_id).do_observe()
                except Exception as e:
                    logger.error(f"心流观察失败: {e}")
                    logger.error(traceback.format_exc())

                info_catcher.catch_after_observe(timing_results["观察"])

                # 思考前使用工具
                update_relationship = ""
                get_mid_memory_id = []
                tool_result_info = {}
                send_emoji = ""
                try:
                    with Timer("思考前使用工具", timing_results):
                        tool_result = await self.tool_user.use_tool(
                            message.processed_plain_text,
                            message.message_info.user_info.user_nickname,
                            chat,
                            heartflow.get_subheartflow(chat.stream_id),
                        )
                        # 如果工具被使用且获得了结果，将收集到的信息合并到思考中
                        # collected_info = ""
                        if tool_result.get("used_tools", False):
                            if "structured_info" in tool_result:
                                tool_result_info = tool_result["structured_info"]
                                # collected_info = ""
                                get_mid_memory_id = []
                                update_relationship = ""

                                # 动态解析工具结果
                                for tool_name, tool_data in tool_result_info.items():
                                    # tool_result_info += f"\n{tool_name} 相关信息:\n"
                                    # for item in tool_data:
                                    # tool_result_info += f"- {item['name']}: {item['content']}\n"

                                    # 特殊判定：mid_chat_mem
                                    if tool_name == "mid_chat_mem":
                                        for mid_memory in tool_data:
                                            get_mid_memory_id.append(mid_memory["content"])

                                    # 特殊判定：change_mood
                                    if tool_name == "change_mood":
                                        for mood in tool_data:
                                            self.mood_manager.update_mood_from_emotion(
                                                mood["content"], global_config.mood_intensity_factor
                                            )

                                    # 特殊判定：change_relationship
                                    if tool_name == "change_relationship":
                                        update_relationship = tool_data[0]["content"]

                                    if tool_name == "send_emoji":
                                        send_emoji = tool_data[0]["content"]

                except Exception as e:
                    logger.error(f"思考前工具调用失败: {e}")
                    logger.error(traceback.format_exc())

                # 处理关系更新
                if update_relationship:
                    stance, emotion = await self.gpt._get_emotion_tags_with_reason(
                        "你还没有回复", message.processed_plain_text, update_relationship
                    )
                    await relationship_manager.calculate_update_relationship_value(
                        chat_stream=message.chat_stream, label=emotion, stance=stance
                    )

                # 思考前脑内状态
                try:
                    with Timer("思考前脑内状态", timing_results):
                        current_mind, past_mind = await heartflow.get_subheartflow(
                            chat.stream_id
                        ).do_thinking_before_reply(
                            message_txt=message.processed_plain_text,
                            sender_info=message.message_info.user_info,
                            chat_stream=chat,
                            obs_id=get_mid_memory_id,
                            extra_info=tool_result_info,
                        )
                except Exception as e:
                    logger.error(f"心流思考前脑内状态失败: {e}")
                    logger.error(traceback.format_exc())
                    # 确保变量被定义，即使在错误情况下
                    current_mind = ""
                    past_mind = ""

                info_catcher.catch_afer_shf_step(timing_results["思考前脑内状态"], past_mind, current_mind)

                # 生成回复
                with Timer("生成回复", timing_results):
                    response_set = await self.gpt.generate_response(message, thinking_id)

                info_catcher.catch_after_generate_response(timing_results["生成回复"])

                if not response_set:
                    logger.info("回复生成失败，返回为空")
                    return

                # 发送消息
                try:
                    with Timer("发送消息", timing_results):
                        first_bot_msg = await self._send_response_messages(message, chat, response_set, thinking_id)
                except Exception as e:
                    logger.error(f"心流发送消息失败: {e}")

                info_catcher.catch_after_response(timing_results["发送消息"], response_set, first_bot_msg)

                info_catcher.done_catch()

                # 处理表情包
                try:
                    with Timer("处理表情包", timing_results):
                        if global_config.emoji_chance == 1:
                            if send_emoji:
                                logger.info(f"麦麦决定发送表情包{send_emoji}")
                                await self._handle_emoji(message, chat, response_set, send_emoji)
                        else:
                            if random() < global_config.emoji_chance:
                                await self._handle_emoji(message, chat, response_set)
                except Exception as e:
                    logger.error(f"心流处理表情包失败: {e}")

                # 思考后脑内状态更新
                try:
                    with Timer("思考后脑内状态更新", timing_results):
                        stream_id = message.chat_stream.stream_id
                        chat_talking_prompt = ""
                        if stream_id:
                            chat_talking_prompt = get_recent_group_detailed_plain_text(
                                stream_id, limit=global_config.MAX_CONTEXT_SIZE, combine=True
                            )

                        await heartflow.get_subheartflow(stream_id).do_thinking_after_reply(
                            response_set, chat_talking_prompt, tool_result_info
                        )
                except Exception as e:
                    logger.error(f"心流思考后脑内状态更新失败: {e}")
                    logger.error(traceback.format_exc())

                # 回复后处理
                await willing_manager.after_generate_reply_handle(message.message_info.message_id)

                # 处理认识关系
                try:
                    is_known = await relationship_manager.is_known_some_one(
                        message.message_info.platform, message.message_info.user_info.user_id
                    )
                    if not is_known:
                        logger.info(f"首次认识用户: {message.message_info.user_info.user_nickname}")
                        await relationship_manager.first_knowing_some_one(
                            message.message_info.platform,
                            message.message_info.user_info.user_id,
                            message.message_info.user_info.user_nickname,
                            message.message_info.user_info.user_cardname
                            or message.message_info.user_info.user_nickname,
                            "",
                        )
                    else:
                        logger.debug(f"已认识用户: {message.message_info.user_info.user_nickname}")
                        if not await relationship_manager.is_qved_name(
                            message.message_info.platform, message.message_info.user_info.user_id
                        ):
                            logger.info(f"更新已认识但未取名的用户: {message.message_info.user_info.user_nickname}")
                            await relationship_manager.first_knowing_some_one(
                                message.message_info.platform,
                                message.message_info.user_info.user_id,
                                message.message_info.user_info.user_nickname,
                                message.message_info.user_info.user_cardname
                                or message.message_info.user_info.user_nickname,
                                "",
                            )
                except Exception as e:
                    logger.error(f"处理认识关系失败: {e}")
                    logger.error(traceback.format_exc())

            except Exception as e:
                logger.error(f"心流处理消息失败: {e}")
                logger.error(traceback.format_exc())

        # 输出性能计时结果
        if do_reply:
            timing_str = " | ".join([f"{step}: {duration:.2f}秒" for step, duration in timing_results.items()])
            trigger_msg = message.processed_plain_text
            response_msg = " ".join(response_set) if response_set else "无回复"
            logger.info(f"触发消息: {trigger_msg[:20]}... | 思维消息: {response_msg[:20]}... | 性能计时: {timing_str}")
        else:
            # 不回复处理
            await willing_manager.not_reply_handle(message.message_info.message_id)

        # 意愿管理器：注销当前message信息
        willing_manager.delete(message.message_info.message_id)

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
