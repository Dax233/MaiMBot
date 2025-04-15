# 定义了来自外部世界的信息
# 外部世界可以是某个聊天 不同平台的聊天 也可以是任意媒体
from datetime import datetime
from src.plugins.models.utils_model import LLM_request
from src.plugins.config.config import global_config
from src.common.database import db
from src.common.logger import get_module_logger
import traceback

logger = get_module_logger("observation")


# 所有观察的基类
class Observation:
    def __init__(self, observe_type, observe_id):
        self.observe_info = ""
        self.observe_type = observe_type
        self.observe_id = observe_id
        self.last_observe_time = datetime.now().timestamp()  # 初始化为当前时间


# 聊天观察
class ChattingObservation(Observation):
    def __init__(self, chat_id):
        super().__init__("chat", chat_id)
        self.chat_id = chat_id

        self.talking_message = []
        self.talking_message_str = ""

        self.name = global_config.BOT_NICKNAME
        self.nick_name = global_config.BOT_ALIAS_NAMES

        self.max_now_obs_len = global_config.observation_context_size
        self.overlap_len = global_config.compressed_length
        self.mid_memorys = []
        self.max_mid_memory_len = global_config.compress_length_limit
        self.mid_memory_info = ""
        self.now_message_info = ""

        self.updating_old = False

        self.llm_summary = LLM_request(
            model=global_config.llm_observation, temperature=0.7, max_tokens=300, request_type="chat_observation"
        )

    # 进行一次观察 返回观察结果observe_info
    def get_observe_info(self, ids=None):
        if ids:
            mid_memory_str = ""
            for id in ids:
                print(f"id：{id}")
                try:
                    for mid_memory in self.mid_memorys:
                        if mid_memory["id"] == id:
                            mid_memory_by_id = mid_memory
                            msg_str = ""
                            for msg in mid_memory_by_id["messages"]:
                                msg_str += f"{msg['detailed_plain_text']}"
                            time_diff = int((datetime.now().timestamp() - mid_memory_by_id["created_at"]) / 60)
                            mid_memory_str += f"距离现在{time_diff}分钟前：\n{msg_str}\n"
                except Exception as e:
                    logger.error(f"获取mid_memory_id失败: {e}")
                    traceback.print_exc()
                    # print(f"获取mid_memory_id失败: {e}")
                    return self.now_message_info

            return mid_memory_str + "现在群里正在聊：\n" + self.now_message_info

        else:
            return self.now_message_info

    async def observe(self):
        # 查找新消息
        new_messages = list(
            db.messages.find({"chat_id": self.chat_id, "time": {"$gt": self.last_observe_time}}).sort("time", 1)
        )  # 按时间正序排列

        if not new_messages:
            return self.observe_info  # 没有新消息，返回上次观察结果

        self.last_observe_time = new_messages[-1]["time"]

        self.talking_message.extend(new_messages)

        # 将新消息转换为字符串格式
        new_messages_str = ""
        for msg in new_messages:
            if "detailed_plain_text" in msg:
                new_messages_str += f"{msg['detailed_plain_text']}"

        # print(f"new_messages_str：{new_messages_str}")

        # 将新消息添加到talking_message，同时保持列表长度不超过20条

        if len(self.talking_message) > self.max_now_obs_len and not self.updating_old:
            self.updating_old = True
            # 计算需要保留的消息数量
            keep_messages_count = self.max_now_obs_len - self.overlap_len
            # 提取所有超出保留数量的最老消息
            oldest_messages = self.talking_message[:-keep_messages_count]
            self.talking_message = self.talking_message[-keep_messages_count:]
            oldest_messages_str = "\n".join([msg["detailed_plain_text"] for msg in oldest_messages])
            oldest_timestamps = [msg["time"] for msg in oldest_messages]

            # 调用 LLM 总结主题
            prompt = f"请总结以下聊天记录的主题：\n{oldest_messages_str}\n主题,用一句话概括包括人物事件和主要信息，不要分点："
            try:
                summary, _ = await self.llm_summary.generate_response_async(prompt)
            except Exception as e:
                print(f"总结主题失败: {e}")
                summary = "无法总结主题"

            mid_memory = {
                "id": str(int(datetime.now().timestamp())),
                "theme": summary,
                "messages": oldest_messages,
                "timestamps": oldest_timestamps,
                "chat_id": self.chat_id,
                "created_at": datetime.now().timestamp(),
            }
            # print(f"mid_memory：{mid_memory}")
            # 存入内存中的 mid_memorys
            self.mid_memorys.append(mid_memory)
            if len(self.mid_memorys) > self.max_mid_memory_len:
                self.mid_memorys.pop(0)

            mid_memory_str = "之前聊天的内容概括是：\n"
            for mid_memory in self.mid_memorys:
                time_diff = int((datetime.now().timestamp() - mid_memory["created_at"]) / 60)
                mid_memory_str += f"距离现在{time_diff}分钟前(聊天记录id:{mid_memory['id']})：{mid_memory['theme']}\n"
            self.mid_memory_info = mid_memory_str

            self.updating_old = False

            # print(f"处理后self.talking_message：{self.talking_message}")

        now_message_str = ""
        now_message_str += self.translate_message_list_to_str(talking_message=self.talking_message)
        self.now_message_info = now_message_str

        logger.debug(f"压缩早期记忆：{self.mid_memory_info}\n现在聊天内容：{self.now_message_info}")

    async def update_talking_summary(self, new_messages_str):
        prompt = ""
        # prompt += f"{personality_info}"
        prompt += f"你的名字叫：{self.name}\n，标识'{self.name}'的都是你自己说的话"
        prompt += f"你正在参与一个qq群聊的讨论，你记得这个群之前在聊的内容是：{self.observe_info}\n"
        prompt += f"现在群里的群友们产生了新的讨论，有了新的发言，具体内容如下：{new_messages_str}\n"
        prompt += """以上是群里在进行的聊天，请你对这个聊天内容进行总结，总结内容要包含聊天的大致内容，目前最新讨论的话题
        以及聊天中的一些重要信息，记得不要分点，精简的概括成一段文本\n"""
        prompt += "总结概括："
        try:
            updated_observe_info, reasoning_content = await self.llm_summary.generate_response_async(prompt)
        except Exception as e:
            print(f"获取总结失败: {e}")
            updated_observe_info = ""

        return updated_observe_info
        # print(f"prompt：{prompt}")
        # print(f"self.observe_info：{self.observe_info}")

    def translate_message_list_to_str(self, talking_message):
        talking_message_str = ""
        for message in talking_message:
            talking_message_str += message["detailed_plain_text"]

        return talking_message_str
