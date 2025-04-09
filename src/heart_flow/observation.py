# 定义了来自外部世界的信息
# 外部世界可以是某个聊天 不同平台的聊天 也可以是任意媒体
from datetime import datetime
from src.plugins.models.utils_model import LLM_request
from src.plugins.config.config import global_config
from src.common.database import db
from src.individuality.individuality import Individuality
import random


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

        self.observe_times = 0

        self.summary_count = 0  # 30秒内的更新次数
        self.max_update_in_30s = 2  # 30秒内最多更新2次
        self.last_summary_time = 0  # 上次更新summary的时间

        self.sub_observe = None

        self.llm_summary = LLM_request(
            model=global_config.llm_observation, temperature=0.7, max_tokens=300, request_type="chat_observation"
        )

    # 进行一次观察 返回观察结果observe_info
    async def observe(self):
        # 查找新消息，限制最多30条
        new_messages = list(
            db.messages.find({"chat_id": self.chat_id, "time": {"$gt": self.last_observe_time}})
            .sort("time", 1)
            .limit(20)
        )  # 按时间正序排列，最多20条

        if not new_messages:
            return self.observe_info  # 没有新消息，返回上次观察结果

        # 将新消息转换为字符串格式
        new_messages_str = ""
        for msg in new_messages:
            if "detailed_plain_text" in msg:
                new_messages_str += f"{msg['detailed_plain_text']}"

        # print(f"new_messages_str：{new_messages_str}")

        # 将新消息添加到talking_message，同时保持列表长度不超过20条
        self.talking_message.extend(new_messages)
        if len(self.talking_message) > 20:
            self.talking_message = self.talking_message[-20:]  # 只保留最新的20条
        self.translate_message_list_to_str()

        # 更新观察次数
        self.observe_times += 1
        self.last_observe_time = new_messages[-1]["time"]

        # 检查是否需要更新summary
        current_time = int(datetime.now().timestamp())
        if current_time - self.last_summary_time >= 30:  # 如果超过30秒，重置计数
            self.summary_count = 0
            self.last_summary_time = current_time

        if self.summary_count < self.max_update_in_30s:  # 如果30秒内更新次数小于2次
            await self.update_talking_summary(new_messages_str)
            self.summary_count += 1

        return self.observe_info

    async def carefully_observe(self):
        # 查找新消息，限制最多40条
        new_messages = list(
            db.messages.find({"chat_id": self.chat_id, "time": {"$gt": self.last_observe_time}})
            .sort("time", 1)
            .limit(30)
        )  # 按时间正序排列，最多30条

        if not new_messages:
            return self.observe_info  # 没有新消息，返回上次观察结果

        # 将新消息转换为字符串格式
        new_messages_str = ""
        for msg in new_messages:
            if "detailed_plain_text" in msg:
                new_messages_str += f"{msg['detailed_plain_text']}\n"

        # 将新消息添加到talking_message，同时保持列表长度不超过30条
        self.talking_message.extend(new_messages)
        if len(self.talking_message) > 30:
            self.talking_message = self.talking_message[-30:]  # 只保留最新的30条
        self.translate_message_list_to_str()

        # 更新观察次数
        self.observe_times += 1
        self.last_observe_time = new_messages[-1]["time"]

        await self.update_talking_summary(new_messages_str)
        return self.observe_info

    async def update_talking_summary(self, new_messages_str):
        # 基于已经有的talking_summary，和新的talking_message，生成一个summary
        # print(f"更新聊天总结：{self.talking_summary}")
        # 开始构建prompt
        prompt_personality = "你"
        # person
        individuality = Individuality.get_instance()

        personality_core = individuality.personality.personality_core
        prompt_personality += personality_core

        personality_sides = individuality.personality.personality_sides
        random.shuffle(personality_sides)
        prompt_personality += f",{personality_sides[0]}"

        identity_detail = individuality.identity.identity_detail
        random.shuffle(identity_detail)
        prompt_personality += f",{identity_detail[0]}"

        personality_info = prompt_personality

        prompt = ""
        prompt += f"{personality_info}，请注意识别你自己的聊天发言"
        prompt += f"你的名字叫：{self.name}，你的昵称是：{self.nick_name}\n"
        prompt += f"你正在参与一个qq群聊的讨论，你记得这个群之前在聊的内容是：{self.observe_info}\n"
        prompt += f"现在群里的群友们产生了新的讨论，有了新的发言，具体内容如下：{new_messages_str}\n"
        prompt += """以上是群里在进行的聊天，请你对这个聊天内容进行总结，总结内容要包含聊天的大致内容，
        以及聊天中的一些重要信息，注意识别你自己的发言，记得不要分点，不要太长，精简的概括成一段文本\n"""
        prompt += "总结概括："
        self.observe_info, reasoning_content = await self.llm_summary.generate_response_async(prompt)
        print(f"prompt：{prompt}")
        print(f"self.observe_info：{self.observe_info}")

    def translate_message_list_to_str(self):
        self.talking_message_str = ""
        for message in self.talking_message:
            self.talking_message_str += message["detailed_plain_text"]
