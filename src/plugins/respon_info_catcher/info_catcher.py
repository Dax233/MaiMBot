from src.plugins.config.config import global_config
from src.plugins.chat.message import MessageRecv, MessageSending, Message
from src.common.database import db
import time
import traceback
from typing import List


class InfoCatcher:
    def __init__(self):
        self.chat_history = []  # 聊天历史，长度为三倍使用的上下文
        self.context_length = global_config.MAX_CONTEXT_SIZE
        self.chat_history_in_thinking = []  # 思考期间的聊天内容
        self.chat_history_after_response = []  # 回复后的聊天内容，长度为一倍上下文

        self.chat_id = ""
        self.response_mode = global_config.response_mode
        self.trigger_response_text = ""
        self.response_text = ""

        self.trigger_response_time = 0
        self.trigger_response_message = None

        self.response_time = 0
        self.response_messages = []

        # 使用字典来存储 heartflow 模式的数据
        self.heartflow_data = {
            "heart_flow_prompt": "",
            "sub_heartflow_before": "",
            "sub_heartflow_now": "",
            "sub_heartflow_after": "",
            "sub_heartflow_model": "",
            "prompt": "",
            "response": "",
            "model": "",
        }

        # 使用字典来存储 reasoning 模式的数据
        self.reasoning_data = {"thinking_log": "", "prompt": "", "response": "", "model": ""}

        # 耗时
        self.timing_results = {
            "interested_rate_time": 0,
            "sub_heartflow_observe_time": 0,
            "sub_heartflow_step_time": 0,
            "make_response_time": 0,
        }

    def catch_decide_to_response(self, message: MessageRecv):
        # 搜集决定回复时的信息
        self.trigger_response_message = message
        self.trigger_response_text = message.detailed_plain_text

        self.trigger_response_time = time.time()

        self.chat_id = message.chat_stream.stream_id

        self.chat_history = self.get_message_from_db_before_msg(message)

    def catch_after_observe(self, obs_duration: float):  # 这里可以有更多信息
        self.timing_results["sub_heartflow_observe_time"] = obs_duration

    # def catch_shf

    def catch_afer_shf_step(self, step_duration: float, past_mind: str, current_mind: str):
        self.timing_results["sub_heartflow_step_time"] = step_duration
        if len(past_mind) > 1:
            self.heartflow_data["sub_heartflow_before"] = past_mind[-1]
            self.heartflow_data["sub_heartflow_now"] = current_mind
        else:
            self.heartflow_data["sub_heartflow_before"] = past_mind[-1]
            self.heartflow_data["sub_heartflow_now"] = current_mind

    def catch_after_llm_generated(self, prompt: str, response: str, reasoning_content: str = "", model_name: str = ""):
        if self.response_mode == "heart_flow":
            self.heartflow_data["prompt"] = prompt
            self.heartflow_data["response"] = response
            self.heartflow_data["model"] = model_name
        elif self.response_mode == "reasoning":
            self.reasoning_data["thinking_log"] = reasoning_content
            self.reasoning_data["prompt"] = prompt
            self.reasoning_data["response"] = response
            self.reasoning_data["model"] = model_name

        self.response_text = response

    def catch_after_generate_response(self, response_duration: float):
        self.timing_results["make_response_time"] = response_duration

    def catch_after_response(
        self, response_duration: float, response_message: List[str], first_bot_msg: MessageSending
    ):
        self.timing_results["make_response_time"] = response_duration
        self.response_time = time.time()
        for msg in response_message:
            self.response_messages.append(msg)

        self.chat_history_in_thinking = self.get_message_from_db_between_msgs(
            self.trigger_response_message, first_bot_msg
        )

    def get_message_from_db_between_msgs(self, message_start: Message, message_end: Message):
        try:
            # 从数据库中获取消息的时间戳
            time_start = message_start.message_info.time
            time_end = message_end.message_info.time
            chat_id = message_start.chat_stream.stream_id

            print(f"查询参数: time_start={time_start}, time_end={time_end}, chat_id={chat_id}")

            # 查询数据库，获取 chat_id 相同且时间在 start 和 end 之间的数据
            messages_between = db.messages.find(
                {"chat_id": chat_id, "time": {"$gt": time_start, "$lt": time_end}}
            ).sort("time", -1)

            result = list(messages_between)
            print(f"查询结果数量: {len(result)}")
            if result:
                print(f"第一条消息时间: {result[0]['time']}")
                print(f"最后一条消息时间: {result[-1]['time']}")
            return result
        except Exception as e:
            print(f"获取消息时出错: {str(e)}")
            return []

    def get_message_from_db_before_msg(self, message: MessageRecv):
        # 从数据库中获取消息
        message_id = message.message_info.message_id
        chat_id = message.chat_stream.stream_id

        # 查询数据库，获取 chat_id 相同且 message_id 小于当前消息的 30 条数据
        messages_before = (
            db.messages.find({"chat_id": chat_id, "message_id": {"$lt": message_id}})
            .sort("time", -1)
            .limit(self.context_length * 3)
        )  # 获取更多历史信息

        return list(messages_before)

    def message_list_to_dict(self, message_list):
        # 存储简化的聊天记录
        result = []
        for message in message_list:
            if not isinstance(message, dict):
                message = self.message_to_dict(message)
            # print(message)

            lite_message = {
                "time": message["time"],
                "user_nickname": message["user_info"]["user_nickname"],
                "processed_plain_text": message["processed_plain_text"],
            }
            result.append(lite_message)

        return result

    def message_to_dict(self, message):
        if not message:
            return None
        if isinstance(message, dict):
            return message
        return {
            # "message_id": message.message_info.message_id,
            "time": message.message_info.time,
            "user_id": message.message_info.user_info.user_id,
            "user_nickname": message.message_info.user_info.user_nickname,
            "processed_plain_text": message.processed_plain_text,
            # "detailed_plain_text": message.detailed_plain_text
        }

    def done_catch(self):
        """将收集到的信息存储到数据库的 thinking_log 集合中"""
        try:
            # 将消息对象转换为可序列化的字典

            thinking_log_data = {
                "chat_id": self.chat_id,
                "response_mode": self.response_mode,
                "trigger_text": self.trigger_response_text,
                "response_text": self.response_text,
                "trigger_info": {
                    "time": self.trigger_response_time,
                    "message": self.message_to_dict(self.trigger_response_message),
                },
                "response_info": {
                    "time": self.response_time,
                    "message": self.response_messages,
                },
                "timing_results": self.timing_results,
                "chat_history": self.message_list_to_dict(self.chat_history),
                "chat_history_in_thinking": self.message_list_to_dict(self.chat_history_in_thinking),
                "chat_history_after_response": self.message_list_to_dict(self.chat_history_after_response),
            }

            # 根据不同的响应模式添加相应的数据
            if self.response_mode == "heart_flow":
                thinking_log_data["mode_specific_data"] = self.heartflow_data
            elif self.response_mode == "reasoning":
                thinking_log_data["mode_specific_data"] = self.reasoning_data

            # 将数据插入到 thinking_log 集合中
            db.thinking_log.insert_one(thinking_log_data)

            return True
        except Exception as e:
            print(f"存储思考日志时出错: {str(e)}")
            print(traceback.format_exc())
            return False


class InfoCatcherManager:
    def __init__(self):
        self.info_catchers = {}

    def get_info_catcher(self, thinking_id: str) -> InfoCatcher:
        if thinking_id not in self.info_catchers:
            self.info_catchers[thinking_id] = InfoCatcher()
        return self.info_catchers[thinking_id]


info_catcher_manager = InfoCatcherManager()
