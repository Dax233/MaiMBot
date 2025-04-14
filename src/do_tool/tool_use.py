from src.plugins.models.utils_model import LLM_request
from src.plugins.config.config import global_config
from src.plugins.chat.chat_stream import ChatStream
from src.common.database import db
import time
import json
from src.common.logger import get_module_logger, TOOL_USE_STYLE_CONFIG, LogConfig
from src.do_tool.tool_can_use import get_all_tool_definitions, get_tool_instance

tool_use_config = LogConfig(
    # 使用消息发送专用样式
    console_format=TOOL_USE_STYLE_CONFIG["console_format"],
    file_format=TOOL_USE_STYLE_CONFIG["file_format"],
)
logger = get_module_logger("tool_use", config=tool_use_config)


class ToolUser:
    def __init__(self):
        self.llm_model_tool = LLM_request(
            model=global_config.llm_heartflow, temperature=0.2, max_tokens=1000, request_type="tool_use"
        )

    async def _build_tool_prompt(self, message_txt: str, sender_name: str, chat_stream: ChatStream, reply_message:str = ""):
        """构建工具使用的提示词

        Args:
            message_txt: 用户消息文本
            sender_name: 发送者名称
            chat_stream: 聊天流对象

        Returns:
            str: 构建好的提示词
        """
        new_messages = list(
            db.messages.find({"chat_id": chat_stream.stream_id, "time": {"$gt": time.time()}}).sort("time", 1).limit(15)
        )
        new_messages_str = ""
        for msg in new_messages:
            if "detailed_plain_text" in msg:
                new_messages_str += f"{msg['detailed_plain_text']}"

        # 这些信息应该从调用者传入，而不是从self获取
        bot_name = global_config.BOT_NICKNAME
        prompt = ""
        prompt += "你正在思考如何回复群里的消息。\n"
        prompt += f"你注意到{sender_name}刚刚说：{message_txt}\n"
        if reply_message:
            prompt += f"你刚刚回复的内容是：{reply_message}\n"
        prompt += f"注意你就是{bot_name}，{bot_name}指的就是你。"

        prompt += "你现在需要对群里的聊天内容进行回复，现在选择工具来对消息和你的回复进行处理，你是否需要额外的信息，或者进行一些动作，比如回忆或者搜寻已有的知识，改变关系和情感，或者了解你现在正在做什么，请输出你需要的工具，或者你需要的额外信息。"
        return prompt

    def _define_tools(self):
        """获取所有已注册工具的定义

        Returns:
            list: 工具定义列表
        """
        return get_all_tool_definitions()

    async def _execute_tool_call(self, tool_call, message_txt: str):
        """执行特定的工具调用

        Args:
            tool_call: 工具调用对象
            message_txt: 原始消息文本

        Returns:
            dict: 工具调用结果
        """
        try:
            function_name = tool_call["function"]["name"]
            function_args = json.loads(tool_call["function"]["arguments"])

            # 获取对应工具实例
            tool_instance = get_tool_instance(function_name)
            if not tool_instance:
                logger.warning(f"未知工具名称: {function_name}")
                return None

            # 执行工具
            result = await tool_instance.execute(function_args, message_txt)
            if result:
                # 根据工具名称确定类型标签
                tool_type = ""
                if "memory" in function_name.lower():
                    tool_type = "memory"
                elif "schedule" in function_name.lower() or "task" in function_name.lower():
                    tool_type = "schedule"
                elif "knowledge" in function_name.lower():
                    tool_type = "knowledge"
                elif "change_relationship" in function_name.lower():
                    tool_type = "change_relationship"
                elif "change_mood" in function_name.lower():
                    tool_type = "change_mood"
                else:
                    tool_type = "other"

                return {
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "name": function_name,
                    "type": tool_type,
                    "content": result["content"],
                }
            return None
        except Exception as e:
            logger.error(f"执行工具调用时发生错误: {str(e)}")
            return None

    async def use_tool(self, message_txt: str, sender_name: str, chat_stream: ChatStream):
        """使用工具辅助思考，判断是否需要额外信息

        Args:
            message_txt: 用户消息文本
            sender_name: 发送者名称
            chat_stream: 聊天流对象

        Returns:
            dict: 工具使用结果，包含结构化的信息
        """
        try:
            # 构建提示词
            prompt = await self._build_tool_prompt(message_txt, sender_name, chat_stream)

            # 定义可用工具
            tools = self._define_tools()
            logger.trace(f"工具定义: {tools}")

            # 使用llm_model_tool发送带工具定义的请求
            payload = {
                "model": self.llm_model_tool.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": global_config.max_response_length,
                "tools": tools,
                "temperature": 0.2,
            }

            logger.trace(f"发送工具调用请求，模型: {self.llm_model_tool.model_name}")
            # 发送请求获取模型是否需要调用工具
            response = await self.llm_model_tool._execute_request(
                endpoint="/chat/completions", payload=payload, prompt=prompt
            )

            # 根据返回值数量判断是否有工具调用
            if len(response) == 3:
                content, reasoning_content, tool_calls = response
                # logger.info(f"工具思考: {tool_calls}")
                # logger.debug(f"工具思考: {content}")

                # 检查响应中工具调用是否有效
                if not tool_calls:
                    logger.debug("模型返回了空的tool_calls列表")
                    return {"used_tools": False}

                tool_calls_str = "" 
                for tool_call in tool_calls:
                    tool_calls_str += f"{tool_call['function']['name']}\n"
                logger.info(f"模型请求调用{len(tool_calls)}个工具: {tool_calls_str}")
                tool_results = []
                structured_info = {
                    "memory": [],
                    "schedule": [],
                    "knowledge": [],
                    "change_relationship": [],
                    "change_mood": [],
                    "other": []
                }

                # 执行所有工具调用
                for tool_call in tool_calls:
                    result = await self._execute_tool_call(tool_call, message_txt)
                    if result:
                        tool_results.append(result)
                        # 将工具结果添加到对应类型的列表中
                        structured_info[result["type"]].append({
                            "name": result["name"],
                            "content": result["content"]
                        })

                # 如果有工具结果，返回结构化的信息
                if any(structured_info.values()):
                    logger.info(f"工具调用收集到结构化信息: {json.dumps(structured_info, ensure_ascii=False)}")
                    return {
                        "used_tools": True,
                        "structured_info": structured_info
                    }
            else:
                # 没有工具调用
                content, reasoning_content = response
                logger.debug("模型没有请求调用任何工具")

            # 如果没有工具调用或处理失败，直接返回原始思考
            return {
                "used_tools": False,
            }

        except Exception as e:
            logger.error(f"工具调用过程中出错: {str(e)}")
            return {
                "used_tools": False,
                "error": str(e),
            }
