from src.plugins.models.utils_model import LLM_request
from src.plugins.config.config import global_config
from src.plugins.chat.chat_stream import ChatStream
from src.common.database import db
import time
import json
from src.common.logger import get_module_logger
from typing import Union
from src.do_tool.tool_can_use import get_all_tool_definitions, get_tool_instance

logger = get_module_logger("tool_use")


class ToolUser:
    def __init__(self):
        self.llm_model_tool = LLM_request(
            model=global_config.llm_heartflow, temperature=0.2, max_tokens=1000, request_type="tool_use"
        )

    async def _build_tool_prompt(self, message_txt:str, sender_name:str, chat_stream:ChatStream):
        """构建工具使用的提示词
        
        Args:
            message_txt: 用户消息文本
            sender_name: 发送者名称
            chat_stream: 聊天流对象
            
        Returns:
            str: 构建好的提示词
        """
        new_messages = list(
            db.messages.find({"chat_id": chat_stream.stream_id, "time": {"$gt": time.time()}})
            .sort("time", 1)
            .limit(15)
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
        prompt += f"注意你就是{bot_name}，{bot_name}指的就是你。"
        prompt += "你现在需要对群里的聊天内容进行回复，现在请你思考，你是否需要额外的信息，或者一些工具来帮你回复，比如回忆或者搜寻已有的知识，或者了解你现在正在做什么，请输出你需要的工具，或者你需要的额外信息。"
        
        return prompt
    
    def _define_tools(self):
        """获取所有已注册工具的定义
        
        Returns:
            list: 工具定义列表
        """
        return get_all_tool_definitions()
    
    async def _execute_tool_call(self, tool_call, message_txt:str):
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
                return {
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "name": function_name,
                    "content": result["content"]
                }
            return None
        except Exception as e:
            logger.error(f"执行工具调用时发生错误: {str(e)}")
            return None
    
    async def use_tool(self, message_txt:str, sender_name:str, chat_stream:ChatStream):
        """使用工具辅助思考，判断是否需要额外信息
        
        Args:
            message_txt: 用户消息文本
            sender_name: 发送者名称
            chat_stream: 聊天流对象
            
        Returns:
            dict: 工具使用结果
        """
        try:
            # 构建提示词
            prompt = await self._build_tool_prompt(message_txt, sender_name, chat_stream)
            
            # 定义可用工具
            tools = self._define_tools()
            
            # 使用llm_model_tool发送带工具定义的请求
            payload = {
                "model": self.llm_model_tool.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": global_config.max_response_length,
                "tools": tools,
                "temperature": 0.2
            }
            
            logger.debug(f"发送工具调用请求，模型: {self.llm_model_tool.model_name}")
            # 发送请求获取模型是否需要调用工具
            response = await self.llm_model_tool._execute_request(
                endpoint="/chat/completions", 
                payload=payload,
                prompt=prompt
            )
            
            # 根据返回值数量判断是否有工具调用
            if len(response) == 3:
                content, reasoning_content, tool_calls = response
                logger.info(f"工具思考: {tool_calls}")
                
                # 检查响应中工具调用是否有效
                if not tool_calls:
                    logger.info("模型返回了空的tool_calls列表")
                    return {"used_tools": False}
                    
                logger.info(f"模型请求调用{len(tool_calls)}个工具")
                tool_results = []
                collected_info = ""
                
                # 执行所有工具调用
                for tool_call in tool_calls:
                    result = await self._execute_tool_call(tool_call, message_txt)
                    if result:
                        tool_results.append(result)
                        # 将工具结果添加到收集的信息中
                        collected_info += f"\n{result['name']}返回结果: {result['content']}\n"
                
                # 如果有工具结果，直接返回收集的信息
                if collected_info:
                    logger.info(f"工具调用收集到信息: {collected_info}")
                    return {
                        "used_tools": True,
                        "collected_info": collected_info,
                    }
            else:
                # 没有工具调用
                content, reasoning_content = response
                logger.info("模型没有请求调用任何工具")
            
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