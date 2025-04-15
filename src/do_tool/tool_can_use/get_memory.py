from src.do_tool.tool_can_use.base_tool import BaseTool
from src.plugins.memory_system.Hippocampus import HippocampusManager
from src.common.logger import get_module_logger
from typing import Dict, Any

logger = get_module_logger("mid_chat_mem_tool")


class GetMemoryTool(BaseTool):
    """从记忆系统中获取相关记忆的工具"""

    name = "mid_chat_mem"
    description = "从记忆系统中获取相关记忆"
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "要查询的相关文本"},
            "max_memory_num": {"type": "integer", "description": "最大返回记忆数量"},
        },
        "required": ["text"],
    }

    async def execute(self, function_args: Dict[str, Any], message_txt: str = "") -> Dict[str, Any]:
        """执行记忆获取

        Args:
            function_args: 工具参数
            message_txt: 原始消息文本

        Returns:
            Dict: 工具执行结果
        """
        try:
            text = function_args.get("text", message_txt)
            max_memory_num = function_args.get("max_memory_num", 2)

            # 调用记忆系统
            related_memory = await HippocampusManager.get_instance().get_memory_from_text(
                text=text, max_memory_num=max_memory_num, max_memory_length=2, max_depth=3, fast_retrieval=False
            )

            memory_info = ""
            if related_memory:
                for memory in related_memory:
                    memory_info += memory[1] + "\n"

            if memory_info:
                content = f"你记得这些事情: {memory_info}"
            else:
                content = f"你不太记得有关{text}的记忆，你对此不太了解"

            return {"name": "mid_chat_mem", "content": content}
        except Exception as e:
            logger.error(f"记忆获取工具执行失败: {str(e)}")
            return {"name": "mid_chat_mem", "content": f"记忆获取失败: {str(e)}"}


# 注册工具
# register_tool(GetMemoryTool)
