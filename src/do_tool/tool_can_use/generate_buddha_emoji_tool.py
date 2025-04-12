from src.do_tool.tool_can_use.base_tool import BaseTool, register_tool
from src.common.logger import get_module_logger
from typing import Dict, Any

logger = get_module_logger("generate_buddha_emoji_tool")

class GenerateBuddhaEmojiTool(BaseTool):
    """生成佛祖颜文字的工具类"""
    name = "generate_buddha_emoji"
    description = "生成一个佛祖的颜文字表情"
    parameters = {
        "type": "object",
        "properties": {
            # 无参数
        },
        "required": []
    }
    
    async def execute(self, function_args: Dict[str, Any], message_txt: str = "") -> Dict[str, Any]:
        """执行工具功能，生成佛祖颜文字
        
        Args:
            function_args: 工具参数
            message_txt: 原始消息文本
            
        Returns:
            Dict: 工具执行结果
        """
        try:
            buddha_emoji = "这是一个佛祖emoji：༼ つ ◕_◕ ༽つ"
            
            return {
                "name": self.name,
                "content": buddha_emoji
            }
        except Exception as e:
            logger.error(f"generate_buddha_emoji工具执行失败: {str(e)}")
            return {
                "name": self.name,
                "content": f"执行失败: {str(e)}"
            }

# 注册工具
register_tool(GenerateBuddhaEmojiTool)