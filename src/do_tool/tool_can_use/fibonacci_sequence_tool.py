from src.do_tool.tool_can_use.base_tool import BaseTool, register_tool
from src.common.logger import get_module_logger
from typing import Dict, Any

logger = get_module_logger("fibonacci_sequence_tool")

class FibonacciSequenceTool(BaseTool):
    """生成斐波那契数列的工具"""
    name = "fibonacci_sequence"
    description = "生成指定长度的斐波那契数列"
    parameters = {
        "type": "object",
        "properties": {
            "n": {
                "type": "integer",
                "description": "斐波那契数列的长度",
                "minimum": 1
            }
        },
        "required": ["n"]
    }
    
    async def execute(self, function_args: Dict[str, Any], message_txt: str = "") -> Dict[str, Any]:
        """执行工具功能
        
        Args:
            function_args: 工具参数
            message_txt: 原始消息文本
            
        Returns:
            Dict: 工具执行结果
        """
        try:
            n = function_args.get("n")
            if n <= 0:
                raise ValueError("参数n必须大于0")
            
            sequence = []
            a, b = 0, 1
            for _ in range(n):
                sequence.append(a)
                a, b = b, a + b
            
            return {
                "name": self.name,
                "content": sequence
            }
        except Exception as e:
            logger.error(f"fibonacci_sequence工具执行失败: {str(e)}")
            return {
                "name": self.name,
                "content": f"执行失败: {str(e)}"
            }

# 注册工具
register_tool(FibonacciSequenceTool)