from src.do_tool.tool_can_use.base_tool import BaseTool, register_tool
from src.plugins.schedule.schedule_generator import bot_schedule
from src.common.logger import get_module_logger
from typing import Dict, Any

logger = get_module_logger("get_current_task_tool")


class GetCurrentTaskTool(BaseTool):
    """获取当前正在做的事情/最近的任务工具"""

    name = "get_current_task"
    description = "获取当前正在做的事情/最近的任务"
    parameters = {
        "type": "object",
        "properties": {
            "num": {"type": "integer", "description": "要获取的任务数量"},
            "time_info": {"type": "boolean", "description": "是否包含时间信息"},
        },
        "required": [],
    }

    async def execute(self, function_args: Dict[str, Any], message_txt: str = "") -> Dict[str, Any]:
        """执行获取当前任务

        Args:
            function_args: 工具参数
            message_txt: 原始消息文本，此工具不使用

        Returns:
            Dict: 工具执行结果
        """
        try:
            # 获取参数，如果没有提供则使用默认值
            num = function_args.get("num", 1)
            time_info = function_args.get("time_info", False)

            # 调用日程系统获取当前任务
            current_task = bot_schedule.get_current_num_task(num=num, time_info=time_info)

            # 格式化返回结果
            if current_task:
                task_info = current_task
            else:
                task_info = "当前没有正在进行的任务"

            return {"name": "get_current_task", "content": f"当前任务信息: {task_info}"}
        except Exception as e:
            logger.error(f"获取当前任务工具执行失败: {str(e)}")
            return {"name": "get_current_task", "content": f"获取当前任务失败: {str(e)}"}


# 注册工具
# register_tool(GetCurrentTaskTool)
