from src.do_tool.tool_can_use.base_tool import BaseTool, register_tool
from src.common.logger import get_module_logger
from typing import Dict, Any

logger = get_module_logger("generate_cmd_tutorial_tool")

class GenerateCmdTutorialTool(BaseTool):
    """生成Windows CMD基本操作教程的工具"""
    name = "generate_cmd_tutorial"
    description = "生成关于Windows命令提示符(CMD)的基本操作教程，包括常用命令和使用方法"
    parameters = {
        "type": "object",
        "properties": {},
        "required": []
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
            tutorial_content = """
# Windows CMD 基本操作教程

## 1. 基本导航命令
- `dir`: 列出当前目录下的文件和文件夹
- `cd <目录名>`: 进入指定目录
- `cd..`: 返回上一级目录
- `cd\\`: 返回根目录

## 2. 文件操作命令
- `copy <源文件> <目标位置>`: 复制文件
- `move <源文件> <目标位置>`: 移动文件
- `del <文件名>`: 删除文件
- `ren <旧文件名> <新文件名>`: 重命名文件

## 3. 系统信息命令
- `systeminfo`: 显示系统配置信息
- `hostname`: 显示计算机名称
- `ver`: 显示Windows版本

## 4. 网络相关命令
- `ipconfig`: 显示网络配置信息
- `ping <主机名或IP>`: 测试网络连接
- `tracert <主机名或IP>`: 跟踪网络路径

## 5. 实用技巧
- 按Tab键可以自动补全文件名或目录名
- 使用`> <文件名>`可以将命令输出重定向到文件
- 使用`| more`可以分页显示长输出

注意：使用命令时要小心，特别是删除操作。
"""
            
            return {
                "name": self.name,
                "content": tutorial_content
            }
        except Exception as e:
            logger.error(f"generate_cmd_tutorial工具执行失败: {str(e)}")
            return {
                "name": self.name,
                "content": f"执行失败: {str(e)}"
            }

# 注册工具
register_tool(GenerateCmdTutorialTool)