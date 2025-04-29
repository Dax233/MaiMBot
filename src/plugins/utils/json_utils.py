import json
import logging
from typing import Any, Dict, TypeVar, List, Union, Tuple

# 定义类型变量用于泛型类型提示
T = TypeVar("T")

# 获取logger
logger = logging.getLogger("json_utils")


def safe_json_loads(json_str: str, default_value: T = None) -> Union[Any, T]:
    """
    安全地解析JSON字符串，出错时返回默认值

    参数:
        json_str: 要解析的JSON字符串
        default_value: 解析失败时返回的默认值

    返回:
        解析后的Python对象，或在解析失败时返回default_value
    """
    if not json_str:
        return default_value

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析失败: {e}, JSON字符串: {json_str[:100]}...")
        return default_value
    except Exception as e:
        logger.error(f"JSON解析过程中发生意外错误: {e}")
        return default_value


def extract_tool_call_arguments(tool_call: Dict[str, Any], default_value: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    从LLM工具调用对象中提取参数

    参数:
        tool_call: 工具调用对象字典
        default_value: 解析失败时返回的默认值

    返回:
        解析后的参数字典，或在解析失败时返回default_value
    """
    default_result = default_value or {}

    if not tool_call or not isinstance(tool_call, dict):
        logger.error(f"无效的工具调用对象: {tool_call}")
        return default_result

    try:
        # 提取function参数
        function_data = tool_call.get("function", {})
        if not function_data or not isinstance(function_data, dict):
            logger.error(f"工具调用缺少function字段或格式不正确: {tool_call}")
            return default_result

        # 提取arguments
        arguments_str = function_data.get("arguments", "{}")
        if not arguments_str:
            return default_result

        # 解析JSON
        return safe_json_loads(arguments_str, default_result)

    except Exception as e:
        logger.error(f"提取工具调用参数时出错: {e}")
        return default_result


def safe_json_dumps(obj: Any, default_value: str = "{}", ensure_ascii: bool = False, pretty: bool = False) -> str:
    """
    安全地将Python对象序列化为JSON字符串

    参数:
        obj: 要序列化的Python对象
        default_value: 序列化失败时返回的默认值
        ensure_ascii: 是否确保ASCII编码(默认False，允许中文等非ASCII字符)
        pretty: 是否美化输出JSON

    返回:
        序列化后的JSON字符串，或在序列化失败时返回default_value
    """
    try:
        indent = 2 if pretty else None
        return json.dumps(obj, ensure_ascii=ensure_ascii, indent=indent)
    except TypeError as e:
        logger.error(f"JSON序列化失败(类型错误): {e}")
        return default_value
    except Exception as e:
        logger.error(f"JSON序列化过程中发生意外错误: {e}")
        return default_value


def normalize_llm_response(response: Any, log_prefix: str = "") -> Tuple[bool, List[Any], str]:
    """
    标准化LLM响应格式，将各种格式（如元组）转换为统一的列表格式

    参数:
        response: 原始LLM响应
        log_prefix: 日志前缀

    返回:
        元组 (成功标志, 标准化后的响应列表, 错误消息)
    """

    logger.debug(f"{log_prefix}原始人 LLM响应: {response}")

    # 检查是否为None
    if response is None:
        return False, [], "LLM响应为None"

    # 记录原始类型
    logger.debug(f"{log_prefix}LLM响应原始类型: {type(response).__name__}")

    # 将元组转换为列表
    if isinstance(response, tuple):
        logger.debug(f"{log_prefix}将元组响应转换为列表")
        response = list(response)

    # 确保是列表类型
    if not isinstance(response, list):
        return False, [], f"无法处理的LLM响应类型: {type(response).__name__}"

    # 处理工具调用部分（如果存在）
    if len(response) == 3:
        content, reasoning, tool_calls = response

        # 将工具调用部分转换为列表（如果是元组）
        if isinstance(tool_calls, tuple):
            logger.debug(f"{log_prefix}将工具调用元组转换为列表")
            tool_calls = list(tool_calls)
            response[2] = tool_calls

    return True, response, ""


def process_llm_tool_calls(
    tool_calls: List[Dict[str, Any]], log_prefix: str = ""
) -> Tuple[bool, List[Dict[str, Any]], str]:
    """
    处理并验证LLM响应中的工具调用列表

    参数:
        tool_calls: 从LLM响应中直接获取的工具调用列表
        log_prefix: 日志前缀

    返回:
        元组 (成功标志, 验证后的工具调用列表, 错误消息)
    """

    # 如果列表为空，表示没有工具调用，这不是错误
    if not tool_calls:
        return True, [], "工具调用列表为空"

    # 验证每个工具调用的格式
    valid_tool_calls = []
    for i, tool_call in enumerate(tool_calls):
        if not isinstance(tool_call, dict):
            logger.warning(f"{log_prefix}工具调用[{i}]不是字典: {type(tool_call).__name__}, 内容: {tool_call}")
            continue

        # 检查基本结构
        if tool_call.get("type") != "function":
            logger.warning(
                f"{log_prefix}工具调用[{i}]不是function类型: type={tool_call.get('type', '未定义')}, 内容: {tool_call}"
            )
            continue

        if "function" not in tool_call or not isinstance(tool_call.get("function"), dict):
            logger.warning(f"{log_prefix}工具调用[{i}]缺少'function'字段或其类型不正确: {tool_call}")
            continue

        func_details = tool_call["function"]
        if "name" not in func_details or not isinstance(func_details.get("name"), str):
            logger.warning(f"{log_prefix}工具调用[{i}]的'function'字段缺少'name'或类型不正确: {func_details}")
            continue
        if "arguments" not in func_details or not isinstance(
            func_details.get("arguments"), str
        ):  # 参数是字符串形式的JSON
            logger.warning(f"{log_prefix}工具调用[{i}]的'function'字段缺少'arguments'或类型不正确: {func_details}")
            continue

        # 可选：尝试解析参数JSON，确保其有效
        args_str = func_details["arguments"]
        try:
            json.loads(args_str)  # 尝试解析，但不存储结果
        except json.JSONDecodeError as e:
            logger.warning(
                f"{log_prefix}工具调用[{i}]的'arguments'不是有效的JSON字符串: {e}, 内容: {args_str[:100]}..."
            )
            continue
        except Exception as e:
            logger.warning(f"{log_prefix}解析工具调用[{i}]的'arguments'时发生意外错误: {e}, 内容: {args_str[:100]}...")
            continue

        valid_tool_calls.append(tool_call)

    if not valid_tool_calls and tool_calls:  # 如果原始列表不为空，但验证后为空
        return False, [], "所有工具调用格式均无效"

    return True, valid_tool_calls, ""
