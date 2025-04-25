import json
import logging
from typing import Any, Dict, TypeVar, List, Union, Callable, Tuple

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


def get_json_value(
    json_obj: Dict[str, Any], key_path: str, default_value: T = None, transform_func: Callable[[Any], T] = None
) -> Union[Any, T]:
    """
    从JSON对象中按照路径提取值，支持点表示法路径，如"data.items.0.name"

    参数:
        json_obj: JSON对象(已解析的字典)
        key_path: 键路径，使用点表示法，如"data.items.0.name"
        default_value: 获取失败时返回的默认值
        transform_func: 可选的转换函数，用于对获取的值进行转换

    返回:
        路径指向的值，或在获取失败时返回default_value
    """
    if not json_obj or not key_path:
        return default_value

    try:
        # 分割路径
        keys = key_path.split(".")
        current = json_obj

        # 遍历路径
        for key in keys:
            # 处理数组索引
            if key.isdigit() and isinstance(current, list):
                index = int(key)
                if 0 <= index < len(current):
                    current = current[index]
                else:
                    return default_value
            # 处理字典键
            elif isinstance(current, dict):
                if key in current:
                    current = current[key]
                else:
                    return default_value
            else:
                return default_value

        # 应用转换函数(如果提供)
        if transform_func and current is not None:
            return transform_func(current)
        return current
    except Exception as e:
        logger.error(f"从JSON获取值时出错: {e}, 路径: {key_path}")
        return default_value


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


def merge_json_objects(*objects: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并多个JSON对象(字典)

    参数:
        *objects: 要合并的JSON对象(字典)

    返回:
        合并后的字典，后面的对象会覆盖前面对象的相同键
    """
    result = {}
    for obj in objects:
        if obj and isinstance(obj, dict):
            result.update(obj)
    return result


def normalize_llm_response(response: Any, log_prefix: str = "") -> Tuple[bool, List[Any], str]:
    """
    标准化LLM响应格式，将各种格式（如元组）转换为统一的列表格式

    参数:
        response: 原始LLM响应
        log_prefix: 日志前缀

    返回:
        元组 (成功标志, 标准化后的响应列表, 错误消息)
    """
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


def process_llm_tool_calls(response: List[Any], log_prefix: str = "") -> Tuple[bool, List[Dict[str, Any]], str]:
    """
    处理并提取LLM响应中的工具调用列表

    参数:
        response: 标准化后的LLM响应列表
        log_prefix: 日志前缀

    返回:
        元组 (成功标志, 工具调用列表, 错误消息)
    """
    # 确保响应格式正确
    if len(response) != 3:
        return False, [], f"LLM响应元素数量不正确: 预期3个元素，实际{len(response)}个"

    # 提取工具调用部分
    tool_calls = response[2]

    # 检查工具调用是否有效
    if tool_calls is None:
        return False, [], "工具调用部分为None"

    if not isinstance(tool_calls, list):
        return False, [], f"工具调用部分不是列表: {type(tool_calls).__name__}"

    if len(tool_calls) == 0:
        return False, [], "工具调用列表为空"

    # 检查工具调用是否格式正确
    valid_tool_calls = []
    for i, tool_call in enumerate(tool_calls):
        if not isinstance(tool_call, dict):
            logger.warning(f"{log_prefix}工具调用[{i}]不是字典: {type(tool_call).__name__}")
            continue

        if tool_call.get("type") != "function":
            logger.warning(f"{log_prefix}工具调用[{i}]不是函数类型: {tool_call.get('type', '未知')}")
            continue

        if "function" not in tool_call or not isinstance(tool_call["function"], dict):
            logger.warning(f"{log_prefix}工具调用[{i}]缺少function字段或格式不正确")
            continue

        valid_tool_calls.append(tool_call)

    # 检查是否有有效的工具调用
    if not valid_tool_calls:
        return False, [], "没有找到有效的工具调用"

    return True, valid_tool_calls, ""


def process_llm_tool_response(
    response: Any, expected_tool_name: str = None, log_prefix: str = ""
) -> Tuple[bool, Dict[str, Any], str]:
    """
    处理LLM返回的工具调用响应，进行常见错误检查并提取参数

    参数:
        response: LLM的响应，预期是[content, reasoning, tool_calls]格式的列表或元组
        expected_tool_name: 预期的工具名称，如不指定则不检查
        log_prefix: 日志前缀，用于标识日志来源

    返回:
        三元组(成功标志, 参数字典, 错误描述)
        - 如果成功解析，返回(True, 参数字典, "")
        - 如果解析失败，返回(False, {}, 错误描述)
    """
    # 使用新的标准化函数
    success, normalized_response, error_msg = normalize_llm_response(response, log_prefix)
    if not success:
        return False, {}, error_msg

    # 使用新的工具调用处理函数
    success, valid_tool_calls, error_msg = process_llm_tool_calls(normalized_response, log_prefix)
    if not success:
        return False, {}, error_msg

    # 检查是否有工具调用
    if not valid_tool_calls:
        return False, {}, "没有有效的工具调用"

    # 获取第一个工具调用
    tool_call = valid_tool_calls[0]

    # 检查工具名称(如果提供了预期名称)
    if expected_tool_name:
        actual_name = tool_call.get("function", {}).get("name")
        if actual_name != expected_tool_name:
            return False, {}, f"工具名称不匹配: 预期'{expected_tool_name}'，实际'{actual_name}'"

    # 提取并解析参数
    try:
        arguments = extract_tool_call_arguments(tool_call, {})
        return True, arguments, ""
    except Exception as e:
        logger.error(f"{log_prefix}解析工具参数时出错: {e}")
        return False, {}, f"解析参数失败: {str(e)}"
