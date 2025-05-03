import asyncio
import json
import random # 添加 random 模块导入
import re
from datetime import datetime
from typing import Tuple, Union, Dict, Any, Set # 引入 Set

import aiohttp
from aiohttp.client import ClientResponse

from src.common.logger import get_module_logger
import base64
from PIL import Image
import io
import os
from ...common.database import db
from ...config.config import global_config
from rich.traceback import install

install(extra_lines=3)

logger = get_module_logger("model_utils")


class PayLoadTooLargeError(Exception):
    """自定义异常类，用于处理请求体过大错误"""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return "请求体过大，请尝试压缩图片或减少输入内容。"


class RequestAbortException(Exception):
    """自定义异常类，用于处理请求中断异常"""

    def __init__(self, message: str, response: ClientResponse):
        super().__init__(message)
        self.message = message
        self.response = response

    def __str__(self):
        return self.message


class PermissionDeniedException(Exception):
    """自定义异常类，用于处理访问拒绝的异常"""

    def __init__(self, message: str, key_identifier: str = None): # 添加 key 标识符
        super().__init__(message)
        self.message = message
        self.key_identifier = key_identifier # 存储导致 403 的 key

    def __str__(self):
        return self.message


# 新增：用于内部标记需要切换 Key 的异常
class _SwitchKeyException(Exception):
    """内部异常，用于标记需要切换Key并且跳过标准等待时间."""
    pass


# 常见Error Code Mapping
error_code_mapping = {
    400: "参数不正确",
    401: "API key 错误，认证失败，请检查/config/bot_config.toml和.env中的配置是否正确哦~", # 401 也可能是 Key 无效
    402: "账号余额不足",
    403: "需要实名,或余额不足,或Key无权限", # 扩展 403 的含义
    404: "Not Found",
    429: "请求过于频繁，请稍后再试",
    500: "服务器内部故障",
    503: "服务器负载过高",
}


async def _safely_record(request_content: Dict[str, Any], payload: Dict[str, Any]):
    image_base64: str = request_content.get("image_base64")
    image_format: str = request_content.get("image_format")
    if (
        image_base64
        and payload
        and isinstance(payload, dict)
        and "messages" in payload
        and len(payload["messages"]) > 0
    ):
        if isinstance(payload["messages"][0], dict) and "content" in payload["messages"][0]:
            content = payload["messages"][0]["content"]
            if isinstance(content, list) and len(content) > 1 and "image_url" in content[1]:
                payload["messages"][0]["content"][1]["image_url"]["url"] = (
                    f"data:image/{image_format.lower() if image_format else 'jpeg'};base64,"
                    f"{image_base64[:10]}...{image_base64[-10:]}"
                )
            # if isinstance(content, str) and len(content) > 100:
            #     payload["messages"][0]["content"] = content[:100]
    return payload


class LLMRequest:
    # 定义需要转换的模型列表，作为类变量避免重复
    MODELS_NEEDING_TRANSFORMATION = [
        "o1",
        "o1-2024-12-17",
        "o1-mini",
        "o1-mini-2024-09-12",
        "o1-preview",
        "o1-preview-2024-09-12",
        "o1-pro",
        "o1-pro-2025-03-19",
        "o3",
        "o3-2025-04-16",
        "o3-mini",
        "o3-mini-2025-01-31o4-mini",
        "o4-mini-2025-04-16",
    ]

    # 类变量，用于存储运行时发现的已失效 Key (避免在同一次运行中重复尝试)
    # 注意：这只在当前进程生命周期内有效
    _abandoned_keys_runtime: Set[str] = set()

    def __init__(self, model: dict, **kwargs):
        # 将大写的配置键转换为小写并从config中获取实际值
        self.model_key_name = model["key"] # 保存原始 key 名称，如 GEMINI_KEY
        try:
            # --- 修改开始: 处理 API Key 配置 (包括废弃列表) ---
            raw_api_key_config = os.environ[self.model_key_name]
            self.base_url = os.environ[model["base_url"]]

            # 1. 解析主 Key 列表/字符串
            parsed_keys = []
            is_list_config = False
            try:
                loaded_keys = json.loads(raw_api_key_config)
                if isinstance(loaded_keys, list):
                    parsed_keys = [str(key) for key in loaded_keys if key]
                    is_list_config = True
                elif isinstance(loaded_keys, str) and loaded_keys:
                    parsed_keys = [loaded_keys]
                else:
                    raise ValueError(f"Parsed API key for {self.model_key_name} is not a valid list or string.")
            except (json.JSONDecodeError, TypeError):
                if isinstance(raw_api_key_config, list): # 直接是列表对象
                    parsed_keys = [str(key) for key in raw_api_key_config if key]
                    is_list_config = True
                elif isinstance(raw_api_key_config, str) and raw_api_key_config: # 是非空字符串
                    parsed_keys = [raw_api_key_config]
                else:
                    raise ValueError(f"Invalid or empty API key config for {self.model_key_name}: {raw_api_key_config}")

            if not parsed_keys:
                raise ValueError(f"No valid API keys found for {self.model_key_name}.")

            # 2. 尝试加载并解析对应的废弃 Key 列表
            abandoned_key_name = f"abandon_{self.model_key_name}"
            abandoned_keys_set = set()
            raw_abandoned_keys = os.environ.get(abandoned_key_name) # 使用 get 避免 Key 不存在时出错

            if raw_abandoned_keys:
                try:
                    loaded_abandoned = json.loads(raw_abandoned_keys)
                    if isinstance(loaded_abandoned, list):
                        abandoned_keys_set.update(str(key) for key in loaded_abandoned if key)
                    elif isinstance(loaded_abandoned, str) and loaded_abandoned: # 也支持单个废弃 key 字符串
                        abandoned_keys_set.add(loaded_abandoned)
                    logger.info(f"模型 {model['name']}: 加载了 {len(abandoned_keys_set)} 个来自配置 '{abandoned_key_name}' 的废弃 Keys。")
                except (json.JSONDecodeError, TypeError):
                    if isinstance(raw_abandoned_keys, list): # 直接是列表
                        abandoned_keys_set.update(str(key) for key in raw_abandoned_keys if key)
                        logger.info(f"模型 {model['name']}: 加载了 {len(abandoned_keys_set)} 个来自配置 '{abandoned_key_name}' (直接列表) 的废弃 Keys。")
                    elif isinstance(raw_abandoned_keys, str) and raw_abandoned_keys: # 是非空字符串
                        abandoned_keys_set.add(raw_abandoned_keys)
                        logger.info(f"模型 {model['name']}: 加载了 1 个来自配置 '{abandoned_key_name}' (字符串) 的废弃 Key。")
                    else:
                        logger.warning(f"无法解析环境变量 '{abandoned_key_name}' 的内容: {raw_abandoned_keys}")

            # 3. 合并运行时废弃列表和配置废弃列表
            all_abandoned_keys = abandoned_keys_set.union(LLMRequest._abandoned_keys_runtime)

            # 4. 从解析的主 Key 列表中过滤掉所有废弃的 Key
            active_keys = [key for key in parsed_keys if key not in all_abandoned_keys]

            if not active_keys:
                logger.error(f"模型 {model['name']}: 所有为 '{self.model_key_name}' 配置的 Keys 都已被废弃或无效。")
                raise ValueError(f"No active API keys available for {self.model_key_name} after filtering abandoned keys.")

            # 5. 最终确定 self._api_key_config
            if is_list_config and len(active_keys) > 1:
                self._api_key_config = active_keys # 存储过滤后的列表
                logger.info(f"模型 {model['name']}: 初始化完成，可用 Keys: {len(self._api_key_config)} (已排除 {len(all_abandoned_keys)} 个废弃 Keys)。")
            elif active_keys: # 只有一个活动 key，或原始配置就是单个 key
                self._api_key_config = active_keys[0] # 存储为单个字符串
                logger.info(f"模型 {model['name']}: 初始化完成，使用单个活动 Key (已排除 {len(all_abandoned_keys)} 个废弃 Keys)。")
            else:
                # 这个分支理论上不会到达，因为前面有 active_keys 的检查
                raise ValueError(f"Unexpected state: No active keys for {self.model_key_name}.")

            # --- 修改结束: 处理 API Key 配置 ---

        except KeyError as e:
            # 处理找不到主 Key 或 Base URL 的情况
            missing_key = str(e).strip("'")
            if missing_key == self.model_key_name:
                logger.error(f"配置错误：找不到 API Key 环境变量 '{self.model_key_name}'")
                raise ValueError(f"配置错误：找不到 API Key 环境变量 '{self.model_key_name}'") from e
            elif missing_key == model["base_url"]:
                logger.error(f"配置错误：找不到 Base URL 环境变量 '{model['base_url']}'")
                raise ValueError(f"配置错误：找不到 Base URL 环境变量 '{model['base_url']}'") from e
            else:
                logger.error(f"配置错误：找不到环境变量 - {str(e)}")
                raise ValueError(f"配置错误：找不到环境变量 - {str(e)}") from e

        except AttributeError as e: # 保留原始的 AttributeError 处理
            logger.error(f"原始 model dict 信息：{model}")
            logger.error(f"配置错误：找不到对应的配置项 - {str(e)}")
            raise ValueError(f"配置错误：找不到对应的配置项 - {str(e)}") from e
        except ValueError as e: # 捕获上面抛出的 ValueError (包括 Key 无效或全部废弃)
            logger.error(f"API Key 或配置初始化错误 for {self.model_key_name}: {str(e)}")
            raise e # 重新抛出

        # --- 初始化其他属性 ---
        self.model_name: str = model["name"]
        self.params = kwargs
        self.stream = model.get("stream", False)
        self.pri_in = model.get("pri_in", 0)
        self.pri_out = model.get("pri_out", 0)
        self._init_database()
        self.request_type = kwargs.pop("request_type", "default")
        # --- 结束其他属性初始化 ---


    @staticmethod
    def _init_database():
        """初始化数据库集合"""
        try:
            # 创建llm_usage集合的索引
            db.llm_usage.create_index([("timestamp", 1)])
            db.llm_usage.create_index([("model_name", 1)])
            db.llm_usage.create_index([("user_id", 1)])
            db.llm_usage.create_index([("request_type", 1)])
        except Exception as e:
            logger.error(f"创建数据库索引失败: {str(e)}")

    def _record_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        user_id: str = "system",
        request_type: str = None,
        endpoint: str = "/chat/completions",
    ):
        """记录模型使用情况到数据库
        Args:
            prompt_tokens: 输入token数
            completion_tokens: 输出token数
            total_tokens: 总token数
            user_id: 用户ID，默认为system
            request_type: 请求类型(chat/embedding/image/topic/schedule)
            endpoint: API端点
        """
        # 如果 request_type 为 None，则使用实例变量中的值
        if request_type is None:
            request_type = self.request_type

        try:
            usage_data = {
                "model_name": self.model_name,
                "user_id": user_id,
                "request_type": request_type,
                "endpoint": endpoint,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost": self._calculate_cost(prompt_tokens, completion_tokens),
                "status": "success",
                "timestamp": datetime.now(),
            }
            db.llm_usage.insert_one(usage_data)
            logger.trace(
                f"Token使用情况 - 模型: {self.model_name}, "
                f"用户: {user_id}, 类型: {request_type}, "
                f"提示词: {prompt_tokens}, 完成: {completion_tokens}, "
                f"总计: {total_tokens}"
            )
        except Exception as e:
            logger.error(f"记录token使用情况失败: {str(e)}")

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """计算API调用成本
        使用模型的pri_in和pri_out价格计算输入和输出的成本

        Args:
            prompt_tokens: 输入token数量
            completion_tokens: 输出token数量

        Returns:
            float: 总成本（元）
        """
        # 使用模型的pri_in和pri_out计算成本
        input_cost = (prompt_tokens / 1000000) * self.pri_in
        output_cost = (completion_tokens / 1000000) * self.pri_out
        return round(input_cost + output_cost, 6)

    async def _prepare_request(
        self,
        endpoint: str,
        prompt: str = None,
        image_base64: str = None,
        image_format: str = None,
        payload: dict = None,
        retry_policy: dict = None,
    ) -> Dict[str, Any]:
        """配置请求参数
        Args:
            endpoint: API端点路径 (如 "chat/completions")
            prompt: prompt文本
            image_base64: 图片的base64编码
            image_format: 图片格式
            payload: 请求体数据
            retry_policy: 自定义重试策略
            request_type: 请求类型
        """

        # 合并重试策略
        default_retry = {
            "max_retries": 3,
            "base_wait": 10,
            "retry_codes": [429, 413, 500, 503], # 403 不再自动重试，除非切换 Key
            "abort_codes": [400, 401, 402, 403], # 403 仍然是中止代码，但会被特殊处理
        }
        policy = {**default_retry, **(retry_policy or {})}

        api_url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        stream_mode = self.stream

        # 构建请求体
        if image_base64:
            payload = await self._build_payload(prompt, image_base64, image_format)
        elif payload is None:
            payload = await self._build_payload(prompt)

        if stream_mode:
            payload["stream"] = stream_mode

        return {
            "policy": policy,
            "payload": payload,
            "api_url": api_url,
            "stream_mode": stream_mode,
            "image_base64": image_base64,  # 保留必要的exception处理所需的原始数据
            "image_format": image_format,
            "prompt": prompt,
        }

    # --- 修改开始: _execute_request (增加 403 处理逻辑) ---
    async def _execute_request(
        self,
        endpoint: str,
        prompt: str = None,
        image_base64: str = None,
        image_format: str = None,
        payload: dict = None,
        retry_policy: dict = None,
        response_handler: callable = None,
        user_id: str = "system",
        request_type: str = None,
    ):
        """统一请求执行入口, 支持列表 key 在 429/403 时自动切换"""
        request_content = await self._prepare_request(
            endpoint, prompt, image_base64, image_format, payload, retry_policy
        )
        policy = request_content["policy"]
        if request_type is None:
            request_type = self.request_type

        current_key = None
        keys_failed_429 = set() # 记录本次请求中因429失败的keys
        keys_abandoned_runtime = set() # 记录本次请求中因403失败的keys (用于本轮尝试)
        key_switch_limit_429 = 3    # 最多因 429 尝试不同的key数量
        key_switch_limit_403 = 3    # 最多因 403 尝试不同的key数量 (可以设为与 429 相同或不同)

        # 准备要尝试的 key 列表 (基于 __init__ 中已过滤的 self._api_key_config)
        available_keys_pool = [] # 用于从中选取 Key
        is_key_list = isinstance(self._api_key_config, list)

        if is_key_list:
            available_keys_pool = list(self._api_key_config) # 使用过滤后的列表
            if not available_keys_pool:
                # 这个错误理论上 __init__ 会捕获，但作为最后防线
                logger.error(f"模型 {self.model_name}: 初始化后无可用活动 Keys。")
                raise ValueError(f"模型 {self.model_name}: 无可用活动 Keys。")
            random.shuffle(available_keys_pool)
            key_switch_limit_429 = min(key_switch_limit_429, len(available_keys_pool))
            key_switch_limit_403 = min(key_switch_limit_403, len(available_keys_pool))
            logger.info(f"模型 {self.model_name}: Key 列表模式，启用 429/403 自动切换（429上限: {key_switch_limit_429}, 403上限: {key_switch_limit_403}）。")
        elif isinstance(self._api_key_config, str):
            available_keys_pool = [self._api_key_config] # 单个 Key
            key_switch_limit_429 = 1
            key_switch_limit_403 = 1
        else:
            # 不应该发生
            logger.error(f"模型 {self.model_name}: 无效的 API Key 配置类型在执行时遇到: {type(self._api_key_config)}")
            raise TypeError(f"模型 {self.model_name}: 无效的 API Key 配置类型")

        last_exception = None # 存储最后遇到的异常

        # 外层循环控制总体重试次数
        for attempt in range(policy["max_retries"]):
            # 选择当前要使用的 key
            if available_keys_pool:
                current_key = available_keys_pool.pop(0)
            elif current_key:
                logger.debug(f"模型 {self.model_name}: 无新 Key 可用或为单 Key 模式，将使用 Key ...{current_key[-4:]} 进行重试 (第 {attempt + 1} 次尝试)")
                pass # 保持 current_key 不变
            else:
                logger.error(f"模型 {self.model_name}: 无法选择 API key (第 {attempt + 1} 次尝试)")
                # 如果是因为所有 Key 都被 403 移除了，应该有个更明确的错误
                if not self._api_key_config or all(k in LLMRequest._abandoned_keys_runtime for k in self._api_key_config if isinstance(self._api_key_config, list)) or (isinstance(self._api_key_config, str) and self._api_key_config in LLMRequest._abandoned_keys_runtime):
                    final_error_msg = f"模型 {self.model_name}: 所有可用 API Keys 均因 403 错误被禁用。"
                    logger.critical(final_error_msg)
                    raise PermissionDeniedException(final_error_msg)
                else:
                    raise RuntimeError(f"模型 {self.model_name}: 无法选择 API key")

            logger.debug(f"模型 {self.model_name}: 尝试使用 Key: ...{current_key[-4:]} (总第 {attempt + 1} 次尝试)")

            try:
                # 使用当前选定的 Key 构建请求头
                headers = await self._build_headers(current_key)
                if request_content["stream_mode"]:
                    headers["Accept"] = "text/event-stream"

                # 发起请求
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        request_content["api_url"], headers=headers, json=request_content["payload"]
                    ) as response:

                        # --- 处理 429 (请求过多) ---
                        if response.status == 429 and is_key_list:
                            logger.warning(f"模型 {self.model_name}: Key ...{current_key[-4:]} 遇到 429 错误。")
                            if current_key not in keys_failed_429:
                                keys_failed_429.add(current_key)
                                logger.info(f"  (因 429 已失败 {len(keys_failed_429)}/{key_switch_limit_429} 个不同 Key)")
                                if available_keys_pool and len(keys_failed_429) < key_switch_limit_429:
                                    logger.info(f"  尝试因 429 切换到下一个可用 Key...")
                                    raise _SwitchKeyException() # 触发切换
                                else:
                                    logger.warning(f"  无更多 Key 可因 429 切换或已达上限。")
                            else:
                                logger.warning(f"  Key ...{current_key[-4:]} 再次遇到 429，按标准重试流程。")

                        # --- 处理 403 (权限拒绝) ---
                        elif response.status == 403 and is_key_list:
                            logger.error(f"模型 {self.model_name}: Key ...{current_key[-4:]} 遇到 403 (权限拒绝) 错误。")
                            if current_key not in keys_abandoned_runtime:
                                keys_abandoned_runtime.add(current_key)
                                # 同时添加到类级别的运行时废弃集合
                                LLMRequest._abandoned_keys_runtime.add(current_key)
                                logger.critical(f"  !! Key ...{current_key[-4:]} 已添加到运行时废弃列表。请考虑将其移至配置中的 'abandon_{self.model_key_name}' !!")

                                # 从当前可用池中移除 (虽然 pop 已经移除了，但以防万一)
                                if current_key in available_keys_pool: available_keys_pool.remove(current_key)

                                if available_keys_pool and len(keys_abandoned_runtime) < key_switch_limit_403:
                                    logger.info(f"  尝试因 403 切换到下一个可用 Key...")
                                    raise _SwitchKeyException() # 触发切换
                                else:
                                    logger.error(f"  无更多 Key 可因 403 切换或已达上限。将中止请求。")
                                    # 抛出 PermissionDeniedException 中止整个请求
                                    await response.read() # 确保读取响应体
                                    raise PermissionDeniedException(f"Key ...{current_key[-4:]} 权限被拒，且无其他可用 Key 切换。", key_identifier=current_key)
                            else:
                                # 这个 Key 在本轮已经被标记为 403 了，不应该再被选中，但以防万一
                                logger.error(f"  Key ...{current_key[-4:]} 再次遇到 403，这不应发生。中止请求。")
                                await response.read()
                                raise PermissionDeniedException(f"Key ...{current_key[-4:]} 重复遇到 403。", key_identifier=current_key)

                        # --- 处理其他 HTTP 错误 ---
                        elif response.status in policy["retry_codes"] or response.status in policy["abort_codes"]:
                            # 调用错误处理函数 (它现在知道 403 是特殊中止)
                            await self._handle_error_response(response, attempt, policy, current_key)
                            # 如果没抛异常，意味着是可等待重试的错误 (如 500, 503, 413)

                        # --- 判断是否需要标准等待重试 ---
                        if response.status in policy["retry_codes"] and attempt < policy["max_retries"] - 1:
                            # 只有非 429/403 的可重试错误才需要等待
                            if response.status not in [429, 403]:
                                wait_time = policy["base_wait"] * (2**attempt)
                                logger.warning(f"模型 {self.model_name}: 遇到可重试错误 {response.status}, 等待 {wait_time} 秒后重试...")
                                await asyncio.sleep(wait_time)
                            # 对于 429/403，如果没触发切换，也在这里进入下一次循环，但不等待
                            last_exception = RuntimeError(f"重试错误 {response.status}")
                            continue # 进入下一次循环

                        # --- 检查是否需要中止 ---
                        # 条件：状态码在中止列表 (包括 403)，或者虽可重试但已是最后一次尝试
                        if response.status in policy["abort_codes"] or (response.status in policy["retry_codes"] and attempt >= policy["max_retries"] - 1):
                            if attempt >= policy["max_retries"] - 1 and response.status in policy["retry_codes"]:
                                logger.error(f"模型 {self.model_name}: 达到最大重试次数，最后一次尝试仍为可重试错误 {response.status}。")

                            # 调用 _handle_error_response 来抛出正确的异常
                            # 它会处理 403 -> PermissionDeniedException, 其他 -> RequestAbortException
                            await self._handle_error_response(response, attempt, policy, current_key)
                            # 如果 _handle_error_response 没有按预期抛出异常（理论上不应发生）
                            await response.read()
                            final_error_msg = f"请求中止或达到最大重试次数，最终状态码: {response.status}"
                            logger.error(final_error_msg)
                            raise RequestAbortException(final_error_msg, response) # 通用中止


                        # --- 请求成功 ---
                        response.raise_for_status() # 确保没有遗漏的错误状态码
                        result = {}
                        if request_content["stream_mode"]:
                            result = await self._handle_stream_output(response)
                        else:
                            result = await response.json()

                        # 成功，处理结果并返回
                        return (
                            response_handler(result)
                            if response_handler
                            else self._default_response_handler(result, user_id, request_type, endpoint)
                        )

            except _SwitchKeyException:
                last_exception = _SwitchKeyException() # 记录切换事件
                logger.debug("捕获到 _SwitchKeyException，立即进行下一次尝试。")
                continue # 立即切换 Key，不等待

            except PermissionDeniedException as e:
                # 捕获由 403 触发的中止异常
                logger.error(f"模型 {self.model_name}: 因权限拒绝 (403) 中止请求: {e}")
                # 如果是列表 key 且是由于无 key 可切换导致的中止，记录一下
                if is_key_list and not available_keys_pool and e.key_identifier:
                    logger.critical(f"  中止原因是 Key ...{e.key_identifier[-4:]} 触发 403 后已无其他 Key 可用。")
                raise e # 直接向外抛出

            except (PayLoadTooLargeError, RequestAbortException) as e:
                # 捕获其他明确要中止的异常
                logger.error(f"模型 {self.model_name}: 请求处理中遇到关键错误，将中止: {e}")
                raise e
            except Exception as e:
                # 捕获其他所有异常
                last_exception = e
                logger.warning(f"模型 {self.model_name}: 第 {attempt + 1} 次尝试中发生非 HTTP 错误: {str(e.__class__.__name__)} - {str(e)}")

                if attempt >= policy["max_retries"] - 1:
                    logger.error(f"模型 {self.model_name}: 达到最大重试次数 ({policy['max_retries']})，因非 HTTP 错误失败。")
                    pass # 让循环结束，最后统一抛出
                else:
                    # 标准重试逻辑（带等待）
                    try:
                        handled_payload, count_delta = await self._handle_exception(e, attempt, request_content)
                        if handled_payload:
                            request_content["payload"] = handled_payload
                            logger.info(f"模型 {self.model_name}: 异常处理更新了 payload，将使用当前 Key 重试。")

                        wait_time = policy["base_wait"] * (2**attempt)
                        logger.warning(f"模型 {self.model_name}: 等待 {wait_time} 秒后重试...")
                        await asyncio.sleep(wait_time)
                        continue

                    except (RequestAbortException, PermissionDeniedException) as abort_exception:
                        logger.error(f"模型 {self.model_name}: 异常处理判断需要中止请求: {abort_exception}")
                        raise abort_exception
                    except RuntimeError as rt_error:
                        logger.error(f"模型 {self.model_name}: 异常处理遇到运行时错误: {rt_error}")
                        raise rt_error

        # --- 循环结束 ---
        logger.error(f"模型 {self.model_name}: 所有重试尝试 ({policy['max_retries']} 次) 均失败。")
        if last_exception:
            # 如果最后一次异常是 PermissionDenied，优先抛出它
            if isinstance(last_exception, PermissionDeniedException):
                logger.error(f"最后遇到的错误是权限拒绝: {str(last_exception)}")
                raise last_exception
            # 否则抛出通用错误
            logger.error(f"最后遇到的错误: {str(last_exception.__class__.__name__)} - {str(last_exception)}")
            raise RuntimeError(f"模型 {self.model_name} 达到最大重试次数，API 请求失败。最后错误: {str(last_exception)}") from last_exception
        else:
            # 如果是因为所有 Key 都被 403 禁用了
            if not available_keys_pool and keys_abandoned_runtime:
                final_error_msg = f"模型 {self.model_name}: 所有可用 API Keys 均因 403 错误被禁用。"
                logger.critical(final_error_msg)
                raise PermissionDeniedException(final_error_msg)
            else:
                # 其他未知原因
                raise RuntimeError(f"模型 {self.model_name} 达到最大重试次数，API 请求失败，原因未知。")
    # --- 修改结束: _execute_request ---


    async def _handle_stream_output(self, response: ClientResponse) -> Dict[str, Any]:
        # (流式输出处理逻辑保持不变，参考上一个版本)
        flag_delta_content_finished = False
        accumulated_content = ""
        usage = None  # 初始化usage变量，避免未定义错误
        reasoning_content = ""
        content = ""
        tool_calls = None  # 初始化工具调用变量

        async for line_bytes in response.content:
            try:
                line = line_bytes.decode("utf-8").strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        if flag_delta_content_finished:
                            chunk_usage = chunk.get("usage", None)
                            if chunk_usage:
                                usage = chunk_usage  # 获取token用量
                        else:
                            delta = chunk["choices"][0]["delta"]
                            delta_content = delta.get("content")
                            if delta_content is None:
                                delta_content = ""
                            accumulated_content += delta_content

                            # 提取工具调用信息
                            if "tool_calls" in delta:
                                if tool_calls is None:
                                    tool_calls = []
                                    for tc in delta["tool_calls"]:
                                        new_tc = dict(tc)
                                        if 'function' in new_tc and 'arguments' not in new_tc['function']:
                                            new_tc['function']['arguments'] = ""
                                        tool_calls.append(new_tc)
                                else:
                                    for i, tc_delta in enumerate(delta["tool_calls"]):
                                        if i < len(tool_calls) and 'function' in tc_delta and 'arguments' in tc_delta['function']:
                                            if 'arguments' in tool_calls[i]['function']:
                                                tool_calls[i]['function']['arguments'] += tc_delta['function']['arguments']
                                            else:
                                                tool_calls[i]['function']['arguments'] = tc_delta['function']['arguments']
                                        elif i >= len(tool_calls):
                                            new_tc = dict(tc_delta)
                                            if 'function' in new_tc and 'arguments' not in new_tc['function']:
                                                new_tc['function']['arguments'] = ""
                                            tool_calls.append(new_tc)


                            # 检测流式输出文本是否结束
                            finish_reason = chunk["choices"][0].get("finish_reason")
                            if delta.get("reasoning_content", None):
                                reasoning_content += delta["reasoning_content"]
                            if finish_reason == "stop" or finish_reason == "tool_calls":
                                chunk_usage = chunk.get("usage", None)
                                if chunk_usage:
                                    usage = chunk_usage
                                    break
                                flag_delta_content_finished = True
                    except json.JSONDecodeError as e:
                        logger.error(f"模型 {self.model_name} 解析流式 JSON 错误: {e} - data: '{data_str}'")
                    except Exception as e:
                        logger.exception(f"模型 {self.model_name} 解析流式输出块错误: {str(e)}")
            except UnicodeDecodeError as e:
                logger.warning(f"模型 {self.model_name} 流式输出解码错误: {e} - bytes: {line_bytes[:50]}...")
            except Exception as e:
                if isinstance(e, GeneratorExit):
                    log_content = f"模型 {self.model_name} 流式输出被中断，正在清理资源..."
                else:
                    log_content = f"模型 {self.model_name} 处理流式输出时发生错误: {str(e)}"
                logger.warning(log_content)
                try:
                    await response.release()
                except Exception as cleanup_error:
                    logger.error(f"清理资源时发生错误: {cleanup_error}")
                content = accumulated_content
                break
        if not content and accumulated_content:
            content = accumulated_content
        think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if think_match:
            reasoning_content = think_match.group(1).strip()
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        message = {
            "content": content,
            "reasoning_content": reasoning_content,
        }

        if tool_calls:
            message["tool_calls"] = tool_calls

        result = {
            "choices": [{"message": message}],
            "usage": usage,
        }
        return result

    # --- 修改开始: _handle_error_response (明确处理 403) ---
    async def _handle_error_response(
        self, response: ClientResponse, retry_count: int, policy: Dict[str, Any], current_key: str = None
    ) -> None: # 通过抛出异常控制流程
        """处理 HTTP 错误响应 (区分 403 和其他错误)"""
        status = response.status
        try:
            error_text = await response.text()
        except Exception as e:
            error_text = f"(无法读取响应体: {e})"

        # --- 特殊处理 403 ---
        if status == 403:
            logger.error(
                f"模型 {self.model_name}: 遇到 403 (权限拒绝) 错误。Key: ...{current_key[-4:] if current_key else 'N/A'}. "
                f"响应: {error_text[:200]}"
            )
            # 直接抛出 PermissionDeniedException，让 _execute_request 捕获并处理切换或中止
            raise PermissionDeniedException(f"模型禁止访问 ({status})", key_identifier=current_key)

        # --- 处理其他可重试错误 (非 403, 非 429) ---
        elif status in policy["retry_codes"] and status != 429:
            if status == 413:
                logger.warning(f"模型 {self.model_name}: 错误码 413 (Payload Too Large)。Key: ...{current_key[-4:] if current_key else 'N/A'}. 尝试压缩...")
                raise PayLoadTooLargeError("请求体过大") # 抛给 _handle_exception 处理压缩
            elif status in [500, 503]:
                logger.error(
                    f"模型 {self.model_name}: 服务器内部错误或过载 ({status})。Key: ...{current_key[-4:] if current_key else 'N/A'}. "
                    f"响应: {error_text[:200]}"
                )
                return # 不抛异常，让主循环等待重试
            else:
                logger.warning(f"模型 {self.model_name}: 遇到可重试错误码: {status}. Key: ...{current_key[-4:] if current_key else 'N/A'}")
                return # 不抛异常，让主循环等待重试

        # --- 处理其他需要中止的错误 (非 403) ---
        elif status in policy["abort_codes"]: # 注意 403 已被上面处理
            logger.error(
                f"模型 {self.model_name}: 遇到需要中止的错误码: {status} - {error_code_mapping.get(status, '未知错误')}. "
                f"Key: ...{current_key[-4:] if current_key else 'N/A'}. 响应: {error_text[:200]}"
            )
            # 抛出通用中止异常
            raise RequestAbortException(f"请求出现错误 {status}，中止处理", response)
        else:
            # 处理未在策略中定义的意外错误码
            logger.error(f"模型 {self.model_name}: 遇到未明确处理的错误码: {status}. Key: ...{current_key[-4:] if current_key else 'N/A'}. 响应: {error_text[:200]}")
            try:
                response.raise_for_status()
                # 如果没抛异常，也强制中止
                raise RequestAbortException(f"未处理的错误状态码 {status}", response)
            except aiohttp.ClientResponseError as e:
                raise RequestAbortException(f"未处理的错误状态码 {status}: {e.message}", response) from e
    # --- 修改结束: _handle_error_response ---


    async def _handle_exception(
        self, exception, retry_count: int, request_content: Dict[str, Any]
    ) -> Union[Tuple[Dict[str, Any], int], Tuple[None, int]]:
        # (这个方法基本保持不变，主要处理 PayLoadTooLarge 和网络错误)
        policy = request_content["policy"]
        payload = request_content["payload"]
        wait_time = policy["base_wait"] * (2**retry_count)
        keep_request = False
        if retry_count < policy["max_retries"] - 1:
            keep_request = True

        if isinstance(exception, PayLoadTooLargeError):
            if keep_request:
                logger.warning("请求体过大 (PayLoadTooLargeError)，尝试压缩图片...")
                image_base64 = request_content.get("image_base64")
                if image_base64:
                    compressed_image_base64 = compress_base64_image_by_scale(image_base64)
                    if compressed_image_base64 != image_base64:
                        new_payload = await self._build_payload(
                            request_content["prompt"], compressed_image_base64, request_content["image_format"]
                        )
                        logger.info("图片压缩成功，将使用压缩后的图片重试。")
                        return new_payload, 0
                    else:
                        logger.warning("图片压缩未改变大小或失败。")
                else:
                    logger.warning("请求体过大但请求中不包含图片，无法压缩。")
                return None, 0 # 返回 None 表示无需修改 payload，让外层等待
            else:
                logger.error("达到最大重试次数，请求体仍然过大。")
                raise RuntimeError("请求体过大，压缩或重试后仍然失败。") from exception

        elif isinstance(exception, aiohttp.ClientError) or isinstance(exception, asyncio.TimeoutError):
            if keep_request:
                logger.error(f"模型 {self.model_name} 网络错误: {str(exception)}")
                return None, 0 # 返回 None，让外层等待
            else:
                logger.critical(f"模型 {self.model_name} 网络错误达到最大重试次数: {str(exception)}")
                raise RuntimeError(f"网络请求失败: {str(exception)}") from exception

        elif isinstance(exception, aiohttp.ClientResponseError):
            if keep_request:
                logger.error(
                    f"模型 {self.model_name} HTTP响应错误 (未被策略覆盖): 状态码: {exception.status}, 错误: {exception.message}"
                )
                try:
                    error_text = await exception.response.text() if hasattr(exception, 'response') else str(exception)
                    logger.error(f"服务器错误响应详情: {error_text[:500]}")
                except Exception as parse_err:
                    logger.warning(f"无法解析服务器错误响应内容: {str(parse_err)}")
                return None, 0 # 返回 None，让外层等待
            else:
                logger.critical(
                    f"模型 {self.model_name} HTTP响应错误达到最大重试次数: 状态码: {exception.status}, 错误: {exception.message}"
                )
                # 尝试安全记录，隐藏 key
                current_key_placeholder = request_content.get("current_key", "******") # 假设 current_key 被传递
                handled_payload = await _safely_record(request_content, payload)
                logger.critical(f"请求头: {await self._build_headers(api_key=current_key_placeholder, no_key=True)} 请求体: {handled_payload}")
                raise RuntimeError(
                    f"模型 {self.model_name} API请求失败: 状态码 {exception.status}, {exception.message}"
                ) from exception

        else:
            # 处理其他所有未预料的异常
            if keep_request:
                logger.error(f"模型 {self.model_name} 遇到未知错误: {str(exception.__class__.__name__)} - {str(exception)}")
                return None, 0 # 返回 None，让外层等待
            else:
                logger.critical(f"模型 {self.model_name} 请求因未知错误失败: {str(exception.__class__.__name__)} - {str(exception)}")
                current_key_placeholder = request_content.get("current_key", "******")
                handled_payload = await _safely_record(request_content, payload)
                logger.critical(f"请求头: {await self._build_headers(api_key=current_key_placeholder, no_key=True)} 请求体: {handled_payload}")
                raise RuntimeError(f"模型 {self.model_name} API请求失败: {str(exception)}") from exception


    async def _transform_parameters(self, params: dict) -> dict:
        """根据模型名称转换参数"""
        new_params = dict(params)
        if self.model_name.lower() in self.MODELS_NEEDING_TRANSFORMATION:
            new_params.pop("temperature", None)
            if "max_tokens" in new_params:
                new_params["max_completion_tokens"] = new_params.pop("max_tokens")
        return new_params

    async def _build_payload(self, prompt: str, image_base64: str = None, image_format: str = None) -> dict:
        """构建请求体"""
        params_copy = await self._transform_parameters(self.params)
        if image_base64:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{image_format.lower() if image_format else 'jpeg'};base64,{image_base64}"},
                        },
                    ],
                }
            ]
        else:
            messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": self.model_name,
            "messages": messages,
            **params_copy,
        }
        if "max_tokens" not in payload and "max_completion_tokens" not in payload:
            if self.model_name.lower() in self.MODELS_NEEDING_TRANSFORMATION:
                payload["max_completion_tokens"] = global_config.model_max_output_length
            else:
                payload["max_tokens"] = global_config.model_max_output_length
        if self.model_name.lower() in self.MODELS_NEEDING_TRANSFORMATION and "max_tokens" in payload:
            payload["max_completion_tokens"] = payload.pop("max_tokens")
        return payload

    def _default_response_handler(
        self, result: dict, user_id: str = "system", request_type: str = None, endpoint: str = "/chat/completions"
    ) -> Tuple:
        """默认响应解析"""
        # (保持不变)
        content = "没有返回结果"
        reasoning_content = ""
        tool_calls = None

        if "choices" in result and result["choices"]:
            message = result["choices"][0].get("message", {})
            raw_content = message.get("content", "")
            content, reasoning = self._extract_reasoning(raw_content if raw_content else "")

            explicit_reasoning = message.get("model_extra", {}).get("reasoning_content", "")
            if not explicit_reasoning:
                explicit_reasoning = message.get("reasoning_content", "")
            reasoning_content = explicit_reasoning if explicit_reasoning else reasoning

            tool_calls = message.get("tool_calls", None)

            usage = result.get("usage", {})
            if usage:
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
                self._record_usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    user_id=user_id,
                    request_type=request_type if request_type is not None else self.request_type,
                    endpoint=endpoint,
                )
            else:
                logger.warning(f"模型 {self.model_name} 的响应中缺少 'usage' 信息。")

            if tool_calls:
                logger.debug(f"检测到工具调用: {tool_calls}")
                return content, reasoning_content, tool_calls
            else:
                return content, reasoning_content
        else:
            logger.warning(f"模型 {self.model_name} 的响应格式不符合预期: {result}")
            return content, reasoning_content


    @staticmethod
    def _extract_reasoning(content: str) -> Tuple[str, str]:
        """CoT思维链提取"""
        # (保持不变)
        if not content:
            return "", ""
        match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        cleaned_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL, count=1).strip()
        if match:
            reasoning = match.group(1).strip()
        else:
            reasoning = ""
        return cleaned_content, reasoning

    # --- 修改: _build_headers (保持接收 api_key) ---
    async def _build_headers(self, api_key: str, no_key: bool = False) -> dict:
        """构建请求头, 使用指定的 API Key"""
        if no_key:
            return {"Authorization": "Bearer **********", "Content-Type": "application/json"}
        else:
            if not api_key:
                logger.error(f"尝试使用无效 (空) 的 API key 为模型 {self.model_name} 构建请求头。")
                raise ValueError(f"无效的 API key 提供给 _build_headers。")
            return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


    async def generate_response(self, prompt: str, user_id: str = "system") -> Tuple:
        """根据输入的提示生成模型的异步响应"""
        # (保持不变, 调用 _execute_request)
        response = await self._execute_request(endpoint="/chat/completions", prompt=prompt, user_id=user_id, request_type="chat")
        if len(response) == 3:
            content, reasoning_content, tool_calls = response
            return content, reasoning_content, self.model_name, tool_calls
        else:
            content, reasoning_content = response
            return content, reasoning_content, self.model_name

    async def generate_response_for_image(self, prompt: str, image_base64: str, image_format: str, user_id: str = "system") -> Tuple:
        """根据输入的提示和图片生成模型的异步响应"""
        # (保持不变, 调用 _execute_request)
        response = await self._execute_request(
            endpoint="/chat/completions", prompt=prompt, image_base64=image_base64, image_format=image_format, user_id=user_id, request_type="vision"
        )
        if len(response) == 3:
            content, reasoning_content, tool_calls = response
            return content, reasoning_content, tool_calls
        else:
            content, reasoning_content = response
            return content, reasoning_content

    async def generate_response_async(self, prompt: str, user_id: str = "system", request_type: str = "chat", **kwargs) -> Union[str, Tuple]:
        """异步方式根据输入的提示生成模型的响应 (通用)"""
        # (保持不变, 调用 _execute_request)
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            **self.params,
            **kwargs,
        }
        if "max_tokens" not in data and "max_completion_tokens" not in data:
            if self.model_name.lower() in self.MODELS_NEEDING_TRANSFORMATION:
                data["max_completion_tokens"] = global_config.model_max_output_length
            else:
                data["max_tokens"] = global_config.model_max_output_length
        response = await self._execute_request(endpoint="/chat/completions", payload=data, prompt=prompt, user_id=user_id, request_type=request_type)
        return response

    async def generate_response_tool_async(self, prompt: str, tools: list, user_id: str = "system", **kwargs) -> tuple[str, str, list]:
        """异步方式根据输入的提示和工具生成模型的响应"""
        # (保持不变, 调用 _execute_request)
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            **self.params,
            **kwargs,
            "tools": tools,
            "tool_choice": "auto",
        }
        if "max_tokens" not in data and "max_completion_tokens" not in data:
            if self.model_name.lower() in self.MODELS_NEEDING_TRANSFORMATION:
                data["max_completion_tokens"] = global_config.model_max_output_length
            else:
                data["max_tokens"] = global_config.model_max_output_length
        response = await self._execute_request(endpoint="/chat/completions", payload=data, prompt=prompt, user_id=user_id, request_type="tool_call")
        logger.debug(f"向模型 {self.model_name} 发送工具调用请求，包含 {len(tools)} 个工具，返回结果: {response}")
        if isinstance(response, tuple) and len(response) == 3:
            content, reasoning_content, tool_calls = response
            if tool_calls:
                logger.debug(f"收到工具调用响应，包含 {len(tool_calls)} 个工具调用")
                return content, reasoning_content, tool_calls
            else:
                logger.debug("收到响应结构但无实际工具调用，视为普通响应")
                return content, reasoning_content, None
        elif isinstance(response, tuple) and len(response) == 2:
            content, reasoning_content = response
            logger.debug("收到普通响应，无工具调用")
            return content, reasoning_content, None
        else:
            logger.error(f"收到来自 _execute_request 的意外响应格式: {response}")
            return "处理响应时出错", "", None


    async def get_embedding(self, text: str, user_id: str = "system") -> Union[list, None]:
        """异步方法：获取文本的embedding向量"""
        # (保持不变, 调用 _execute_request)
        if len(text) < 1:
            logger.debug("该消息没有长度，不再发送获取embedding向量的请求")
            return None

        def embedding_handler(result):
            embedding_value = None
            if "data" in result and len(result["data"]) > 0:
                embedding_value = result["data"][0].get("embedding", None)
            usage = result.get("usage", {})
            if usage:
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
                self._record_usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    user_id=user_id,
                    request_type="embedding",
                    endpoint="/embeddings",
                )
            else:
                logger.warning(f"模型 {self.model_name} (Embedding) 的响应中缺少 'usage' 信息。")
            return embedding_value

        embedding = await self._execute_request(
            endpoint="/embeddings",
            prompt=text,
            payload={"model": self.model_name, "input": text, "encoding_format": "float"},
            retry_policy={"max_retries": 2, "base_wait": 6},
            response_handler=embedding_handler,
            user_id=user_id,
            request_type="embedding"
        )
        return embedding


def compress_base64_image_by_scale(base64_data: str, target_size: int = 0.8 * 1024 * 1024) -> str:
    """压缩base64格式的图片到指定大小"""
    # (保持不变)
    try:
        image_data = base64.b64decode(base64_data)
        if len(image_data) <= target_size * 1.05:
            logger.info(f"图片大小 {len(image_data) / 1024:.1f}KB 已足够小，无需压缩。")
            return base64_data
        img = Image.open(io.BytesIO(image_data))
        img_format = img.format
        original_width, original_height = img.size
        scale = max(0.2, min(1.0, (target_size / len(image_data)) ** 0.5))
        new_width = max(1, int(original_width * scale))
        new_height = max(1, int(original_height * scale))
        output_buffer = io.BytesIO()
        save_format = img_format # Default to original format

        if getattr(img, "is_animated", False) and img.n_frames > 1:
            frames = []
            durations = []
            loop = img.info.get('loop', 0)
            disposal = img.info.get('disposal', 2)
            logger.info(f"检测到 GIF 动图 ({img.n_frames} 帧)，尝试按比例压缩...")
            for frame_idx in range(img.n_frames):
                img.seek(frame_idx)
                current_duration = img.info.get('duration', 100)
                durations.append(current_duration)
                new_frame = img.convert("RGBA").copy()
                resized_frame = new_frame.resize((new_width, new_height), Image.Resampling.LANCZOS)
                frames.append(resized_frame)
            if frames:
                frames[0].save(
                    output_buffer, format="GIF", save_all=True, append_images=frames[1:],
                    optimize=False, duration=durations, loop=loop, disposal=disposal,
                    transparency=img.info.get('transparency', None), background=img.info.get('background', None)
                )
                save_format = "GIF"
            else:
                logger.warning("未能处理 GIF 帧。")
                return base64_data
        else:
            if img.mode in ("RGBA", "LA") or 'transparency' in img.info:
                resized_img = img.convert("RGBA").resize((new_width, new_height), Image.Resampling.LANCZOS)
                save_format = "PNG"
                save_params = {"optimize": True}
            else:
                resized_img = img.convert("RGB").resize((new_width, new_height), Image.Resampling.LANCZOS)
                if img_format and img_format.upper() == "JPEG":
                    save_format = "JPEG"
                    save_params = {"quality": 85, "optimize": True}
                else:
                    save_format = "PNG"
                    save_params = {"optimize": True}
            resized_img.save(output_buffer, format=save_format, **save_params)

        compressed_data = output_buffer.getvalue()
        logger.success(f"压缩图片: {original_width}x{original_height} -> {new_width}x{new_height} ({img.format} -> {save_format})")
        logger.info(f"压缩前大小: {len(image_data) / 1024:.1f}KB, 压缩后大小: {len(compressed_data) / 1024:.1f}KB (目标: {target_size / 1024:.1f}KB)")
        if len(compressed_data) < len(image_data) * 0.95:
            return base64.b64encode(compressed_data).decode("utf-8")
        else:
            logger.info("压缩效果不明显或反而增大，返回原始图片。")
            return base64_data
    except Exception as e:
        logger.error(f"压缩图片失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return base64_data