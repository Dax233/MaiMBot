from .observation import Observation
from src.plugins.models.utils_model import LLMRequest
from src.config.config import global_config
import time
import traceback
from src.common.logger import get_module_logger, LogConfig, SUB_HEARTFLOW_STYLE_CONFIG  # noqa: E402
from src.individuality.individuality import Individuality
import random
from ..plugins.utils.prompt_builder import Prompt, global_prompt_manager
from src.do_tool.tool_use import ToolUser
from src.plugins.utils.json_utils import safe_json_dumps, normalize_llm_response, process_llm_tool_calls
from src.heart_flow.chat_state_info import ChatStateInfo


subheartflow_config = LogConfig(
    console_format=SUB_HEARTFLOW_STYLE_CONFIG["console_format"],
    file_format=SUB_HEARTFLOW_STYLE_CONFIG["file_format"],
)
logger = get_module_logger("subheartflow", config=subheartflow_config)


def init_prompt():
    prompt = ""
    prompt += "{extra_info}\n"
    prompt += "{prompt_personality}\n"
    prompt += "刚刚你的内心想法是：{current_thinking_info}\n"
    prompt += "-----------------------------------\n"
    prompt += "现在是{time_now}，你正在上网，和qq群里的网友们聊天，以下是正在进行的聊天内容：\n{chat_observe_info}\n"
    prompt += "\n你现在{mood_info}\n"
    prompt += "现在请你生成你的内心想法，要求思考群里正在进行的话题，之前大家聊过的话题，群里成员的关系。"
    prompt += "请你思考，要不要对群里的话题进行回复，以及如何对群聊内容进行回复\n"
    prompt += "回复的要求是：不要总是重复自己提到过的话题，如果你要回复，最好只回复一个人的一个话题\n"
    prompt += "请注意不要输出多余内容(包括前后缀，冒号和引号，括号， 表情，等)，不要回复自己的发言\n"
    prompt += "现在请你先输出想法，{hf_do_next}，不要分点输出,文字不要浮夸"
    prompt += "在输出完想法后，请你思考应该使用什么工具。工具可以帮你取得一些你不知道的信息，或者进行一些操作。"
    prompt += "如果你需要做某件事，来对消息和你的回复进行处理，请使用工具。\n"

    Prompt(prompt, "sub_heartflow_prompt_before")


class SubMind:
    def __init__(self, subheartflow_id: str, chat_state: ChatStateInfo, observations: Observation):
        self.subheartflow_id = subheartflow_id

        self.llm_model = LLMRequest(
            model=global_config.llm_sub_heartflow,
            temperature=global_config.llm_sub_heartflow["temp"],
            max_tokens=800,
            request_type="sub_heart_flow",
        )

        self.chat_state = chat_state
        self.observations = observations

        self.current_mind = ""
        self.past_mind = []
        self.structured_info = {}

    async def do_thinking_before_reply(self):
        """
        在回复前进行思考，生成内心想法并收集工具调用结果

        返回:
            tuple: (current_mind, past_mind) 当前想法和过去的想法列表
        """
        # 更新活跃时间
        self.last_active_time = time.time()

        # ---------- 1. 准备基础数据 ----------
        # 获取现有想法和情绪状态
        current_thinking_info = self.current_mind
        mood_info = self.chat_state.mood

        # 获取观察对象
        observation = self.observations[0]
        if not observation:
            logger.error(f"[{self.subheartflow_id}] 无法获取观察对象")
            self.update_current_mind("(我没看到任何聊天内容...)")
            return self.current_mind, self.past_mind

        # 获取观察内容
        chat_observe_info = observation.get_observe_info()

        # ---------- 2. 准备工具和个性化数据 ----------
        # 初始化工具
        tool_instance = ToolUser()
        tools = tool_instance._define_tools()

        # 获取个性化信息
        individuality = Individuality.get_instance()

        # 构建个性部分
        prompt_personality = f"你的名字是{individuality.personality.bot_nickname}，你"
        prompt_personality += individuality.personality.personality_core

        # 随机添加个性侧面
        if individuality.personality.personality_sides:
            random_side = random.choice(individuality.personality.personality_sides)
            prompt_personality += f"，{random_side}"

        # 随机添加身份细节
        if individuality.identity.identity_detail:
            random_detail = random.choice(individuality.identity.identity_detail)
            prompt_personality += f"，{random_detail}"

        # 获取当前时间
        time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # ---------- 3. 构建思考指导部分 ----------
        # 创建本地随机数生成器，基于分钟数作为种子
        local_random = random.Random()
        current_minute = int(time.strftime("%M"))
        local_random.seed(current_minute)

        # 思考指导选项和权重
        hf_options = [
            ("继续生成你在这个聊天中的想法，在原来想法的基础上继续思考，但是不要纠结于同一个话题", 0.6),
            ("生成你在这个聊天中的想法，在原来的想法上尝试新的话题", 0.1),
            ("生成你在这个聊天中的想法，不要太深入", 0.2),
            ("继续生成你在这个聊天中的想法，进行深入思考", 0.1),
        ]

        # 加权随机选择思考指导
        hf_do_next = local_random.choices(
            [option[0] for option in hf_options], weights=[option[1] for option in hf_options], k=1
        )[0]

        # ---------- 4. 构建最终提示词 ----------
        # 获取提示词模板并填充数据
        prompt = (await global_prompt_manager.get_prompt_async("sub_heartflow_prompt_before")).format(
            extra_info="",  # 可以在这里添加额外信息
            prompt_personality=prompt_personality,
            bot_name=individuality.personality.bot_nickname,
            current_thinking_info=current_thinking_info,
            time_now=time_now,
            chat_observe_info=chat_observe_info,
            mood_info=mood_info,
            hf_do_next=hf_do_next,
        )

        logger.debug(f"[{self.subheartflow_id}] 心流思考提示词构建完成")

        # ---------- 5. 执行LLM请求并处理响应 ----------
        content = ""  # 初始化内容变量
        _reasoning_content = ""  # 初始化推理内容变量

        try:
            # 调用LLM生成响应
            response = await self.llm_model.generate_response_tool_async(prompt=prompt, tools=tools)

            # 标准化响应格式
            success, normalized_response, error_msg = normalize_llm_response(
                response, log_prefix=f"[{self.subheartflow_id}] "
            )

            if not success:
                # 处理标准化失败情况
                logger.warning(f"[{self.subheartflow_id}] {error_msg}")
                content = "LLM响应格式无法处理"
            else:
                # 从标准化响应中提取内容
                if len(normalized_response) >= 2:
                    content = normalized_response[0]
                    _reasoning_content = normalized_response[1] if len(normalized_response) > 1 else ""

                # 处理可能的工具调用
                if len(normalized_response) == 3:
                    # 提取并验证工具调用
                    success, valid_tool_calls, error_msg = process_llm_tool_calls(
                        normalized_response, log_prefix=f"[{self.subheartflow_id}] "
                    )

                    if success and valid_tool_calls:
                        # 记录工具调用信息
                        tool_calls_str = ", ".join(
                            [call.get("function", {}).get("name", "未知工具") for call in valid_tool_calls]
                        )
                        logger.info(
                            f"[{self.subheartflow_id}] 模型请求调用{len(valid_tool_calls)}个工具: {tool_calls_str}"
                        )

                        # 收集工具执行结果
                        await self._execute_tool_calls(valid_tool_calls, tool_instance)
                    elif not success:
                        logger.warning(f"[{self.subheartflow_id}] {error_msg}")
        except Exception as e:
            # 处理总体异常
            logger.error(f"[{self.subheartflow_id}] 执行LLM请求或处理响应时出错: {e}")
            logger.error(traceback.format_exc())
            content = "思考过程中出现错误"

        # 记录最终思考结果
        logger.debug(f"[{self.subheartflow_id}] 心流思考结果:\n{content}\n")

        # 处理空响应情况
        if not content:
            content = "(不知道该想些什么...)"
            logger.warning(f"[{self.subheartflow_id}] LLM返回空结果，思考失败。")

        # ---------- 6. 更新思考状态并返回结果 ----------
        # 更新当前思考内容
        self.update_current_mind(content)

        return self.current_mind, self.past_mind

    async def _execute_tool_calls(self, tool_calls, tool_instance):
        """
        执行一组工具调用并收集结果

        参数:
            tool_calls: 工具调用列表
            tool_instance: 工具使用器实例
        """
        tool_results = []
        structured_info = {}  # 动态生成键

        # 执行所有工具调用
        for tool_call in tool_calls:
            try:
                result = await tool_instance._execute_tool_call(tool_call)
                if result:
                    tool_results.append(result)

                    # 使用工具名称作为键
                    tool_name = result["name"]
                    if tool_name not in structured_info:
                        structured_info[tool_name] = []

                    structured_info[tool_name].append({"name": result["name"], "content": result["content"]})
            except Exception as tool_e:
                logger.error(f"[{self.subheartflow_id}] 工具执行失败: {tool_e}")

        # 如果有工具结果，记录并更新结构化信息
        if structured_info:
            logger.debug(f"工具调用收集到结构化信息: {safe_json_dumps(structured_info, ensure_ascii=False)}")
            self.structured_info = structured_info

    def update_current_mind(self, response):
        self.past_mind.append(self.current_mind)
        self.current_mind = response


init_prompt()
