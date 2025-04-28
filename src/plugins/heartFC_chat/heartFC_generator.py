from typing import List, Optional


from ..models.utils_model import LLMRequest
from ...config.config import global_config
from ..chat.message import MessageRecv
from .heartflow_prompt_builder import prompt_builder
from ..chat.utils import process_llm_response
from src.common.logger_manager import get_logger
from src.plugins.respon_info_catcher.info_catcher import info_catcher_manager
from ..utils.timer_calculator import Timer

from src.plugins.moods.moods import MoodManager


logger = get_logger("llm")


class HeartFCGenerator:
    def __init__(self):
        self.model_normal = LLMRequest(
            model=global_config.llm_normal,
            temperature=global_config.llm_normal["temp"],
            max_tokens=256,
            request_type="response_heartflow",
        )

        self.model_sum = LLMRequest(
            model=global_config.llm_summary_by_topic, temperature=0.6, max_tokens=2000, request_type="relation"
        )
        self.current_model_type = "r1"  # 默认使用 R1
        self.current_model_name = "unknown model"

    async def generate_response(
        self,
        structured_info: str,
        current_mind_info: str,
        reason: str,
        message: MessageRecv,
        thinking_id: str,
    ) -> Optional[List[str]]:
        """根据当前模型类型选择对应的生成函数"""

        arousal_multiplier = MoodManager.get_instance().get_arousal_multiplier()

        current_model = self.model_normal
        current_model.temperature = global_config.llm_normal["temp"] * arousal_multiplier  # 激活度越高，温度越高
        model_response = await self._generate_response_with_model(
            structured_info, current_mind_info, reason, message, current_model, thinking_id
        )

        if model_response:
            model_processed_response = await self._process_response(model_response)

            return model_processed_response
        else:
            logger.info(f"{self.current_model_type}思考，失败")
            return None

    async def _generate_response_with_model(
        self,
        structured_info: str,
        current_mind_info: str,
        reason: str,
        message: MessageRecv,
        model: LLMRequest,
        thinking_id: str,
    ) -> str:
        info_catcher = info_catcher_manager.get_info_catcher(thinking_id)

        with Timer() as _build_prompt:
            prompt = await prompt_builder.build_prompt(
                build_mode="focus",
                reason=reason,
                current_mind_info=current_mind_info,
                structured_info=structured_info,
                message_txt="",
                sender_name="",
                chat_stream=message.chat_stream,
            )
        # logger.info(f"构建prompt时间: {t_build_prompt.human_readable}")

        try:
            content, reasoning_content, self.current_model_name = await model.generate_response(prompt)

            logger.info(f"\nprompt:{prompt}\n生成回复{content}\n")

            info_catcher.catch_after_llm_generated(
                prompt=prompt, response=content, reasoning_content=reasoning_content, model_name=self.current_model_name
            )

        except Exception:
            logger.exception("生成回复时出错")
            return None

        return content

    async def _process_response(self, content: str) -> List[str]:
        """处理响应内容，返回处理后的内容和情感标签"""
        if not content:
            return None

        processed_response = process_llm_response(content)

        # print(f"得到了处理后的llm返回{processed_response}")

        return processed_response
