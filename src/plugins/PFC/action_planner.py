from typing import Tuple
from src.common.logger import get_module_logger
from ..models.utils_model import LLM_request
from ..config.config import global_config
from .chat_observer import ChatObserver
from .pfc_utils import get_items_from_json
from src.individuality.individuality import Individuality
from .observation_info import ObservationInfo
from .conversation_info import ConversationInfo

logger = get_module_logger("action_planner")


class ActionPlannerInfo:
    def __init__(self):
        self.done_action = []
        self.goal_list = []
        self.knowledge_list = []
        self.memory_list = []


class ActionPlanner:
    """行动规划器"""

    def __init__(self, stream_id: str):
        self.llm = LLM_request(
            model=global_config.llm_normal,
            temperature=global_config.llm_normal["temp"],
            max_tokens=1000,
            request_type="action_planning",
        )
        self.personality_info = Individuality.get_instance().get_prompt(type="personality", x_person=2, level=2)
        self.name = global_config.BOT_NICKNAME
        self.chat_observer = ChatObserver.get_instance(stream_id)

    async def plan(self, observation_info: ObservationInfo, conversation_info: ConversationInfo) -> Tuple[str, str]:
        """规划下一步行动

        Args:
            observation_info: 决策信息
            conversation_info: 对话信息

        Returns:
            Tuple[str, str]: (行动类型, 行动原因)
        """
        # 构建提示词
        logger.debug(f"开始规划行动：当前目标: {conversation_info.goal_list}")

        # 构建对话目标
        goals_str = ""
        if conversation_info.goal_list:
            for goal_reason in conversation_info.goal_list:
                # 处理字典或元组格式
                if isinstance(goal_reason, tuple):
                    # 假设元组的第一个元素是目标，第二个元素是原因
                    goal = goal_reason[0]
                    reasoning = goal_reason[1] if len(goal_reason) > 1 else "没有明确原因"
                elif isinstance(goal_reason, dict):
                    goal = goal_reason.get("goal")
                    reasoning = goal_reason.get("reasoning", "没有明确原因")
                else:
                    # 如果是其他类型，尝试转为字符串
                    goal = str(goal_reason)
                    reasoning = "没有明确原因"

                goal_str = f"目标：{goal}，产生该对话目标的原因：{reasoning}\n"
                goals_str += goal_str
        else:
            goal = "目前没有明确对话目标"
            reasoning = "目前没有明确对话目标，最好思考一个对话目标"
            goals_str = f"目标：{goal}，产生该对话目标的原因：{reasoning}\n"

        # 获取聊天历史记录
        chat_history_list = (
            observation_info.chat_history[-20:]
            if len(observation_info.chat_history) >= 20
            else observation_info.chat_history
        )
        chat_history_text = ""
        for msg in chat_history_list:
            chat_history_text += f"{msg.get('detailed_plain_text', '')}\n"

        if observation_info.new_messages_count > 0:
            new_messages_list = observation_info.unprocessed_messages

            chat_history_text += f"有{observation_info.new_messages_count}条新消息：\n"
            for msg in new_messages_list:
                chat_history_text += f"{msg.get('detailed_plain_text', '')}\n"

            observation_info.clear_unprocessed_messages()

        personality_text = f"你的名字是{self.name}，{self.personality_info}"

        # 构建action历史文本
        action_history_list = (
            conversation_info.done_action[-10:]
            if len(conversation_info.done_action) >= 10
            else conversation_info.done_action
        )
        action_history_text = "你之前做的事情是："
        for action in action_history_list:
            if isinstance(action, dict):
                action_type = action.get("action")
                action_reason = action.get("reason")
                action_status = action.get("status")
                if action_status == "recall":
                    action_history_text += (
                        f"原本打算：{action_type}，但是因为有新消息，你发现这个行动不合适，所以你没做\n"
                    )
                elif action_status == "done":
                    action_history_text += f"你之前做了：{action_type}，原因：{action_reason}\n"
            elif isinstance(action, tuple):
                # 假设元组的格式是(action_type, action_reason, action_status)
                action_type = action[0] if len(action) > 0 else "未知行动"
                action_reason = action[1] if len(action) > 1 else "未知原因"
                action_status = action[2] if len(action) > 2 else "done"
                if action_status == "recall":
                    action_history_text += (
                        f"原本打算：{action_type}，但是因为有新消息，你发现这个行动不合适，所以你没做\n"
                    )
                elif action_status == "done":
                    action_history_text += f"你之前做了：{action_type}，原因：{action_reason}\n"

        prompt = f"""{personality_text}。现在你在参与一场QQ聊天，请分析以下内容，根据信息决定下一步行动：

当前对话目标：{goals_str}

{action_history_text}

最近的对话记录：
{chat_history_text}

请你接下去想想要你要做什么，可以发言，可以等待，可以倾听，可以调取知识。注意不同行动类型的要求，不要重复发言：
行动类型：
fetch_knowledge: 需要调取知识，当需要专业知识或特定信息时选择
wait: 当你做出了发言,对方尚未回复时暂时等待对方的回复
listening: 倾听对方发言，当你认为对方发言尚未结束时采用
direct_reply: 不符合上述情况，回复对方，注意不要过多或者重复发言
rethink_goal: 重新思考对话目标，当发现对话目标不合适时选择，会重新思考对话目标
end_conversation: 结束对话，长时间没回复或者当你觉得谈话暂时结束时选择，停止该场对话

请以JSON格式输出，包含以下字段：
1. action: 行动类型，注意你之前的行为
2. reason: 选择该行动的原因，注意你之前的行为（简要解释）

注意：请严格按照JSON格式输出，不要包含任何其他内容。"""

        logger.debug(f"发送到LLM的提示词: {prompt}")
        try:
            content, _ = await self.llm.generate_response_async(prompt)
            logger.debug(f"LLM原始返回内容: {content}")

            # 使用简化函数提取JSON内容
            success, result = get_items_from_json(
                content, "action", "reason", default_values={"action": "direct_reply", "reason": "没有明确原因"}
            )

            if not success:
                return "direct_reply", "JSON解析失败，选择直接回复"

            action = result["action"]
            reason = result["reason"]

            # 验证action类型
            if action not in [
                "direct_reply",
                "fetch_knowledge",
                "wait",
                "listening",
                "rethink_goal",
                "end_conversation",
            ]:
                logger.warning(f"未知的行动类型: {action}，默认使用listening")
                action = "listening"

            logger.info(f"规划的行动: {action}")
            logger.info(f"行动原因: {reason}")
            return action, reason

        except Exception as e:
            logger.error(f"规划行动时出错: {str(e)}")
            return "direct_reply", "发生错误，选择直接回复"
