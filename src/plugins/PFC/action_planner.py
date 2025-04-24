import time
from typing import Tuple, List, Dict, Any, Optional # 确保导入了必要的类型
from src.common.logger import get_module_logger
from ..models.utils_model import LLMRequest
from ...config.config import global_config
from .chat_observer import ChatObserver
from .pfc_utils import get_items_from_json
from src.individuality.individuality import Individuality
from .observation_info import ObservationInfo
from .conversation_info import ConversationInfo

logger = get_module_logger("action_planner")

# 注意：这个 ActionPlannerInfo 类似乎没有在 ActionPlanner 中使用，
# 如果确实没用，可以考虑移除，但暂时保留以防万一。
class ActionPlannerInfo:
    def __init__(self):
        self.done_action = []
        self.goal_list = []
        self.knowledge_list = []
        self.memory_list = []

# ActionPlanner 类定义，顶格
class ActionPlanner:
    """行动规划器"""
    def __init__(self, stream_id: str):
        self.llm = LLMRequest(
            model=global_config.llm_PFC_action_planner,
            temperature=global_config.llm_PFC_action_planner["temp"],
            max_tokens=1500,
            request_type="action_planning",
        )
        self.personality_info = Individuality.get_instance().get_prompt(type="personality", x_person=2, level=3)
        self.identity_detail_info = Individuality.get_instance().get_prompt(type="identity", x_person=2, level=2)
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
        # --- 获取 Bot 上次发言时间信息 ---
        time_since_last_bot_message_info = ""
        try:
            bot_id = str(global_config.BOT_QQ)
            if hasattr(observation_info, 'chat_history') and observation_info.chat_history:
                for i in range(len(observation_info.chat_history) - 1, -1, -1):
                    msg = observation_info.chat_history[i]
                    if not isinstance(msg, dict):
                        continue
                    sender_info = msg.get('user_info', {})
                    sender_id = str(sender_info.get('user_id')) if isinstance(sender_info, dict) else None
                    msg_time = msg.get('time')
                    if sender_id == bot_id and msg_time:
                        time_diff = time.time() - msg_time
                        if time_diff < 60.0:
                            time_since_last_bot_message_info = f"提示：你上一条成功发送的消息是在 {time_diff:.1f} 秒前。\n"
                        break
            else:
                logger.debug("Observation info chat history is empty or not available for bot time check.")
        except AttributeError:
            logger.warning("ObservationInfo object might not have chat_history attribute yet for bot time check.")
        except Exception as e:
            logger.warning(f"获取 Bot 上次发言时间时出错: {e}")
        # --- 获取 Bot 上次发言时间信息结束 ---

        timeout_context = ""
        try: # 添加 try-except 以增加健壮性
            if hasattr(conversation_info, 'goal_list') and conversation_info.goal_list:
                last_goal_tuple = conversation_info.goal_list[-1]
                if isinstance(last_goal_tuple, tuple) and len(last_goal_tuple) > 0:
                    last_goal_text = last_goal_tuple[0]
                    if isinstance(last_goal_text, str) and "分钟，思考接下来要做什么" in last_goal_text:
                        try:
                            timeout_minutes_text = last_goal_text.split('，')[0].replace('你等待了','')
                            timeout_context = f"重要提示：你刚刚因为对方长时间（{timeout_minutes_text}）没有回复而结束了等待，这可能代表在对方看来本次聊天已结束，请基于此情况规划下一步，不要重复等待前的发言。\n"
                        except Exception:
                            timeout_context = f"重要提示：你刚刚因为对方长时间没有回复而结束了等待，这可能代表在对方看来本次聊天已结束，请基于此情况规划下一步，不要重复等待前的发言。\n"
            else:
                logger.debug("Conversation info goal_list is empty or not available for timeout check.")
        except AttributeError:
             logger.warning("ConversationInfo object might not have goal_list attribute yet for timeout check.")
        except Exception as e:
             logger.warning(f"检查超时目标时出错: {e}")

        # 构建提示词
        logger.debug(f"开始规划行动：当前目标: {getattr(conversation_info, 'goal_list', '不可用')}") # 使用 getattr

        # 构建对话目标 (goals_str)
        goals_str = ""
        try: # 添加 try-except
            if hasattr(conversation_info, 'goal_list') and conversation_info.goal_list:
                for goal_reason in conversation_info.goal_list:
                    if isinstance(goal_reason, tuple) and len(goal_reason) > 0:
                        goal = goal_reason[0]
                        reasoning = goal_reason[1] if len(goal_reason) > 1 else "没有明确原因"
                    elif isinstance(goal_reason, dict):
                        goal = goal_reason.get("goal", "目标内容缺失")
                        reasoning = goal_reason.get("reasoning", "没有明确原因")
                    else:
                        goal = str(goal_reason)
                        reasoning = "没有明确原因"
                    goal = str(goal) if goal is not None else "目标内容缺失"
                    reasoning = str(reasoning) if reasoning is not None else "没有明确原因"
                    goal_str += f"- 目标：{goal}\n  原因：{reasoning}\n"
            if not goals_str: # 如果循环后 goals_str 仍为空
                goals_str = "- 目前没有明确对话目标，请考虑设定一个。\n"
        except AttributeError:
             logger.warning("ConversationInfo object might not have goal_list attribute yet.")
             goals_str = "- 获取对话目标时出错。\n"
        except Exception as e:
             logger.error(f"构建对话目标字符串时出错: {e}")
             goals_str = "- 构建对话目标时出错。\n"

        # 获取聊天历史记录 (chat_history_text)
        chat_history_text = ""
        try:
            if hasattr(observation_info, 'chat_history') and observation_info.chat_history:
                chat_history_list = observation_info.chat_history[-20:]
                for msg in chat_history_list:
                    if isinstance(msg, dict) and 'detailed_plain_text' in msg:
                        chat_history_text += f"{msg.get('detailed_plain_text', '')}\n"
                    elif isinstance(msg, str):
                        chat_history_text += f"{msg}\n"
                if not chat_history_text: # 如果历史记录是空列表
                     chat_history_text = "还没有聊天记录。\n"
            else:
                chat_history_text = "还没有聊天记录。\n"

            if hasattr(observation_info, 'new_messages_count') and observation_info.new_messages_count > 0:
                if hasattr(observation_info, 'unprocessed_messages') and observation_info.unprocessed_messages:
                    new_messages_list = observation_info.unprocessed_messages
                    chat_history_text += f"--- 以下是 {observation_info.new_messages_count} 条新消息 ---\n"
                    for msg in new_messages_list:
                        if isinstance(msg, dict) and 'detailed_plain_text' in msg:
                            chat_history_text += f"{msg.get('detailed_plain_text', '')}\n"
                        elif isinstance(msg, str):
                            chat_history_text += f"{msg}\n"
                    # 清理消息应该由调用者或 observation_info 内部逻辑处理，这里不再调用 clear
                    # if hasattr(observation_info, 'clear_unprocessed_messages'):
                    #    observation_info.clear_unprocessed_messages()
                else:
                    logger.warning("ObservationInfo has new_messages_count > 0 but unprocessed_messages is empty or missing.")
        except AttributeError:
            logger.warning("ObservationInfo object might be missing expected attributes for chat history.")
            chat_history_text = "获取聊天记录时出错。\n"
        except Exception as e:
            logger.error(f"处理聊天记录时发生未知错误: {e}")
            chat_history_text = "处理聊天记录时出错。\n"


        # 构建 Persona 文本 (persona_text)
        identity_details_only = self.identity_detail_info
        identity_addon = ""
        if isinstance(identity_details_only, str):
            pronouns = ["你", "我", "他"]
            original_details = identity_details_only
            for p in pronouns:
                if identity_details_only.startswith(p):
                    identity_details_only = identity_details_only[len(p):]
                    break
            if identity_details_only.endswith("。"):
                 identity_details_only = identity_details_only[:-1]
            cleaned_details = identity_details_only.strip(',， ')
            if cleaned_details:
                identity_addon = f"并且{cleaned_details}"
        persona_text = f"你的名字是{self.name}，{self.personality_info}{identity_addon}。"

        # --- 构建更清晰的行动历史和上一次行动结果 ---
        action_history_summary = "你最近执行的行动历史：\n"
        last_action_context = "关于你【上一次尝试】的行动：\n"

        action_history_list = []
        try: # 添加 try-except
            if hasattr(conversation_info, 'done_action') and conversation_info.done_action:
                action_history_list = conversation_info.done_action[-5:]
            else:
                logger.debug("Conversation info done_action is empty or not available.")
        except AttributeError:
             logger.warning("ConversationInfo object might not have done_action attribute yet.")
        except Exception as e:
             logger.error(f"访问行动历史时出错: {e}")

        if not action_history_list:
            action_history_summary += "- 还没有执行过行动。\n"
            last_action_context += "- 这是你规划的第一个行动。\n"
        else:
            for i, action_data in enumerate(action_history_list):
                action_type = "未知"
                plan_reason = "未知"
                status = "未知"
                final_reason = ""
                action_time = ""

                if isinstance(action_data, dict):
                    action_type = action_data.get("action", "未知")
                    plan_reason = action_data.get("plan_reason", "未知规划原因")
                    status = action_data.get("status", "未知")
                    final_reason = action_data.get("final_reason", "")
                    action_time = action_data.get("time", "")
                elif isinstance(action_data, tuple):
                    if len(action_data) > 0: action_type = action_data[0]
                    if len(action_data) > 1: plan_reason = action_data[1]
                    if len(action_data) > 2: status = action_data[2]
                    if status == "recall" and len(action_data) > 3: final_reason = action_data[3]

                reason_text = f", 失败/取消原因: {final_reason}" if final_reason else ""
                summary_line = f"- 时间:{action_time}, 尝试行动:'{action_type}', 状态:{status}{reason_text}"
                action_history_summary += summary_line + "\n"

                if i == len(action_history_list) - 1:
                    last_action_context += f"- 上次【规划】的行动是: '{action_type}'\n"
                    last_action_context += f"- 当时规划的【原因】是: {plan_reason}\n"
                    if status == "done":
                        last_action_context += f"- 该行动已【成功执行】。\n"
                    elif status == "recall":
                        last_action_context += f"- 但该行动最终【未能执行/被取消】。\n"
                        if final_reason:
                             last_action_context += f"- 【重要】失败/取消的具体原因是: “{final_reason}”\n"
                        else:
                             last_action_context += f"- 【重要】失败/取消原因未明确记录。\n"
                    else:
                        last_action_context += f"- 该行动当前状态: {status}\n"

        # --- 构建最终的 Prompt ---
        prompt = f"""{persona_text}。现在你在参与一场QQ聊天，请根据以下【所有信息】审慎决策下一步行动，可以发言，可以等待，可以倾听，可以调取知识：

【当前对话目标】
{goals_str if goals_str.strip() else "- 目前没有明确对话目标，请考虑设定一个。"}


【最近行动历史概要】
{action_history_summary}
【上一次行动的详细情况和结果】
{last_action_context}
【时间和超时提示】
{time_since_last_bot_message_info}{timeout_context}
【最近的对话记录】(包括你已成功发送的消息 和 新收到的消息)
{chat_history_text if chat_history_text.strip() else "还没有聊天记录。"}

--- 行动决策指南 ---
1.  **仔细分析【上一次行动的详细情况和结果】**。如果上次行动是 direct_reply 且因“内容与你上一条发言完全相同”或“高度相似”而被取消(status: recall)，那么【绝对不要】立即再次规划 direct_reply。在这种特定情况下，你应该优先考虑 wait (等待用户的新回应) 或 rethink_goal (如果对话似乎因此卡住了)。
2.  结合【当前对话目标】和【最近的对话记录】来判断是否需要回应、回应什么。如果【最近的对话记录】中有新的用户消息，通常需要 direct_reply。如果上次行动成功，或者上次失败的原因不是重复，可以根据对话内容考虑 direct_reply。
3.  注意【时间和超时提示】，如果对方长时间未回复（例如在 timeout_context 中提示），end_conversation 可能更合适。
4.  只有在你确信需要发言（比如回应新消息、追问、深入话题），并且上一次行动没有因重复被拒时，才应优先选择 direct_reply。

--- 可选行动类型 ---
fetch_knowledge: 需要调取知识，当需要专业知识或特定信息时选择
wait: 等待对方回复（尤其是在你刚发言后、或上次发言因重复被拒时、或不确定做什么时，这是较安全的选择）
listening: 倾听对方发言，当你认为对方发言尚未结束时采用
direct_reply: 直接回复或发送新消息，允许适当的追问和深入话题，**但是请务必遵守上面的决策指南，避免在因重复被拒后立即使用，也不要在对方没有回复的情况下过多的“消息轰炸”或重复发言**
rethink_goal: 重新思考对话目标，当发现对话目标不再适用或对话卡住时选择，注意私聊的环境是灵活的，有可能需要经常选择
end_conversation: 决定结束对话，对方长时间没回复或者当你觉得谈话暂时结束时可以选择

请以JSON格式输出你的决策：
{{
    "action": "选择的行动类型 (必须是上面列表中的一个)",
    "reason": "选择该行动的详细原因 (必须解释你是如何根据“上一次行动结果”、“对话记录”和“决策指南”做出判断的)"
}}

注意：请严格按照JSON格式输出，不要包含任何其他内容。"""

        logger.debug(f"发送到LLM的提示词 (已更新): {prompt}")
        try:
            content, _ = await self.llm.generate_response_async(prompt)
            logger.debug(f"LLM原始返回内容: {content}")

            success, result = get_items_from_json(
                content, "action", "reason",
                default_values={"action": "wait", "reason": "LLM返回格式错误或未提供原因，默认等待"}
            )

            action = result.get("action", "wait")
            reason = result.get("reason", "LLM未提供原因，默认等待")


            # 验证action类型
            valid_actions = ["direct_reply", "fetch_knowledge", "wait", "listening", "rethink_goal", "end_conversation"]
            if action not in valid_actions:
                logger.warning(f"LLM返回了未知的行动类型: '{action}'，强制改为 wait")
                reason = f"(原始行动'{action}'无效，已强制改为wait) {reason}"
                action = "wait"

            logger.info(f"规划的行动: {action}")
            logger.info(f"行动原因: {reason}")
            return action, reason

        except Exception as e:
            logger.error(f"规划行动时调用 LLM 或处理结果出错: {str(e)}")
            return "wait", f"行动规划处理中发生错误，暂时等待: {str(e)}"