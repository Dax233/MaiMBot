"""
基于聊天记录的人格特征分析系统
"""

from typing import Dict, List
import json
import os
from pathlib import Path
from dotenv import load_dotenv
import sys
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib.font_manager as fm

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent
env_path = project_root / ".env"

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(root_path)

from src.plugins.personality.scene import get_scene_by_factor, PERSONALITY_SCENES  # noqa: E402
from src.plugins.personality.questionnaire import FACTOR_DESCRIPTIONS  # noqa: E402
from src.plugins.personality.offline_llm import LLMModel  # noqa: E402
from src.plugins.personality.who_r_u import MessageAnalyzer  # noqa: E402

# 加载环境变量
if env_path.exists():
    print(f"从 {env_path} 加载环境变量")
    load_dotenv(env_path)
else:
    print(f"未找到环境变量文件: {env_path}")
    print("将使用默认配置")


class ChatBasedPersonalityEvaluator:
    def __init__(self):
        self.personality_traits = {"开放性": 0, "严谨性": 0, "外向性": 0, "宜人性": 0, "神经质": 0}
        self.scenarios = []
        self.message_analyzer = MessageAnalyzer()
        self.llm = LLMModel()
        self.trait_scores_history = defaultdict(list)  # 记录每个特质的得分历史

        # 为每个人格特质获取对应的场景
        for trait in PERSONALITY_SCENES:
            scenes = get_scene_by_factor(trait)
            if not scenes:
                continue
            scene_keys = list(scenes.keys())
            selected_scenes = random.sample(scene_keys, min(3, len(scene_keys)))

            for scene_key in selected_scenes:
                scene = scenes[scene_key]
                other_traits = [t for t in PERSONALITY_SCENES if t != trait]
                secondary_trait = random.choice(other_traits)
                self.scenarios.append(
                    {"场景": scene["scenario"], "评估维度": [trait, secondary_trait], "场景编号": scene_key}
                )

    def analyze_chat_context(self, messages: List[Dict]) -> str:
        """
        分析一组消息的上下文，生成场景描述
        """
        context = ""
        for msg in messages:
            nickname = msg.get("user_info", {}).get("user_nickname", "未知用户")
            content = msg.get("processed_plain_text", msg.get("detailed_plain_text", ""))
            if content:
                context += f"{nickname}: {content}\n"
        return context

    def evaluate_chat_response(
        self, user_nickname: str, chat_context: str, dimensions: List[str] = None
    ) -> Dict[str, float]:
        """
        评估聊天内容在各个人格维度上的得分
        """
        # 使用所有维度进行评估
        dimensions = list(self.personality_traits.keys())

        dimension_descriptions = []
        for dim in dimensions:
            desc = FACTOR_DESCRIPTIONS.get(dim, "")
            if desc:
                dimension_descriptions.append(f"- {dim}：{desc}")

        dimensions_text = "\n".join(dimension_descriptions)

        prompt = f"""请根据以下聊天记录，评估"{user_nickname}"在大五人格模型中的维度得分（1-6分）。

聊天记录：
{chat_context}

需要评估的维度说明：
{dimensions_text}

请按照以下格式输出评估结果，注意，你的评价对象是"{user_nickname}"（仅输出JSON格式）：
{{
    "开放性": 分数,
    "严谨性": 分数,
    "外向性": 分数,
    "宜人性": 分数,
    "神经质": 分数
}}

评分标准：
1 = 非常不符合该维度特征
2 = 比较不符合该维度特征
3 = 有点不符合该维度特征
4 = 有点符合该维度特征
5 = 比较符合该维度特征
6 = 非常符合该维度特征

如果你觉得某个维度没有相关信息或者无法判断，请输出0分

请根据聊天记录的内容和语气，结合维度说明进行评分。如果维度可以评分，确保分数在1-6之间。如果没有体现，请输出0分"""

        try:
            ai_response, _ = self.llm.generate_response(prompt)
            start_idx = ai_response.find("{")
            end_idx = ai_response.rfind("}") + 1
            if start_idx != -1 and end_idx != 0:
                json_str = ai_response[start_idx:end_idx]
                scores = json.loads(json_str)
                return {k: max(0, min(6, float(v))) for k, v in scores.items()}
            else:
                print("AI响应格式不正确，使用默认评分")
                return {dim: 0 for dim in dimensions}
        except Exception as e:
            print(f"评估过程出错：{str(e)}")
            return {dim: 0 for dim in dimensions}

    def evaluate_user_personality(self, qq_id: str, num_samples: int = 10, context_length: int = 5) -> Dict:
        """
        基于用户的聊天记录评估人格特征

        Args:
            qq_id (str): 用户QQ号
            num_samples (int): 要分析的聊天片段数量
            context_length (int): 每个聊天片段的上下文长度

        Returns:
            Dict: 评估结果
        """
        # 获取用户的随机消息及其上下文
        chat_contexts, user_nickname = self.message_analyzer.get_user_random_contexts(
            qq_id, num_messages=num_samples, context_length=context_length
        )
        if not chat_contexts:
            return {"error": f"没有找到QQ号 {qq_id} 的消息记录"}

        # 初始化评分
        final_scores = defaultdict(float)
        dimension_counts = defaultdict(int)
        chat_samples = []

        # 清空历史记录
        self.trait_scores_history.clear()

        # 分析每个聊天上下文
        for chat_context in chat_contexts:
            # 评估这段聊天内容的所有维度
            scores = self.evaluate_chat_response(user_nickname, chat_context)

            # 记录样本
            chat_samples.append(
                {"聊天内容": chat_context, "评估维度": list(self.personality_traits.keys()), "评分": scores}
            )

            # 更新总分和历史记录
            for dimension, score in scores.items():
                if score > 0:  # 只统计大于0的有效分数
                    final_scores[dimension] += score
                    dimension_counts[dimension] += 1
                self.trait_scores_history[dimension].append(score)

        # 计算平均分
        average_scores = {}
        for dimension in self.personality_traits:
            if dimension_counts[dimension] > 0:
                average_scores[dimension] = round(final_scores[dimension] / dimension_counts[dimension], 2)
            else:
                average_scores[dimension] = 0  # 如果没有有效分数，返回0

        # 生成趋势图
        self._generate_trend_plot(qq_id, user_nickname)

        result = {
            "用户QQ": qq_id,
            "用户昵称": user_nickname,
            "样本数量": len(chat_samples),
            "人格特征评分": average_scores,
            "维度评估次数": dict(dimension_counts),
            "详细样本": chat_samples,
            "特质得分历史": {k: v for k, v in self.trait_scores_history.items()},
        }

        # 保存结果
        os.makedirs("results", exist_ok=True)
        result_file = f"results/personality_result_{qq_id}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return result

    def _generate_trend_plot(self, qq_id: str, user_nickname: str):
        """
        生成人格特质累计平均分变化趋势图
        """
        # 查找系统中可用的中文字体
        chinese_fonts = []
        for f in fm.fontManager.ttflist:
            try:
                if "简" in f.name or "SC" in f.name or "黑" in f.name or "宋" in f.name or "微软" in f.name:
                    chinese_fonts.append(f.name)
            except Exception:
                continue

        if chinese_fonts:
            plt.rcParams["font.sans-serif"] = chinese_fonts + ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
        else:
            # 如果没有找到中文字体，使用默认字体，并将中文昵称转换为拼音或英文
            try:
                from pypinyin import lazy_pinyin

                user_nickname = "".join(lazy_pinyin(user_nickname))
            except ImportError:
                user_nickname = "User"  # 如果无法转换为拼音，使用默认英文

        plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

        plt.figure(figsize=(12, 6))
        plt.style.use("bmh")  # 使用内置的bmh样式，它有类似seaborn的美观效果

        colors = {
            "开放性": "#FF9999",
            "严谨性": "#66B2FF",
            "外向性": "#99FF99",
            "宜人性": "#FFCC99",
            "神经质": "#FF99CC",
        }

        # 计算每个维度在每个时间点的累计平均分
        cumulative_averages = {}
        for trait, scores in self.trait_scores_history.items():
            if not scores:
                continue

            averages = []
            total = 0
            valid_count = 0
            for score in scores:
                if score > 0:  # 只计算大于0的有效分数
                    total += score
                    valid_count += 1
                    if valid_count > 0:
                        averages.append(total / valid_count)
                else:
                    # 如果当前分数无效，使用前一个有效的平均分
                    if averages:
                        averages.append(averages[-1])
                    else:
                        continue  # 跳过无效分数

            if averages:  # 只有在有有效分数的情况下才添加到累计平均中
                cumulative_averages[trait] = averages

        # 绘制每个维度的累计平均分变化趋势
        for trait, averages in cumulative_averages.items():
            x = range(1, len(averages) + 1)
            plt.plot(x, averages, "o-", label=trait, color=colors.get(trait), linewidth=2, markersize=8)

            # 添加趋势线
            z = np.polyfit(x, averages, 1)
            p = np.poly1d(z)
            plt.plot(x, p(x), "--", color=colors.get(trait), alpha=0.5)

        plt.title(f"{user_nickname} 的人格特质累计平均分变化趋势", fontsize=14, pad=20)
        plt.xlabel("评估次数", fontsize=12)
        plt.ylabel("累计平均分", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.ylim(0, 7)
        plt.tight_layout()

        # 保存图表
        os.makedirs("results/plots", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = f"results/plots/personality_trend_{qq_id}_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()


def analyze_user_personality(qq_id: str, num_samples: int = 10, context_length: int = 5) -> str:
    """
    分析用户人格特征的便捷函数

    Args:
        qq_id (str): 用户QQ号
        num_samples (int): 要分析的聊天片段数量
        context_length (int): 每个聊天片段的上下文长度

    Returns:
        str: 格式化的分析结果
    """
    evaluator = ChatBasedPersonalityEvaluator()
    result = evaluator.evaluate_user_personality(qq_id, num_samples, context_length)

    if "error" in result:
        return result["error"]

    # 格式化输出
    output = f"QQ号 {qq_id} ({result['用户昵称']}) 的人格特征分析结果：\n"
    output += "=" * 50 + "\n\n"

    output += "人格特征评分：\n"
    for trait, score in result["人格特征评分"].items():
        if score == 0:
            output += f"{trait}: 数据不足，无法判断 (评估次数: {result['维度评估次数'].get(trait, 0)})\n"
        else:
            output += f"{trait}: {score}/6 (评估次数: {result['维度评估次数'].get(trait, 0)})\n"

        # 添加变化趋势描述
        if trait in result["特质得分历史"] and len(result["特质得分历史"][trait]) > 1:
            scores = [s for s in result["特质得分历史"][trait] if s != 0]  # 过滤掉无效分数
            if len(scores) > 1:  # 确保有足够的有效分数计算趋势
                trend = np.polyfit(range(len(scores)), scores, 1)[0]
                if abs(trend) < 0.1:
                    trend_desc = "保持稳定"
                elif trend > 0:
                    trend_desc = "呈上升趋势"
                else:
                    trend_desc = "呈下降趋势"
                output += f"    变化趋势: {trend_desc} (斜率: {trend:.2f})\n"

    output += f"\n分析样本数量：{result['样本数量']}\n"
    output += f"结果已保存至：results/personality_result_{qq_id}.json\n"
    output += "变化趋势图已保存至：results/plots/目录\n"

    return output


if __name__ == "__main__":
    # 测试代码
    # test_qq = ""  # 替换为要测试的QQ号
    # print(analyze_user_personality(test_qq, num_samples=30, context_length=20))
    # test_qq = ""
    # print(analyze_user_personality(test_qq, num_samples=30, context_length=20))
    test_qq = "1026294844"
    print(analyze_user_personality(test_qq, num_samples=30, context_length=30))
