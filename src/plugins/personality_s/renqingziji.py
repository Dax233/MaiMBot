"""
The definition of artificial personality in this paper follows the dispositional para-digm and adapts a definition of
personality developed for humans [17]:
Personality for a human is the "whole and organisation of relatively stable tendencies and patterns of experience and
behaviour within one person (distinguishing it from other persons)". This definition is modified for artificial
personality:
Artificial personality describes the relatively stable tendencies and patterns of behav-iour of an AI-based machine that
can be designed by developers and designers via different modalities, such as language, creating the impression
of individuality of a humanized social agent when users interact with the machine."""

from typing import Dict, List
import json
import os
from pathlib import Path
from dotenv import load_dotenv
import sys

"""
第一种方案：基于情景评估的人格测定
"""
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent
env_path = project_root / ".env"

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(root_path)

from src.plugins.personality.scene import get_scene_by_factor, PERSONALITY_SCENES  # noqa: E402
from src.plugins.personality.questionnaire import FACTOR_DESCRIPTIONS  # noqa: E402
from src.plugins.personality.offline_llm import LLMModel  # noqa: E402

# 加载环境变量
if env_path.exists():
    print(f"从 {env_path} 加载环境变量")
    load_dotenv(env_path)
else:
    print(f"未找到环境变量文件: {env_path}")
    print("将使用默认配置")


class PersonalityEvaluator_direct:
    def __init__(self):
        self.personality_traits = {"开放性": 0, "严谨性": 0, "外向性": 0, "宜人性": 0, "神经质": 0}
        self.scenarios = []

        # 为每个人格特质获取对应的场景
        for trait in PERSONALITY_SCENES:
            scenes = get_scene_by_factor(trait)
            if not scenes:
                continue

            # 从每个维度选择3个场景
            import random

            scene_keys = list(scenes.keys())
            selected_scenes = random.sample(scene_keys, min(3, len(scene_keys)))

            for scene_key in selected_scenes:
                scene = scenes[scene_key]

                # 为每个场景添加评估维度
                # 主维度是当前特质，次维度随机选择一个其他特质
                other_traits = [t for t in PERSONALITY_SCENES if t != trait]
                secondary_trait = random.choice(other_traits)

                self.scenarios.append(
                    {"场景": scene["scenario"], "评估维度": [trait, secondary_trait], "场景编号": scene_key}
                )

        self.llm = LLMModel()

    def evaluate_response(self, scenario: str, response: str, dimensions: List[str]) -> Dict[str, float]:
        """
        使用 DeepSeek AI 评估用户对特定场景的反应
        """
        # 构建维度描述
        dimension_descriptions = []
        for dim in dimensions:
            desc = FACTOR_DESCRIPTIONS.get(dim, "")
            if desc:
                dimension_descriptions.append(f"- {dim}：{desc}")

        dimensions_text = "\n".join(dimension_descriptions)

        prompt = f"""请根据以下场景和用户描述，评估用户在大五人格模型中的相关维度得分（1-6分）。

场景描述：
{scenario}

用户回应：
{response}

需要评估的维度说明：
{dimensions_text}

请按照以下格式输出评估结果（仅输出JSON格式）：
{{
    "{dimensions[0]}": 分数,
    "{dimensions[1]}": 分数
}}

评分标准：
1 = 非常不符合该维度特征
2 = 比较不符合该维度特征
3 = 有点不符合该维度特征
4 = 有点符合该维度特征
5 = 比较符合该维度特征
6 = 非常符合该维度特征

请根据用户的回应，结合场景和维度说明进行评分。确保分数在1-6之间，并给出合理的评估。"""

        try:
            ai_response, _ = self.llm.generate_response(prompt)
            # 尝试从AI响应中提取JSON部分
            start_idx = ai_response.find("{")
            end_idx = ai_response.rfind("}") + 1
            if start_idx != -1 and end_idx != 0:
                json_str = ai_response[start_idx:end_idx]
                scores = json.loads(json_str)
                # 确保所有分数在1-6之间
                return {k: max(1, min(6, float(v))) for k, v in scores.items()}
            else:
                print("AI响应格式不正确，使用默认评分")
                return {dim: 3.5 for dim in dimensions}
        except Exception as e:
            print(f"评估过程出错：{str(e)}")
            return {dim: 3.5 for dim in dimensions}


def main():
    print("欢迎使用人格形象创建程序！")
    print("接下来，您将面对一系列场景（共15个）。请根据您想要创建的角色形象，描述在该场景下可能的反应。")
    print("每个场景都会评估不同的人格维度，最终得出完整的人格特征评估。")
    print("评分标准：1=非常不符合，2=比较不符合，3=有点不符合，4=有点符合，5=比较符合，6=非常符合")
    print("\n准备好了吗？按回车键开始...")
    input()

    evaluator = PersonalityEvaluator_direct()
    final_scores = {"开放性": 0, "严谨性": 0, "外向性": 0, "宜人性": 0, "神经质": 0}
    dimension_counts = {trait: 0 for trait in final_scores.keys()}

    for i, scenario_data in enumerate(evaluator.scenarios, 1):
        print(f"\n场景 {i}/{len(evaluator.scenarios)} - {scenario_data['场景编号']}:")
        print("-" * 50)
        print(scenario_data["场景"])
        print("\n请描述您的角色在这种情况下会如何反应：")
        response = input().strip()

        if not response:
            print("反应描述不能为空！")
            continue

        print("\n正在评估您的描述...")
        scores = evaluator.evaluate_response(scenario_data["场景"], response, scenario_data["评估维度"])

        # 更新最终分数
        for dimension, score in scores.items():
            final_scores[dimension] += score
            dimension_counts[dimension] += 1

        print("\n当前评估结果：")
        print("-" * 30)
        for dimension, score in scores.items():
            print(f"{dimension}: {score}/6")

        if i < len(evaluator.scenarios):
            print("\n按回车键继续下一个场景...")
            input()

    # 计算平均分
    for dimension in final_scores:
        if dimension_counts[dimension] > 0:
            final_scores[dimension] = round(final_scores[dimension] / dimension_counts[dimension], 2)

    print("\n最终人格特征评估结果：")
    print("-" * 30)
    for trait, score in final_scores.items():
        print(f"{trait}: {score}/6")
        print(f"测试场景数：{dimension_counts[trait]}")

    # 保存结果
    result = {"final_scores": final_scores, "dimension_counts": dimension_counts, "scenarios": evaluator.scenarios}

    # 确保目录存在
    os.makedirs("results", exist_ok=True)

    # 保存到文件
    with open("results/personality_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n结果已保存到 results/personality_result.json")


if __name__ == "__main__":
    main()
