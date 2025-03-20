from typing import Dict, List
import json
import os
from pathlib import Path
from dotenv import load_dotenv
import sys

current_dir = Path(__file__).resolve().parent
# 获取项目根目录（上三层目录）
project_root = current_dir.parent.parent.parent
# env.dev文件路径
env_path = project_root / ".env.prod"

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(root_path)

from src.plugins.personality.offline_llm import LLMModel  # noqa E402

# 加载环境变量
if env_path.exists():
    print(f"从 {env_path} 加载环境变量")
    load_dotenv(env_path)
else:
    print(f"未找到环境变量文件: {env_path}")
    print("将使用默认配置")


class PersonalityEvaluator:
    def __init__(self):
        self.personality_traits = {"开放性": 0, "尽责性": 0, "外向性": 0, "宜人性": 0, "神经质": 0}
        self.scenarios = [
            {
                "场景": "在团队项目中，你发现一个同事的工作质量明显低于预期，这可能会影响整个项目的进度。",
                "评估维度": ["尽责性", "宜人性"],
            },
            {"场景": "你被邀请参加一个完全陌生的社交活动，现场都是不认识的人。", "评估维度": ["外向性", "神经质"]},
            {
                "场景": "你的朋友向你推荐了一个新的艺术展览，但风格与你平时接触的完全不同。",
                "评估维度": ["开放性", "外向性"],
            },
            {"场景": "在工作中，你遇到了一个技术难题，需要学习全新的技术栈。", "评估维度": ["开放性", "尽责性"]},
            {"场景": "你的朋友因为个人原因情绪低落，向你寻求帮助。", "评估维度": ["宜人性", "神经质"]},
        ]
        self.llm = LLMModel()

    def evaluate_response(self, scenario: str, response: str, dimensions: List[str]) -> Dict[str, float]:
        """
        使用 DeepSeek AI 评估用户对特定场景的反应
        """
        prompt = f"""请根据以下场景和用户描述，评估用户在大五人格模型中的相关维度得分（0-10分）。
场景：{scenario}
用户描述：{response}

需要评估的维度：{", ".join(dimensions)}

请按照以下格式输出评估结果（仅输出JSON格式）：
{{
    "维度1": 分数,
    "维度2": 分数
}}

评估标准：
- 开放性：对新事物的接受程度和创造性思维
- 尽责性：计划性、组织性和责任感
- 外向性：社交倾向和能量水平
- 宜人性：同理心、合作性和友善程度
- 神经质：情绪稳定性和压力应对能力

请确保分数在0-10之间，并给出合理的评估理由。"""

        try:
            ai_response, _ = self.llm.generate_response(prompt)
            # 尝试从AI响应中提取JSON部分
            start_idx = ai_response.find("{")
            end_idx = ai_response.rfind("}") + 1
            if start_idx != -1 and end_idx != 0:
                json_str = ai_response[start_idx:end_idx]
                scores = json.loads(json_str)
                # 确保所有分数在0-10之间
                return {k: max(0, min(10, float(v))) for k, v in scores.items()}
            else:
                print("AI响应格式不正确，使用默认评分")
                return {dim: 5.0 for dim in dimensions}
        except Exception as e:
            print(f"评估过程出错：{str(e)}")
            return {dim: 5.0 for dim in dimensions}


def main():
    print("欢迎使用人格形象创建程序！")
    print("接下来，您将面对一系列场景。请根据您想要创建的角色形象，描述在该场景下可能的反应。")
    print("每个场景都会评估不同的人格维度，最终得出完整的人格特征评估。")
    print("\n准备好了吗？按回车键开始...")
    input()

    evaluator = PersonalityEvaluator()
    final_scores = {"开放性": 0, "尽责性": 0, "外向性": 0, "宜人性": 0, "神经质": 0}
    dimension_counts = {trait: 0 for trait in final_scores.keys()}

    for i, scenario_data in enumerate(evaluator.scenarios, 1):
        print(f"\n场景 {i}/{len(evaluator.scenarios)}:")
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
            print(f"{dimension}: {score}/10")

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
        print(f"{trait}: {score}/10")

    # 保存结果
    result = {"final_scores": final_scores, "scenarios": evaluator.scenarios}

    # 确保目录存在
    os.makedirs("results", exist_ok=True)

    # 保存到文件
    with open("results/personality_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n结果已保存到 results/personality_result.json")


if __name__ == "__main__":
    main()
