from typing import Dict
import json
import os
from pathlib import Path
import sys
from datetime import datetime
import random
from scipy import stats  # 添加scipy导入用于t检验

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent
env_path = project_root / ".env"

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(root_path)

from src.plugins.personality.big5_test import BigFiveTest  # noqa: E402
from src.plugins.personality.renqingziji import PersonalityEvaluator_direct  # noqa: E402
from src.plugins.personality.questionnaire import FACTOR_DESCRIPTIONS, PERSONALITY_QUESTIONS  # noqa: E402


class CombinedPersonalityTest:
    def __init__(self):
        self.big5_test = BigFiveTest()
        self.scenario_test = PersonalityEvaluator_direct()
        self.dimensions = ["开放性", "严谨性", "外向性", "宜人性", "神经质"]

    def run_combined_test(self):
        """运行组合测试"""
        print("\n=== 人格特征综合评估系统 ===")
        print("\n本测试将通过两种方式评估人格特征：")
        print("1. 传统问卷测评（约40题）")
        print("2. 情景反应测评（15个场景）")
        print("\n两种测评完成后，将对比分析结果的异同。")
        input("\n准备好开始第一部分（问卷测评）了吗？按回车继续...")

        # 运行问卷测试
        print("\n=== 第一部分：问卷测评 ===")
        print("本部分采用六级评分，请根据每个描述与您的符合程度进行打分：")
        print("1 = 完全不符合")
        print("2 = 比较不符合")
        print("3 = 有点不符合")
        print("4 = 有点符合")
        print("5 = 比较符合")
        print("6 = 完全符合")
        print("\n重要提示：您可以选择以下两种方式之一来回答问题：")
        print("1. 根据您自身的真实情况来回答")
        print("2. 根据您想要扮演的角色特征来回答")
        print("\n无论选择哪种方式，请保持一致并认真回答每个问题。")
        input("\n按回车开始答题...")

        questionnaire_results = self.run_questionnaire()

        # 转换问卷结果格式以便比较
        questionnaire_scores = {factor: data["得分"] for factor, data in questionnaire_results.items()}

        # 运行情景测试
        print("\n=== 第二部分：情景反应测评 ===")
        print("接下来，您将面对一系列具体场景，请描述您在每个场景中可能的反应。")
        print("每个场景都会评估不同的人格维度，共15个场景。")
        print("您可以选择提供自己的真实反应，也可以选择扮演一个您创作的角色来回答。")
        input("\n准备好开始了吗？按回车继续...")

        scenario_results = self.run_scenario_test()

        # 比较和展示结果
        self.compare_and_display_results(questionnaire_scores, scenario_results)

        # 保存结果
        self.save_results(questionnaire_scores, scenario_results)

    def run_questionnaire(self):
        """运行问卷测试部分"""
        # 创建题目序号到题目的映射
        questions_map = {q["id"]: q for q in PERSONALITY_QUESTIONS}

        # 获取所有题目ID并随机打乱顺序
        question_ids = list(questions_map.keys())
        random.shuffle(question_ids)

        answers = {}
        total_questions = len(question_ids)

        for i, question_id in enumerate(question_ids, 1):
            question = questions_map[question_id]
            while True:
                try:
                    print(f"\n问题 [{i}/{total_questions}]")
                    print(f"{question['content']}")
                    score = int(input("您的评分（1-6）: "))
                    if 1 <= score <= 6:
                        answers[question_id] = score
                        break
                    else:
                        print("请输入1-6之间的数字！")
                except ValueError:
                    print("请输入有效的数字！")

            # 每10题显示一次进度
            if i % 10 == 0:
                print(f"\n已完成 {i}/{total_questions} 题 ({int(i / total_questions * 100)}%)")

        return self.calculate_questionnaire_scores(answers)

    def calculate_questionnaire_scores(self, answers):
        """计算问卷测试的维度得分"""
        results = {}
        factor_questions = {"外向性": [], "神经质": [], "严谨性": [], "开放性": [], "宜人性": []}

        # 将题目按因子分类
        for q in PERSONALITY_QUESTIONS:
            factor_questions[q["factor"]].append(q)

        # 计算每个维度的得分
        for factor, questions in factor_questions.items():
            total_score = 0
            for q in questions:
                score = answers[q["id"]]
                # 处理反向计分题目
                if q["reverse_scoring"]:
                    score = 7 - score  # 6分量表反向计分为7减原始分
                total_score += score

            # 计算平均分
            avg_score = round(total_score / len(questions), 2)
            results[factor] = {"得分": avg_score, "题目数": len(questions), "总分": total_score}

        return results

    def run_scenario_test(self):
        """运行情景测试部分"""
        final_scores = {"开放性": 0, "严谨性": 0, "外向性": 0, "宜人性": 0, "神经质": 0}
        dimension_counts = {trait: 0 for trait in final_scores.keys()}

        # 随机打乱场景顺序
        scenarios = self.scenario_test.scenarios.copy()
        random.shuffle(scenarios)

        for i, scenario_data in enumerate(scenarios, 1):
            print(f"\n场景 [{i}/{len(scenarios)}] - {scenario_data['场景编号']}")
            print("-" * 50)
            print(scenario_data["场景"])
            print("\n请描述您在这种情况下会如何反应：")
            response = input().strip()

            if not response:
                print("反应描述不能为空！")
                continue

            print("\n正在评估您的描述...")
            scores = self.scenario_test.evaluate_response(scenario_data["场景"], response, scenario_data["评估维度"])

            # 更新分数
            for dimension, score in scores.items():
                final_scores[dimension] += score
                dimension_counts[dimension] += 1

            # print("\n当前场景评估结果：")
            # print("-" * 30)
            # for dimension, score in scores.items():
            #     print(f"{dimension}: {score}/6")

            # 每5个场景显示一次总进度
            if i % 5 == 0:
                print(f"\n已完成 {i}/{len(scenarios)} 个场景 ({int(i / len(scenarios) * 100)}%)")

            if i < len(scenarios):
                input("\n按回车继续下一个场景...")

        # 计算平均分
        for dimension in final_scores:
            if dimension_counts[dimension] > 0:
                final_scores[dimension] = round(final_scores[dimension] / dimension_counts[dimension], 2)

        return final_scores

    def compare_and_display_results(self, questionnaire_scores: Dict, scenario_scores: Dict):
        """比较和展示两种测试的结果"""
        print("\n=== 测评结果对比分析 ===")
        print("\n" + "=" * 60)
        print(f"{'维度':<8} {'问卷得分':>10} {'情景得分':>10} {'差异':>10} {'差异程度':>10}")
        print("-" * 60)

        # 收集每个维度的得分用于统计分析
        questionnaire_values = []
        scenario_values = []
        diffs = []

        for dimension in self.dimensions:
            q_score = questionnaire_scores[dimension]
            s_score = scenario_scores[dimension]
            diff = round(abs(q_score - s_score), 2)

            questionnaire_values.append(q_score)
            scenario_values.append(s_score)
            diffs.append(diff)

            # 计算差异程度
            diff_level = "低" if diff < 0.5 else "中" if diff < 1.0 else "高"
            print(f"{dimension:<8} {q_score:>10.2f} {s_score:>10.2f} {diff:>10.2f} {diff_level:>10}")

        print("=" * 60)

        # 计算整体统计指标
        mean_diff = sum(diffs) / len(diffs)
        std_diff = (sum((x - mean_diff) ** 2 for x in diffs) / (len(diffs) - 1)) ** 0.5

        # 计算效应量 (Cohen's d)
        pooled_std = (
            (
                sum((x - sum(questionnaire_values) / len(questionnaire_values)) ** 2 for x in questionnaire_values)
                + sum((x - sum(scenario_values) / len(scenario_values)) ** 2 for x in scenario_values)
            )
            / (2 * len(self.dimensions) - 2)
        ) ** 0.5

        if pooled_std != 0:
            cohens_d = abs(mean_diff / pooled_std)

            # 解释效应量
            if cohens_d < 0.2:
                effect_size = "微小"
            elif cohens_d < 0.5:
                effect_size = "小"
            elif cohens_d < 0.8:
                effect_size = "中等"
            else:
                effect_size = "大"

        # 对所有维度进行整体t检验
        t_stat, p_value = stats.ttest_rel(questionnaire_values, scenario_values)
        print("\n整体统计分析:")
        print(f"平均差异: {mean_diff:.3f}")
        print(f"差异标准差: {std_diff:.3f}")
        print(f"效应量(Cohen's d): {cohens_d:.3f}")
        print(f"效应量大小: {effect_size}")
        print(f"t统计量: {t_stat:.3f}")
        print(f"p值: {p_value:.3f}")

        if p_value < 0.05:
            print("结论: 两种测评方法的结果存在显著差异 (p < 0.05)")
        else:
            print("结论: 两种测评方法的结果无显著差异 (p >= 0.05)")

        print("\n维度说明：")
        for dimension in self.dimensions:
            print(f"\n{dimension}:")
            desc = FACTOR_DESCRIPTIONS[dimension]
            print(f"定义：{desc['description']}")
            print(f"特征词：{', '.join(desc['trait_words'])}")

        # 分析显著差异
        significant_diffs = []
        for dimension in self.dimensions:
            diff = abs(questionnaire_scores[dimension] - scenario_scores[dimension])
            if diff >= 1.0:  # 差异大于等于1分视为显著
                significant_diffs.append(
                    {
                        "dimension": dimension,
                        "diff": diff,
                        "questionnaire": questionnaire_scores[dimension],
                        "scenario": scenario_scores[dimension],
                    }
                )

        if significant_diffs:
            print("\n\n显著差异分析：")
            print("-" * 40)
            for diff in significant_diffs:
                print(f"\n{diff['dimension']}维度的测评结果存在显著差异：")
                print(f"问卷得分：{diff['questionnaire']:.2f}")
                print(f"情景得分：{diff['scenario']:.2f}")
                print(f"差异值：{diff['diff']:.2f}")

                # 分析可能的原因
                if diff["questionnaire"] > diff["scenario"]:
                    print("可能原因：在问卷中的自我评价较高，但在具体情景中的表现较为保守。")
                else:
                    print("可能原因：在具体情景中表现出更多该维度特征，而在问卷自评时较为保守。")

    def save_results(self, questionnaire_scores: Dict, scenario_scores: Dict):
        """保存测试结果"""
        results = {
            "测试时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "问卷测评结果": questionnaire_scores,
            "情景测评结果": scenario_scores,
            "维度说明": FACTOR_DESCRIPTIONS,
        }

        # 确保目录存在
        os.makedirs("results", exist_ok=True)

        # 生成带时间戳的文件名
        filename = f"results/personality_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # 保存到文件
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n完整的测评结果已保存到：{filename}")


def load_existing_results():
    """检查并加载已有的测试结果"""
    results_dir = "results"
    if not os.path.exists(results_dir):
        return None

    # 获取所有personality_combined开头的文件
    result_files = [f for f in os.listdir(results_dir) if f.startswith("personality_combined_") and f.endswith(".json")]

    if not result_files:
        return None

    # 按文件修改时间排序，获取最新的结果文件
    latest_file = max(result_files, key=lambda f: os.path.getmtime(os.path.join(results_dir, f)))

    print(f"\n发现已有的测试结果：{latest_file}")
    try:
        with open(os.path.join(results_dir, latest_file), "r", encoding="utf-8") as f:
            results = json.load(f)
        return results
    except Exception as e:
        print(f"读取结果文件时出错：{str(e)}")
        return None


def main():
    test = CombinedPersonalityTest()

    # 检查是否存在已有结果
    existing_results = load_existing_results()

    if existing_results:
        print("\n=== 使用已有测试结果进行分析 ===")
        print(f"测试时间：{existing_results['测试时间']}")

        questionnaire_scores = existing_results["问卷测评结果"]
        scenario_scores = existing_results["情景测评结果"]

        # 直接进行结果对比分析
        test.compare_and_display_results(questionnaire_scores, scenario_scores)
    else:
        print("\n未找到已有的测试结果，开始新的测试...")
        test.run_combined_test()


if __name__ == "__main__":
    main()
