#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# from .questionnaire import PERSONALITY_QUESTIONS, FACTOR_DESCRIPTIONS

import os
import sys
from pathlib import Path
import random

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent
env_path = project_root / ".env"

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(root_path)

from src.plugins.personality.questionnaire import PERSONALITY_QUESTIONS, FACTOR_DESCRIPTIONS  # noqa: E402


class BigFiveTest:
    def __init__(self):
        self.questions = PERSONALITY_QUESTIONS
        self.factors = FACTOR_DESCRIPTIONS

    def run_test(self):
        """运行测试并收集答案"""
        print("\n欢迎参加中国大五人格测试！")
        print("\n本测试采用六级评分，请根据每个描述与您的符合程度进行打分：")
        print("1 = 完全不符合")
        print("2 = 比较不符合")
        print("3 = 有点不符合")
        print("4 = 有点符合")
        print("5 = 比较符合")
        print("6 = 完全符合")
        print("\n请认真阅读每个描述，选择最符合您实际情况的选项。\n")

        # 创建题目序号到题目的映射
        questions_map = {q["id"]: q for q in self.questions}

        # 获取所有题目ID并随机打乱顺序
        question_ids = list(questions_map.keys())
        random.shuffle(question_ids)

        answers = {}
        total_questions = len(question_ids)

        for i, question_id in enumerate(question_ids, 1):
            question = questions_map[question_id]
            while True:
                try:
                    print(f"\n[{i}/{total_questions}] {question['content']}")
                    score = int(input("您的评分（1-6）: "))
                    if 1 <= score <= 6:
                        answers[question_id] = score
                        break
                    else:
                        print("请输入1-6之间的数字！")
                except ValueError:
                    print("请输入有效的数字！")

        return self.calculate_scores(answers)

    def calculate_scores(self, answers):
        """计算各维度得分"""
        results = {}
        factor_questions = {"外向性": [], "神经质": [], "严谨性": [], "开放性": [], "宜人性": []}

        # 将题目按因子分类
        for q in self.questions:
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

    def get_factor_description(self, factor):
        """获取因子的详细描述"""
        return self.factors[factor]


def main():
    test = BigFiveTest()
    results = test.run_test()

    print("\n测试结果：")
    print("=" * 50)
    for factor, data in results.items():
        print(f"\n{factor}:")
        print(f"平均分: {data['得分']} (总分: {data['总分']}, 题目数: {data['题目数']})")
        print("-" * 30)
        description = test.get_factor_description(factor)
        print("维度说明:", description["description"][:100] + "...")
        print("\n特征词:", ", ".join(description["trait_words"]))
    print("=" * 50)


if __name__ == "__main__":
    main()
