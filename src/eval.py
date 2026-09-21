"""RAG 检索质量评估：把评估集跑一遍，看三层检索到底找得准不准。

用法（在项目根目录执行）：
    venv\\Scripts\\python src\\eval.py

评估集在 data/eval/questions.json，每条带一个 level：
- basic    基础题，用词比较接近条款本身
- hard     难例题，纯大白话、不带专业术语，像真实用户随口问的
- negative 案例库里故意没有的问题，看系统会不会硬凑一条出来

指标：
- Recall@1 / Recall@K：期望案例排在 第1名 / 前K名 的比例（找得准不准）
- MRR：期望案例名次的倒数平均，排第1得1分、第2得0.5分、第3得0.33分（排得好不好）
- 拒答正确率：负向题里一条都没返回的比例（会不会硬凑）

basic 和 hard 分开统计 —— 两组的差距就是"用词一变，检索掉多少分"。

注意：每跑一次都要真实调用大模型（每条一次向量化 + 一次精排），是花钱的，别频繁空跑。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag import init_rag, rag_search  # noqa: E402

# 路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_FILE = os.path.join(BASE_DIR, "data", "eval", "questions.json")

# 取前几条，跟线上保持一致（retrieve_node 调 rag_search 用的就是 3）
TOP_K = 3


def run_positive_group(title: str, items: list) -> tuple:
    """跑一组正向题（库里确实有答案的），打印每条结果，返回 (Recall@1, Recall@K, MRR)。"""
    print(f"\n【{title}】共 {len(items)} 条")
    print("-" * 72)

    hit1 = 0
    hit_k = 0
    mrr_sum = 0.0
    for q in items:
        cases = rag_search(q["question"], top_k=TOP_K)
        ids = [c["id"] for c in cases]
        expected = q["expected_id"]

        if expected in ids:
            rank = ids.index(expected) + 1
            hit_k += 1
            mrr_sum += 1.0 / rank
            if rank == 1:
                hit1 += 1
            mark = f"命中，排第 {rank}"
        else:
            mark = f"没命中（实际返回：{ids if ids else '空'}）"
        print(f"{q['id']}  {mark}\n     {q['question']}")

    n = len(items)
    return hit1 / n, hit_k / n, mrr_sum / n


def run_negative_group(items: list) -> float:
    """跑负向题：案例库里本来就没有，系统应该一条都不给（宁可说没有，也不能硬凑）。"""
    print(f"\n【负向题：案例库里故意没有的】共 {len(items)} 条")
    print("-" * 72)

    refused = 0
    for q in items:
        cases = rag_search(q["question"], top_k=TOP_K)
        if cases:
            detail = "误召回：" + "、".join(f"{c['id']}({c['final_score']}分)" for c in cases)
        else:
            detail = "正确拒答"
            refused += 1
        print(f"{q['id']}  {detail}\n     {q['question']}")

    return refused / len(items)


def main():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    init_rag()

    basic = [q for q in questions if q["level"] == "basic"]
    hard = [q for q in questions if q["level"] == "hard"]
    negative = [q for q in questions if q["level"] == "negative"]

    print(f"\n评估集共 {len(questions)} 条：基础 {len(basic)} / 难例 {len(hard)} / 负向 {len(negative)}，检索取前 {TOP_K} 条")

    b1, bk, bmrr = run_positive_group("基础题（用词接近条款）", basic)
    h1, hk, hmrr = run_positive_group("难例题（纯大白话、无术语）", hard)
    refuse = run_negative_group(negative)

    print("\n" + "=" * 72)
    print("汇总")
    print("=" * 72)
    print(f"基础题    Recall@1 = {b1:.2f}   Recall@{TOP_K} = {bk:.2f}   MRR = {bmrr:.3f}")
    print(f"难例题    Recall@1 = {h1:.2f}   Recall@{TOP_K} = {hk:.2f}   MRR = {hmrr:.3f}")
    print(f"负向题    拒答正确率 = {refuse:.2f}")


if __name__ == "__main__":
    # Windows 控制台默认 GBK，强制 UTF-8 避免中文乱码
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    main()
