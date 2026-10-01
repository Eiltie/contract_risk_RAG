import json

from rag import generate_answer, rag_search, to_simple_cases
from state import AgentState


def retrieve_node(state: AgentState) -> dict:
    """节点①：检索。调用三层检索找相似案例，把结果写进状态。"""
    print("\n[检索节点] 正在用三层检索找案例...")

    cases = rag_search(state["question"], top_k=3)

    # 裁成生成结论需要的几个字段，转成 JSON 字符串存进状态。
    # 裁剪规则放在 rag.to_simple_cases —— 流式接口走的是同一个函数，
    # 两边字段必须一致，不然会出现「网页和命令行显示不一样」的问题。
    return {"retrieved_cases": json.dumps(to_simple_cases(cases), ensure_ascii=False, indent=2)}


def generate_node(state: AgentState) -> dict:
    """节点②：生成。读检索到的案例，让大模型写结论。"""
    print("[生成节点] 正在结合案例写结论...")

    cases = json.loads(state["retrieved_cases"])   # JSON 字符串 → 列表
    answer = generate_answer(state["question"], cases)
    return {"answer": answer}