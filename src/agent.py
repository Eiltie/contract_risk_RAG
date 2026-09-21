import json

from rag import generate_answer, rag_search
from state import AgentState


def retrieve_node(state: AgentState) -> dict:
    """节点①：检索。调用三层检索找相似案例，把结果写进状态。"""
    print("\n[检索节点] 正在用三层检索找案例...")

    cases = rag_search(state["question"], top_k=3)

    # 只保留生成结论需要的几个字段，转成 JSON 字符串存进状态
    # final_score（精排分）也要留着：生成时要靠它判断这条依据有多硬，前端也靠它显示相关度
    simple_cases = [
        {
            "id": c["id"],
            "title": c["title"],
            "risk_type": c["risk_type"],
            "risk_level": c["risk_level"],
            "final_score": c.get("final_score"),
            "analysis": c["analysis"],
            "suggestion": c["suggestion"],
        }
        for c in cases
    ]
    return {"retrieved_cases": json.dumps(simple_cases, ensure_ascii=False, indent=2)}


def generate_node(state: AgentState) -> dict:
    """节点②：生成。读检索到的案例，让大模型写结论。"""
    print("[生成节点] 正在结合案例写结论...")

    cases = json.loads(state["retrieved_cases"])   # JSON 字符串 → 列表
    answer = generate_answer(state["question"], cases)
    return {"answer": answer}