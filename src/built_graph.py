from langgraph.graph import END, StateGraph

from agent import generate_node, retrieve_node
from state import AgentState


def build_graph():
    """把节点和边拼成一张图，compile 成一个 agent。"""
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve_node)   # 节点① 检索
    graph.add_node("generate", generate_node)   # 节点② 生成

    graph.set_entry_point("retrieve")           # 入口：先检索
    graph.add_edge("retrieve", "generate")      # 检索完 → 生成
    graph.add_edge("generate", END)             # 生成完 → 结束

    return graph.compile()