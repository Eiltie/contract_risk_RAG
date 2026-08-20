from typing import TypedDict


class AgentState(TypedDict):
    """状态字典：节点之间靠它传数据。每个节点只改自己负责的字段。"""
    question: str           # 用户问题
    retrieved_cases: str    # 检索到的案例（JSON 字符串）
    answer: str             # 最终答案