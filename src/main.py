import json
import sys

from built_graph import build_graph
from rag import init_rag

def run():
    """交互式：你问一句，agent 走一遍图，答一句；输入 exit 退出。"""
    app = build_graph()

    print("=" * 50)
    print("合同风险条款检索助手（输入 exit 退出）")
    print("=" * 50)

    # 启动时就把检索准备好（建索引、建/加载向量库），
    # 用户输入后直接检索，不再临时建库。
    init_rag()

    while True:
        query = input("\n你：").strip()

        if query.lower() in ("exit", "quit", "退出", "q"):
            print("再见。")
            break

        if not query:
            continue

        # 把问题丢进图，走一遍「检索 → 生成」
        result = app.invoke({"question": query})

        # 打印检索节点找到的案例
        print("\n【检索节点】找到的案例")
        cases = json.loads(result["retrieved_cases"])

        if not cases:
            print("  （案例库中没有找到相近的案例，下面的回答是通用建议）")

        for i, c in enumerate(cases, 1):
            score = c.get("final_score")
            score_text = f"  相关度 {score}/10" if score is not None else ""
            print(f"  {i}. {c['title']}（{c['id']}）{score_text}")

        # 打印生成节点给出的结论
        print("\n【生成节点】回答")
        
        for line in result["answer"].split("\n"):
            print("  " + line)


if __name__ == "__main__":
    # Windows 控制台默认 GBK，强制 UTF-8 避免中文乱码
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    run()
