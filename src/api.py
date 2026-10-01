import json
import os
import sys

# 确保无论从哪个目录启动，都能 import 到本目录（src）下的 rag、built_graph 等
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from built_graph import build_graph
from rag import generate_answer_stream, init_rag, rag_search_stream

# 路径：本文件在 src/ 下，上一级才是项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# 启动时就把检索准备好（建索引、建/加载向量库），之后请求直接检索
init_rag()
_graph = build_graph()   # 图编译一次，全局复用

app = FastAPI(title="合同风险条款检索助手")

# 前端用的静态文件（style.css 等）都从 static/ 目录发出去。
# 挂上之后，页面里写 /static/style.css 就能取到；
# 以后再加 page.js 之类的文件，丢进 static/ 就行，后端不用动。
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    """返回前端页面。"""
    return FileResponse(os.path.join(STATIC_DIR, "page.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


class AskRequest(BaseModel):
    question: str   # 用户描述的风险情形


@app.post("/chat")
def chat(req: AskRequest):
    """问一句，等结论写完，一次性返回答案 + 命中的案例。

    不走流式，适合脚本调用、接口调试。
    """
    result = _graph.invoke({"question": req.question})
    return {
        "question": req.question,
        "answer": result["answer"],
        "retrieved_cases": json.loads(result["retrieved_cases"]),
    }


def _sse(event: str, data) -> str:
    """把一条消息拼成 SSE（Server-Sent Events）格式。

    SSE 规定「一条消息以空行结尾」，所以 data 里绝对不能出现裸换行——
    这里统一先用 json.dumps 压成一行。ensure_ascii=False 是为了让中文
    原样发出去，别变成 \\uXXXX 一堆转义码（那样前端还得再解一次）。
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/chat/stream")
def chat_stream(req: AskRequest):
    """流式版：检索每走一步都实时往外推，最后再一个字一个字推结论。

    后端只负责「如实上报」每一步的进展和结果 —— 页面上先显示谁、怎么排，
    全部由前端决定（现在是「先结论、后依据」）。想换显示顺序只改前端。

    为什么要有流式：
    「精排」是整条链里最慢的一段（大模型要给十几二十条候选逐条打分），
    老做法是等它全跑完才吭声，用户点完按钮只能盯着一片空白。
    现在改成边走边报 —— 总时长没变，但页面有时间做各种反馈。

    这里不再自己扛分段逻辑，只干一件事：把 rag_search_stream 吐出的事件
    转成 SSE 格式发给前端。「哪一步报什么」都收在 rag.py 里，跟检索逻辑放一起。
    """
    def gen():
        try:
            for name, payload in rag_search_stream(req.question, top_k=3):
                yield _sse(name, payload)

                # 案例一算好就发出去（页面上什么时候画、按什么顺序画，前端说了算）
                if name == "cases":
                    for piece in generate_answer_stream(req.question, payload):
                        yield _sse("token", piece)

            yield _sse("done", {})

        except Exception as e:
            # 出错也得告诉前端一声，不然页面会一直转圈傻等
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
