import json
import os
import sys

# 确保无论从哪个目录启动，都能 import 到本目录（src）下的 rag、built_graph 等
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import retrieve_node
from built_graph import build_graph
from rag import generate_answer_stream, init_rag

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
    """流式版：先推检索到的案例，再一个字一个字推生成的结论。

    为什么要分两段推：
    检索里的「精排」必须等大模型把整张打分表吐完，程序才能解析、排序、过门槛，
    所以这一步天生流不了，也是整个请求里最慢的一段。既然躲不掉，
    就让它别白等——精排一结束就把案例先推给前端，用户马上有东西看，
    而不是盯着空白干等到结论也写完。
    """
    def gen():
        try:
            # 第 1 段：检索。
            # 直接复用图里那个检索节点，这样字段筛选逻辑跟 /chat 完全一致，
            # 不会出现「流式接口少返回一个字段」这种两边对不上的问题。
            state = retrieve_node({"question": req.question})
            cases = json.loads(state["retrieved_cases"])
            yield _sse("cases", cases)

            # 第 2 段：逐字推结论
            for piece in generate_answer_stream(req.question, cases):
                yield _sse("token", piece)

            yield _sse("done", {})

        except Exception as e:
            # 出错也得告诉前端一声，不然页面会一直转圈傻等
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
