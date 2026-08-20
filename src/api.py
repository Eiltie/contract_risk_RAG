# ============================================================
# api.py —— FastAPI 接口（后端 + 一个简单前端页面）
#
# 运行（任选其一）：
#   方式一： cd src && ..\venv\Scripts\uvicorn api:app --reload --port 8000
#   方式二： 在项目根目录执行  venv\Scripts\uvicorn src.api:app --reload --port 8000
#   方式三： python src\api.py
#
# 接口 / 页面：
#   GET  /            给普通用户看的页面（static/index.html）
#   GET  /health      健康检查
#   POST /chat        提交问题，返回 {question, answer, retrieved_cases}
# ============================================================
import json
import os
import sys

# 确保无论从哪个目录启动，都能 import 到本目录（src）下的 rag、built_graph 等
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from built_graph import build_graph
from rag import init_rag

# 启动时就把检索准备好（建索引、建/加载向量库），之后请求直接检索
init_rag()
_graph = build_graph()   # 图编译一次，全局复用

app = FastAPI(title="合同风险条款检索助手")

# 前端页面目录（项目根目录下的 static）
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    """返回给普通用户看的页面。"""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


class AskRequest(BaseModel):
    question: str   # 用户描述的风险情形


@app.post("/chat")
def chat(req: AskRequest):
    """问一句，返回最终答案 + 命中的案例。"""
    result = _graph.invoke({"question": req.question})
    return {
        "question": req.question,
        "answer": result["answer"],
        "retrieved_cases": json.loads(result["retrieved_cases"]),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
