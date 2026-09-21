import json
import os
import sys

# 确保无论从哪个目录启动，都能 import 到本目录（src）下的 rag、built_graph 等
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from built_graph import build_graph
from rag import init_rag

# 启动时就把检索准备好（建索引、建/加载向量库），之后请求直接检索
init_rag()
_graph = build_graph()   # 图编译一次，全局复用

app = FastAPI(title="合同风险条款检索助手")

# 极简调试页：只为了能在浏览器里手动试一下 /chat，不做样式美化，也不依赖外部页面文件。
# 渲染一律用 textContent 填内容，不拼 HTML 字符串 —— 大模型的输出不会被当成 HTML 执行。
PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>合同风险条款检索助手</title>
</head>
<body>
<h3>合同风险条款检索助手</h3>
<textarea id="q" rows="3" cols="60" placeholder="例如：乙方要我们承担无限赔偿责任"></textarea>
<br>
<button id="go">分析风险</button>
<span id="status"></span>
<div id="out"></div>
<script>
var out = document.getElementById('out');
var status = document.getElementById('status');
var q = document.getElementById('q');

function line(text, bold) {
  var d = document.createElement('div');
  d.textContent = text;
  if (bold) { d.style.fontWeight = 'bold'; }
  return d;
}

function render(data) {
  out.innerHTML = '';                        // 清空上一次的结果
  var pre = document.createElement('pre');   // pre 保留大模型输出的换行
  pre.textContent = data.answer || '（无结果）';
  out.appendChild(pre);

  out.appendChild(line('参考案例：', true));
  var cases = data.retrieved_cases || [];
  if (!cases.length) {
    out.appendChild(line('案例库中没有找到相近的案例，上面的回答是通用建议。'));
    return;
  }
  cases.forEach(function (c) {
    var score = (c.final_score === null || c.final_score === undefined) ? '' : '  相关度 ' + c.final_score + '/10';
    out.appendChild(line('[' + c.id + '] ' + c.title + '（' + c.risk_type + ' / ' + c.risk_level + '）' + score, true));
    out.appendChild(line('分析：' + c.analysis));
    out.appendChild(line('建议：' + c.suggestion));
  });
}

document.getElementById('go').onclick = function () {
  var text = q.value.trim();
  if (!text) { q.focus(); return; }

  status.textContent = '检索分析中...';
  fetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question: text })
  }).then(function (resp) {
    if (!resp.ok) { throw new Error('服务返回错误：' + resp.status); }
    return resp.json();
  }).then(function (data) {
    status.textContent = '';
    render(data);
  }).catch(function (e) {
    status.textContent = '';
    out.innerHTML = '';
    out.appendChild(line('出错了：' + e.message));
  });
};
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    """返回内置的极简调试页面。"""
    return PAGE


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
