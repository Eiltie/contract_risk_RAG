# 合同风险条款检索分析助手

一个基于 **RAG（检索增强生成）+ LangGraph** 的合同风险分析助手：输入一句风险描述，系统自动完成「关键词检索 → 语义检索 → LLM 精排」三层召回，结合 40 条真实合同风险案例库，输出**风险判断、案例依据、处理建议**三段结论。

提供 **CLI 命令行** 与 **Web 页面** 两种交互方式。

---

## 功能特性

- **三层检索**：BM25 关键词检索（认字面）+ 向量语义检索（认意思）+ LLM 精排（认质量），兼顾精确匹配与语义泛化。
- **混合检索融合**：用 RRF（Reciprocal Rank Fusion）融合两路排名，解决「BM25 分与余弦相似度分不可直接比较」的问题。
- **门槛过滤**：LLM 精排后按阈值丢弃弱相关/不相关案例，实现「宁缺毋滥」。
- **双模型分工**：DeepSeek 负责精排打分与结论生成，智谱 embedding 负责文本向量化。
- **案例库**：40 条真实(并非真实真实就违法了xd)合同风险案例，覆盖 11 类合同，每条含条款原文、风险类型、风险等级、法条依据分析、修改建议。

---

## 检索流程（三层）

```
用户输入（风险描述）
      │
      ▼
  ① 关键词检索：jieba 分词 + BM25，召回字面匹配的案例
      │
      ▼
  ② 语义检索：embedding 向量化 + ChromaDB，召回意思相近的案例
      │
      ▼
  ③ 合并精排：RRF 融合两路排名 → LLM 逐条打分 → 门槛过滤
      │
      ▼
  生成节点：结合命中的案例，输出 风险判断 / 案例依据 / 处理建议
```

---

## 技术栈

- **编排框架**：LangGraph（StateGraph 状态图，检索 → 生成两个节点）
- **检索**：rank-bm25（BM25）、jieba（分词）、ChromaDB（持久化向量库）
- **模型**：DeepSeek（生成 + 精排）、智谱 embedding-3（向量化），经 LangChain 接入
- **服务**：FastAPI + Uvicorn
- **前端**：原生 HTML / CSS / JS

---

## 目录结构

```
.
├── src/
│   ├── main.py            # CLI 入口（交互式问答）
│   ├── api.py             # FastAPI 入口（Web 服务）
│   ├── built_graph.py     # LangGraph 图编排（检索→生成）
│   ├── agent.py           # 检索节点、生成节点
│   ├── rag.py             # 三层检索 + 答案生成核心逻辑
│   ├── llm.py             # 模型封装（DeepSeek / 智谱）
│   ├── state.py           # 状态定义（节点间数据传递）
│   └── total_prompts.py   # Prompt 模板（精排 / 生成）
├── data/
│   ├── risk_cases/cases.json   # 案例库（40 条）
│   └── chroma_db/              # 向量库（首次运行自动生成）
├── static/
│   └── index.html         # Web 前端页面
├── requirements.txt       # 依赖
├── .env                   # 密钥配置（需自行填写，见下文）
└── README.md
```

---

## 快速开始

### 1. 环境要求

- Python 3.9+
- 有效的 DeepSeek API Key 与智谱（Zhipu）API Key

### 2. 安装依赖

```bash
# 创建并激活虚拟环境（Windows）
python -m venv venv
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置 .env

在项目根目录创建 `.env` 文件，填入以下内容：

```ini
DEEPSEEK_API_KEY=你的DeepSeek密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash

ZHIPU_API_KEY=你的智谱密钥
```

> 向量模型固定使用智谱 `embedding-3`（在 `src/llm.py` 中配置，可通过 `ZHIPU_API_KEY` 使用）。

### 4. 运行

**方式一：CLI 命令行**

```bash
cd src
..\venv\Scripts\python main.py
```

输入问题后回车即可，输入 `exit` / `quit` / `退出` 退出。

**方式二：Web 服务**

在项目根目录执行（任选其一）：

```bash
# 方式 A：uvicorn 启动
venv\Scripts\uvicorn src.api:app --reload --port 8000

# 方式 B：直接运行
venv\Scripts\python src\api.py
```

浏览器打开 <http://127.0.0.1:8000>，在输入框描述风险情形，点击「分析风险」。

---

## 接口说明

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 返回前端页面 |
| GET | `/health` | 健康检查，返回 `{"status": "ok"}` |
| POST | `/chat` | 提交问题，返回答案与命中案例 |

`POST /chat` 请求示例：

```json
{ "question": "乙方要我们承担无限赔偿责任" }
```

响应：

```json
{
  "question": "乙方要我们承担无限赔偿责任",
  "answer": "…（风险判断 / 案例依据 / 处理建议 三段）…",
  "retrieved_cases": [ { "id": "C003", "title": "…", "risk_type": "…", "risk_level": "…", "analysis": "…", "suggestion": "…" } ]
}
```

---

## 案例库说明

案例库位于 `data/risk_cases/cases.json`，当前含 40 条案例，覆盖采购、销售、劳动、租赁、民间借贷、担保、买卖、建设工程、技术、居间、通用条款等 11 类合同。

每条案例字段：

| 字段 | 说明 |
|------|------|
| `id` | 案例唯一标识 |
| `category` | 合同类别 |
| `title` | 风险案例标题 |
| `clause_text` | 条款原文 |
| `risk_type` | 风险类型 |
| `risk_level` | 风险等级（high / medium / low） |
| `analysis` | 法条依据与风险分析 |
| `suggestion` | 修改建议 |
| `keywords` | 检索关键词 |

**新增案例**：直接在 `cases.json` 中按上述结构追加即可。首次运行（或案例数量变化后）程序会自动重建向量库，无需手动处理。

---

## 配置项说明

| 配置项 | 位置 | 说明 |
|------|------|------|
| 精排门槛 `MIN_FINAL_SCORE` | `src/rag.py` | 精排分低于此值的案例被过滤，默认 `6.0` |
| 精排温度 `0.0` | `src/rag.py` | 打分需稳定，故温度置 0 |
| 生成温度 `0.3` | `src/rag.py` | 写结论，温度稍高更自然 |
| RRF 参数 `k=60` | `src/rag.py`（`merge_rankings`） | 名次融合的平滑常数 |
