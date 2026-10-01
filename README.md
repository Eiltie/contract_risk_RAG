# 合同风险条款检索分析助手

一个基于 **RAG（检索增强生成）+ LangGraph** 的合同风险分析助手：输入一句风险描述，系统自动完成「关键词检索 → 语义检索 → LLM 精排」三层召回，结合 40 条真实合同风险案例库，输出**风险判断、案例依据、处理建议**三段结论。

提供 **CLI 命令行** 与 **Web 服务** 两种使用方式。

---

## 功能特性

- **三层检索**：BM25 关键词检索（认字面）+ 向量语义检索（认意思）+ LLM 精排（认质量），兼顾精确匹配与语义泛化。
- **混合检索融合**：用 RRF（Reciprocal Rank Fusion）融合两路排名，解决「BM25 分与余弦相似度分不可直接比较」的问题。
- **门槛过滤**：LLM 精排后按阈值丢弃弱相关/不相关案例，实现「宁缺毋滥」。
- **双模型分工**：DeepSeek 负责精排打分与结论生成，智谱 embedding 负责文本向量化。
- **案例库**：40 条真实合同风险案例，覆盖 12 类合同，每条含条款原文、风险类型、风险等级、法条依据分析、修改建议。

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
- **页面**：原生 HTML / CSS / JS（`static/`），不引入前端框架与构建工具
- **流式输出**：`/chat/stream` 用 SSE 推送——检索一结束先把命中的案例推给前端，再逐字推生成的结论

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
│   ├── total_prompts.py   # Prompt 模板（精排 / 生成）
│   └── eval.py            # 检索质量评估（不参与线上运行，手动跑，见下文）
├── static/                # 前端页面（不参与 Python 逻辑，由 api.py 以静态文件发出去）
│   ├── page.html          # 页面结构 + 交互逻辑（JS 内联在文件末尾）
│   └── style.css          # 样式
├── data/
│   ├── risk_cases/cases.json   # 案例库（40 条）
│   ├── eval/questions.json     # 评估集（60 道题，给 eval.py 判卷用）
│   └── chroma_db/              # 向量库（首次运行自动生成）
├── package/
│   └── requirements.txt   # 依赖
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
pip install -r package\requirements.txt
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
# 方式 A：python -m uvicorn 启动（可加 --reload，改代码后自动重启）
venv\Scripts\python -m uvicorn src.api:app --reload --port 8000

# 方式 B：直接运行脚本
venv\Scripts\python src\api.py
```

浏览器打开 <http://127.0.0.1:8000>，在输入框描述风险情形，点击「分析风险」。

> **⚠️ 别用 `venv\Scripts\uvicorn`**：本项目目录曾改过名，venv 里 pip 生成的那批 `.exe` 启动器
> （`uvicorn.exe`、`pip.exe` 等共 28 个）内部把解释器路径写死了，指向改名前的旧地址，
> 启动时会**静默失败、一个字都不报**。凡是要用这类命令，一律改成 `python -m 模块名` 的写法，
> 比如装包用 `venv\Scripts\python -m pip install xxx`。

> **端口被占用**：如果启动时报 `[Errno 10048]`，说明 8000 端口已经被别的程序占着。
> 换个端口即可：`venv\Scripts\python -m uvicorn src.api:app --port 8001`。

---

## 接口说明

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 返回前端页面（`static/page.html`） |
| GET | `/static/*` | 前端静态文件（`style.css` 等） |
| GET | `/health` | 健康检查，返回 `{"status": "ok"}` |
| POST | `/chat` | 等结论写完，一次性返回答案与命中案例 |
| POST | `/chat/stream` | 流式返回：先推案例，再逐字推结论（SSE） |

`POST /chat` 请求示例：

```json
{ "question": "乙方要我们承担无限赔偿责任" }
```

响应：

```json
{
  "question": "乙方要我们承担无限赔偿责任",
  "answer": "…（风险判断 / 案例依据 / 处理建议 三段）…",
  "retrieved_cases": [ { "id": "C003", "title": "…", "risk_type": "…", "risk_level": "…", "final_score": 10, "analysis": "…", "suggestion": "…" } ]
}
```

### `POST /chat/stream`（流式）

请求体和 `/chat` 完全一样。响应是 SSE（`text/event-stream`）：一条消息以空行结尾，
内容是 `event: 事件名` 加上 `data: JSON`。一共有四种事件：

| 事件 | data | 什么时候发 |
|------|------|-----------|
| `cases` | 命中的案例数组 | 检索（含精排）一结束，立刻发 |
| `token` | 一小段文字 | 生成过程中反复发，前端按顺序拼起来就是完整结论 |
| `done` | `{}` | 全部结束 |
| `error` | `{"message": "…"}` | 中途出错（比如模型接口挂了） |

**为什么分成两段推**：检索里的「精排」必须等大模型把整张打分表吐完，程序才能解析、
排序、过门槛，所以这一步天生流不了，而且它是整个请求里最慢的一段（实测约 6 秒，
占总耗时的三分之二）。既然躲不掉，就让它别白等——精排一结束先把案例推给前端，
用户马上有东西看，而不是盯着空白干等到结论也写完。

前端（`static/page.html`）用 `fetch` 读响应流并自己拆 SSE，没有用浏览器原生的
`EventSource`——因为 `EventSource` 只支持 GET，而这里得把问题 POST 过去。

---

## 案例库说明

案例库位于 `data/risk_cases/cases.json`，当前含 40 条案例，覆盖劳动合同（9 条）、通用条款（4 条）以及采购、销售、服务、租赁、民间借贷、担保、买卖、建设工程、技术、居间等共 12 类合同，每类 1–3 条。

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
| 召回宽度 `RECALL_K` | `src/rag.py` | 关键词、语义两路各召回多少条候选，默认 `10` |

---

## 检索质量评估

调参数（改召回宽度、改门槛、换 embedding 模型、往案例库里加案例）之后，别凭感觉判断好坏，跑一遍评估集看数字。

```bash
venv\Scripts\python src\eval.py
```

评估集在 `data/eval/questions.json`，共 60 道题，每题带一个 `level`：

- `basic`（40 道）：用词接近条款原文，属于"好搜"的问题
- `hard`（15 道）：纯大白话、不带专业术语，模仿真实用户随口问
- `negative`（5 道）：案例库里故意没有对应案例，看系统会不会硬凑一条出来

打印的指标：

| 指标 | 含义 |
|------|------|
| Recall@1 | 期望案例正好排第 1 名的比例 |
| Recall@K | 期望案例出现在返回的前 3 条里（找得到就行） |
| MRR | 名次倒数的平均值，排第 1 得 1 分、第 2 得 0.5 分，衡量"排得好不好" |
| 拒答正确率 | 负向题里一条都没返回的比例，衡量"会不会硬凑" |

`basic` 和 `hard` 分开统计，两组的差距就是"用户换个说法会掉多少分"。

> 注意：跑一次每条题都要真实调用一次 embedding + 一次精排，**是花钱的**，别频繁空跑。
