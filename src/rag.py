import json
import os
import chromadb
import jieba
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from rank_bm25 import BM25Okapi
from llm import get_chat_model, get_embed_model
from total_prompts import ANSWER_PROMPT, NO_HIT_PROMPT, RERANK_PROMPT

# 路径
# 本文件在 src/ 下，所以 BASE_DIR 要取上一级才是项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES_FILE = os.path.join(BASE_DIR, "data", "risk_cases", "cases.json")   # 案例库
VECTOR_DB_DIR = os.path.join(BASE_DIR, "data", "chroma_db")                # 向量库

load_dotenv(os.path.join(BASE_DIR, ".env"))  # 读 .env 里的 API Key


# 精排分门槛：大模型精排分低于这个值的案例不返回（宁缺毋滥）。
# 对应 RERANK_PROMPT 的评分标准：6-8 比较相关，3-5 弱相关，0-2 不相关。
MIN_FINAL_SCORE = 6.0

# 每一路（关键词/语义）各召回多少条候选，交给精排去挑。
# 为什么不止取三五条：用户说大白话时，正确答案经常排得很靠后
# （实测 15 条口语化难例题里，9 条的正确答案在 5 名开外），召回太少精排根本见不到它。
# 实测 10 条的性价比最高：K=5 只能覆盖 6/15 条，K=10 覆盖 11/15，再往上收益就很小了。
RECALL_K = 10


# 读案例库 

def load_cases() -> list[dict]:
    """从 json 文件读入全部历史案例，返回一个列表。"""
    with open(CASES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def make_case_text(case: dict) -> str:
    """把一条案例拼成一段可检索的文本。

    关键词故意写两遍，是为了在「关键词检索」里给关键词更高权重，
    这样搜关键词时这条案例更容易排前面。
    """
    title = case["title"]
    risk_type = case["risk_type"]
    keywords = " ".join(case["keywords"])          # 关键词用空格连起来
    clause = case["clause_text"]                   # 条款原文
    return f"{title} {risk_type} {keywords} {keywords} {clause}"

# 索引：只建一次，用全局变量记住（None 表示还没建）

_cases = None          # 全部案例
_by_id = None          # id → 案例，方便按 id 查
_bm25 = None           # 关键词索引（第一层用）
_embed_model = None    # 向量模型
_collection = None     # 向量库（第二层用）
_rerank_llm = None     # 精排用的大模型
_answer_llm = None     # 生成结论用的大模型


def _ensure_ready():
    """确保索引都建好了。第一次调用才真正建，之后直接复用。"""
    global _cases, _by_id, _bm25, _embed_model, _collection, _rerank_llm, _answer_llm

    if _bm25 is not None:
        return   # 已经建过，跳过

    print("初始化检索...")
    _cases = load_cases()
    _by_id = {c["id"]: c for c in _cases}
    _embed_model = get_embed_model()
    _rerank_llm = get_chat_model(temperature=0.0)   # 打分要稳定，温度 0
    _answer_llm = get_chat_model(temperature=0.3)   # 写结论，温度稍高更自然

    # 建关键词索引：把每条案例先分词，再交给 BM25 统计词频
    docs = [jieba.lcut(make_case_text(c)) for c in _cases]
    _bm25 = BM25Okapi(docs)

    # 建/加载 向量库
    _collection = _get_collection()

    print("检索就绪。\n")


def init_rag():
    """程序启动时调用一次：提前加载案例、建好关键词索引和向量库。

    这样用户输入后直接检索，不用再等第一次建库。
    幂等，重复调用也不会重复建（内部有判断）。
    """
    _ensure_ready()


def _get_collection():
    """拿到向量库。已经存在就直接加载，不存在才向量化（只做一次）。"""
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

    # 先看看向量库是不是已经建好了
    try:
        collection = client.get_collection("cases")

        if collection.count() == len(_cases):   # 数量对得上，说明是最新的
            print("检测到已存在的向量库，直接加载（不重复向量化）")
            return collection
        # 数量对不上（案例库改过），删掉重建
        else:
            client.delete_collection("cases")

    except Exception:
        pass   # 没找到，说明还没建过，走下面新建

    # 走到这里说明要新建：把每条案例逐条向量化（条数跟着案例库走，别写死）
    collection = client.create_collection(name="cases")

    print(f"向量库不存在，正在向量化 {len(_cases)} 条案例...")
    for c in _cases:
        vec = _embed_model.embed_query(make_case_text(c))
        collection.add(
            ids=[c["id"]],
            embeddings=[vec],
            metadatas=[{"title": c["title"], "risk_type": c["risk_type"]}],
        )
    print("向量化完成。")
    return collection

# RAG 三层检索（全是函数）

def keyword_search(query: str, top_k: int = 3) -> list[dict]:
    """① 第一层：关键词检索，认「字面」。

    把查询切词，去每个案例里数这些词出现多少次，出现越多分越高。
    返回最像的 top_k 条案例，每条带 keyword_score。
    """
    _ensure_ready()

    words = jieba.lcut(query)               # 查询先分词
    scores = _bm25.get_scores(words)        # 每个案例得一个分

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    result = []
    for i, score in ranked[:top_k]:
        if score <= 0:          # 0 分 = 一个词都没匹配上，跳过
            continue
        c = _cases[i].copy()
        c["keyword_score"] = round(score, 4)
        result.append(c)
    return result


def semantic_search(query: str, top_k: int = 3) -> list[dict]:
    """② 第二层：语义检索，认「意思」。

    把查询变成向量，去向量库里找距离最近的（距离越小越像）。
    返回最像的 top_k 条案例，每条带 semantic_score（0~1，越大越像）。
    """
    _ensure_ready()

    query_vec = _embed_model.embed_query(query)   # 查询 → 向量
    res = _collection.query(query_embeddings=[query_vec], n_results=top_k)

    ids = res["ids"][0]          # 命中的案例 id
    dists = res["distances"][0]  # 对应的距离（越小越像）

    result = []
    for cid, dist in zip(ids, dists):
        c = _by_id[cid].copy()
        c["semantic_score"] = round(1.0 - dist, 4)   # 距离越小 → 相似度越高
        result.append(c)
    return result


def merge_rankings(list_of_results):
    """③ 第三层前半：合并排名。

    把关键词、向量两路的结果，合成一个总排名。
    规则：
    - 不看两路各自的原始分数（关键词分和向量分数值不一样，没法比）
    - 只看「名次」：第1名加 1/61 分，第2名加 1/62 分……名次越靠前加得越多
    - 同一个案例两路都出现，分就叠加，所以两路都靠前的案例总分最高
    返回 {案例id: 合并分}，分越高代表越该排前面。
    """
    k = 60
    total = {}                        # 案例id -> 合并分

    for one_road in list_of_results:  # 关键词一路、向量一路，各走一遍
        rank = 1                      # 名次，从第1名开始
        for case in one_road:         # 这一路里逐条看（已经是按分数排好序的）
            cid = case["id"]
            if cid not in total:      # 这个案例第一次出现，先记 0 分
                total[cid] = 0
            total[cid] += 1.0 / (k + rank)   # 按名次加分
            rank += 1                 # 名次往后挪

    return total


def _pull_scores(text: str, pos: int):
    """从 text 的 pos 位置往后找完整的 {"id": ..., "score": ...}，抠一条算一条。

    这是给「流式精排」配的小工具。模型还在往外吐的时候，整段文本是不完整的，
    JSON 随时可能被切在两段之间（比如这一块收到的是 `{"id": "C0`，
    下一块才是 `12", "score": 8}`），所以没法直接 json.loads 整段，
    只能一小段一小段地抠：找到一个 { 再找到配对的 }，中间那段拿来试解析。

    返回 (这一轮抠出来的 [(id, 分数), ...], 下次接着扫的位置)。
    位置必须带出去（不能每块都从头扫），不然同一条分数会被反复上报。
    """
    found = []
    while True:
        start = text.find("{", pos)
        if start < 0:            # 后面没有 { 了，下次从末尾接着扫
            return found, len(text)

        end = text.find("}", start)
        if end < 0:              # 这个 { 还没等到它配对的 }，先停在这，等后面的内容
            return found, start

        try:
            item = json.loads(text[start:end + 1])
            cid, score = item["id"], float(item["score"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pos = start + 1      # 不是一条合法分数（比如模型废话里带个 {），跳过继续找
            continue

        pos = end + 1
        found.append((cid, score))


def rerank_stream(query: str, candidates: list[dict]):
    """③ 第三层后半：精排（流式版）。跟 rerank 排出来的结果一模一样，只是边打分边往外报。

    每从模型嘴里抠出一条分数，就 yield 一个 (案例id, 分数) 出去 ——
    调用方拿去立刻推给前端，用户看着分数一条条亮，不用干等十几秒才知道打了什么分。

    函数跑完时，candidates 会被写上 final_score 并排好序（跟 rerank 完全一致）。
    """
    _ensure_ready()

    if len(candidates) <= 1:        # 只有一个候选，没排序意义
        for c in candidates:
            c["final_score"] = None   # 未打分，交给 rag_search 直接返回
        return

    # 把候选整理成清单给大模型看
    lines = [f"  [{c['id']}] {c['title']}（{c['risk_type']}）" for c in candidates]
    user = f"查询：{query}\n\n候选案例：\n" + "\n".join(lines) + "\n\n请对每个候选案例打分。"

    messages = [
        SystemMessage(content=RERANK_PROMPT),
        HumanMessage(content=user),
    ]

    # ① 边收边报：模型吐一块，我们抠一块，抠到一条就 yield 一条
    #    （这一步纯粹是「报进度」，不参与最终结果，抠漏了也没关系）
    buf = ""
    pos = 0
    for chunk in _rerank_llm.stream(messages):
        buf += chunk.content
        got, pos = _pull_scores(buf, pos)
        for cid, score in got:
            yield cid, score

    # ② 收完了，走原来那套完整解析，拿「权威结果」来排序
    #    （上面边收边抠的只是给用户看的进度，最终名次以这里为准）
    content = buf.strip()

    # 大模型偶尔把 JSON 包在 ``` 里，先剥掉
    if content.startswith("```"):
        content = content.strip("`")

    try:
        score_map = {item["id"]: item["score"] for item in json.loads(content)}
    except (json.JSONDecodeError, KeyError, TypeError):
        # 解析失败：标记未打分，退回按合并分排序（门槛过滤会跳过）
        for c in candidates:
            c["final_score"] = None
        candidates.sort(key=lambda c: c["merged_score"], reverse=True)
        return

    for c in candidates:
        c["final_score"] = score_map.get(c["id"], 0.0)
    candidates.sort(key=lambda c: c["final_score"], reverse=True)


def rerank(query: str, candidates: list[dict]) -> list[dict]:
    """精排：让大模型给候选案例打分，排出最终顺序。

    内部就是跑一遍流式版、把中间的分数事件丢掉 ——
    老接口（rag_search / 命令行 / 评估脚本）不关心过程，等最终结果就行。
    """
    for _cid, _score in rerank_stream(query, candidates):
        pass
    return candidates


def _recall(query: str) -> list[dict]:
    """第 1、2 层 + 第 3 层前半：两路召回 → 合并名次 → 组装候选池。

    只负责「找出候选」，不精排、不截断 —— 精排那一步最慢，
    单独拎出来是为了让流式接口能先把这批候选报出去，用户不用干等。
    """
    # 第 1、2 层：两路各自召回（各取前 RECALL_K 条，候选池大一些，精排才有得挑）
    by_keyword = keyword_search(query, top_k=RECALL_K)
    by_semantic = semantic_search(query, top_k=RECALL_K)

    # 第 3 层前半：合并两路排名
    merged = merge_rankings([by_keyword, by_semantic])
    ranked_ids = sorted(merged, key=lambda cid: merged[cid], reverse=True)

    candidates = []
    for cid in ranked_ids:
        c = _by_id[cid].copy()
        c["merged_score"] = round(merged[cid], 6)
        candidates.append(c)
    return candidates


def _finalize(candidates: list[dict], top_k: int) -> list[dict]:
    """精排之后的收尾：过门槛 + 截断成 top_k。两个检索入口共用这一段，保证结果一致。"""
    # 门槛过滤：精排成功后，把弱相关/不相关（分低于门槛）的踢掉，宁缺毋滥
    if candidates and all(c.get("final_score") is not None for c in candidates):
        candidates = [c for c in candidates if c["final_score"] >= MIN_FINAL_SCORE]

    return candidates[:top_k]


def rag_search(query: str, top_k: int = 3) -> list[dict]:
    """RAG 三层检索：关键词 → 语义 → 合并精排，一步走完，返回最终案例列表。

    这就是给「检索节点」调用的那个 function。
    """
    _ensure_ready()

    candidates = _recall(query)              # 前两层召回 + 第三层前半合并
    candidates = rerank(query, candidates)   # 第三层后半：精排
    return _finalize(candidates, top_k)      # 过门槛、截断


def rag_search_stream(query: str, top_k: int = 3):
    """RAG 三层检索（流式版）：跑的东西跟 rag_search 一模一样，区别是「边跑边汇报」。

    每走一步就 yield 一个 (事件名, 数据) 出去，前端拿到就能立刻更新页面：

      ("stage",      "正在…")          —— 当前在干嘛，显示在状态栏
      ("candidates", [ {id, title, …}]) —— 候选预览：精排还没开始，先把待打分的案例摆出来
      ("score",      {"id":…, "score":…}) —— 精排每打出一条分数，点亮对应卡片
      ("cases",      [ … ])             —— 最终案例，内容跟 rag_search 的返回完全一致

    为什么能先报候选：两路召回 + 合并只花不到一秒，候选池这时候就已经定了；
    慢的是精排，所以先把候选摆出去，让用户在那十几秒里有东西可看。
    """
    _ensure_ready()

    yield "stage", "正在检索案例库…"
    candidates = _recall(query)

    # 候选预览：只报「一会儿要打分的都有谁」，分数还没算出来
    yield "candidates", [
        {"id": c["id"], "title": c["title"], "risk_type": c["risk_type"]}
        for c in candidates
    ]

    yield "stage", f"正在给 {len(candidates)} 条候选逐条打分排序…"
    for cid, score in rerank_stream(query, candidates):
        yield "score", {"id": cid, "score": score}

    yield "cases", to_simple_cases(_finalize(candidates, top_k))


def to_simple_cases(cases: list[dict]) -> list[dict]:
    """把检索结果裁成「生成结论 + 前端展示」用得着的那几个字段。

    为什么单独抽出来：流式接口和老接口都要裁，
    共用这一处才能保证两边裁得一模一样，不会出现「网页少显示一个字段」这种对不上的问题。

    final_score（精排分）要留着：生成时要靠它判断这条依据有多硬，前端也靠它显示相关度。
    """
    return [
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


def _build_answer_messages(query: str, cases: list[dict]) -> list:
    """把「用户问题 + 检索到的案例」拼成给大模型的消息列表。

    分两种走法：
    - 检索到了案例：把案例（连精排分一起）交给大模型，让它总结成三段结论；
    - 一条都没检索到：换 NO_HIT_PROMPT，要求它如实说"没有案例依据"，
      只给不依赖案例库的通用建议 —— 不然它会用自己脑子里的知识硬凑一段"案例依据"出来。

    为什么单独抽出来：写结论有两条路（一次性返回 / 流式逐字返回），
    两条路必须用一模一样的提示词，所以拼消息这段共用 —— 以后改提示词只用改这一处。
    """
    # 无命中：案例库覆盖不到这个问题，宁可说没有，也不能装作有依据
    if not cases:
        return [
            SystemMessage(content=NO_HIT_PROMPT),
            HumanMessage(content=f"用户描述：{query}"),
        ]

    parts = []
    for c in cases:
        score = c.get("final_score")
        # 把精排分一起给大模型看，它才知道这条依据有多硬
        score_text = f"，相关度 {score}/10" if score is not None else ""
        parts.append(
            f"[{c['id']}] {c['title']}（{c['risk_type']}{score_text}）\n"
            f"  分析：{c['analysis']}\n"
            f"  建议：{c['suggestion']}"
        )
    case_text = "\n".join(parts)

    user = f"用户描述：{query}\n\n历史相似案例：\n{case_text}"
    return [
        SystemMessage(content=ANSWER_PROMPT),
        HumanMessage(content=user),
    ]


def generate_answer(query: str, cases: list[dict]) -> str:
    """最后一步（RAG 的 G）：拿「问题 + 检索到的案例」让大模型写结论。

    这是「一次性拿完整答案」的版本：命令行版和 /chat 接口用它。
    invoke 会一直等到整段结论写完才返回。
    """
    _ensure_ready()

    resp = _answer_llm.invoke(_build_answer_messages(query, cases))
    return resp.content.strip()


def generate_answer_stream(query: str, cases: list[dict]):
    """同上，但「一边写一边往外吐」（网页的流式接口用它）。

    跟 generate_answer 的区别只有一个：把 invoke 换成 stream。
    invoke 等整段写完一次性返回；stream 每生成一小块就 yield 一块，
    调用方拿去立刻发给浏览器，用户就能看着字一个一个往外冒。
    """
    _ensure_ready()

    for chunk in _answer_llm.stream(_build_answer_messages(query, cases)):
        piece = chunk.content
        if piece:          # 有些分块是空的（只带元信息没有正文），跳过
            yield piece
