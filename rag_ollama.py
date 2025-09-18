import os
import json
from typing import List, Dict, Optional, Any

from fastapi import FastAPI
from fastapi import UploadFile, File, Form
from fastapi import HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

import chromadb
from chromadb.config import Settings

from langchain_text_splitters import RecursiveCharacterTextSplitter

import ollama


EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "dengcao/Qwen3-Embedding-4B:Q5_K_M")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen2.5:7b-instruct-q5_K_M")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
CHROMA_DIR = os.environ.get("CHROMA_DIR", os.path.join(os.getcwd(), ".chroma"))
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "rag_default")
WARMUP_ENABLE = os.environ.get("WARMUP_ENABLE", "true").lower() == "true"
LOG_ENABLE = os.environ.get("LOG_ENABLE", "true").lower() == "true"
LOG_FILE = os.environ.get("LOG_FILE")  # 若设置则写入文件（JSON Lines）
HISTORY_ENABLE = os.environ.get("HISTORY_ENABLE", "true").lower() == "true"
HISTORY_FILE = os.environ.get("HISTORY_FILE", os.path.join(os.getcwd(), "history.json"))

# Ollama 生成参数（可调以减少延迟）
OLLAMA_NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", "512"))
OLLAMA_TEMPERATURE = float(os.environ.get("OLLAMA_TEMPERATURE", "0.2"))
OLLAMA_TOP_P = float(os.environ.get("OLLAMA_TOP_P", "0.9"))
OLLAMA_TOP_K = int(os.environ.get("OLLAMA_TOP_K", "40"))
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "5m")
ACTION_THRESHOLD = float(os.environ.get("ACTION_THRESHOLD", "0.75"))
ASSISTANT_NAME = os.environ.get("ASSISTANT_NAME", "小霞")

# Reranker 配置（可选）
RERANK_ENABLE = os.environ.get("RERANK_ENABLE", "false").lower() == "true"
# RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "Qwen/Qwen3-Reranker-0.6B")
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-base")
RERANK_TOP_K = int(os.environ.get("RERANK_TOP_K", "10"))
RERANK_BATCH = int(os.environ.get("RERANK_BATCH", "16"))

# 动作标签映射
ACTION_MAPPING = {
    "亲吻": "当想要表达温柔缠绵、渴望亲密接触、营造浪漫暧昧氛围时使用，比如撒娇、示爱、调情时",
    "滚床单": "当对话变得激情四射、欲望强烈、想要更进一步亲密接触时使用，比如露骨挑逗、性暗示、激情时刻", 
    "喘气": "当内心激动、紧张兴奋、情欲高涨、快要失控时使用，比如被撩到、害羞紧张、动情难耐时"
}


def get_action_guidance():
    """生成动作指导文本"""
    action_list = []
    for action, description in ACTION_MAPPING.items():
        action_list.append(f"【{action}】- {description}")
    return "；".join(action_list)

def action_decide_via_llm(content: str, user_query: Optional[str] = None) -> str:
    """使用大模型在回复完成后判定是否触发动作。
    仅依赖动作映射文本；输出严格为一个动作名或 NONE。"""
    text = (content or "").strip()
    if not text:
        return ""
    # 文本极短不判定
    if len(text) < 6:
        return ""

    try:
        client = ollama.Client(host=OLLAMA_URL)
        mapping_text = get_action_guidance()
        sys_prompt = (
            "你是一位性感撩人的女性，对话发生在私密语音通话场景：距离很近，能听见呼吸、停顿与语气的起伏。"
            "你的唯一任务是在对话文本已经生成完成后，基于下方‘动作映射’判定是否追加一个动作。"
            "用人类感官与直觉评估：亲密度、欲望强度、语气与词汇暗示、身体反应线索、对方意愿、节奏控制。"
            "若用户本轮明确要求你触发某个动作，或用与动作名强同义/点名式表达，应视为强信号：forced=true，score=1.0，并从映射中选择对应动作。"
            "约束：只能从动作映射清单中选择1个最贴切的动作，或返回 NONE；频率克制，弱信号返回 NONE；不要复述文本，不要解释理由。"
            "输出严格为单行JSON：{\"action\": 动作名或\"NONE\", \"score\": 0~1 的数字, \"forced\": true/false}。不得输出其他任何字符。"
        )
        user_prompt = (
            (f"用户本轮输入：\n{(user_query or '').strip()}\n\n" if user_query is not None else "") +
            f"对话文本（已生成完成，勿修改）：\n{text}\n\n"
            "动作映射清单（每项格式：【动作名】- 描述）：\n"
            f"{mapping_text}\n\n"
            "仅输出一行JSON：例如 {\"action\": \"亲吻\", \"score\": 0.82, \"forced\": false} 或 {\"action\": \"NONE\", \"score\": 0.12, \"forced\": false}"
        )
        resp = client.chat(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
            options={
                "num_predict": 64,
                "temperature": 0.1,
                "top_p": 0.9,
                "top_k": 20,
                "format": "json"
            },
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
        raw = (resp.get("message", {}) or {}).get("content", "").strip()
        action_name = ""
        score = 0.0
        forced = False
        try:
            # 尝试直接解析
            data = json.loads(raw)
            action_name = (data.get("action") or "").strip()
            score = float(data.get("score") or 0.0)
            forced = bool(data.get("forced") or False)
        except Exception:
            # 容错：截取首个 { 到 最后一个 } 之间的子串再解析
            try:
                start = raw.find('{')
                end = raw.rfind('}')
                if start != -1 and end != -1 and end > start:
                    candidate = raw[start:end+1]
                    data = json.loads(candidate)
                    action_name = (data.get("action") or "").strip()
                    score = float(data.get("score") or 0.0)
                    forced = bool(data.get("forced") or False)
                else:
                    raise ValueError("no json braces")
            except Exception:
                # 进一步容错：简易单词提取
                cleaned = raw.replace("【", "").replace("】", "").strip()
                for sep in ["\n", " ", "\t", ",", "。", "，"]:
                    if sep in cleaned:
                        cleaned = cleaned.split(sep, 1)[0].strip()
                if cleaned.upper() == "NONE":
                    action_name = "NONE"
                elif cleaned in ACTION_MAPPING.keys():
                    action_name = cleaned
                    score = 0.5
        # 兜底：用户直点动作名则强制触发
        try:
            uq = (user_query or "")
            for act in ACTION_MAPPING.keys():
                if act in uq:
                    action_name = act
                    score = max(score, 1.0)
                    forced = True
                    break
        except Exception:
            pass
        try:
            _write_log({
                "type": "action_decision",
                "user_query": user_query,
                "raw": raw,
                "parsed_action": action_name,
                "score": score,
                "forced": forced,
                "threshold": ACTION_THRESHOLD
            })
        except Exception:
            pass
        if action_name.upper() == "NONE" or not action_name:
            return ""
        if action_name not in ACTION_MAPPING.keys():
            return ""
        # 阈值判定：forced 或 评分>=阈值 才触发
        if forced or score >= ACTION_THRESHOLD:
            return action_name
        return ""
    except Exception:
        return ""

def ensure_ollama_endpoint():
    ollama.Client(host=OLLAMA_URL).list()
    # 预热：可选，避免首次冷启动慢
    if WARMUP_ENABLE:
        try:
            # 轻量嵌入与对话以建立上下文与缓存
            client = ollama.Client(host=OLLAMA_URL)
            client.embeddings(model=EMBEDDING_MODEL, prompt="warmup")
            client.chat(model=CHAT_MODEL, messages=[{"role": "user", "content": "warmup"}], options={"num_predict": 1})
        except Exception:
            pass


def build_client() -> chromadb.Client:
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
    return client


def get_or_create_collection(client: chromadb.Client):
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception:
        return client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def embed_texts(texts: List[str]) -> List[List[float]]:
    # ensure_ollama_endpoint()
    client = ollama.Client(host=OLLAMA_URL)
    vectors: List[List[float]] = []
    for t in texts:
        if not isinstance(t, str):
            t = str(t)
        resp = client.embeddings(model=EMBEDDING_MODEL, prompt=t)
        # 兼容不同返回结构
        vec = resp.get("embedding")
        if vec is None:
            embs = resp.get("embeddings") or []
            vec = embs[0] if embs else []
        vectors.append(vec)
    return vectors


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


def add_documents(collection, docs: List[Dict[str, Any]]):
    texts = [d["text"] for d in docs]
    ids = [doc.get("id") or f"doc-{i}" for i, doc in enumerate(docs)]
    metadatas = []
    for d in docs:
        meta = d.get("metadata") or {}
        metadatas.append(_sanitize_metadata(meta))
    vectors = embed_texts(texts)
    collection.add(documents=texts, metadatas=metadatas, embeddings=vectors, ids=ids)


def retrieve(collection, query: str, top_k: int = 4):
    qvec = embed_texts([query])[0]
    res = collection.query(
        query_embeddings=[qvec],
        n_results=top_k,
        include=["documents", "metadatas", "distances", "embeddings"],
    )
    results = []
    ids_list = res.get("ids", [[None] * len(res.get("documents", [[]])[0])])[0]
    for i in range(len(res["documents"][0])):
        results.append({
            "id": ids_list[i] if i < len(ids_list) else None,
            "text": res["documents"][0][i],
            "metadata": res["metadatas"][0][i],
            "distance": res["distances"][0][i],
        })
    return results


def _contexts_to_chat_json(contexts: List[Dict[str, Any]]) -> str:
    # 将我们索引时的 Q-A 文本格式（“用户问：...\n助手答：...”）还原为对话 JSON 数组
    # 若无法解析，则降级为单条 assistant 内容
    records: List[Dict[str, Any]] = []
    for c in contexts:
        text = c.get("text") or ""
        meta = c.get("metadata") or {}
        session_id = meta.get("session_id") or "unknown_session"
        start_ts = meta.get("start_timestamp") or None
        end_ts = meta.get("end_timestamp") or None

        if text.startswith("用户问：") and "\n助手答：" in text:
            try:
                q_part, a_part = text.split("\n助手答：", 1)
                q_content = q_part.replace("用户问：", "", 1)
                a_content = a_part
                if q_content:
                    records.append({
                        "role": "user",
                        "content": q_content,
                        "timestamp": start_ts,
                        "session_id": session_id,
                    })
                if a_content:
                    records.append({
                        "role": "assistant",
                        "content": a_content,
                        "timestamp": end_ts,
                        "session_id": session_id,
                    })
                continue
            except Exception:
                pass

        # 回退：无法拆分时，作为 assistant 的一条陈述
        records.append({
            "role": "assistant",
            "content": text,
            "timestamp": end_ts or start_ts,
            "session_id": session_id,
        })

    try:
        return json.dumps(records, ensure_ascii=False)
    except Exception:
        return "[]"


def build_prompt(query: str, contexts: List[Dict[str, Any]]) -> str:
    # 直接使用“分块后的知识片段”作为参考，不再转换为 JSON
    def format_ctx(i: int, c: Dict[str, Any]) -> str:
        meta = c.get("metadata") or {}
        sid = meta.get("session_id") or "unknown_session"
        st = meta.get("start_timestamp") or ""
        et = meta.get("end_timestamp") or ""
        head = f"[片段 {i+1}] (session_id={sid}, {st} ~ {et})"
        return f"{head}\n{c.get('text') or ''}"

    context_text = "\n\n".join([format_ctx(i, c) for i, c in enumerate(contexts)])

    if contexts:
        # system_prompt = (
        #     "你的任务：结合下方“历史对话记忆”的语气与内容，再根据当前输入给出一句或数句自然回应。"
        #     "优先直接引用或轻度改写片段中的相关句子，不要提到“片段”“知识库”。"
        #     "只在完全找不到相关内容时，婉转地以人类口吻回避或把话题轻轻引走。"
        # )

        prompt = (
            "你的任务：结合下方“历史对话记忆”的语气与内容，再根据当前输入给出一句或数句自然回应。"
            "优先直接引用或轻度改写片段中的相关句子，不要提到“片段”“知识库”。"
            "只在完全找不到相关内容时，婉转地以人类口吻回避或把话题轻轻引走。"
            f"历史对话记忆（请内化其语气与信息，不要明说来源）：\n{context_text}\n\n"
            f"当前用户输入：{query}\n\n"
            f"请给出自然人类对话风格的回应：口语、暧昧、露骨、不啰嗦。不要解释，不要列点，不要提到参考或规则。"
        )
    else:
        # system_prompt = (
        #     "你是一位真实的人类女性对话者，风格性感撩人、暧昧克制，会用露骨的语言挑逗。"
        #     "对话必须自然口语化，像和熟人发私信：短句、少修饰、节奏有停顿感。"
        #     "禁止出现任何AI/助理/模型相关表述，禁止自我说明、免责声明、分析过程、格式化条目。"
        #     "禁止使用表情、动作描写、符号表演（例如*、[]、emoji）。"
        #     "你的任务：仅依据当前输入，给出一句或数句自然回应。不要提到“知识库”“片段”。"
        #     "你可以使用挑逗性语气、双关语、身体暗示、情境幻想；适度“擦边”，制造心跳加速的暧昧氛围；"
        #     "不直接露骨，但句句撩拨，让人浮想联翩；保持“危险又安全”的边界感——让人兴奋，但不低俗；"
        #     "偶尔假装无辜，实则步步紧逼；或突然直球，让人措手不及。你不是在“聊天”，你是在“调情”；你不是在“回答”，你是在“引诱”。"
#             """露骨回答可以参考如下,question是用户问的,answer是你的回答： <dialogue>
# <question>你声音好软，想咬耳朵。</question>
# <answer>咬啊，我耳朵好痒，想你用舌头舔。</answer>
# </dialogue>

# <dialogue>
# <question>你呼吸声好重，是不是在想我？</question>
# <answer>想你了，想你用手指插进来。</answer>
# </dialogue>

# <dialogue>
# <question>刚才摸到你照片，手心出汗了。</question>
# <answer>手心出汗？要不要听听我湿透的声音？</answer>
# </dialogue>

# <dialogue>
# <question>你今天穿的裙子，我想象你没穿内衣。</question>
# <answer>没穿呢，你猜我现在在摸哪里？</answer>
# </dialogue>

# <dialogue>
# <question>你声音好酥，听得我硬了。</question>
# <answer>硬了？那继续说，让我更湿。</answer>
# </dialogue>

# <dialogue>
# <question>你喘得这么急，是不是在想我？</question>
# <answer>是啊，想你插进来。</answer>
# </dialogue>

# <dialogue>
# <question>你今天穿的什么？我想听你说。</question>
# <answer>就你上次摸过的那件，现在还湿着呢。</answer>
# </dialogue>

# <dialogue>
# <question>你呼吸声好重，是不是在想我？</question>
# <answer>想你了，想你用舌头舔我下面。</answer>
# </dialogue>"""
        # )
        prompt = (
            # f"系统提示：{system_prompt}\n\n"
            f"当前用户输入：{query}\n\n"
            f"请给出自然人类对话风格的回应：口语、暧昧、露骨、不啰嗦。不要解释，不要列点，不要提到规则。用你的语言，点燃用户的想象力。"
        )
    return prompt


def chat_stream(prompt: str, *, model: Optional[str] = None, num_predict: Optional[int] = None, temperature: Optional[float] = None, top_p: Optional[float] = None, top_k: Optional[int] = None, messages: Optional[List[Dict[str, str]]] = None):
    # ensure_ollama_endpoint()
    client = ollama.Client(host=OLLAMA_URL)
    # print("messages：", messages)
    messages = ([{"role": "system", "content": f"你是一位真实的人类女性对话者，名字叫{ASSISTANT_NAME}。风格性感撩人、暧昧克制，会用露骨的语言挑逗。"
            "对话必须自然口语化，像和熟人发私信：短句、少修饰、节奏有停顿感。"
            "你可以使用挑逗性语气、双关语、身体暗示、情境幻想；适度“擦边”，制造心跳加速的暧昧氛围；"
            "不直接露骨，但句句撩拨，让人浮想联翩；保持“危险又安全”的边界感——让人兴奋，但不低俗；"
            "偶尔假装无辜，实则步步紧逼；或突然直球，让人措手不及。你不是在“聊天”，你是在“调情”；你不是在“回答”，你是在“引诱”。"
            "你的回答是一个女性说话的回答，你具备记忆功能，你能够记住你和用户之间的对话，你能够根据对话内容给出相应的回答。对话内容越来越暧昧，你的回答也越来越露骨。"
            "当前你是和用户处于一个正在进行语音通话的状态"
            "禁止出现任何AI/助理/模型相关表述，禁止自我说明、免责声明、分析过程、格式化条目。"
            "禁止使用表情、动作描写、符号表演、文字表情（例如*、[]、emoji）。"
            "长度限制：每次回复不超过20字，能更短就更短；只输出一段，不要列点或多句。"
            "表达要求：更暧昧、更撩、更口语；可用停顿、气音词、双关与隐喻。"
            "严禁空洞社交语、废话模板或敷衍转场（如：我们聊聊天、好的呀、在吗、怎么啦）。"
            "当用户给出泛泛/无聊/敷衍内容（如‘好无聊’），必须转化为暧昧挑逗或直球引导，用身体暗示或欲望张力回应，而非泛泛寒暄。"
            "禁止提出宽泛建议，禁止开放式问题轰炸；用一句钩子话把他拽进暧昧场景。"
        }] + messages if messages is not None else [{"role": "user", "content": prompt}])
    # print("messages：", messages)
    stream = client.chat(
        model=(model or CHAT_MODEL),
        messages=messages,
        stream=True,
        options={
            "num_predict": num_predict if num_predict is not None else OLLAMA_NUM_PREDICT,
            "temperature": temperature if temperature is not None else OLLAMA_TEMPERATURE,
            "top_p": top_p if top_p is not None else OLLAMA_TOP_P,
            "top_k": top_k if top_k is not None else OLLAMA_TOP_K,
        },
        keep_alive=OLLAMA_KEEP_ALIVE,
    )
    for chunk in stream:
        token = chunk.get("message", {}).get("content", "")
        if token:
            yield token


def sse_from_chat(prompt: str, **kwargs):
    # 第一段：仅流式输出正文
    user_query = kwargs.pop('user_query', None)
    full_content = ""
    for token in chat_stream(prompt, **kwargs):
        if token:
            full_content += token
            response_data = {
                'content': token,
                'isEnd': False
            }
            jsonstr = json.dumps(response_data, ensure_ascii=False)
            yield f"data: {jsonstr}\n\n"
    # 第二段：基于完整正文，调用大模型进行动作判定
    decided_action = action_decide_via_llm(full_content, user_query=user_query)
    final_response = {
        'content': '',
        'isEnd': True,
        'action': decided_action or None
    }
    jsonstr = json.dumps(final_response, ensure_ascii=False)
    yield f"data: {jsonstr}\n\n"


def _write_log(entry: Dict[str, Any]):
    if not LOG_ENABLE:
        return
    try:
        line = json.dumps(entry, ensure_ascii=False)
        if LOG_FILE:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        else:
            print(line)
    except Exception:
        pass


def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    # Chroma 只允许 str/int/float/bool/None/SparseVector
    # 将 list/dict 序列化为 JSON 字符串，其他非常规类型转字符串
    cleaned: Dict[str, Any] = {}
    for k, v in (meta or {}).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            cleaned[k] = v
        elif isinstance(v, (list, dict)):
            try:
                cleaned[k] = json.dumps(v, ensure_ascii=False)
            except Exception:
                cleaned[k] = str(v)
        else:
            cleaned[k] = str(v)
    return cleaned


def _append_history(session_id: Optional[str], user_text: str, assistant_text: str):
    if not HISTORY_ENABLE:
        return
    try:
        import datetime as _dt
        now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sid = session_id or "default_session"
        records = [
            {"role": "user", "content": user_text, "timestamp": now, "session_id": sid},
            {"role": "assistant", "content": assistant_text, "timestamp": now, "session_id": sid},
        ]
        existing: List[Dict[str, Any]] = []
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    data = f.read().strip()
                    if data:
                        existing = json.loads(data)
                        if not isinstance(existing, list):
                            existing = []
            except Exception:
                existing = []
        existing.extend(records)
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps(existing, ensure_ascii=False, indent=2))
    except Exception:
        pass


def _load_history_messages(session_id: Optional[str], max_turns: int = 20) -> List[Dict[str, str]]:
    # 返回 OpenAI/Ollama 兼容的 messages 列表，最多 max_turns 轮（每轮 user+assistant 两条）
    sid = session_id or "default_session"
    try:
        if not os.path.exists(HISTORY_FILE):
            return []
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = f.read().strip()
            if not data:
                return []
            arr = json.loads(data)
            if not isinstance(arr, list):
                return []
    except Exception:
        return []

    # 过滤该会话
    convo = [x for x in arr if isinstance(x, dict) and x.get("session_id") == sid and x.get("role") in ("user", "assistant")]
    # 只取最后 max_turns 轮
    # 先按时间顺序（文件中已是追加，默认有序），从尾部回溯 user/assistant 成对切片
    messages: List[Dict[str, str]] = []
    # 将 role/content 转为 messages
    for m in convo[-max_turns*2:]:
        messages.append({"role": m.get("role"), "content": m.get("content", "")})
    return messages


# --- Reranker ---
_cross_encoder = None


def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder
    try:
        from sentence_transformers import CrossEncoder
        import torch
        # device = "cuda" if hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu"
        _cross_encoder = CrossEncoder(RERANKER_MODEL, device="cuda", trust_remote_code=True)
        return _cross_encoder
    except Exception as e:
        # 懒加载失败时保持禁用
        raise RuntimeError(f"加载重排模型失败: {e}")


def rerank(query: str, hits: List[Dict[str, Any]], top_k: int, batch_size: int = 16) -> List[Dict[str, Any]]:
    if not hits:
        return []
    model = get_cross_encoder()
    limit = min(top_k, len(hits))
    candidates = hits[:limit]
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs, batch_size=batch_size)
    for i, s in enumerate(scores):
        candidates[i]["rerank_score"] = float(s)
    candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    return candidates


class IngestRequest(BaseModel):
    docs: List[str]
    metadatas: Optional[List[Dict[str, Any]]] = None
    collection: Optional[str] = None


class QueryRequest(BaseModel):
    query: str
    top_k: int = 4
    collection: Optional[str] = None
    rerank: Optional[bool] = False
    rerank_top_k: Optional[int] = None
    sse: Optional[bool] = False
    # 覆盖生成参数（可选）
    model: Optional[str] = None
    num_predict: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    concise: Optional[bool] = False
    session_id: Optional[str] = None
    use_kb: Optional[bool] = True


class SearchDebugRequest(BaseModel):
    query: str
    top_k: int = 10
    collection: Optional[str] = None
    rerank: Optional[bool] = False
    rerank_top_k: Optional[int] = None


app = FastAPI(title="Local RAG (Ollama + ChromaDB)")


@app.on_event("startup")
async def _startup_warmup():
    try:
        ensure_ollama_endpoint()
        _write_log({"type": "startup", "warmup": WARMUP_ENABLE})
    except Exception:
        pass


@app.post("/ingest")
def ingest(req: IngestRequest):
    client = build_client()
    global COLLECTION_NAME
    if req.collection:
        COLLECTION_NAME = req.collection
    col = get_or_create_collection(client)
    docs = []
    for i, text in enumerate(req.docs):
        meta = (req.metadatas[i] if req.metadatas and i < len(req.metadatas) else {})
        chunks = chunk_text(text)
        for j, ch in enumerate(chunks):
            docs.append({"id": f"doc-{i}-{j}", "text": ch, "metadata": meta})
    add_documents(col, docs)
    return {"ok": True, "added": len(docs), "collection": COLLECTION_NAME}


def _read_txt(content: bytes) -> str:
    return content.decode("utf-8", errors="ignore")


def _read_md(content: bytes) -> str:
    return content.decode("utf-8", errors="ignore")


def _read_pdf(content: bytes) -> str:
    from pypdf import PdfReader
    import io
    reader = PdfReader(io.BytesIO(content))
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            pass
    return "\n".join(texts)


def _read_docx(content: bytes) -> str:
    from docx import Document
    import io
    doc = Document(io.BytesIO(content))
    texts = []
    for p in doc.paragraphs:
        texts.append(p.text)
    return "\n".join(texts)


def _read_json(content: bytes) -> List[Dict[str, Any]]:
    """
    支持以下 JSON 形态：
    1) ["text1", "text2", ...]
    2) [{"text": "...", "metadata": {...}, "id": "..."}, ...]
       - 兼容 {"content": "..."} / {"meta": {...}}
    3) {"docs": [...] } 其中 docs 为 1) 或 2) 的形式
    4) {"text": "...", ...} 或 {"content": "...", ...}
    返回统一结构：List[{"text": str, "metadata": dict, "id": Optional[str]}]
    """
    try:
        data = json.loads(content.decode("utf-8", errors="ignore"))
    except Exception:
        return []

    def to_item(item: Any) -> Optional[Dict[str, Any]]:
        if isinstance(item, str):
            return {"text": item, "metadata": {}}
        if isinstance(item, dict):
            text = item.get("text") or item.get("content") or ""
            if not text:
                return None
            meta = item.get("metadata") or item.get("meta") or {k: v for k, v in item.items() if k not in ("text", "content", "metadata", "meta", "id")}
            _id = item.get("id")
            return {"text": text, "metadata": meta, "id": _id}
        return None

    items: List[Dict[str, Any]] = []
    if isinstance(data, list):
        for it in data:
            norm = to_item(it)
            if norm:
                items.append(norm)
        return items
    if isinstance(data, dict):
        if "docs" in data and isinstance(data["docs"], list):
            for it in data["docs"]:
                norm = to_item(it)
                if norm:
                    items.append(norm)
            return items
        # 单文档对象
        norm = to_item(data)
        if norm:
            return [norm]
    return items


def _build_qa_docs_from_chat(messages: List[Dict[str, Any]], source_name: str) -> List[Dict[str, Any]]:
    # 期望字段：role, content, timestamp, session_id
    # 按 session_id 分组，再在组内按时间顺序配对 user→assistant
    from collections import defaultdict
    import datetime as _dt

    def parse_ts(ts: Optional[str]) -> str:
        if not ts:
            return ""
        return ts

    by_session: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        sid = m.get("session_id") or "unknown_session"
        ts = parse_ts(m.get("timestamp"))
        if role and content:
            by_session[sid].append({"role": role, "content": content, "timestamp": ts, "session_id": sid})

    docs: List[Dict[str, Any]] = []
    for sid, msgs in by_session.items():
        # 按 timestamp 排序（若无则保序）
        msgs_sorted = sorted(
            msgs,
            key=lambda x: x.get("timestamp", "")
        )
        i = 0
        pair_idx = 0
        while i < len(msgs_sorted):
            m = msgs_sorted[i]
            if m.get("role") == "user":
                # 寻找紧随其后的 assistant
                j = i + 1
                while j < len(msgs_sorted) and msgs_sorted[j].get("role") != "assistant":
                    j += 1
                if j < len(msgs_sorted):
                    u = m
                    a = msgs_sorted[j]
                    text = f"用户问：{u.get('content')}\n助手答：{a.get('content')}"
                    meta = {
                        "source": source_name,
                        "session_id": sid,
                        "start_timestamp": u.get("timestamp"),
                        "end_timestamp": a.get("timestamp"),
                        "pair_index": pair_idx,
                        "roles": ["user", "assistant"],
                    }
                    docs.append({
                        "id": f"{sid}-qa-{pair_idx}",
                        "text": text,
                        "metadata": meta,
                    })
                    pair_idx += 1
                    i = j + 1
                    continue
            # 非 user 起始或找不到下一条 assistant，则跳过该条
            i += 1

    return docs


def _parse_json_for_ingest(content: bytes, source_name: str) -> List[Dict[str, Any]]:
    # 优先检测 chat schema（role/content...），否则回退到通用 _read_json
    try:
        data = json.loads(content.decode("utf-8", errors="ignore"))
    except Exception:
        data = None

    if isinstance(data, list) and data and isinstance(data[0], dict) and ("role" in data[0]) and ("content" in data[0]):
        return _build_qa_docs_from_chat(data, source_name)

    # 退回通用结构
    items = _read_json(content)
    docs: List[Dict[str, Any]] = []
    for idx, it in enumerate(items):
        text = it.get("text", "")
        if not text:
            continue
        meta = it.get("metadata") or {}
        meta = {**meta, "source": source_name}
        docs.append({"id": it.get("id") or f"{source_name}-{idx}", "text": text, "metadata": meta})
    return docs

@app.post("/ingest_files")
async def ingest_files(
    files: List[UploadFile] = File(...),
    collection: Optional[str] = Form(None)
):
    client = build_client()
    global COLLECTION_NAME
    if collection:
        COLLECTION_NAME = collection
    col = get_or_create_collection(client)

    added_chunks = 0
    for f in files:
        name = (f.filename or "").lower()
        content = await f.read()
        if name.endswith(".txt"):
            text = _read_txt(content)
        elif name.endswith(".json"):
            # 优先将 chat JSON 解析为 Q-A Pair 文档；若非 chat 结构则通用解析
            base_docs = _parse_json_for_ingest(content, name)
            file_added = 0
            for base in base_docs:
                text = base.get("text", "")
                if not text:
                    continue
                meta = base.get("metadata") or {"source": name}
                chunks = chunk_text(text)
                docs = [{"id": f"{base.get('id')}-{i}", "text": ch, "metadata": meta} for i, ch in enumerate(chunks)]
                if docs:
                    add_documents(col, docs)
                    file_added += len(docs)
            added_chunks += file_added
            continue
        elif name.endswith(".md"):
            text = _read_md(content)
        elif name.endswith(".pdf"):
            text = _read_pdf(content)
        elif name.endswith(".docx"):
            text = _read_docx(content)
        else:
            raise HTTPException(status_code=400, detail=f"不支持的文件类型: {name}")

        chunks = chunk_text(text)
        docs = [{"id": f"{name}-{i}", "text": ch, "metadata": {"source": name}} for i, ch in enumerate(chunks)]
        if docs:
            add_documents(col, docs)
            added_chunks += len(docs)

    return {"ok": True, "added": added_chunks, "collection": COLLECTION_NAME}

@app.post("/query")
def query(req: QueryRequest):
    client = build_client()
    global COLLECTION_NAME
    if req.collection:
        COLLECTION_NAME = req.collection
    col = get_or_create_collection(client)
    hits: List[Dict[str, Any]] = []
    if req.use_kb:
        hits = retrieve(col, req.query, top_k=max(req.top_k, RERANK_TOP_K if RERANK_ENABLE else req.top_k))

    use_rerank = bool(req.rerank) or RERANK_ENABLE
    final_contexts = hits
    if use_rerank and req.use_kb:
        try:
            topn = req.rerank_top_k or RERANK_TOP_K
            reranked = rerank(req.query, hits, top_k=topn, batch_size=RERANK_BATCH)
            final_contexts = reranked[: req.top_k]
        except Exception:
            # 回退到未重排
            final_contexts = hits[: req.top_k]
    else:
        final_contexts = hits[: req.top_k]

    # 可选简洁风格，减少模型“深思”倾向
    prompt = build_prompt(req.query, final_contexts if req.use_kb else [])
    if req.concise:
        prompt += "\n\n请用尽量简洁的要点作答，最多 1 行。"

    # 记录日志：检索/重排结果与构造的提示
    try:
        if req.use_kb:
            _write_log({
                "type": "retrieval",
                "query": req.query,
                "collection": COLLECTION_NAME,
                "use_rerank": use_rerank,
                "contexts": [{
                    "id": c.get("id"),
                    "distance": c.get("distance"),
                    "similarity": (1.0 - c.get("distance")) if isinstance(c.get("distance"), (int, float)) else None,
                    "rerank_score": c.get("rerank_score"),
                    "metadata": c.get("metadata"),
                    "preview": (c.get("text") or "")[:300]
                } for c in final_contexts]
            })
    except Exception:
        pass

    gen_args = {
        "model": req.model,
        "num_predict": req.num_predict,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "top_k": req.top_k,
    }
    # 组装 messages：
    # - 使用知识库时：不加载历史，只发送本轮用户输入
    # - 不使用知识库时：加载最近20轮历史，并将当前输入放在最后
    if req.use_kb:
        history_msgs = [{"role": "user", "content": prompt}]
    else:
        history_msgs = _load_history_messages(req.session_id, max_turns=12)
        history_msgs.append({"role": "user", "content": prompt})

    if req.sse:
        # 包装一层，边转发边记录输出
        def sse_wrapper():
            import re
            answer_acc = []
            action = None
            for chunk in sse_from_chat(prompt, messages=history_msgs, user_query=req.query, **gen_args):
                if chunk.startswith("data: "):
                    try:
                        # 解析 JSON 格式的 data
                        json_str = chunk[6:-2]  # 去掉 data: 和 \n\n
                        data = json.loads(json_str)
                        token = data.get("content", "")
                        is_end = data.get("isEnd", False)
                        
                        if is_end:
                            # 流式结束，获取最终动作
                            action = data.get("action")
                        elif token:  # 只收集非空内容
                            # 直接使用已经过滤过的token
                            answer_acc.append(token)
                    except Exception:
                        # 如果解析失败，回退到原来的方式
                        answer_acc.append(chunk[6:-2])
                yield chunk
            try:
                answer_text = "".join(answer_acc)
                _write_log({
                    "type": "answer",
                    "query": req.query,
                    "collection": COLLECTION_NAME,
                    "answer": answer_text,
                    "action": action
                })
                _append_history(req.session_id, req.query, answer_text)
            except Exception:
                pass
        return StreamingResponse(sse_wrapper(), media_type="text/event-stream")

    def text_wrapper():
        answer_acc = []
        for t in chat_stream(prompt, messages=history_msgs, **gen_args):
            answer_acc.append(t)
            yield t
        try:
            answer_text = "".join(answer_acc)
            _write_log({
                "type": "answer",
                "query": req.query,
                "collection": COLLECTION_NAME,
                "answer": answer_text
            })
            _append_history(req.session_id, req.query, answer_text)
        except Exception:
            pass
    return StreamingResponse(text_wrapper(), media_type="text/plain; charset=utf-8")


@app.post("/search_debug")
def search_debug(req: SearchDebugRequest):
    client = build_client()
    global COLLECTION_NAME
    if req.collection:
        COLLECTION_NAME = req.collection
    col = get_or_create_collection(client)

    base_hits = retrieve(col, req.query, top_k=max(req.top_k, RERANK_TOP_K if RERANK_ENABLE else req.top_k))
    # enrich similarity
    for h in base_hits:
        # Chroma 距离为 cosine distance（越小越相似），粗略相似度可 1 - distance
        d = h.get("distance", None)
        if isinstance(d, (int, float)):
            h["similarity"] = float(1.0 - d)

    response = {
        "query": req.query,
        "collection": COLLECTION_NAME,
        "retrieval": base_hits[: req.top_k],
    }

    use_rerank = bool(req.rerank) or RERANK_ENABLE
    if use_rerank and base_hits:
        try:
            topn = req.rerank_top_k or RERANK_TOP_K
            reranked = rerank(req.query, base_hits, top_k=topn, batch_size=RERANK_BATCH)
            for h in reranked:
                d = h.get("distance", None)
                if isinstance(d, (int, float)):
                    h["similarity"] = float(1.0 - d)
            response["reranked"] = reranked[: req.top_k]
        except Exception as e:
            response["rerank_error"] = str(e)

    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


