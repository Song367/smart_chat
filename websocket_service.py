import asyncio
import json
import re
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

import rag_ollama
import tts_service


app = FastAPI(title="RAG + TTS WebSocket Service")


class Session:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.cancel_event = asyncio.Event()
        self.request_id: Optional[str] = None
        self.pipeline_task: Optional[asyncio.Task] = None

    def new_request(self) -> str:
        self.request_id = uuid.uuid4().hex
        self.cancel_event.clear()
        return self.request_id


async def _send_json(ws: WebSocket, data: Dict[str, Any]):
    try:
        if ws.client_state == WebSocketState.CONNECTED and ws.application_state == WebSocketState.CONNECTED:
            await ws.send_text(json.dumps(data, ensure_ascii=False))
    except Exception:
        # 连接可能已关闭，静默失败，让上层根据 cancel_event 结束
        try:
            rid = data.get("request_id") if isinstance(data, dict) else None
            print(f"[ws][send_json_failed] rid={rid} data_type={data.get('type') if isinstance(data, dict) else type(data)}")
        except Exception:
            pass
        raise


async def _send_binary(ws: WebSocket, data: bytes):
    try:
        if ws.client_state == WebSocketState.CONNECTED and ws.application_state == WebSocketState.CONNECTED:
            await ws.send_bytes(data)
    except Exception:
        # 连接可能已关闭
        try:
            print(f"[ws][send_binary_failed] bytes={len(data)}")
        except Exception:
            pass
        raise


async def _generate_text_and_segment(session: Session, user_text: str, out_queue: asyncio.Queue):
    """生成 LLM 文本：
    - 按 token 下发 partial_text
    - 基于标点与长度(≥6)切分句子，投递到 out_queue 进行 TTS
    - 若最终总长度 ≤6 或没有任何切分产生，则在结束时投递整段
    返回最终拼接的文本。
    """
    request_id = session.request_id or ""
    buf: list[str] = []
    all_tokens: list[str] = []
    emitted_any = False
    # 中英文句子终止/停顿标点
    # 句子边界符集合（按最后一个边界切分）：中文句号/叹号/问号/逗号，及对应英文 ! ? ,
    boundary_any = re.compile(r"[。！？，!? ,]")
    seg_index = 0

    def _iter_tokens():
        for token in rag_ollama.chat_stream(user_text):
            if session.cancel_event.is_set():
                break
            if token:
                yield token

    loop = asyncio.get_running_loop()

    async for token in _async_iter_tokens(loop, _iter_tokens):
        all_tokens.append(token)
        buf.append(token)

        # 增量判断边界：若缓冲中出现任意边界符，则按“最后一个边界符”切分
        cand = "".join(buf)
        if cand:
            last_idx = None
            for m in boundary_any.finditer(cand):
                last_idx = m.end()  # 边界符之后的位置（切片右开）
            if last_idx is not None:
                left = cand[:last_idx].strip()
                right = cand[last_idx:]  # 先不 strip，保留后续拼接的自然性
                if len(left) >= 6:
                    try:
                        await _send_json(session.websocket, {
                            "type": "segment_text",
                            "request_id": request_id,
                            "index": seg_index,
                            "text": left,
                        })
                        # 服务器侧调试日志：打印分段内容
                        try:
                            print(f"[ws][segment] rid={request_id} idx={seg_index} text={left}")
                        except Exception:
                            pass
                        await out_queue.put({"index": seg_index, "text": left})
                        emitted_any = True
                        seg_index += 1
                        buf.clear()
                        if right:
                            buf.append(right)
                    except Exception:
                        pass

    final_text = "".join(all_tokens).strip()
    # 结束时：若有剩余片段
    tail = "".join(buf).strip()
    if not session.cancel_event.is_set():
        if tail:
            if len(tail) >= 6:
                await _send_json(session.websocket, {
                    "type": "segment_text",
                    "request_id": request_id,
                    "index": seg_index,
                    "text": tail,
                })
                try:
                    print(f"[ws][segment] rid={request_id} idx={seg_index} text={tail}")
                except Exception:
                    pass
                await out_queue.put({"index": seg_index, "text": tail})
                emitted_any = True
                seg_index += 1
            else:
                # 尾部不到6字，若此前未触发过且总长≤6，则整体触发一次
                if not emitted_any and len(final_text) <= 6:
                    await _send_json(session.websocket, {
                        "type": "segment_text",
                        "request_id": request_id,
                        "index": seg_index,
                        "text": final_text,
                    })
                    try:
                        print(f"[ws][segment] rid={request_id} idx={seg_index} text={final_text}")
                    except Exception:
                        pass
                    await out_queue.put({"index": seg_index, "text": final_text})
                    seg_index += 1
                # 否则忽略不足6字的尾巴（避免过碎 TTS）

        await _send_json(session.websocket, {
            "type": "llm_end",
            "request_id": request_id,
        })
        try:
            print(f"[ws][llm_end] rid={request_id} total_len={len(final_text)}")
        except Exception:
            pass

    return final_text


async def _async_iter_tokens(loop: asyncio.AbstractEventLoop, sync_gen_factory):
    """将同步生成器在执行器中迭代，异步逐项产出。"""
    def consume_next(it):
        try:
            return next(it)
        except StopIteration:
            return None

    it = sync_gen_factory()
    while True:
        token = await loop.run_in_executor(None, consume_next, it)
        if token is None:
            break
        yield token


async def _stream_tts(session: Session, text: str, *, meta: Optional[Dict[str, Any]] = None):
    """调用 tts_service 的构造体与迭代器，按二进制分片发送音频。"""
    request_id = session.request_id or ""
    req = tts_service.OnlyTextRequest(text=text)
    body = tts_service._build_stream_body(req)  # noqa: SLF001 使用内部函数，因本地服务集成

    loop = asyncio.get_running_loop()

    def _iter_audio():
        for chunk in tts_service._iter_minimaxi_stream(body):  # noqa: SLF001
            if session.cancel_event.is_set():
                break
            if chunk:
                yield chunk

    seq = 0
    async for audio_bytes in _async_iter_bytes(loop, _iter_audio):
        # PCM 边收边播：直接下发二进制，同时可选发送一条极简元信息
        await _send_binary(session.websocket, audio_bytes)
        await _send_json(session.websocket, {
            "type": "tts_chunk",
            "request_id": request_id,
            "index": (meta or {}).get("index"),
            "seq": seq,
            "size": len(audio_bytes),
            "format": "pcm",
            "sample_rate": 32000,
            "channels": 1,
            "width_bytes": 2
        })
        seq += 1
    # 不在此处发送 tts_end，由上层消费者统一发送携带 index 的 tts_end


async def _async_iter_bytes(loop: asyncio.AbstractEventLoop, sync_gen_factory):
    def consume_next(it):
        try:
            return next(it)
        except StopIteration:
            return None

    it = sync_gen_factory()
    while True:
        data = await loop.run_in_executor(None, consume_next, it)
        if data is None:
            break
        yield data


@app.websocket("/ws")
async def ws_handler(websocket: WebSocket):
    await websocket.accept()
    session = Session(websocket)
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                await _send_json(websocket, {"type": "error", "message": "invalid json"})
                continue

            mtype = msg.get("type")
            if mtype == "start":
                text = (msg.get("text") or "").strip()
                if not text:
                    await _send_json(websocket, {"type": "error", "message": "text is required"})
                    continue
                request_id = session.new_request()
                await _send_json(websocket, {"type": "ack", "request_id": request_id})

                async def _pipeline():
                    try:
                        out_queue: asyncio.Queue = asyncio.Queue()

                        async def producer():
                            try:
                                await _generate_text_and_segment(session, text, out_queue)
                            finally:
                                # 结束信号
                                await out_queue.put(None)  # type: ignore

                        async def consumer():
                            while not session.cancel_event.is_set():
                                seg = await out_queue.get()
                                if seg is None:
                                    break
                                if not seg:
                                    continue
                                if isinstance(seg, dict):
                                    seg_idx = seg.get("index")
                                    seg_text = seg.get("text") or ""
                                else:
                                    # 兼容旧格式
                                    seg_idx = 0
                                    seg_text = str(seg)
                                # 标记当前分段开始
                                await _send_json(websocket, {
                                    "type": "tts_start",
                                    "request_id": session.request_id,
                                    "index": seg_idx,
                                    "text": seg_text,
                                })
                                await _stream_tts(session, seg_text, meta={"index": seg_idx})
                                # 标记当前分段完成
                                await _send_json(websocket, {
                                    "type": "tts_end",
                                    "request_id": session.request_id,
                                    "index": seg_idx,
                                })

                        await asyncio.gather(producer(), consumer())
                    except Exception as e:
                        await _send_json(websocket, {"type": "error", "request_id": request_id, "message": str(e)})

                # 后台运行管道
                session.pipeline_task = asyncio.create_task(_pipeline())
            elif mtype == "cancel":
                session.cancel_event.set()
                try:
                    print(f"[ws][cancel] rid={session.request_id}")
                except Exception:
                    pass
                await _send_json(websocket, {"type": "cancel_ack", "request_id": session.request_id})
            else:
                await _send_json(websocket, {"type": "error", "message": f"unknown type: {mtype}"})
    except WebSocketDisconnect:
        # 连接断开即标记取消
        session.cancel_event.set()
        try:
            if session.pipeline_task and not session.pipeline_task.done():
                session.pipeline_task.cancel()
        except Exception:
            pass
        try:
            print(f"[ws][disconnect] rid={session.request_id} reason=client_closed")
        except Exception:
            pass
    except Exception:
        session.cancel_event.set()
        try:
            if session.pipeline_task and not session.pipeline_task.done():
                session.pipeline_task.cancel()
        except Exception:
            pass
        try:
            print(f"[ws][handler_exception] rid={session.request_id}")
        except Exception:
            pass


@app.get("/health")
def health():
    return {"status": "ok"}


@app.on_event("startup")
async def _startup_ollama_warmup():
    try:
        # 预热 Ollama，确保模型端点可用，减少首个请求延迟
        rag_ollama.ensure_ollama_endpoint()
    except Exception:
        # 忽略预热失败，后续请求仍会按需尝试
        pass


