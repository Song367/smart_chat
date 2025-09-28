import asyncio
import json
import re
import uuid
import os
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
from fastapi.staticfiles import StaticFiles
import wave
import io

import rag_ollama
import tts_service

# 服务配置
SERVICE_HOST = os.environ.get("SERVICE_HOST", "127.0.0.1")
SERVICE_PORT = os.environ.get("SERVICE_PORT", "9000")
SERVICE_SCHEME = os.environ.get("SERVICE_SCHEME", "http")
BASE_URL = f"{SERVICE_SCHEME}://{SERVICE_HOST}:{SERVICE_PORT}"


# 性能监控类
class ServerPerformanceMonitor:
    def __init__(self):
        self.timestamps = {}
        self.start_time = None
        
    def mark(self, event_name: str, description: str = "", request_id: str = ""):
        """记录时间戳"""
        current_time = time.time()
        if self.start_time is None:
            self.start_time = current_time
            print(f"🚀 [SERVER_PERF] 开始计时 - {event_name}: {description} (rid={request_id})")
        else:
            elapsed = current_time - self.start_time
            print(f"⏱️  [SERVER_PERF] {event_name} - 总耗时: {elapsed:.3f}s - {description} (rid={request_id})")
        
        self.timestamps[event_name] = current_time
        
    def get_interval(self, from_event: str, to_event: str) -> float:
        """获取两个事件之间的时间间隔"""
        if from_event in self.timestamps and to_event in self.timestamps:
            return self.timestamps[to_event] - self.timestamps[from_event]
        return 0.0
        
    def log_summary(self, request_id: str = ""):
        """输出性能摘要"""
        if not self.timestamps:
            return
            
        print("=" * 60)
        print(f"📊 服务端性能分析摘要 (rid={request_id}):")
        
        events = list(self.timestamps.keys())
        for i in range(len(events) - 1):
            current = events[i]
            next_event = events[i + 1]
            interval = self.get_interval(current, next_event)
            print(f"  {current} → {next_event}: {interval:.3f}s")
            
        total_time = self.timestamps[events[-1]] - self.start_time if events else 0
        print(f"  总耗时: {total_time:.3f}s")
        print("=" * 60)


app = FastAPI(title="RAG + TTS WebSocket Service (Optimized)")
# 挂载本地静态目录：/assets → ./assets （其中包含 audio/ 与 images/）
app.mount("/assets", StaticFiles(directory="assets"), name="assets")


def _resolve_media_path(path: str) -> Optional[str]:
    """尽力解析本地媒体路径，兼容 assets/ 与工作目录直挂的 audio/images。
    返回可用的绝对路径，找不到返回 None。
    """
    if not path:
        return None
    candidates = []
    # 原始
    candidates.append(path)
    # 相对工作目录
    candidates.append(os.path.join(os.getcwd(), path))
    # 兼容 assets/audio → audio, assets/images → images
    if path.startswith("assets/"):
        candidates.append(path.replace("assets/audio", "audio").replace("assets\\audio", "audio"))
        candidates.append(os.path.join(os.getcwd(), path.replace("assets/audio", "audio").replace("assets\\audio", "audio")))
        candidates.append(path.replace("assets/images", "images").replace("assets\\images", "images"))
        candidates.append(os.path.join(os.getcwd(), path.replace("assets/images", "images").replace("assets\\images", "images")))
    # 大小写扩展（.wav/.WAV）
    base, ext = os.path.splitext(path)
    if ext.lower() == ".wav":
        candidates.append(base + ".wav")
        candidates.append(base + ".WAV")
        candidates.append(os.path.join(os.getcwd(), base + ".wav"))
        candidates.append(os.path.join(os.getcwd(), base + ".WAV"))
    # 去重保持顺序
    seen = set()
    uniq = []
    for c in candidates:
        c2 = os.path.normpath(c)
        if c2 not in seen:
            seen.add(c2)
            uniq.append(c2)
    for c in uniq:
        try:
            if os.path.exists(c):
                return c
        except Exception:
            continue
    return None


class Session:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.cancel_event = asyncio.Event()
        self.request_id: Optional[str] = None
        self.pipeline_task: Optional[asyncio.Task] = None
        self.session_id: str = uuid.uuid4().hex  # 每个连接分配唯一会话ID
        self.history_messages: list = []  # 内存中存储该会话的历史对话
        self._history_loaded = False  # 标记是否已加载历史
        self.performance_monitor = ServerPerformanceMonitor()

    def new_request(self) -> str:
        self.request_id = uuid.uuid4().hex
        self.cancel_event.clear()
        self.performance_monitor = ServerPerformanceMonitor()  # 重置性能监控
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
    session.performance_monitor.mark("llm_start", "开始LLM文本生成", request_id)
    
    buf: list[str] = []
    all_tokens: list[str] = []
    emitted_any = False
    # 中英文句子终止/停顿标点
    # 句子边界符集合（按最后一个边界切分）：中文句号/叹号/问号/逗号，及对应英文 ! ? ,
    boundary_any = re.compile(r"[。！？，!? ,]")
    seg_index = 0

    def _iter_tokens():
        # 首次对话时加载历史到内存
        if not session._history_loaded:
            try:
                session.performance_monitor.mark("history_load_start", "开始加载历史对话", request_id)
                session.history_messages = rag_ollama._load_history_messages(session.session_id, max_turns=12)
                session._history_loaded = True
                session.performance_monitor.mark("history_load_end", f"历史对话加载完成: {len(session.history_messages)}条", request_id)
                print(f"[ws][history_loaded] sid={session.session_id} count={len(session.history_messages)}")
            except Exception as e:
                print(f"[ws][history_load_failed] sid={session.session_id} error={e}")
                session.history_messages = []
        
        # 使用内存中的历史对话
        current_messages = session.history_messages.copy()
        current_messages.append({"role": "user", "content": user_text})
        
        session.performance_monitor.mark("llm_call_start", "开始调用LLM API", request_id)
        for token in rag_ollama.chat_stream("", messages=current_messages):
            if session.cancel_event.is_set():
                break
            if token:
                yield token
        session.performance_monitor.mark("llm_call_end", "LLM API调用完成", request_id)

    loop = asyncio.get_running_loop()

    async for token in _async_iter_tokens(loop, _iter_tokens):
        all_tokens.append(token)
        buf.append(token)

        # 增量判断边界：若缓冲中出现任意边界符，则按"最后一个边界符"切分
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
        
        # 追加到内存中的历史对话
        try:
            session.history_messages.append({"role": "user", "content": user_text})
            session.history_messages.append({"role": "assistant", "content": final_text})
            # 限制历史长度，保留最近12轮对话
            if len(session.history_messages) > 24:  # 12轮 * 2条/轮
                session.history_messages = session.history_messages[-24:]
            print(f"[ws][history_updated] rid={request_id} sid={session.session_id} count={len(session.history_messages)}")
        except Exception as e:
            print(f"[ws][history_update_failed] rid={request_id} error={e}")
        try:
            print(f"[ws][llm_end] rid={request_id} total_len={len(final_text)}")
        except Exception:
            pass

    session.performance_monitor.mark("llm_complete", f"LLM文本生成完成: {len(final_text)}字符", request_id)
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
    session.performance_monitor.mark(f"tts_start_{meta.get('index', 0)}", f"开始TTS生成: '{text[:20]}...'", request_id)
    
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
    first_chunk = True
    async for audio_bytes in _async_iter_bytes(loop, _iter_audio):
        # 记录第一个音频块的时间
        if first_chunk:
            session.performance_monitor.mark(f"tts_first_chunk_{meta.get('index', 0)}", f"收到第一个TTS音频块: {len(audio_bytes)}字节", request_id)
            first_chunk = False
            
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
    
    session.performance_monitor.mark(f"tts_complete_{meta.get('index', 0)}", f"TTS生成完成: {seq}个音频块", request_id)


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
                session.performance_monitor.mark("request_received", f"收到用户请求: '{text}'", request_id)
                await _send_json(websocket, {"type": "ack", "request_id": request_id})

                async def _pipeline():
                    try:
                        out_queue: asyncio.Queue = asyncio.Queue()
                        segment_count = 0
                        finished_count = 0
                        early_action_name: Optional[str] = None
                        early_action_task: Optional[asyncio.Task] = None
                        early_started: bool = False
                        early_checked: bool = False

                        async def producer():
                            nonlocal early_started, early_action_name, early_action_task
                            try:
                                # 生成文本并分段；期间会发送多条 segment_text
                                await _generate_text_and_segment(session, text, out_queue)
                            finally:
                                # 结束信号
                                await out_queue.put(None)  # type: ignore

                        async def consumer():
                            nonlocal segment_count, finished_count, early_started, early_action_name, early_action_task, early_checked
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
                                # 在第一段到来时判定并启动预制音频（与 TTS 并行），避免空播与提前判定延迟
                                if not early_checked:
                                    early_checked = True
                                    try:
                                        session.performance_monitor.mark("action_prediction_start", "开始动作预测", request_id)
                                        pre_name2 = rag_ollama.predict_action_via_llm_quick(text)
                                        session.performance_monitor.mark("action_prediction_end", f"动作预测完成: {pre_name2}", request_id)
                                        if pre_name2:
                                            cfg2 = getattr(rag_ollama, "ACTION_REGISTRY", {}).get(pre_name2)
                                            kind2 = (cfg2.get("kind") or "").lower() if isinstance(cfg2, dict) else ""
                                            if isinstance(cfg2, dict) and kind2 == "audio":
            
                                                early_action_name = pre_name2
                                                async def _early_play2():
                                                    # 延迟 500ms 再发送预制音频，错峰与 TTS 首包
                                                    try:
                                                        await asyncio.sleep(0.5)
                                                        print(f"[ws][early_action_delay_done] rid={session.request_id} action={pre_name2}")
                                                    except Exception:
                                                        pass
                                                    media = cfg2.get("media") or {}
                                                    raw_path = media.get("path") or ""
                                                    path = _resolve_media_path(raw_path) or raw_path
                                                    await _send_json(websocket, {
                                                        "type": "action_audio_start",
                                                        "action": pre_name2,
                                                        "id": media.get("id"),
                                                        "format": "pcm",
                                                        "sample_rate": 32000,
                                                        "channels": 1,
                                                        "width_bytes": 2,
                                                    })
                                                    try:
                                                        if not path or (not os.path.exists(path)):
                                                            raise FileNotFoundError(path or "<empty>")
                                                        seq = 0
                                                        with wave.open(path, "rb") as wf:
                                                            src_channels = wf.getnchannels()
                                                            src_rate = wf.getframerate()
                                                            src_width = wf.getsampwidth()
                                                            if src_width != 2:
                                                                raise RuntimeError("only 16-bit WAV supported")
                                                            import audioop
                                                            chunk_size = 4096
                                                            while not session.cancel_event.is_set():
                                                                data = wf.readframes(chunk_size // (src_channels * src_width) if src_channels and src_width else chunk_size)
                                                                if not data:
                                                                    break
                                                                mono = data if src_channels == 1 else audioop.tomono(data, 2, 0.5, 0.5)
                                                                pcm = mono if src_rate == 32000 else audioop.ratecv(mono, 2, 1, src_rate, 32000, None)[0]
                                                                if pcm:
                                                                    await _send_binary(websocket, pcm)
                                                                    await _send_json(websocket, {"type": "action_audio_chunk", "seq": seq, "size": len(pcm)})
                                                                    seq += 1
                                                    except Exception as e:
                                                        await _send_json(websocket, {"type": "action_error", "reason": "asset_not_found", "action": pre_name2, "message": str(e)})
                                                    finally:
                                                        await _send_json(websocket, {"type": "action_audio_end", "action": pre_name2})
                                                early_action_task = asyncio.create_task(_early_play2())
                                                early_started = True
                                                try:
                                                    print(f"[ws][early_action_start_fallback] rid={session.request_id} action={pre_name2}")
                                                except Exception:
                                                    pass
                                            elif isinstance(cfg2, dict) and kind2 == "image":
                                                # 图片类动作：在首段到来时直接返回图片 URL（同样延迟 500ms 以避免与 TTS 首包抢占）
                                                try:
                                                    await asyncio.sleep(0.5)
                                                except Exception:
                                                    pass
                                                media2 = cfg2.get("media") or {}
                                                raw_path2 = media2.get("path") or ""
                                                path2 = _resolve_media_path(raw_path2) or raw_path2
                                                url2 = None
                                                try:
                                                    if path2 and os.path.exists(path2):
                                                        norm_path2 = path2.replace("\\", "/")
                                                        if "/assets/" in norm_path2:
                                                            # 提取相对路径
                                                            relative_path = norm_path2[norm_path2.find("/assets/"):]
                                                            # 构建完整URL
                                                            url2 = f"{BASE_URL}{relative_path}"
                                                        elif "/images/" in norm_path2:
                                                            # 提取文件名并构建完整URL
                                                            filename = norm_path2.split("/images/")[-1]
                                                            url2 = f"{BASE_URL}/assets/images/{filename}"
                                                except Exception:
                                                    url2 = None
                                                await _send_json(websocket, {
                                                    "type": "action_image",
                                                    "action": pre_name2,
                                                    "url": url2 or path2 or ""
                                                })
                                                early_started = True
                                                try:
                                                    print(f"[ws][early_image_sent] rid={session.request_id} action={pre_name2} url={url2 or path2}")
                                                except Exception:
                                                    pass
                                    except Exception:
                                        pass
                                # 标记当前分段开始
                                await _send_json(websocket, {
                                    "type": "tts_start",
                                    "request_id": session.request_id,
                                    "index": seg_idx,
                                    "text": seg_text,
                                })
                                try:
                                    print(f"[ws][tts_segment_start] rid={session.request_id} idx={seg_idx} early_started={early_started}")
                                except Exception:
                                    pass
                                segment_count += 1
                                await _stream_tts(session, seg_text, meta={"index": seg_idx})
                                # 标记当前分段完成
                                await _send_json(websocket, {
                                    "type": "tts_end",
                                    "request_id": session.request_id,
                                    "index": seg_idx,
                                })
                                finished_count += 1

                            # 后置兜动作：禁用，仅保留前置并行动作
                            try:
                                print(f"[ws][post_action_disabled] rid={session.request_id}")
                            except Exception:
                                pass

                        await asyncio.gather(producer(), consumer())
                        
                        # 输出性能摘要
                        session.performance_monitor.log_summary(request_id)
                        
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
        # 连接断开时保存历史到文件（可选）
        try:
            if session.history_messages:
                # 这里可以添加保存到文件的逻辑，或者依赖定期保存
                pass
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
        if rag_ollama.USE_OPENAI_CLIENT:
            print("WebSocket 服务：使用 OpenAI 客户端，跳过 Ollama 初始化")
        else:
            # 预热 Ollama，确保模型端点可用，减少首个请求延迟
            rag_ollama.ensure_ollama_endpoint()
            print("WebSocket 服务：Ollama 端点已就绪")
    except Exception as e:
        print(f"WebSocket 服务初始化失败: {e}")
        # 忽略预热失败，后续请求仍会按需尝试


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("WebSocket + 静态文件服务启动 (优化版)")
    print("=" * 60)
    print("服务地址:")
    print(f"  WebSocket: ws://{SERVICE_HOST}:{SERVICE_PORT}/ws")
    print(f"  静态文件: {BASE_URL}/assets/")
    print(f"  图片访问: {BASE_URL}/assets/images/")
    print(f"  音频访问: {BASE_URL}/assets/audio/")
    print("=" * 60)
    
    # 启动服务
    uvicorn.run(
        "websocket_service_optimized:app",
        host=SERVICE_HOST,
        port=int(SERVICE_PORT),
        log_level="info",
        reload=False
    )
