import asyncio
import json
from collections import defaultdict
from io import BytesIO

import websockets  # pip install websockets
from pydub import AudioSegment  # pip install pydub  (需要本机安装 ffmpeg)
import pyaudio  # pip install pyaudio


WS_URL = "ws://127.0.0.1:9000/ws"
TEST_TEXT = "工作日：9点–17点（16:30停止取号）"


def play_mp3_bytes(mp3_bytes: bytes):
    # 解码 MP3 -> PCM（依赖 ffmpeg）
    seg = AudioSegment.from_file(BytesIO(mp3_bytes), format="mp3")
    p = pyaudio.PyAudio()
    stream = p.open(
        format=p.get_format_from_width(seg.sample_width),
        channels=seg.channels,
        rate=seg.frame_rate,
        output=True,
    )
    try:
        stream.write(seg.raw_data)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


async def main():
    async with websockets.connect(WS_URL, max_size=None) as ws:
        await ws.send(json.dumps({"type": "start", "text": TEST_TEXT}, ensure_ascii=False))

        # 为每段保存累积的音频数据
        buffers = defaultdict(bytearray)
        active_indices = set()

        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                # 原始音频帧：追加到“最近一个活动分段”（按最后一个 tts_start 的 index）
                if not active_indices:
                    # 若没有活动分段，丢弃或忽略
                    continue
                current_index = max(active_indices)
                buffers[current_index].extend(msg)
                continue

            # 处理 JSON 控制事件
            try:
                data = json.loads(msg)
            except Exception:
                print("[text]", msg)
                continue

            t = data.get("type")
            if t == "ack":
                print("[ack]", data)
            elif t == "segment_text":
                print("[segment]", data.get("index"), data.get("text"))
            elif t == "tts_start":
                idx = int(data.get("index") or 0)
                active_indices.add(idx)
                print(f"[tts_start] index={idx}")
            elif t == "tts_chunk":
                # 可选：仅日志
                idx = data.get("index")
                seq = data.get("seq")
                size = data.get("size")
                print(f"[tts_chunk] index={idx} seq={seq} size={size}")
            elif t == "tts_end":
                idx = int(data.get("index") or 0)
                print(f"[tts_end] index={idx}, bytes={len(buffers[idx])}")
                # 播放该段
                try:
                    play_mp3_bytes(bytes(buffers[idx]))
                except Exception as e:
                    print("[play_error]", e)
                # 清理
                buffers.pop(idx, None)
                active_indices.discard(idx)
            elif t == "llm_end":
                print("[llm_end]")
                # 可选择在所有活动段播放完再退出
                if not active_indices:
                    break
            elif t == "error":
                print("[error]", data)
                break


if __name__ == "__main__":
    asyncio.run(main())


