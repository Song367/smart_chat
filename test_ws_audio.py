import asyncio
import json
from collections import defaultdict
from io import BytesIO

import websockets  # pip install websockets
import pyaudio  # pip install pyaudio


WS_URL = "ws://127.0.0.1:9000/ws"
TEST_TEXT = "工作日：9点–17点（16:30停止取号）"


class PcmPlayer:
    def __init__(self, sample_rate: int = 32000, channels: int = 1, width_bytes: int = 2):
        self.sample_rate = sample_rate
        self.channels = channels
        self.width_bytes = width_bytes
        self._p = pyaudio.PyAudio()
        self._stream = self._p.open(
            format=self._p.get_format_from_width(width_bytes),
            channels=channels,
            rate=sample_rate,
            output=True,
        )

    def write(self, pcm_bytes: bytes):
        self._stream.write(pcm_bytes)

    def close(self):
        try:
            self._stream.stop_stream()
            self._stream.close()
        finally:
            self._p.terminate()


async def main():
    async with websockets.connect(WS_URL, max_size=None) as ws:
        await ws.send(json.dumps({"type": "start", "text": TEST_TEXT}, ensure_ascii=False))

        player = PcmPlayer(sample_rate=32000, channels=1, width_bytes=2)
        active_indices = set()

        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                # 直接边收边播（PCM）
                player.write(msg)
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
                fmt = data.get("format")
                print(f"[tts_chunk] index={idx} seq={seq} size={size} fmt={fmt}")
            elif t == "tts_end":
                idx = int(data.get("index") or 0)
                print(f"[tts_end] index={idx}")
            elif t == "llm_end":
                print("[llm_end]")
                # 继续等待剩余段的 tts_end，或自行退出
            elif t == "error":
                print("[error]", data)
                break


if __name__ == "__main__":
    asyncio.run(main())


