import os
import json
from typing import Any, Dict, Iterator, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
MINIMAXI_API_KEY: str = os.getenv("MINIMAXI_API_KEY", "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiLkuIrmtbfpopzpgJTnp5HmioDmnInpmZDlhazlj7giLCJVc2VyTmFtZSI6IuadqOmqpSIsIkFjY291bnQiOiIiLCJTdWJqZWN0SUQiOiIxNzI4NzEyMzI0OTc5NjI2ODM5IiwiUGhvbmUiOiIxMzM4MTU1OTYxOCIsIkdyb3VwSUQiOiIxNzI4NzEyMzI0OTcxMjM4MjMxIiwiUGFnZU5hbWUiOiIiLCJNYWlsIjoiIiwiQ3JlYXRlVGltZSI6IjIwMjUtMDYtMTYgMTY6Mjk6NTkiLCJUb2tlblR5cGUiOjEsImlzcyI6Im1pbmltYXgifQ.D_JF0-nO89NdMZCYq4ocEyqxtZ9SeEdtMvbeSkZTWspt0XfX2QpPAVh-DI3MCPZTeSmjNWLf4fA_Th2zpVrj4UxWMbGKBeLZWLulNpwAHGMUTdqenuih3daCDPCzs0duhlFyQnZgGcEOGQ476HL72N2klujP8BUy_vfAh_Zv0po-aujQa5RxardDSOsbs49NTPEw0SQEXwaJ5bVmiZ5s-ysJ9pZWSEiyJ6SX9z3JeZHKj9DxHdOw5roZR8izo54e4IoqyLlzEfhOMW7P15-ffDH3M6HGiEmeBaGRYGAIciELjZS19ONNMKsTj-wXNGWtKG-sjAB1uuqkkT5Ul9Dunw")
MINIMAXI_TTS_URL: str = os.getenv("MINIMAXI_TTS_URL", "https://api.minimaxi.com/v1/t2a_v2")
DEFAULT_MODEL: str = os.getenv("MINIMAXI_TTS_MODEL", "speech-02-turbo")


app = FastAPI(title="TTS Streaming Service", version="1.0.0")


class OnlyTextRequest(BaseModel):
    text: str = Field(..., description="要合成的文本")


def _build_headers(api_key: str) -> Dict[str, str]:
    # 去除环境变量中可能意外包含的引号与空白
    api_key = (api_key or "").strip().strip('"').strip("'")
    if not api_key:
        raise HTTPException(status_code=500, detail="MINIMAXI_API_KEY 未配置")
    return {
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }


def _build_stream_body(req: OnlyTextRequest) -> str:
    body = {
        "model": DEFAULT_MODEL,
        "text": req.text,
        "stream": True,
        "language_boost": "auto",
        "voice_setting": {
            "voice_id": "yantu-qinggang-demo2-male-4",
            "speed": 1,
            "vol": 1,
            "pitch": 0
        },
        "audio_setting": {
            "sample_rate": 32000,
            "bitrate": 128000,
            "format": "mp3"
        }
    }
    return json.dumps(body, ensure_ascii=False)


def _build_nonstream_body(req: OnlyTextRequest) -> str:
    body = {
        "model": DEFAULT_MODEL,
        "text": req.text,
        "stream": False,
        "language_boost": "auto",
        "output_format": "url",
        "voice_setting": {
            "voice_id": "yantu-qinggang-demo2-male-4",
            "speed": 1,
            "vol": 1,
            "pitch": 0
        },
        "audio_setting": {
            "sample_rate": 32000,
            "bitrate": 128000,
            "format": "mp3"
        }
    }
    return json.dumps(body, ensure_ascii=False)


def _iter_minimaxi_stream(body: str) -> Iterator[bytes]:
    headers = _build_headers(MINIMAXI_API_KEY)
    with requests.post(MINIMAXI_TTS_URL, headers=headers, data=body, stream=True) as resp:
        if resp.status_code != 200:
            try:
                detail = resp.json()
            except Exception:
                detail = {"status_code": resp.status_code, "text": resp.text}
            raise HTTPException(status_code=resp.status_code, detail=detail)

        # 逐行解析，避免将最终总结块再次输出导致重复
        for line in resp.iter_lines(decode_unicode=False):
            if not line:
                continue
            if not line.startswith(b"data:"):
                continue
            try:
                data_obj = json.loads(line[5:])
            except Exception:
                continue

            payload = data_obj.get("data") or {}
            status = payload.get("status")
            audio_hex = payload.get("audio")

            # 仅在分片块（status == 1）输出
            if status == 1 and audio_hex:
                try:
                    yield bytes.fromhex(audio_hex)
                except ValueError:
                    continue
            # 收到结束块（status == 2）后退出循环
            if status == 2:
                break


@app.post("/tts/stream")
def tts_stream(req: OnlyTextRequest):
    body = _build_stream_body(req)
    generator = _iter_minimaxi_stream(body)
    # 返回原始音频字节流，客户端可直接播放/保存
    return StreamingResponse(generator, media_type="audio/mpeg")


@app.post("/tts/synthesize")
def tts_synthesize(req: OnlyTextRequest):
    headers = _build_headers(MINIMAXI_API_KEY)
    body = _build_nonstream_body(req)
    resp = requests.post(MINIMAXI_TTS_URL, headers=headers, data=body)

    if resp.status_code != 200:
        try:
            detail = resp.json()
        except Exception:
            detail = {"status_code": resp.status_code, "text": resp.text}
        raise HTTPException(status_code=resp.status_code, detail=detail)

    try:
        payload = resp.json()
    except Exception:
        raise HTTPException(status_code=500, detail="TTS 返回解析失败")

    # 非流式：固定 output_format=url，直接透传官方 JSON（包含可下载链接）
    return JSONResponse(content=payload)


@app.get("/health")
def health():
    return {"status": "ok"}


"""
运行示例（PowerShell / NuShell 类似）：

  setx MINIMAXI_API_KEY "<你的api_key>"
  uvicorn tts_service:app --host 0.0.0.0 --port 8000

调用示例：

  POST http://localhost:8000/tts/stream
  {
    "text": "今天是不是很开心呀，当然了！",
    "voice_setting": {"voice_id": "male-qn-qingse", "emotion": "happy"}
  }

  POST http://localhost:8000/tts/synthesize
  {
    "text": "测试非流式返回 URL",
    "output_format": "url"
  }
"""


