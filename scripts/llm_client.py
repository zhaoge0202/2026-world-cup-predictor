"""
LLM 客户端 — OpenAI 兼容（grok-4.3-high）

配置从本地 .env 读取（GROK_API_BASE / GROK_API_KEY / GROK_MODEL），
.env 已被 .gitignore 忽略，密钥绝不进入代码或 git。
"""

import os
import json
import time
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_env(path):
    cfg = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    cfg[k.strip()] = v.strip()
    return cfg


_ENV = _load_env(os.path.join(ROOT, ".env"))
BASE = os.environ.get("GROK_API_BASE") or _ENV.get("GROK_API_BASE", "")
KEY = os.environ.get("GROK_API_KEY") or _ENV.get("GROK_API_KEY", "")
MODEL = os.environ.get("GROK_MODEL") or _ENV.get("GROK_MODEL", "grok-4.3-high")

_CONFIGURED = bool(BASE and KEY)


def is_configured() -> bool:
    return _CONFIGURED


def chat(messages, temperature=0.2, max_tokens=2000, timeout=120,
         json_mode=False, retries=2):
    """调用 LLM，返回文本内容。失败抛异常（调用方决定降级）。"""
    if not _CONFIGURED:
        raise RuntimeError("LLM 未配置（缺 .env 中 GROK_API_BASE/KEY）")
    url = BASE.rstrip("/") + "/chat/completions"
    payload = {
        "model": MODEL, "messages": messages,
        "temperature": temperature, "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
    last = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            last = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise last


def chat_json(messages, **kw):
    """要求返回 JSON，解析后返回 dict（容错提取 {...}）"""
    txt = chat(messages, json_mode=True, **kw)
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        a, b = txt.find("{"), txt.rfind("}")
        if a >= 0 and b > a:
            return json.loads(txt[a:b + 1])
        raise


if __name__ == "__main__":
    print(f"BASE={BASE}  MODEL={MODEL}  configured={_CONFIGURED}")
    t0 = time.time()
    out = chat([{"role": "user", "content": "用一句话回答：2024 年欧洲杯冠军是哪支球队？只回答队名。"}],
               max_tokens=100)
    print(f"[基础连通 {time.time()-t0:.1f}s] {out!r}")

    t0 = time.time()
    j = chat_json([{"role": "user",
                    "content": '返回 JSON：{"home_win":0.x,"draw":0.x,"away_win":0.x} '
                               '估计西班牙 vs 摩洛哥 的胜平负概率（西班牙为 home）。只返回 JSON。'}],
                  max_tokens=200)
    print(f"[JSON 模式 {time.time()-t0:.1f}s] {j}")
