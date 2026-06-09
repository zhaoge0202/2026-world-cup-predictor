"""
2026 世界杯赛程抓取脚本

数据源: openfootball/worldcup.json (GitHub 公开，无 API Key、无限流)
URL:    https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json

字段结构: { name, matches: [{round, date, time, team1, team2, group, ground, score?}] }
"""

import requests
import json
import os
import time
import threading
from datetime import datetime, date
from typing import Optional, List, Dict, Any, Callable

OPENFOOTBALL_2026_URL = (
    "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"
)

HEADERS = {
    "User-Agent": "WorldCupPredictorBot/1.0 (Hermes AI Agent; educational project)"
}


def fetch_fixtures() -> Optional[Dict[str, Any]]:
    """从 openfootball 拉取 2026 赛程 JSON"""
    try:
        resp = requests.get(OPENFOOTBALL_2026_URL, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"⚠️ 无法从 openfootball 获取赛程数据: {e}")
        return None


def save_fixtures(data: Dict[str, Any], cache_path: str) -> None:
    """保存赛程到本地缓存，附加 fetch_date 元信息"""
    payload = {
        "fetch_date": datetime.now().isoformat(timespec="seconds"),
        "source": "openfootball/worldcup.json",
        "name": data.get("name", "World Cup 2026"),
        "matches": data.get("matches", []),
    }
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_fixtures(cache_path: str) -> Optional[Dict[str, Any]]:
    """从本地缓存加载赛程；不存在返回 None"""
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, encoding="utf-8") as f:
        return json.load(f)


def fetch_and_save(cache_path: str) -> Optional[Dict[str, Any]]:
    """拉取并保存；返回完整 payload（含 fetch_date）"""
    data = fetch_fixtures()
    if data is None:
        return None
    save_fixtures(data, cache_path)
    return load_fixtures(cache_path)


def get_matches(cache_path: str, auto_refresh: bool = True) -> List[Dict[str, Any]]:
    """读缓存中的 matches 数组；不存在则自动拉取"""
    cached = load_fixtures(cache_path)
    if cached is None and auto_refresh:
        cached = fetch_and_save(cache_path)
    if cached is None:
        return []
    return cached.get("matches", [])


# ── 后台定时刷新 daemon ──────────────────────────────────────────────
# 赛事窗口内（含 score.ft 回填）刷得勤，平时一天一次
TOURNAMENT_START = date(2026, 6, 11)
TOURNAMENT_END = date(2026, 7, 19)
TOURNAMENT_INTERVAL = 1800   # 赛事期间 30 分钟
IDLE_INTERVAL = 86400        # 平时 24 小时


def _current_interval(tournament_interval: int, idle_interval: int) -> int:
    today = date.today()
    if TOURNAMENT_START <= today <= TOURNAMENT_END:
        return tournament_interval
    return idle_interval


def start_refresh_daemon(
    cache_path: str,
    on_update: Optional[Callable[[Dict[str, Any]], None]] = None,
    tournament_interval: int = TOURNAMENT_INTERVAL,
    idle_interval: int = IDLE_INTERVAL,
) -> threading.Thread:
    """
    启动后台 daemon 线程定时刷新赛程。
    每轮: 按当前日期决定 sleep 间隔 → 拉取 → 回填本地缓存 → on_update 回调。
    daemon=True，主进程退出时自动结束。
    """
    def _loop():
        while True:
            interval = _current_interval(tournament_interval, idle_interval)
            time.sleep(interval)
            try:
                payload = fetch_and_save(cache_path)
                if payload is not None:
                    matches = payload.get("matches", [])
                    played = sum(1 for m in matches if m.get("score"))
                    print(f"🔄 赛程已刷新: {len(matches)} 场（{played} 场有比分）")
                    if on_update is not None:
                        on_update(payload)
            except Exception as e:
                print(f"⚠️ 赛程刷新失败: {e}")

    t = threading.Thread(target=_loop, name="fixtures-refresh", daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="抓取 2026 世界杯赛程")
    parser.add_argument(
        "--output",
        default="data/wc2026_fixtures.json",
        help="输出 JSON 文件路径",
    )
    args = parser.parse_args()

    print(f"📥 拉取 openfootball 2026 赛程 → {args.output}")
    payload = fetch_and_save(args.output)
    if payload is None:
        print("❌ 抓取失败")
        raise SystemExit(1)

    matches = payload.get("matches", [])
    print(f"✅ 共 {len(matches)} 场比赛")
    if matches:
        first = matches[0]
        last = matches[-1]
        print(f"   首场: {first.get('date')} {first.get('team1')} vs {first.get('team2')}")
        print(f"   末场: {last.get('date')} {last.get('team1')} vs {last.get('team2')}")
