"""
世界杯 2026 实时比分提供器

双源策略:
  主源: TheSportsDB (公开 key=123, 30 req/min)
  备源: Football-Data.org (env FOOTBALL_DATA_KEY, 10 req/min)

特性:
  - 内存缓存 60s TTL（前端 60s 轮询时，对外 API 调用 1 次/分钟）
  - 双源失败返回空数组，不抛异常
  - 统一输出格式
"""

import os
import time
import requests
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Any, Optional

THESPORTSDB_URL = "https://www.thesportsdb.com/api/v1/json/123/eventsday.php"
FOOTBALL_DATA_URL = "https://api.football-data.org/v4/competitions/WC/matches"
ROOT = Path(__file__).resolve().parents[1]

HEADERS = {
    "User-Agent": "WorldCupPredictorBot/1.0 (Hermes AI Agent; educational project)"
}

CACHE_TTL_SECONDS = 60

# 自适应 TTL：有 live 比赛刷得快，空闲刷得慢
TTL_LIVE = 3        # 有进行中比赛（压在 TheSportsDB 30/min 限流下：20 req/min）
TTL_SCHEDULED = 30  # 当日有未开赛比赛
TTL_IDLE = 300      # 当日无比赛


def _is_live_status(status: str) -> bool:
    s = (status or "").upper()
    return (
        "LIVE" in s
        or "PLAY" in s
        or s in ("1H", "2H", "HT", "ET", "P", "PAUSED", "BT")
    )


def _is_scheduled_status(status: str) -> bool:
    s = (status or "").upper()
    return s in ("NS", "SCHEDULED", "TIMED", "TBD", "")


def _normalize_thesportsdb(event: Dict[str, Any]) -> Dict[str, Any]:
    """TheSportsDB 字段 → 统一格式"""
    home_score = event.get("intHomeScore")
    away_score = event.get("intAwayScore")
    return {
        "date": event.get("dateEvent"),
        "time": (event.get("strTimestamp") or "")[11:16],
        "team_home": event.get("strHomeTeam") or "",
        "team_away": event.get("strAwayTeam") or "",
        "score_home": int(home_score) if home_score not in (None, "") else None,
        "score_away": int(away_score) if away_score not in (None, "") else None,
        "status": event.get("strStatus") or "NS",
        "minute": event.get("strProgress") or "",
        "venue": event.get("strVenue") or "",
        "league": event.get("strLeague") or "",
        "red_home": None,
        "red_away": None,
        "yellow_home": None,
        "yellow_away": None,
        "xg_home": None,
        "xg_away": None,
        "referee": event.get("strReferee") or None,
        "source": "thesportsdb",
    }


def _normalize_football_data(match: Dict[str, Any]) -> Dict[str, Any]:
    """Football-Data.org 字段 → 统一格式"""
    score = (match.get("score") or {}).get("fullTime") or {}
    utc = match.get("utcDate") or ""
    return {
        "date": utc[:10],
        "time": utc[11:16],
        "team_home": (match.get("homeTeam") or {}).get("name") or "",
        "team_away": (match.get("awayTeam") or {}).get("name") or "",
        "score_home": score.get("home"),
        "score_away": score.get("away"),
        "status": match.get("status") or "SCHEDULED",
        "minute": str(match.get("minute") or ""),
        "venue": match.get("venue") or "",
        "league": "FIFA World Cup",
        "red_home": None,
        "red_away": None,
        "yellow_home": None,
        "yellow_away": None,
        "xg_home": None,
        "xg_away": None,
        "referee": match.get("referees", [{}])[0].get("name") if match.get("referees") else None,
        "source": "football-data",
    }


def _fetch_thesportsdb(target_date: str) -> Optional[List[Dict[str, Any]]]:
    """主源: 抓某日 Soccer 赛事并过滤出 FIFA World Cup"""
    try:
        resp = requests.get(
            THESPORTSDB_URL,
            params={"d": target_date, "s": "Soccer"},
            headers=HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json() or {}
        events = data.get("events") or []
        wc = [
            _normalize_thesportsdb(e)
            for e in events
            if "World Cup" in (e.get("strLeague") or "")
        ]
        return wc
    except Exception as e:
        print(f"⚠️ TheSportsDB 抓取失败: {e}")
        return None


def _fetch_football_data(target_date: str) -> Optional[List[Dict[str, Any]]]:
    """备源: 需要 env FOOTBALL_DATA_KEY，未设置直接跳过"""
    key = _football_data_key()
    if not key:
        return None
    try:
        resp = requests.get(
            FOOTBALL_DATA_URL,
            params={"dateFrom": target_date, "dateTo": target_date},
            headers={**HEADERS, "X-Auth-Token": key},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json() or {}
        return [_normalize_football_data(m) for m in (data.get("matches") or [])]
    except Exception as e:
        print(f"⚠️ Football-Data.org 抓取失败: {e}")
        return None


def _football_data_key(env_path: os.PathLike = ROOT / ".env") -> str:
    key = os.environ.get("FOOTBALL_DATA_KEY")
    if key:
        return key
    try:
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "FOOTBALL_DATA_KEY":
                    return v.strip()
    except FileNotFoundError:
        pass
    return ""


class LiveScoresProvider:
    """带 60s 内存缓存的实时比分提供器"""

    def __init__(self, ttl: int = CACHE_TTL_SECONDS):
        self.ttl = ttl
        self._cache: Dict[str, List[Dict[str, Any]]] = {}
        self._cache_ts: Dict[str, float] = {}
        self._cache_ttl: Dict[str, int] = {}

    def _cache_key(self, target_date: str) -> str:
        return f"scores:{target_date}"

    @staticmethod
    def _ttl_for(scores: List[Dict[str, Any]]) -> int:
        """按内容决定下次刷新间隔"""
        if any(_is_live_status(s.get("status", "")) for s in scores):
            return TTL_LIVE
        if any(_is_scheduled_status(s.get("status", "")) for s in scores):
            return TTL_SCHEDULED
        return TTL_IDLE

    def get_scores(self, target_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取某日所有世界杯比赛（默认今日，本地时区）
        返回统一格式数组；双源失败返回 []
        缓存 TTL 自适应：live 3s / 未开赛 30s / 空闲 300s
        """
        if target_date is None:
            target_date = date.today().isoformat()

        key = self._cache_key(target_date)
        now = time.time()
        ttl = self._cache_ttl.get(key, self.ttl)
        if key in self._cache and (now - self._cache_ts.get(key, 0)) < ttl:
            return self._cache[key]

        # 主源 TheSportsDB
        scores = _fetch_thesportsdb(target_date)

        # fallback Football-Data.org
        if scores is None or len(scores) == 0:
            fd = _fetch_football_data(target_date)
            if fd is not None and len(fd) > 0:
                scores = fd
            elif scores is None:
                scores = []

        self._cache[key] = scores
        self._cache_ts[key] = now
        self._cache_ttl[key] = self._ttl_for(scores)
        return scores

    def get_today_scores(self) -> List[Dict[str, Any]]:
        return self.get_scores(None)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="查询某日世界杯实时比分")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD，默认今日")
    args = parser.parse_args()

    provider = LiveScoresProvider()
    scores = provider.get_scores(args.date)
    print(f"📊 {args.date or 'today'} — {len(scores)} 场比赛")
    print(json.dumps(scores, ensure_ascii=False, indent=2))

    # 缓存命中验证
    t0 = time.time()
    provider.get_scores(args.date)
    print(f"\n⚡ 第二次调用（缓存）: {(time.time()-t0)*1000:.1f}ms")
