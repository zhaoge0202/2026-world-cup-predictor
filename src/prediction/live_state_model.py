from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Optional, Tuple

from .match_model import MAX_GOALS, build_score_matrix, top_scorelines

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_REFEREE_PATH = os.path.join(ROOT, "data", "referee_profiles.json")

LIVE_STATUSES = {"LIVE", "IN_PLAY", "1H", "2H", "HT", "ET", "PAUSED", "BT"}


def _float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    parsed = _float(value)
    return int(parsed) if parsed is not None else default


def _minute(value: Any) -> int:
    if isinstance(value, str):
        digits = "".join(ch for ch in value if ch.isdigit())
        if not digits:
            return 0
        value = digits
    return max(0, min(130, _int(value, 0)))


def _is_live(status: Any) -> bool:
    s = str(status or "").upper()
    return s in LIVE_STATUSES or "LIVE" in s or "PLAY" in s


def _home_maps_to_team1(base_match: dict, live_state: dict) -> bool:
    home = str(live_state.get("team_home") or "").strip()
    away = str(live_state.get("team_away") or "").strip()
    team1 = str(base_match.get("team1") or "").strip()
    team2 = str(base_match.get("team2") or "").strip()
    if home == team1 or away == team2:
        return True
    if home == team2 or away == team1:
        return False
    return True


def _current_score(base_match: dict, live_state: dict) -> Tuple[int, int]:
    home_score = _int(live_state.get("score_home"), 0)
    away_score = _int(live_state.get("score_away"), 0)
    if _home_maps_to_team1(base_match, live_state):
        return home_score, away_score
    return away_score, home_score


def _cards(base_match: dict, live_state: dict) -> Tuple[int, int]:
    red_home = _int(live_state.get("red_home"), 0)
    red_away = _int(live_state.get("red_away"), 0)
    if _home_maps_to_team1(base_match, live_state):
        return red_home, red_away
    return red_away, red_home


def _xg(base_match: dict, live_state: dict) -> Tuple[Optional[float], Optional[float]]:
    xg_home = _float(live_state.get("xg_home"))
    xg_away = _float(live_state.get("xg_away"))
    if _home_maps_to_team1(base_match, live_state):
        return xg_home, xg_away
    return xg_away, xg_home


def _renormalize(a: float, d: float, b: float) -> Tuple[float, float, float]:
    total = a + d + b
    if total <= 0:
        return 1 / 3, 1 / 3, 1 / 3
    return a / total, d / total, b / total


def load_referee_profiles(path: str = DEFAULT_REFEREE_PATH) -> Dict[str, dict]:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "referees" in payload:
        rows = payload.get("referees") or []
        return {str(r.get("name") or ""): r for r in rows if r.get("name")}
    if isinstance(payload, dict):
        return payload
    return {}


def _referee_adjustment(referee: Optional[dict]) -> float:
    if not referee:
        return 1.0
    red_rate = _float(referee.get("red_per_match"), 0.0) or 0.0
    penalty_rate = _float(referee.get("penalty_rate"), 0.0) or 0.0
    return max(0.9, min(1.12, 1.0 + red_rate * 0.04 + penalty_rate * 0.03))


def _remaining_lambdas(base_match: dict, live_state: dict, referee: Optional[dict]) -> Tuple[float, float, Dict[str, dict]]:
    minute = _minute(live_state.get("minute"))
    remaining = max(0.0, (96.0 - min(minute, 96)) / 96.0)
    lam1 = max(0.01, float(base_match.get("lambda_team1") or 1.2) * remaining)
    lam2 = max(0.01, float(base_match.get("lambda_team2") or 1.2) * remaining)
    adjustments: Dict[str, dict] = {"time_decay": {"minute": minute, "remaining_ratio": remaining}}

    red1, red2 = _cards(base_match, live_state)
    if red1 or red2:
        lam1 *= 0.82 ** red1
        lam2 *= 0.82 ** red2
        lam1 *= 1.10 ** red2
        lam2 *= 1.10 ** red1
        adjustments["red_card"] = {"team1_red": red1, "team2_red": red2}

    xg1, xg2 = _xg(base_match, live_state)
    minute_for_rate = max(15, minute)
    if xg1 is not None and xg2 is not None and minute > 0:
        projected1 = max(0.05, xg1 / minute_for_rate * 96.0)
        projected2 = max(0.05, xg2 / minute_for_rate * 96.0)
        remaining_xg1 = max(0.01, projected1 * remaining)
        remaining_xg2 = max(0.01, projected2 * remaining)
        lam1 = 0.55 * lam1 + 0.45 * remaining_xg1
        lam2 = 0.55 * lam2 + 0.45 * remaining_xg2
        adjustments["xg"] = {"team1_xg": xg1, "team2_xg": xg2, "weight": 0.45}

    ref_mult = _referee_adjustment(referee)
    if referee:
        lam1 *= ref_mult
        lam2 *= ref_mult
        adjustments["referee"] = {"name": referee.get("name"), "tempo_multiplier": ref_mult}

    return lam1, lam2, adjustments


def _live_score_matrix(current1: int, current2: int, remaining_lam1: float, remaining_lam2: float) -> Iterable[dict]:
    remaining_matrix = build_score_matrix(remaining_lam1, remaining_lam2, max_goals=MAX_GOALS)
    for row in remaining_matrix:
        g1 = current1 + int(row["team1_goals"])
        g2 = current2 + int(row["team2_goals"])
        yield {
            "team1_goals": g1,
            "team2_goals": g2,
            "score": f"{g1}-{g2}",
            "prob": row["prob"],
        }


def build_live_match_prediction(
    base_match: dict,
    live_state: Optional[dict] = None,
    referee_profiles: Optional[Dict[str, dict]] = None,
) -> dict:
    live_state = live_state or {}
    if not _is_live(live_state.get("status")):
        payload = dict(base_match)
        payload["source"] = "schedule"
        payload["live_available"] = False
        return payload

    referee_profiles = referee_profiles if referee_profiles is not None else load_referee_profiles()
    referee_name = live_state.get("referee")
    referee = referee_profiles.get(referee_name) if referee_name else None
    current1, current2 = _current_score(base_match, live_state)
    lam1, lam2, adjustments = _remaining_lambdas(base_match, live_state, referee)
    matrix = list(_live_score_matrix(current1, current2, lam1, lam2))
    team1_win = sum(r["prob"] for r in matrix if r["team1_goals"] > r["team2_goals"])
    draw = sum(r["prob"] for r in matrix if r["team1_goals"] == r["team2_goals"])
    team2_win = sum(r["prob"] for r in matrix if r["team1_goals"] < r["team2_goals"])
    team1_win, draw, team2_win = _renormalize(team1_win, draw, team2_win)
    expected1 = current1 + lam1
    expected2 = current2 + lam2

    return {
        **base_match,
        "source": "live-state",
        "live_available": True,
        "team1_win": team1_win,
        "draw": draw,
        "team2_win": team2_win,
        "lambda_team1": lam1,
        "lambda_team2": lam2,
        "current_score": {"team1": current1, "team2": current2},
        "expected_final_score": {
            "team1": round(expected1, 2),
            "team2": round(expected2, 2),
            "display": f"{expected1:.1f}-{expected2:.1f}",
        },
        "top_scores": top_scorelines(matrix, limit=6),
        "adjustments": adjustments,
        "data_quality": {
            "has_red_cards": live_state.get("red_home") is not None and live_state.get("red_away") is not None,
            "has_xg": live_state.get("xg_home") is not None and live_state.get("xg_away") is not None,
            "has_referee": bool(referee),
        },
    }
