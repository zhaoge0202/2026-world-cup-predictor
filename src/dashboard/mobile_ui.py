"""
世界杯预测 — 移动端独立版（纯 HTML/CSS/JS，无 Gradio）
Apple Sports 深黑主题，7 Tab 完整功能
Champion | Factor | Mystic | H2H | Squad | Polymarket | Info

用法:
    cd ~/Desktop/world_cup_predictor
    python -m src.dashboard.mobile_ui
    本地访问: http://localhost:7862
"""

import http.server
import socketserver
import argparse
import os
import sys
import json
import time
import random
import threading
from datetime import datetime
from typing import Dict, List

# ── 项目路径 ───────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.models.player_scoring import Player, Squad
from src.models.team_scoring import score_all_teams, ModelWeights
from src.models.mystic_factor import MysticFactorEngine
from src.models.ucl_final_mentality import (
    compute_country_ucl_mentality_bonus,
    compute_final_mentality_signal,
)
from src.models.conformal import ConformalPredictor
from src.models.feature_attribution import attribute_all_teams
from src.models.match_data import load_match_data, FlashscoreParser
from scripts.elo_scraper import load_elo_cache
from scripts.ingest_wikipedia_squads import normalize_position
from scripts.fixtures_fetcher import load_fixtures, fetch_and_save, start_refresh_daemon
from scripts.live_scores_provider import LiveScoresProvider
from src.prediction.live_state_model import build_live_match_prediction
from src.prediction.schedule_model import generate_schedule_predictions
try:
    from scripts.realtime_predictor import (
        adjust_champion_probs,
        predict_match,
        CACHE as REALTIME_CACHE,
        CHAMPION_TTL,
        MATCH_TTL,
        _fresh,
        _load_cache,
    )
    _RT_AVAILABLE = True
except Exception as _e:
    _RT_AVAILABLE = False
    REALTIME_CACHE = os.path.join(ROOT, "data", "realtime_cache.json")
    CHAMPION_TTL = 10 * 60
    MATCH_TTL = 3 * 3600

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ── 常量 ───────────────────────────────────────────────────────────────
WIKI_DATA = os.path.join(ROOT, "data", "wc2026_players_processed.json")
ELO_CACHE = os.path.join(ROOT, "data", "elo_cache_2026.json")
FIXTURES_CACHE = os.path.join(ROOT, "data", "wc2026_fixtures.json")
FINAL_PRED = os.path.join(ROOT, "data", "wc2026_prediction_final.json")
SCHEDULE_PRED = os.path.join(ROOT, "data", "wc2026_schedule_predictions.json")
SCHEDULE_PRED_REFRESH_SECONDS = int(os.environ.get("WC_SCHEDULE_PRED_REFRESH_SECONDS", "60"))
SCHEDULE_PRED_SIMULATIONS = int(os.environ.get("WC_SCHEDULE_PRED_SIMULATIONS", "3000"))
ANALYSIS_REFRESH_SECONDS = int(os.environ.get("WC_ANALYSIS_REFRESH_SECONDS", "300"))

QUALIFIED_TEAMS = [
    "Argentina", "Brazil", "Uruguay", "Colombia", "Ecuador", "Paraguay",
    "France", "Germany", "Spain", "England", "Portugal", "Netherlands",
    "Belgium", "Croatia", "Switzerland", "Austria", "Czech Republic", "Turkey",
    "Sweden", "Morocco", "Senegal", "Algeria", "Egypt", "Ghana",
    "Ivory Coast", "Tunisia", "DR Congo", "Cape Verde", "Japan", "South Korea",
    "Iran", "Iraq", "Qatar", "Saudi Arabia", "Australia", "Uzbekistan", "Jordan",
    "USA", "Mexico", "Canada", "Panama", "Curaçao", "Haiti",
    "New Zealand", "Norway", "South Africa", "Bosnia and Herzegovina", "Scotland",
]




HOST_COUNTRY = "USA"
DEFENDING_CHAMPION = "Argentina"

FLAG = {
    "Brazil": "🇧🇷", "Argentina": "🇦🇷", "France": "🇫🇷",
    "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Germany": "🇩🇪", "Spain": "🇪🇸",
    "Portugal": "🇵🇹", "Netherlands": "🇳🇱",
    "Belgium": "🇧🇪", "Croatia": "🇭🇷", "Switzerland": "🇨🇭",
    "Uruguay": "🇺🇾", "Colombia": "🇨🇴", "Mexico": "🇲🇽", "USA": "🇺🇸",
    "Japan": "🇯🇵", "South Korea": "🇰🇷", "Australia": "🇦🇺", "Iran": "🇮🇷",
    "Morocco": "🇲🇦", "Senegal": "🇸🇳", "Egypt": "🇪🇬",
    "Austria": "🇦🇹",
    "Czech Republic": "🇨🇿", "Turkey": "🇹🇷", "Sweden": "🇸🇪",
    "Ecuador": "🇪🇨", "Paraguay": "🇵🇾", "Saudi Arabia": "🇸🇦", "Qatar": "🇶🇦",
    "Ivory Coast": "🇨🇮", "Ghana": "🇬🇭", "Tunisia": "🇹🇳",
    "Algeria": "🇩🇿", "DR Congo": "🇨🇩", "Cape Verde": "🇨🇻",
    "Uzbekistan": "🇺🇿", "Jordan": "🇯🇴", "Panama": "🇵🇦",
    "Curaçao": "🇨🇼", "Haiti": "🇭🇹",
    "New Zealand": "🇳🇿", "Iraq": "🇮🇶", "Norway": "🇳🇴", "South Africa": "🇿🇦",
    "Canada": "🇨🇦",
    "Bosnia and Herzegovina": "🇧🇦", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
}

# ── 辅助函数 ───────────────────────────────────────────────────────────
def _infer_tournaments(caps: int, age: int):
    if age >= 25 and caps >= 10:
        return ["2022"]
    if age >= 23 and caps >= 20:
        return ["2022"]
    if age >= 22 and caps >= 30:
        return ["2022"]
    return []

def _estimate_mv(pos: str, caps: int, age: int) -> float:
    base = {"GK": 8, "DF": 12, "MF": 15, "FW": 20}.get(pos, 10)
    caps_factor = min(2.0, caps / 30 + 0.5)
    age_factor = 1.5 if 27 <= age <= 29 else (1.2 if 24 <= age <= 26 else (0.8 if age > 31 else 0.9))
    return round(base * caps_factor * age_factor, 1)

def _build_sample(country: str, elo: float):
    """
    Deterministic sample generator keyed on country name.
    Ensures consistent exp_score per ELO band across runs:
      - elo > 1850 (top tier): 18/23 tournament players -> exp ~0.063
      - 1750 < elo <= 1850 (mid tier): 12/23 tournament players -> exp ~0.042
      - elo <= 1750 (low tier): 6/23 tournament players -> exp ~0.021
    Caps are deterministic per country name (no random variation across restarts).
    """
    seed = hash(country) % 1000
    rng = random.Random(seed)

    if elo > 1850:
        # 18 experienced (caps 30-150), 5 developing (caps 0-25)
        caps_bands = [30, 60, 90, 120, 150] * 3 + [60] * 3  # 18
        caps_bands += [0, 5, 10, 15, 20, 25][:5]              # 5 = 23 total
        age_mean, age_std = 27, 4
        age_min, age_max = 22, 34
        tournaments = ["2022"] * 18 + [[]] * 5
    elif elo > 1750:
        # 12 experienced (caps 20-80), 11 developing (caps 0-35)
        caps_bands = [20, 35, 50, 65, 80] * 2 + [25, 40, 55, 70]  # 14
        caps_bands += [0, 5, 10, 15, 20, 25, 30, 35, 40][:9]        # 9 = 23 total
        age_mean, age_std = 26, 5
        age_min, age_max = 21, 35
        tournaments = ["2022"] * 12 + [[]] * 11
    else:
        # 6 experienced (caps 15-65), 17 youth (caps 0-15)
        caps_bands = [15, 25, 35, 45, 55, 65]                        # 6
        extra_youth = [0, 3, 6, 9, 12, 15]
        for _ in range(2):
            extra_youth += [0, 3, 6, 9, 12, 15]
        caps_bands += extra_youth[:17]                                 # 17 = 23 total
        age_mean, age_std = 25, 5
        age_min, age_max = 20, 36
        tournaments = ["2022"] * 6 + [[]] * 17

    rng.shuffle(caps_bands)
    positions = ['GK', 'CB', 'CB', 'LB', 'RB', 'DM', 'CM', 'CM', 'AM', 'LW', 'RW', 'ST']
    rng.shuffle(positions)

    players = []
    for i in range(23):
        caps = max(0, caps_bands[i] + rng.randint(-5, 5))
        age = max(age_min, min(age_max, int(rng.gauss(age_mean, age_std))))
        has_tournament = bool(tournaments[i])
        players.append({
            "name": f"P{i+1}_{country[:3]}",
            "age": age,
            "position": positions[i % len(positions)],
            "club": "Club",
            "market_value": max(0.5, rng.uniform(5, 60)),
            "national_caps": caps,
            "national_goals": rng.randint(0, caps // 3) if caps > 0 else 0,
            "tournaments": ["2022"] if has_tournament else [],
        })

    coaching_factor = 0.4 + (hash(country) % 1000) / 1000.0 * 0.5
    return {
        "country": country,
        "players": players,
        "elo": elo,
        "coaching_factor": coaching_factor,
    }

# ── UCL 心态数据加载 ────────────────────────────────────────────────────
def _load_ucl_data():
    """返回 {国家: {total_bonus, description, players}}"""
    UCL_COUNTRIES = ["France", "England", "Georgia", "Portugal", "Italy", "Argentina"]
    UCL_DESCS = {
        "France": "PSG 5-0 Inter Milan - Mentality Explosion",
        "Argentina": "Inter 0-5 PSG final loss - pressure signal",
    }
    result = {}
    for eng_name in UCL_COUNTRIES:
        bonus = compute_country_ucl_mentality_bonus(eng_name)
        if bonus.get("signal_count", 0) > 0:
            players = []
            for sig in bonus.get("signals", []):
                players.append({
                    "name": sig.player_name,
                    "mentality_signal": sig.mentality_score,
                    "framework": sig.nearest_framework,
                    "tier": sig.tier_label,
                    "wc_adjustment": sig.wc_prob_adjustment,
                })
            wc_adj = bonus.get("wc_total_adjustment", 0.0)
            result[eng_name] = {
                "total_bonus": wc_adj,
                "description": UCL_DESCS.get(eng_name, "UCL final mentality signal"),
                "players": players,
            }
    return result

# ── 主数据加载 ──────────────────────────────────────────────────────────
_cached_results = None

def _load_analysis():
    """返回 (results, ucl_data)

    results 每条包含:
      country, elo, prob, final_prob, shift, logical_prob,
      verdict, zen, tao, iching, confidence,
      contrarian, fav_curse,
      elo_score, age_score, exp_score, form_score, coach_score, mystic_score,
      narrative, ci_low, ci_high,
      players (top 15 by caps)
    """
    global _cached_results
    if _cached_results is not None:
        return _cached_results

    # 1. Wiki data
    wiki_data = {}
    if os.path.exists(WIKI_DATA):
        with open(WIKI_DATA, encoding="utf-8") as f:
            wiki_data = json.load(f)

    # 2. Elo
    elo_dict = load_elo_cache(ELO_CACHE) or {}

    # 3. Build squads (as dict first, for JSON serialization)
    teams_data = wiki_data.get("teams", {})
    squad_dicts = {}
    for country in QUALIFIED_TEAMS:
        elo = elo_dict.get(country, 1650.0)
        if country in teams_data:
            players_raw = teams_data[country].get("players", [])
            players = []
            for p in players_raw:
                age = p.get("age")
                if not age:
                    continue
                caps = p.get("caps", 0)
                pos = normalize_position(p.get("position", "MF"))
                tournaments = _infer_tournaments(caps, age)
                mv = _estimate_mv(pos, caps, age)
                players.append({
                    "name": p["name"],
                    "age": age,
                    "position": pos,
                    "club": p.get("club", "Unknown"),
                    "market_value": mv,
                    "national_goals": p.get("goals", 0),
                    "national_caps": caps,
                    "tournaments": tournaments,
                })
            players.sort(key=lambda x: x["national_caps"], reverse=True)
            players = players[:15]  # top 15 for JSON size
            if players:
                coach_hash = hash(country) % 1000 / 1000.0
                squad_dicts[country] = {
                    "country": country,
                    "players": players,
                    "elo": elo,
                    "coaching_factor": 0.4 + coach_hash * 0.5,
                }
                continue
        squad_dicts[country] = _build_sample(country, elo)

    # 4. Build Squad objects for scoring
    # Set tournament_history for pending teams (those using synthetic samples)
    # based on ELO tier, compensating for data gaps vs real-data teams:
    #   ELO > 1850: "Semi" (+2% via world_cup_semi)
    #   1750 < ELO <= 1850: "Quarter" (+1% via world_cup_quarter)
    #   ELO <= 1750: no boost
    #
    # 2026-05-30 数据层修复：手动覆盖已知真实历史的队
    # （这些队缺Wikipedia数据，但历史成就明确，不能按ELO档位自动分配）
    MANUAL_TOURNAMENT_HISTORY = {
        "Spain":       ["Semi", "Final"],     # 2018四强（早），2010冠军（晚）
        "Portugal":    ["Quarter", "Semi"],   # 2022十六强（早），2018四强（晚）
        "Germany":     ["Group"],              # 2022小组赛，2014冠军已是过去
        "England":     ["Semi", "Quarter"],    # 2018四强（早），2022十六强（晚）
        "Belgium":     ["Group", "Quarter"],  # 2022小组赛（早），2018季军（晚）
        "Netherlands": ["Quarter"],            # 2022十六强
    }
    for country in QUALIFIED_TEAMS:
        sq = squad_dicts[country]
        # Only override if not already set (real-data teams already have tournament_history=["2022"])
        if sq.get("tournament_history") is None:
            if country in MANUAL_TOURNAMENT_HISTORY:
                sq["tournament_history"] = MANUAL_TOURNAMENT_HISTORY[country]
            else:
                elo = sq.get("elo", 1650)
                if elo > 1850:
                    sq["tournament_history"] = ["Semi"]
                elif elo > 1750:
                    sq["tournament_history"] = ["Quarter"]
                else:
                    sq["tournament_history"] = []

    squad_objs = []
    for country in QUALIFIED_TEAMS:
        sq = squad_dicts[country]
        pl_objs = [
            Player(
                name=pp["name"],
                age=pp["age"],
                position=pp["position"],
                club=pp["club"],
                market_value=pp["market_value"],
                national_goals=pp["national_goals"],
                national_caps=pp["national_caps"],
                tournaments=pp["tournaments"],
            )
            for pp in sq["players"]
        ]
        squad_objs.append(Squad(
            country=sq["country"],
            players=pl_objs,
            elo=sq["elo"],
            recent_win_rate=0.3 + (sq["elo"] - 1500) / 1000 * 0.5,
            coaching_factor=sq["coaching_factor"],
            tournament_history=sq.get("tournament_history", ["2022"] if sq["country"] == DEFENDING_CHAMPION else []),
        ))

    # 4b. Load match data (Flashscore — WC results, WC fixtures, friendlies)
    # Build recent_results dict: { country -> [match_dict, ...] }
    # Each match_dict has: team_a, team_b, score_a, score_b (or None for fixtures)
    all_match_data = load_match_data()  # returns (fixtures, results, friendly_results)
    match_fixtures, match_results, friendly_results = all_match_data

    # Build per-team match lists for form_score calculation
    recent_results: Dict[str, List[Dict]] = {}
    for m in match_results + friendly_results:
        for team in [m["team_a"], m["team_b"]]:
            if team not in recent_results:
                recent_results[team] = []
            recent_results[team].append(m)

    # 5. Score
    weights = ModelWeights()
    scored = score_all_teams(
        squad_objs,
        weights=weights,
        host_team=HOST_COUNTRY,
        defending_champ=DEFENDING_CHAMPION,
        recent_results=recent_results,
    )

    # 6. UCL 心态 override 精确调参
    # 框架含义：
    #   正心态 → favorite_curse↑（减少强队诅咒压制）, contrarian↑（不过度自信）
    #   France（4 PSG 球员大胜5-0，心态强势）: +2% 位移目标
    #   Argentina（劳塔罗进球强势，但 Inter 输了，心态次强势）: -1.5% 位移目标
    #   England（萨卡阿森纳半决赛失利）: -1.5% 位移目标
    ucl_overrides = {
        "France": {
            "contrarian": 0.015,
            "favorite_curse": 0.025,
            "gs_volatility": 0.008,
            "knockout_unc": 0.003,
        },
        "Argentina": {
            "contrarian": 0.015,
            "favorite_curse": 0.025,
            "gs_volatility": 0.008,
            "knockout_unc": 0.003,
        },
        # Brazil: 5次世界杯冠军(1958/62/70/94/2002)，近届持续4强以内，但22年8强意外出局→略低于Argentina
        "Brazil": {
            "contrarian": 0.012,
            "favorite_curse": 0.020,
            "gs_volatility": 0.006,
            "knockout_unc": 0.002,
        },
        # England: 2024欧洲杯亚军，萨卡/贝林厄姆新生代崛起，心态强势→修正为正值
        "England": {
            "contrarian": 0.012,
            "favorite_curse": 0.020,
            "gs_volatility": 0.005,
            "knockout_unc": 0.002,
        },
    }

    engine = MysticFactorEngine()
    mystic_teams = [{
        "country": t.country,
        "elo": squad_dicts.get(t.country, {}).get("elo", 1700),
        "prob": t.final_probability,
        "avg_age": 27.0,
        "exp_ratio": 0.5,
        "is_host": (t.country == HOST_COUNTRY),
        "is_defending": (t.country == DEFENDING_CHAMPION),
        "is_first_tournament": (t.final_probability < 0.01),
    } for t in scored]

    mystic_results = engine.analyze(
        mystic_teams,
        stage="tournament",
        ucl_mentality_overrides=ucl_overrides,
    )
    mystic_map = {r.country: r for r in mystic_results}

    # 7. Merge results（含 factor scores + squad players）
    results = []
    for t in scored:
        r = mystic_map.get(t.country)
        sq_dict = squad_dicts.get(t.country, {})
        results.append({
            "country": t.country,
            "elo": sq_dict.get("elo", 1700),
            "mod_elo": t.mod_elo or sq_dict.get("elo", 1700),  # 因子修正 Elo，同步到 H2H
            "prob": t.final_probability,
            "final_prob": r.mystic_prob if r else t.final_probability,
            "shift": (r.mystic_prob - t.final_probability) if r else 0,
            "logical_prob": t.final_probability,
            "verdict": r.verdict if r else "—",
            "zen": r.zen.final_recommendation if r else "—",
            "tao": r.tao.tao_recommendation if r else "—",
            "iching": "".join(r.iching.hexagram[:2]) if r else "—",
            "confidence": r.confidence if r else 0.5,
            "contrarian": r.contrarian_shift if r else 0,
            "fav_curse": r.favorite_curse if r else 0,
            # Factor scores（来自 TeamResult）
            "elo_score": t.elo_score,
            "age_score": t.age_score,
            "exp_score": t.experience_score,
            "form_score": t.form_score,
            "coach_score": t.coaching_score,
            "mystic_score": t.mystic_score,
            "narrative": getattr(t, 'narrative', '') or '',
            "ci_low": t.confidence_interval[0] if t.confidence_interval else 0,
            "ci_high": t.confidence_interval[1] if t.confidence_interval else 0,
            # Squad players（top 15 by caps）
            "players": sq_dict.get("players", [])[:15],
        })

    results.sort(key=lambda x: x["final_prob"], reverse=True)

    # 9. Conformal Prediction — 冠军概率置信区间
    conformal = ConformalPredictor()
    cal_info = conformal.calibrate()
    champion_intervals = conformal.predict_champion_intervals(results)
    interval_map = {c.country: c for c in champion_intervals}
    for r in results:
        ci = interval_map.get(r["country"])
        if ci:
            r["conformal_ci_low"] = ci.conformal_interval[0]
            r["conformal_ci_high"] = ci.conformal_interval[1]
            r["conformal_uncertainty"] = ci.uncertainty_level
        else:
            r["conformal_ci_low"] = r["ci_low"]
            r["conformal_ci_high"] = r["ci_high"]
            r["conformal_uncertainty"] = "medium"

    # 10. Feature Attribution — 因子绝对贡献归因
    attributions = attribute_all_teams(results)
    attr_map = {a["country"]: a for a in attributions}
    for r in results:
        r["attribution"] = attr_map.get(r["country"])

    # 11. H2H Conformal Prediction — 所有球队两两 H2H 的预测集
    # 为每个 team 预计算其对所有其他队的 H2H conformal 结果
    # 格式: { teamA: { teamB: { prediction_set, set_size, confidence, explanation } } }
    h2h_conformal_map: Dict = {}
    elo_dict_h2h = {r["country"]: r.get("mod_elo", r.get("elo", 1700)) for r in results}
    for r in results:
        country_a = r["country"]
        elo_a = r.get("mod_elo", r.get("elo", 1700))
        h2h_conformal_map[country_a] = {}
        for r2 in results:
            if r2["country"] == country_a:
                continue
            country_b = r2["country"]
            elo_b = r2.get("mod_elo", r2.get("elo", 1700))
            cp = conformal.predict_h2h(country_a, country_b, elo_a, elo_b)
            h2h_conformal_map[country_a][country_b] = cp.to_dict()

    # 8. UCL
    ucl_data = _load_ucl_data()

    _cached_results = (results, ucl_data, h2h_conformal_map,
                       match_fixtures, match_results, friendly_results)
    return results, ucl_data, h2h_conformal_map, match_fixtures, match_results, friendly_results


# ═══════════════════════════════════════════════════════════════════════════
# 纯 HTML/CSS/JS 移动端界面（7 Tab）
# ═══════════════════════════════════════════════════════════════════════════

HTML_BODY = r'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>世界杯 2026 / WC 2026</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#000;--s:#111;--s2:#1c1c1e;--bd:#2c2c2e;--tx:#fff;--tx2:#8e8e93;--tx3:#48484a;--bl:#0a84ff;--gr:#30d158;--rd:#ff453a;--gd:#ffd60a;--sl:#98989d;--br:#ac8e68}
html,body{height:100%;background:var(--bg);color:var(--tx);font-family:"Inter",-apple-system,sans-serif;overflow:hidden}
.hdr{position:fixed;top:0;left:0;right:0;z-index:100;background:rgba(0,0,0,0.9);backdrop-filter:blur(24px);border-bottom:0.5px solid var(--bd);padding:14px 20px 12px}
.hdr-title{font-size:19px;font-weight:800;letter-spacing:-0.4px}
.hdr-sub{font-size:11px;color:var(--tx2);margin-top:3px}
.tabbar{position:fixed;bottom:0;left:0;right:0;z-index:100;background:rgba(0,0,0,0.9);backdrop-filter:blur(24px);border-top:0.5px solid var(--bd);display:flex}
.tab{flex:1;display:flex;flex-direction:column;align-items:center;padding:10px 0 8px;gap:3px;border:none;background:none;color:var(--tx3);font-size:9px;font-weight:600;cursor:pointer;-webkit-tap-highlight-color:transparent;transition:color 0.15s}
.tab.on{color:var(--bl)}
.ico{font-size:20px;line-height:1}
.pg{display:none;height:100vh;overflow-y:auto;padding:68px 16px 88px;-webkit-overflow-scrolling:touch}
.pg.on{display:block}
.card{background:var(--s);border-radius:16px;border:0.5px solid var(--bd);padding:16px;margin-bottom:12px}
.card-title{font-size:10px;font-weight:700;color:var(--tx2);text-transform:uppercase;letter-spacing:1.2px;margin-bottom:14px}
/* Leaderboard */
.lb{display:flex;flex-direction:column}
.lb-r{display:flex;align-items:center;padding:11px 0;border-bottom:0.5px solid var(--bd);gap:10px}
.lb-r:last-child{border-bottom:none}
.lb-rk{font-size:14px;font-weight:800;color:var(--tx2);width:26px;text-align:center;flex-shrink:0}
.lb-rk.t1{color:var(--gd)}.lb-rk.t2{color:var(--sl)}.lb-rk.t3{color:var(--br)}
.lb-fl{font-size:22px;flex-shrink:0;width:30px;text-align:center}
.lb-inf{flex:1;min-width:0}
.lb-nm{font-size:15px;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.lb-el{font-size:11px;color:var(--tx2);margin-top:2px}
.pb{height:3px;background:var(--bd);border-radius:2px;margin-top:5px}
.pb-fi{height:100%;border-radius:2px}
.lb-pr{text-align:right;flex-shrink:0;min-width:60px}
.lb-pct{font-size:17px;font-weight:800;font-variant-numeric:tabular-nums}
.lb-pct.vh{color:var(--bl)}
.lb-sh{font-size:11px;font-weight:600;margin-top:2px}
/* Dynamic match sections */
/* ── SofaScore-inspired Match Cards ── */
.dyn-list{display:flex;flex-direction:column;gap:2px}
.dyn-date-hd{font-size:11px;font-weight:800;color:var(--tx2);padding:10px 0 4px 0;text-transform:uppercase;letter-spacing:.8px;border-bottom:.5px solid var(--bd);margin-bottom:4px}
.dyn-mo{color:var(--tx2);font-weight:400}

/* Match Card Row */
.mc-row{display:grid;grid-template-columns:60px 1fr auto 1fr;align-items:center;gap:6px;padding:10px 0;border-bottom:.5px solid var(--bd)}
.mc-row:last-child{border-bottom:none}
.mc-row.mc-frn{opacity:.9}

/* Round Badge */
.mc-rd{flex-shrink:0}
.mc-badge{display:inline-block;font-size:9px;font-weight:800;padding:3px 6px;border-radius:4px;color:#fff;white-space:nowrap;text-transform:uppercase;letter-spacing:.4px}

/* Team */
.mc-team{display:flex;align-items:center;gap:5px;min-width:0}
.mc-team.mc-tl{justify-content:flex-start}
.mc-team.mc-tr{justify-content:flex-end}
.mc-fl{font-size:18px;line-height:1;flex-shrink:0}
.mc-nm{font-size:13px;font-weight:700;color:var(--tx1);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:90px}
.mc-nm b{font-weight:800}
.mc-winner .mc-nm{color:var(--gr)}

/* Center: VS / Time / Score */
.mc-ct{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:1px;min-width:52px}
.mc-vs{font-size:10px;font-weight:800;color:var(--tx2);letter-spacing:.5px}
.mc-kick{font-size:12px;font-weight:700;color:var(--tx1)}
.mc-score-ct{flex-direction:row;gap:4px}
.mc-scr{font-size:16px;font-weight:800;color:var(--tx1);min-width:16px;text-align:center}
.mc-scr-sep{font-size:14px;font-weight:700;color:var(--tx2)}
.mc-scr.mc-sc-awin,.mc-scr.mc-sc-bwin{color:var(--gd)}
.mc-ft{font-size:9px;font-weight:800;color:var(--tx2);margin-top:1px}

/* Old dyn-* styles (keep for fallback) */
.dyn-sect{padding:0}
.dyn-empty{color:var(--tx2);font-size:13px;text-align:center;padding:16px 0}
.dyn-team{color:var(--tx1)}
.dyn-team.dyn-win{color:var(--gr);font-weight:800}
.dyn-vs{color:var(--tx2);font-size:12px;font-weight:400}
.dyn-score{font-size:16px;font-weight:800;color:var(--gd);white-space:nowrap}
.dyn-meta{display:flex;gap:8px;font-size:11px;color:var(--tx2)}
.dyn-date{font-weight:600}
.dyn-time{opacity:0.7}
.dyn-rnd{opacity:0.7}
/* Factor breakdown */
.fb-r{display:flex;flex-direction:column;padding:12px 0;border-bottom:0.5px solid var(--bd);cursor:pointer}
.fb-r:last-child{border-bottom:none}
.fb-hd{display:flex;align-items:center;gap:8px;margin-bottom:6px}
.fb-fl{font-size:20px}
.fb-nm{font-size:14px;font-weight:700;flex:1}
.fb-pct{font-size:14px;font-weight:800;color:var(--bl)}
.fb-bars{display:flex;flex-direction:column;gap:5px}
.fb-bar{display:flex;align-items:center;gap:8px}
.fb-lbl{font-size:10px;color:var(--tx2);width:52px;flex-shrink:0;font-weight:600}
.fb-track{height:4px;background:var(--bd);border-radius:2px;flex:1}
.fb-fill{height:4px;border-radius:2px;transition:width 0.3s}
.fb-val{font-size:10px;font-weight:700;width:34px;text-align:right;flex-shrink:0}
.fb-expanded{display:none;padding:8px 0 4px}
.fb-expanded.on{display:block}
.fb-narrative{font-size:11px;color:var(--tx2);line-height:1.5;margin-top:6px;font-style:italic}
/* Mystic */
.mc-r{display:flex;align-items:center;gap:10px;padding:12px 0;border-bottom:0.5px solid var(--bd);cursor:pointer}
.mc-r:last-child{border-bottom:none}
.mc-fl{font-size:26px}
.mc-nm{font-size:15px;font-weight:700}
.mc-mt{font-size:12px;color:var(--tx2);margin-top:2px}
.mc-dt{display:none;padding:12px 0 4px}
.mc-dt.on{display:block}
.tags{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px}
.tag{font-size:11px;font-weight:700;padding:4px 10px;border-radius:20px}
.tag.pos{background:rgba(48,209,88,0.15);color:var(--gr)}
.tag.neg{background:rgba(255,69,58,0.15);color:var(--rd)}
.tag.neu{background:rgba(142,142,147,0.15);color:var(--tx2)}
.tag.mystic{background:rgba(255,214,10,0.15);color:var(--gd)}
.mtrics{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.mtric{background:var(--s2);border-radius:10px;padding:10px 12px}
.mtric-lbl{font-size:9px;color:var(--tx2);font-weight:700;text-transform:uppercase;letter-spacing:0.6px}
.mtric-val{font-size:16px;font-weight:800;margin-top:4px}
.mtric-val.pos{color:var(--gr)}
.mtric-val.neg{color:var(--rd)}
/* UCL */
.ucard{background:var(--s);border-radius:16px;border:0.5px solid var(--bd);padding:16px;margin-bottom:12px}
.ucard-fl{font-size:36px;margin-bottom:6px}
.ucard-nm{font-size:20px;font-weight:800}
.ucard-bns{font-size:15px;font-weight:800;margin-top:6px}
.ucard-bns.pos{color:var(--gr)}
.ucard-bns.neg{color:var(--rd)}
.ucard-dsc{font-size:11px;color:var(--tx2);margin-top:4px}
.urow{display:flex;align-items:center;gap:10px;padding:10px 0;border-bottom:0.5px solid var(--bd)}
.urow:last-child{border-bottom:none}
.unm{font-size:14px;font-weight:600}
.uclub{font-size:12px;color:var(--tx2)}
.ums{font-size:13px;font-weight:700;margin-left:auto;flex-shrink:0}
.ums.pos{color:var(--gr)}
.ums.neg{color:var(--rd)}
.fw{background:var(--s2);border-radius:12px;padding:14px}
.fw-tl{font-size:12px;font-weight:700;color:var(--gd);margin-bottom:8px}
.fw-it{font-size:12px;color:var(--tx2);line-height:1.8}
/* H2H */
.h2h Teams{display:flex;flex-direction:column;gap:12px;margin-bottom:16px}
.h2h Team{display:flex;flex-direction:column}
.h2h Team label{display:block;font-size:12px;font-weight:700;color:var(--tx2);text-transform:uppercase;letter-spacing:0.6px;margin-bottom:8px}
.h2h Team select{width:100%;background:var(--s2);color:var(--tx);border:0.5px solid var(--bd);border-radius:14px;padding:16px 18px;font-size:17px;font-weight:700;appearance:none;-webkit-appearance:none;cursor:pointer;line-height:1.4}
.h2h-vs{font-size:24px;font-weight:900;color:var(--gd);text-align:center;padding:4px 0}
.h2h-bar{display:flex;align-items:center;gap:0;border-radius:16px;overflow:hidden;height:52px;background:var(--s2);margin-bottom:14px}
.h2h-bar-a{flex:0 0 auto;display:flex;align-items:center;justify-content:center;padding:0 12px;height:100%;font-size:14px;font-weight:800;color:var(--tx);background:var(--bl);min-width:0;overflow:hidden}
.h2h-bar-d{flex:0 0 auto;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;color:var(--s);padding:0 10px;height:100%;background:var(--gd);min-width:0;overflow:hidden}
.h2h-bar-b{flex:0 0 auto;display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:800;color:var(--tx);padding:0 12px;background:var(--s2);min-width:0;overflow:hidden}
.h2h-3m{display:flex;gap:8px;margin-bottom:16px}
.h2h-3m .h2h-3m-it{flex:1;background:var(--s2);border-radius:12px;padding:12px 0;text-align:center}
.h2h-3m .h2h-3m-v{font-size:18px;font-weight:800;color:var(--tx)}
.h2h-3m .h2h-3m-l{font-size:10px;font-weight:600;color:var(--tx2);text-transform:uppercase;margin-top:4px}
.h2h-fc{display:flex;flex-direction:column;gap:8px;margin-bottom:16px}
.h2h-fr{display:flex;align-items:center;gap:8px;font-size:12px}
.h2h-fr-lbl{flex:0 0 70px;font-weight:700;color:var(--tx2)}
.h2h-fr-bar{flex:1;height:24px;background:var(--s2);border-radius:6px;overflow:hidden;display:flex}
.h2h-fr-a{height:100%;transition:width 0.4s}
.h2h-fr-b{height:100%;transition:width 0.4s}
.h2h-fr-val{display:flex;flex:0 0 60px;font-size:12px;font-weight:700;justify-content:flex-end;gap:4px}
.h2h-note{background:var(--s2);border-radius:12px;padding:14px 16px;font-size:13px;color:var(--tx2);line-height:1.7;margin-top:4px}
.h2h-note strong{color:var(--gd)}
.h2h-wl{font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:0.4px;padding:3px 7px;border-radius:6px;display:inline-block}
.h2h-wl.w{background:rgba(48,209,88,0.15);color:var(--gr)}
.h2h-wl.l{background:rgba(255,69,58,0.15);color:var(--rd)}
.h2h-wl.d{background:rgba(255,214,10,0.15);color:var(--gd)}
.h2h-matchup{padding:10px 0;border-bottom:0.5px solid var(--bd)}
.h2h-matchup:last-child{border-bottom:none}
.h2h-mu-pos{font-size:10px;font-weight:700;color:var(--tx2);text-transform:uppercase;letter-spacing:0.6px;margin-bottom:6px}
.h2h-mu-row{display:flex;align-items:center;gap:8px}
.h2h-mu-p{flex:1;font-size:13px;font-weight:600}
.h2h-mu-p .h2h-wl{margin-left:6px}
.h2h-mu-s{font-size:12px;font-weight:700;color:var(--gd);width:28px;text-align:center}
.h2h-mu-r{text-align:right;flex:1}
.h2h-mu-r .h2h-wl{margin-right:6px}
/* Conformal Prediction */
.cp-set-box{margin-top:4px}
.cp-set-hd{display:flex;align-items:center;justify-content:space-between;margin-bottom:6px}
.cp-set-lbl{font-size:10px;font-weight:700;color:var(--tx2);text-transform:uppercase;letter-spacing:0.8px}
.cp-set-badge{font-size:11px;font-weight:800;padding:3px 8px;border-radius:6px;letter-spacing:0.3px}
.cp-set-exp{font-size:12px;color:var(--tx2);margin-bottom:4px}
.cp-set-conf{font-size:11px;color:var(--tx3)}
/* Factor Attribution */
.attr-sect{margin-bottom:14px}
.attr-row{display:flex;align-items:center;gap:8px;font-size:12px;margin-bottom:5px}
.attr-lbl{flex:0 0 80px;font-weight:700;color:var(--tx2)}
.attr-bar{flex:1;height:18px;background:var(--s2);border-radius:4px;overflow:hidden;display:flex}
.attr-seg{height:100%;transition:width 0.3s}
.attr-seg-a{background:var(--bl)}
.attr-seg-d{background:var(--rd)}
.attr-seg-val{flex:0 0 48px;font-size:11px;font-weight:700;text-align:right;justify-content:flex-end;display:flex;gap:2px}
.attr-delta-p{font-size:11px;color:var(--gr)}
.attr-delta-n{font-size:11px;color:var(--rd)}
.attr-meta{font-size:10px;color:var(--tx3);margin-top:6px;padding-top:6px;border-top:0.5px solid var(--bd)}
.attr-meta span{margin-right:12px}
.attr-note{font-size:11px;color:var(--tx3);font-style:italic;margin-top:4px}
/* Squad */
.sel{width:100%;background:var(--s2);color:var(--tx);border:0.5px solid var(--bd);border-radius:12px;padding:12px 16px;font-size:15px;font-weight:600;margin-bottom:12px;appearance:none;-webkit-appearance:none}
.sel-wrap{position:relative}
.sel-wrap::after{content:"▼";position:absolute;right:16px;top:50%;transform:translateY(-50%);font-size:10px;color:var(--tx2);pointer-events:none}
.sq-card{background:var(--s);border-radius:16px;border:0.5px solid var(--bd);overflow:hidden;margin-bottom:12px}
.sq-ph{background:var(--s2);padding:12px 16px;display:flex;align-items:center;gap:12px}
.sq-ph-fl{font-size:28px}
.sq-ph-nm{font-size:16px;font-weight:800}
.sq-ph-elo{font-size:12px;color:var(--tx2);margin-top:2px}
.sq-table{width:100%}
.sq-th{background:var(--s2);padding:8px 12px;font-size:9px;font-weight:700;color:var(--tx2);text-transform:uppercase;letter-spacing:0.6px;text-align:left}
.sq-td{padding:9px 12px;font-size:13px;border-bottom:0.5px solid var(--bd)}
.sq-td:last-child{border-bottom:none}
.sq-pos{font-size:10px;font-weight:700;color:var(--tx2);width:28px}
.sq-name{font-weight:600}
.sq-club{font-size:11px;color:var(--tx2)}
.sq-mv{font-size:12px;font-weight:700;color:var(--gd);white-space:nowrap}
.sq-caps{text-align:right;font-variant-numeric:tabular-nums}
.sq-goals{text-align:right;font-variant-numeric:tabular-nums;color:var(--tx2)}
/* Info */
.info-sec{margin-bottom:24px}
.info-tl{font-size:11px;font-weight:700;color:var(--tx2);text-transform:uppercase;letter-spacing:1px;margin-bottom:10px}
.info-row{background:var(--s);border-radius:12px;padding:14px 16px;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center}
.info-lbl{font-size:14px;color:var(--tx2)}
.info-val{font-size:14px;font-weight:700;text-align:right}
.calibration{background:var(--s);border-radius:12px;padding:16px;margin-bottom:10px}
.cal-hd{font-size:13px;font-weight:800;margin-bottom:8px;display:flex;align-items:center;gap:8px}
.cal-bd{font-size:12px;color:var(--tx2);line-height:1.7}

/* H2H Team Picker */
.h2h-teams{display:flex;gap:12px;margin-bottom:18px;align-items:center}
.h2h-team{flex:1}
.h2h-team label{display:block;font-size:10px;font-weight:700;color:var(--tx2);text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px}
.h2h-pick{display:flex;align-items:center;gap:10px;background:var(--s2);border:1px solid var(--bd);border-radius:14px;padding:14px;min-height:72px;cursor:pointer;-webkit-tap-highlight-color:transparent;transition:border-color .15s}
.h2h-pick:active{background:var(--bd)}
.h2h-pick-fl{font-size:28px;flex-shrink:0;line-height:1}
.h2h-pick-info{flex:1;min-width:0}
.h2h-pick-nm{font-size:15px;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.h2h-pick-pr{font-size:12px;color:var(--tx2);margin-top:3px}
.h2h-pick-chevron{font-size:16px;color:var(--tx3);flex-shrink:0}
.h2h-vs{font-size:18px;font-weight:900;color:var(--gd);text-align:center;flex-shrink:0;padding-bottom:20px}
/* Picker overlay */
.pick-overlay{position:fixed;inset:0;z-index:200;background:rgba(0,0,0,.7);backdrop-filter:blur(8px);display:none;flex-direction:column;justify-content:flex-end}
.pick-overlay.on{display:flex}
.pick-sheet{background:var(--s);border-radius:20px 20px 0 0;max-height:75vh;display:flex;flex-direction:column;overflow:hidden}
.pick-sheet-hd{background:var(--s2);padding:16px 20px 14px;display:flex;align-items:center;justify-content:space-between;border-bottom:.5px solid var(--bd);flex-shrink:0}
.pick-sheet-tl{font-size:15px;font-weight:800}
.pick-sheet-close{background:none;border:none;color:var(--bl);font-size:14px;font-weight:700;cursor:pointer;padding:4px 8px}
.pick-search-wrap{position:relative;padding:12px 16px;flex-shrink:0}
.pick-search{width:100%;background:var(--bd);border:none;border-radius:10px;padding:10px 14px 10px 36px;font-size:14px;color:var(--tx);box-sizing:border-box}
.pick-search::placeholder{color:var(--tx3)}
.pick-search-wrap::before{content:"🔍";position:absolute;left:26px;top:50%;transform:translateY(-50%);font-size:13px;pointer-events:none}
.pick-list{flex:1;overflow-y:auto;-webkit-overflow-scrolling:touch;padding:4px 0}
.pick-item{display:flex;align-items:center;gap:12px;padding:14px 20px;border-bottom:.5px solid var(--bd);cursor:pointer;-webkit-tap-highlight-color:var(--s2)}
.pick-item:last-child{border-bottom:none}
.pick-item:active{background:var(--s2)}
.pick-item-fl{font-size:26px;flex-shrink:0;width:34px;text-align:center}
.pick-item-info{flex:1;min-width:0}
.pick-item-nm{font-size:15px;font-weight:700}
.pick-item-pr{font-size:12px;color:var(--tx2);margin-top:2px}
.pick-item-chk{font-size:16px;color:var(--bl);flex-shrink:0;display:none}
.pick-item.sel .pick-item-chk{display:block}
/* Score Prediction */
.sc-pred{margin-bottom:16px}
.sc-pred-r{display:flex;align-items:center;gap:0;border-radius:12px;overflow:hidden;height:56px;background:var(--s2);margin-bottom:16px;padding:0 4px}
.sc-team{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:0 8px}
.sc-team-nm{font-size:11px;font-weight:700;margin-bottom:4px;max-width:80px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sc-goals{display:flex;align-items:center;gap:4px;flex-shrink:0}
.sc-gl{font-size:28px;font-weight:800;line-height:1;color:var(--tx);min-width:28px;text-align:center}
.sc-sep{font-size:22px;font-weight:800;color:var(--tx3);flex-shrink:0}
.sc-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-bottom:16px}
.sc-cell{background:var(--s2);border-radius:10px;padding:8px 6px;text-align:center}
.sc-cell.top{background:rgba(10,132,255,.15);border:1px solid rgba(10,132,255,.3)}
.sc-s{font-size:17px;font-weight:800;color:var(--tx)}
.sc-p{font-size:10px;color:var(--tx2);margin-top:3px;font-weight:600}
.sc-most-likely{background:var(--s);border-radius:12px;padding:14px 16px;margin-bottom:16px}
.sc-ml-hd{font-size:10px;font-weight:700;color:var(--tx2);text-transform:uppercase;letter-spacing:.8px;margin-bottom:10px}
.sc-ml-row{display:flex;align-items:center;justify-content:space-between;padding:7px 0;border-bottom:.5px solid var(--bd)}
.sc-ml-row:last-child{border-bottom:none}
.sc-ml-sc{font-size:15px;font-weight:800}
.sc-ml-od{font-size:11px;font-weight:600;color:var(--tx2)}
.sc-ml-d{font-size:11px;font-weight:700;color:var(--gd);min-width:40px;text-align:right}
.sc-hd{display:flex;align-items:center;justify-content:space-between;padding:8px 0 6px;font-size:11px;font-weight:700;color:var(--tx2);text-transform:uppercase;letter-spacing:.5px}
.sc-hd-hi{color:var(--yl)}
.sc-hd-sub{font-size:10px;font-weight:800;color:var(--gr)}
.sc-grid-hi .sc-cell-hi{background:rgba(255,204,0,.08);border:1px solid rgba(255,204,0,.2)}
.sc-note{font-size:11px;color:var(--tx2);line-height:1.6;margin-top:8px;font-style:italic}
.h2h-matchup{padding:10px 0;border-bottom:0.5px solid var(--bd)}
.h2h-matchup:last-child{border-bottom:none}
.h2h-mu-pos{font-size:10px;font-weight:700;color:var(--tx2);text-transform:uppercase;letter-spacing:0.6px;margin-bottom:6px}
.h2h-mu-row{display:flex;align-items:center;gap:8px}
.h2h-mu-p{flex:1;font-size:13px;font-weight:600}
.h2h-mu-p .h2h-wl{margin-left:6px}
.h2h-mu-s{font-size:12px;font-weight:700;color:var(--gd);width:28px;text-align:center}
.h2h-mu-r{text-align:right;flex:1}
.h2h-mu-r .h2h-wl{margin-right:6px}
/* Conformal Prediction */
.cp-set-box{margin-top:4px}
.cp-set-hd{display:flex;align-items:center;justify-content:space-between;margin-bottom:6px}
.cp-set-lbl{font-size:10px;font-weight:700;color:var(--tx2);text-transform:uppercase;letter-spacing:0.8px}
.cp-set-badge{font-size:11px;font-weight:800;padding:3px 8px;border-radius:6px;letter-spacing:0.3px}
.cp-set-exp{font-size:12px;color:var(--tx2);margin-bottom:4px}
.cp-set-conf{font-size:11px;color:var(--tx3)}
/* Factor Attribution */
.attr-sect{margin-bottom:14px}
.attr-row{display:flex;align-items:center;gap:8px;font-size:12px;margin-bottom:5px}
.attr-lbl{flex:0 0 80px;font-weight:700;color:var(--tx2)}
.attr-bar{flex:1;height:18px;background:var(--s2);border-radius:4px;overflow:hidden;display:flex}
.attr-seg{height:100%;transition:width 0.3s}
.attr-seg-a{background:var(--bl)}
.attr-seg-d{background:var(--rd)}
.attr-seg-val{flex:0 0 48px;font-size:11px;font-weight:700;text-align:right;justify-content:flex-end;display:flex;gap:2px}
.attr-delta-p{font-size:11px;color:var(--gr)}
.attr-delta-n{font-size:11px;color:var(--rd)}
.attr-meta{font-size:10px;color:var(--tx3);margin-top:6px;padding-top:6px;border-top:0.5px solid var(--bd)}
.attr-meta span{margin-right:12px}
.attr-note{font-size:11px;color:var(--tx3);font-style:italic;margin-top:4px}
/* Squad */
.sel{width:100%;background:var(--s2);color:var(--tx);border:0.5px solid var(--bd);border-radius:12px;padding:12px 16px;font-size:15px;font-weight:600;margin-bottom:12px;appearance:none;-webkit-appearance:none}
.sel-wrap{position:relative}
.sel-wrap::after{content:"▼";position:absolute;right:16px;top:50%;transform:translateY(-50%);font-size:10px;color:var(--tx2);pointer-events:none}
.sq-card{background:var(--s);border-radius:16px;border:0.5px solid var(--bd);overflow:hidden;margin-bottom:12px}
.sq-ph{background:var(--s2);padding:12px 16px;display:flex;align-items:center;gap:12px}
.sq-ph-fl{font-size:28px}
.sq-ph-nm{font-size:16px;font-weight:800}
.sq-ph-elo{font-size:12px;color:var(--tx2);margin-top:2px}
.sq-table{width:100%}
.sq-th{background:var(--s2);padding:8px 12px;font-size:9px;font-weight:700;color:var(--tx2);text-transform:uppercase;letter-spacing:0.6px;text-align:left}
.sq-td{padding:9px 12px;font-size:13px;border-bottom:0.5px solid var(--bd)}
.sq-td:last-child{border-bottom:none}
.sq-pos{font-size:10px;font-weight:700;color:var(--tx2);width:28px}
.sq-name{font-weight:600}
.sq-club{font-size:11px;color:var(--tx2)}
.sq-mv{font-size:12px;font-weight:700;color:var(--gd);white-space:nowrap}
.sq-caps{text-align:right;font-variant-numeric:tabular-nums}
.sq-goals{text-align:right;font-variant-numeric:tabular-nums;color:var(--tx2)}
/* Info */
.info-sec{margin-bottom:24px}
.info-tl{font-size:11px;font-weight:700;color:var(--tx2);text-transform:uppercase;letter-spacing:1px;margin-bottom:10px}
.info-row{background:var(--s);border-radius:12px;padding:14px 16px;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center}
.info-lbl{font-size:14px;color:var(--tx2)}
.info-val{font-size:14px;font-weight:700;text-align:right}
.calibration{background:var(--s);border-radius:12px;padding:16px;margin-bottom:10px}
.cal-hd{font-size:13px;font-weight:800;margin-bottom:8px;display:flex;align-items:center;gap:8px}
.cal-bd{font-size:12px;color:var(--tx2);line-height:1.7}
/* Polymarket Market Comparison */
.pm-hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px}
.pm-sub{font-size:11px;color:var(--tx2);margin-bottom:14px}
.pm-mkt{margin-bottom:20px}
.pm-mkt-tl{font-size:10px;font-weight:700;color:var(--tx2);text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px}
.pm-row{display:flex;align-items:center;gap:10px;padding:10px 0;border-bottom:.5px solid var(--bd)}
.pm-fl{font-size:20px;line-height:1;flex-shrink:0}
.pm-nm{flex:1;font-size:13px;font-weight:600;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.pm-bar{flex:1;height:6px;background:var(--s2);border-radius:3px;overflow:hidden}
.pm-bar-in{height:100%;border-radius:3px;transition:width .4s}
.pm-val{width:64px;text-align:right;font-size:12px;font-weight:700;flex-shrink:0}
.pm-val.pos{color:var(--gr)}
.pm-val.neg{color:var(--rd)}
.pm-val.neu{color:var(--tx2)}
.pm-legend{display:flex;gap:14px;margin-top:10px;font-size:11px;color:var(--tx2)}
.pm-legend span::before{content:'▪ ';font-size:8px;vertical-align:middle}
.pm-sum{margin-top:12px;padding:12px;background:var(--bg2);border-radius:8px;border-left:3px solid var(--bl)}
.pm-sum-tl{font-size:12px;font-weight:600;color:var(--bl);margin-bottom:8px}
.pm-sum-grp{display:flex;flex-direction:column;gap:5px;margin-bottom:10px}
.pm-sum-row{display:flex;align-items:center;gap:8px;font-size:13px}
.pm-sum-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.pm-sum-val{font-weight:700;font-size:12px;flex-shrink:0}
.pm-sum-lbl{color:var(--tx2)}
.pm-sum-empty{font-size:13px;color:var(--tx2);padding:4px 0}
/* Fixtures */
.fx-tabs{display:flex;gap:4px;margin-bottom:14px;background:var(--s2);padding:4px;border-radius:10px}
.fx-tab{flex:1;text-align:center;padding:9px 0;font-size:12px;font-weight:700;color:var(--tx2);border-radius:8px;cursor:pointer;-webkit-tap-highlight-color:transparent;transition:all 0.15s}
.fx-tab.on{background:var(--bl);color:#fff}
.fx-day-grp{margin-bottom:18px}
.fx-day-hd{font-size:11px;font-weight:800;color:var(--gd);margin-bottom:8px;padding-bottom:4px;border-bottom:0.5px solid var(--bd);text-transform:uppercase;letter-spacing:0.8px}
.fx-day-hd.today{color:var(--rd)}
.fx-day-hd.group{color:var(--bl)}
.fx-match{background:var(--s2);border-radius:12px;padding:11px 13px;margin-bottom:7px;border:0.5px solid var(--bd)}
.fx-match.live{border-color:var(--rd);background:rgba(255,69,58,0.06)}
.fx-meta{display:flex;justify-content:space-between;font-size:9px;color:var(--tx2);margin-bottom:8px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px}
.fx-meta-grp{color:var(--bl)}
.fx-row{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:8px}
.fx-team{display:flex;align-items:center;gap:7px;min-width:0}
.fx-team.away{justify-content:flex-end}
.fx-team-fl{font-size:10px;font-weight:800;color:var(--tx2);background:var(--bd);padding:3px 6px;border-radius:5px;flex-shrink:0;letter-spacing:0.5px}
.fx-team-nm{font-size:13px;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0}
.fx-score{font-size:17px;font-weight:800;font-variant-numeric:tabular-nums;color:var(--tx);min-width:60px;text-align:center;padding:0 6px;line-height:1}
.fx-score.ns{color:var(--tx3);font-size:13px;font-weight:600}
.fx-score.live{color:var(--rd)}
.fx-ft{display:flex;justify-content:space-between;font-size:10px;margin-top:7px;color:var(--tx2);font-weight:600}
.fx-status.live{color:var(--rd);font-weight:800}
.fx-status.live::before{content:"● ";animation:fxpulse 1.5s infinite}
@keyframes fxpulse{0%,100%{opacity:1}50%{opacity:0.3}}
.fx-empty{text-align:center;padding:32px 16px;color:var(--tx2);font-size:13px;background:var(--s2);border-radius:12px;margin-bottom:12px;line-height:1.7}
/* Standings */
.st-grp{margin-bottom:16px}
.st-hd{display:flex;justify-content:space-between;align-items:center;font-size:11px;font-weight:800;color:var(--bl);margin-bottom:6px;text-transform:uppercase;letter-spacing:0.6px}
.st-prog{font-size:9px;color:var(--tx2);font-weight:600}
.st-tbl{background:var(--s2);border-radius:10px;overflow:hidden}
.st-thr,.st-row{display:grid;grid-template-columns:16px 1fr 18px 16px 16px 16px 26px 26px;align-items:center;gap:2px;padding:7px 7px}
.st-thr{font-size:8px;color:var(--tx2);font-weight:700;background:rgba(255,255,255,0.03)}
.st-row{border-top:0.5px solid var(--bd);font-size:11px}
.st-row.z1,.st-row.z2{background:rgba(48,209,88,0.08)}
.st-row.z3{background:rgba(255,214,10,0.07)}
.st-row.z4{background:rgba(255,69,58,0.05)}
.st-pos{font-weight:800;color:var(--tx2);text-align:center}
.st-tm{display:flex;align-items:center;gap:5px;min-width:0}
.st-tm-fl{font-size:9px;font-weight:800;color:var(--tx2);background:var(--bd);padding:2px 4px;border-radius:4px;flex-shrink:0}
.st-tm-nm{font-size:12px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.st-q{font-size:9px;flex-shrink:0}
.st-n{text-align:center;font-variant-numeric:tabular-nums;color:var(--tx2)}
.st-pts{text-align:center;font-weight:800;color:var(--tx);font-variant-numeric:tabular-nums}
.st-third-row{display:flex;align-items:center;gap:8px;padding:7px 9px;border-top:0.5px solid var(--bd);font-size:12px}
.st-third-row.in{background:rgba(48,209,88,0.08)}
.st-third-rk{width:20px;font-weight:800;color:var(--tx2);text-align:center;flex-shrink:0}
.st-third-nm{flex:1;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.st-third-st{font-size:11px;font-weight:700;flex-shrink:0}
.st-legend{display:flex;flex-wrap:wrap;gap:10px;font-size:10px;color:var(--tx2);margin:4px 0 14px;padding:0 2px}
.st-legend span{display:flex;align-items:center;gap:4px}
.st-dot{width:9px;height:9px;border-radius:2px;flex-shrink:0}
/* 实时冠军卡片 */
.fin-row{display:flex;align-items:center;padding:11px 0;border-bottom:0.5px solid var(--bd);gap:10px;cursor:pointer}
.fin-row:last-child{border-bottom:none}
.fin-fact{display:none;font-size:11px;color:var(--gd);margin-top:7px;line-height:1.7;background:rgba(255,214,10,0.06);border-radius:8px;padding:8px 10px}
.fin-fact.on{display:block}
</style>
</head>
<body>
<div class="hdr">
  <div class="hdr-title">世界杯 2026 / WC 2026</div>
  <div class="hdr-sub" id="upd"></div>
</div>

<div class="tabbar">
  <button class="tab on" id="tb-home" onclick="showTab('home')"><span class="ico"><svg width="22" height="22" viewBox="0 0 22 22" fill="none"><path d="M11 2L13.5 7.5L19.5 8.5L15 13L16 19L11 16L6 19L7 13L2.5 8.5L8.5 7.5L11 2Z" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/><path d="M8 19H14V21H8V19Z" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/></svg></span><span>冠军</span></button>
  <button class="tab" id="tb-factor" onclick="showTab('factor')"><span class="ico"><svg width="22" height="22" viewBox="0 0 22 22" fill="none"><rect x="3" y="10" width="3.5" height="9" rx="1" stroke="currentColor" stroke-width="1.6"/><rect x="9.25" y="6" width="3.5" height="13" rx="1" stroke="currentColor" stroke-width="1.6"/><rect x="15.5" y="2" width="3.5" height="17" rx="1" stroke="currentColor" stroke-width="1.6"/></svg></span><span>因子</span></button>
  <button class="tab" id="tb-mystic" onclick="showTab('mystic')"><span class="ico"><svg width="22" height="22" viewBox="0 0 22 22" fill="none"><circle cx="11" cy="11" r="8.5" stroke="currentColor" stroke-width="1.6"/><circle cx="11" cy="11" r="3.5" fill="currentColor" opacity="0.4"/><path d="M11 2.5V5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/><path d="M11 17V19.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/><path d="M2.5 11H5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/><path d="M17 11H19.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg></span><span>玄学</span></button>
  <button class="tab" id="tb-h2h" onclick="showTab('h2h')"><span class="ico"><svg width="22" height="22" viewBox="0 0 22 22" fill="none"><path d="M4 11H10M10 11L7 8M10 11L7 14" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/><path d="M18 11H12M12 11L15 8M12 11L15 14" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg></span><span>对战</span></button>
  <button class="tab" id="tb-squad" onclick="showTab('squad')"><span class="ico"><svg width="22" height="22" viewBox="0 0 22 22" fill="none"><circle cx="7" cy="5.5" r="2.5" stroke="currentColor" stroke-width="1.6"/><path d="M2 17.5C2 14.4624 4.23858 12 7 12H7C9.76142 12 12 14.4624 12 17.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/><circle cx="15" cy="5.5" r="2.5" stroke="currentColor" stroke-width="1.6"/><path d="M10 17.5C10 14.4624 12.2386 12 15 12H15C17.7614 12 20 14.4624 20 17.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg></span><span>球队</span></button>
  <button class="tab" id="tb-poly" onclick="showTab('poly')"><span class="ico"><svg width="22" height="22" viewBox="0 0 22 22" fill="none"><path d="M3 17L8 10L13 14L19 5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/><circle cx="19" cy="5" r="2" stroke="currentColor" stroke-width="1.6"/></svg></span><span>市场</span></button>
  <button class="tab" id="tb-fixtures" onclick="showTab('fixtures')"><span class="ico"><svg width="22" height="22" viewBox="0 0 22 22" fill="none"><rect x="3" y="5" width="16" height="14" rx="2" stroke="currentColor" stroke-width="1.6"/><path d="M3 9H19" stroke="currentColor" stroke-width="1.6"/><path d="M7 3V7" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/><path d="M15 3V7" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg></span><span>赛程</span></button>
  <button class="tab" id="tb-info" onclick="showTab('info')"><span class="ico"><svg width="22" height="22" viewBox="0 0 22 22" fill="none"><circle cx="11" cy="11" r="8.5" stroke="currentColor" stroke-width="1.6"/><path d="M11 10V16" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><circle cx="11" cy="6.5" r="0.9" fill="currentColor"/></svg></span><span>说明</span></button>
</div>

<!-- TAB: Champion -->
<div class="pg on" id="pg-home">
  <div class="card" id="final-card" style="border-color:var(--gd)">
    <div class="card-title" style="color:var(--gd)">🎯 最精准预测 / Market-Calibrated — 市场共识 + 数据模型融合</div>
    <div class="lb" id="finlb"></div>
    <div id="fin-note" style="font-size:10px;color:var(--tx2);margin-top:10px;line-height:1.6"></div>
  </div>
  <div class="card">
    <div class="card-title">冠军概率 TOP 6 / Champion Prob</div>
    <div class="lb" id="lb"></div>
  </div>
  <!-- Section 2: WC Fixtures (within 7 days) -->
  <div class="card">
    <div class="card-title">⚔️ 赛程 / Upcoming Fixtures</div>
    <div class="dyn-sect" id="sec-fix"></div>
  </div>
  <!-- Section 3: WC Results (completed) -->
  <div class="card">
    <div class="card-title">📊 赛果 / Match Results</div>
    <div class="dyn-sect" id="sec-wcr"></div>
  </div>
  <!-- Section 4: Friendly Results -->
  <div class="card">
    <div class="card-title">🤝 热身赛 / Friendly Results</div>
    <div class="dyn-sect" id="sec-frn"></div>
  </div>
</div>

<!-- TAB: Factor Breakdown -->
<div class="pg" id="pg-factor">
  <div class="card">
    <div class="card-title">因子拆解 / Factor Breakdown — 点击展开详情</div>
    <div id="fb"></div>
  </div>
</div>

<!-- TAB: Mystic -->
<div class="pg" id="pg-mystic">
  <div class="card">
    <div class="card-title">玄学分析 / Mystic Analysis — 点击展开详情</div>
    <div id="ml"></div>
  </div>
</div>

<!-- TAB: H2H -->
<div class="pg" id="pg-h2h">
  <div class="card">
    <div class="card-title">对战预测 / H2H Predictor</div>
    <div class="sel-wrap" style="margin-bottom:12px">
      <select class="sel" id="h2h-match" onchange="applyScheduleMatch()"></select>
    </div>
    <div class="h2h-teams">
      <div class="h2h-team">
        <label>Team A / 球队A</label>
        <div class="h2h-pick" id="h2h-pick-a" onclick="openPicker('a')">
          <span class="h2h-pick-fl" id="h2h-pick-fl-a"></span>
          <span class="h2h-pick-info">
            <span class="h2h-pick-nm" id="h2h-pick-nm-a"></span>
            <span class="h2h-pick-pr" id="h2h-pick-pr-a"></span>
          </span>
          <span class="h2h-pick-chevron">›</span>
        </div>
      </div>
      <div class="h2h-vs">⚔️</div>
      <div class="h2h-team">
        <label>Team B / 球队B</label>
        <div class="h2h-pick" id="h2h-pick-b" onclick="openPicker('b')">
          <span class="h2h-pick-fl" id="h2h-pick-fl-b"></span>
          <span class="h2h-pick-info">
            <span class="h2h-pick-nm" id="h2h-pick-nm-b"></span>
            <span class="h2h-pick-pr" id="h2h-pick-pr-b"></span>
          </span>
          <span class="h2h-pick-chevron">›</span>
        </div>
      </div>
    </div>
    <select id="h2h-a" onchange="h2hChange()" style="display:none"></select>
    <select id="h2h-b" onchange="h2hChange()" style="display:none"></select>
    <div class="h2h-bar" id="h2h-bar">
      <div class="h2h-bar-a" id="h2h-bar-a" style="width:45%"></div>
      <div class="h2h-bar-d" id="h2h-bar-d" style="width:10%">—</div>
      <div class="h2h-bar-b" id="h2h-bar-b" style="width:45%"></div>
    </div>
    <div id="h2h-content"></div>
  </div>
</div>

<!-- Picker Overlay -->
<div class="pick-overlay" id="pick-overlay" onclick="closePicker(event)">
  <div class="pick-sheet" onclick="event.stopPropagation()">
    <div class="pick-sheet-hd">
      <span class="pick-sheet-tl" id="pick-title">Select Team</span>
      <button class="pick-sheet-close" onclick="closePicker()">Done</button>
    </div>
    <div class="pick-search-wrap">
      <input class="pick-search" id="pick-search" placeholder="Search team..." oninput="filterPickList()">
    </div>
    <div class="pick-list" id="pick-list"></div>
  </div>
</div>

<!-- TAB: Squad -->
<div class="pg" id="pg-squad">
  <div class="card">
    <div class="card-title">球员阵容 / Squad Roster</div>
    <div class="sel-wrap">
      <select class="sel" id="sq-sel" onchange="sqChange()"></select>
    </div>
    <div id="sq-content"></div>
  </div>
</div>

<!-- TAB: Polymarket Market -->
<div class="pg" id="pg-poly">
  <div class="card">
    <div class="card-title">市场博弈 / Polymarket</div>
    <div class="pm-hdr">
      <span style="font-size:13px;color:var(--tx2)">我的概率 vs 市场定价</span>
      <span id="poly-upd" style="font-size:11px;color:var(--tx2)"></span>
    </div>
    <div class="pm-sub">红色=你的模型比市场更乐观，潜在高估 &nbsp;|&nbsp; 绿色=你的模型比市场更保守，潜在低估</div>
    <div class="pm-mkt">
      <div class="pm-mkt-tl">🏆 冠军投注 / Winner</div>
      <div id="poly-winner"></div>
    </div>
    <div class="pm-sum" id="poly-sum"></div>
    <div class="pm-legend">
      <span style="color:var(--gr)">●</span><span style="font-size:12px;color:var(--tx2)">你的概率 &gt; 市场&nbsp;</span>
      <span style="color:var(--rd)">●</span><span style="font-size:12px;color:var(--tx2)">你的概率 &lt; 市场&nbsp;</span>
      <span style="color:var(--tx2)">●</span><span style="font-size:12px;color:var(--tx2)">|差|&le;2%（中性）</span>
    </div>
  </div>
</div>

<!-- TAB: Fixtures -->
<div class="pg" id="pg-fixtures">
  <div class="card">
    <div class="card-title">赛程比分 / Fixtures &amp; Live Scores</div>
    <div class="fx-tabs">
      <div class="fx-tab on" data-fxv="today" onclick="switchFxView('today')">今日</div>
      <div class="fx-tab" data-fxv="upcoming" onclick="switchFxView('upcoming')">未来 7 天</div>
      <div class="fx-tab" data-fxv="standings" onclick="switchFxView('standings')">积分榜</div>
      <div class="fx-tab" data-fxv="all" onclick="switchFxView('all')">全部</div>
    </div>
    <div id="fx-body"></div>
    <div id="fx-upd" style="text-align:center;font-size:10px;color:var(--tx2);margin-top:14px"></div>
  </div>
</div>

<!-- TAB: Info -->
<div class="pg" id="pg-info">
  <div class="info-sec">
    <div class="info-tl">模型 / Model</div>
    <div class="info-row"><span class="info-lbl">数据源 / Data Sources</span><span class="info-val">Wikipedia + FiveThirtyEight Elo</span></div>
    <div class="info-row"><span class="info-lbl">评分维度 / Dimensions</span><span class="info-val">5 — Elo/年龄/经验/状态/教练</span></div>
    <div class="info-row"><span class="info-lbl">玄学框架 / Mystic</span><span class="info-val">易经·道德经·悖论三重</span></div>
    <div class="info-row"><span class="info-lbl">更新时间 / Updated</span><span class="info-val" id="infTime"></span></div>
  </div>
  <div class="info-sec">
    <div class="info-tl">校准框架 / Calibration</div>
    <div class="calibration">
      <div class="cal-hd"><span>🇧🇷</span><span style="color:var(--rd)">Brazil 2014 (1-7 Germany)</span></div>
      <div class="cal-bd">半决赛1-7惨败 — 心理崩溃框架 / Psychological collapse.<br>
      参数: <b style="color:var(--rd)">pressure +0.05</b>, <b>amplification ×1.5</b><br>
      影响: 热门诅咒强化 — 成为强队时触发自我强化的压力循环</div>
    </div>
    <div class="calibration">
      <div class="cal-hd"><span>🇫🇷</span><span style="color:var(--gr)">France 2018 (4-2 Croatia)</span></div>
      <div class="cal-bd">决赛4-2克罗地亚 — 势能爆发框架 / Momentum explosion.<br>
      参数: <b style="color:var(--gr)">pressure +0.05</b>, <b>conversion +0.05</b><br>
      影响: 逆袭信心加成 — 突破强敌后建立势能，后续效率提升</div>
    </div>
  </div>
  <div class="info-sec">
    <div class="info-tl">悖论框架 / Paradox</div>
    <div class="calibration">
      <div class="cal-bd">
        <b>热门诅咒 / FavCurse</b>: 热度越高，外部期待形成反向压力，概率被系统压低<br><br>
        <b>逆向思维 / Contrarian</b>: 表面弱势实际被低估；表面强势实际被高估<br><br>
        <b>淘汰赛不确定性 / Knockout Unc</b>: 小组赛逻辑无法预测单场淘汰赛的随机性<br><br>
        <b>势能天花板 / Luck Ceiling</b>: 纸面实力有上限，运气是冠军的必要非充分条件
      </div>
    </div>
  </div>
  <div class="info-sec">
    <div class="info-tl">欧冠调参 / UCL Tuning</div>
    <div id="ucl-info"></div>
  </div>
  <div class="info-sec">
    <div class="info-tl">版本 / Version</div>
    <div class="calibration">
      <div class="cal-bd">
        <b>World Cup 2026 Predictor v2</b><br>
        Pure HTML/CSS/JS Mobile UI — No Gradio dependency<br>
        mystic_factor_ucl_integration: True<br>
        UCL Final v2: PSG 5-0 Inter Milan
      </div>
    </div>
  </div>
</div>

<script>
var D=__DATA__;
var U=__UCL__;
var F=__FIXTURES__;
var FN=__FINAL__;
var SP=__SCHEDULE_PRED__;
var RT=__REALTIME__;
var HC=__H2H_CONF__;
var MF=__MATCH_FIXTURES__;
var WCR=__WC_RESULTS__;
var FRN=__FRIENDLIES__;
var FL={
  "Argentina":"🇦🇷","Brazil":"🇧🇷","France":"🇫🇷","Germany":"🇩🇪","Spain":"🇪🇸",
  "England":"🏴󠁧󠁢󠁥󠁮󠁧󠁿","Portugal":"🇵🇹","Netherlands":"🇳🇱","Belgium":"🇧🇪",
  "Croatia":"🇭🇷","Switzerland":"🇨🇭","Austria":"🇦🇹","Czech Republic":"🇨🇿","Turkey":"🇹🇷",
  "Sweden":"🇸🇪","Morocco":"🇲🇦","Senegal":"🇸🇳","Egypt":"🇪🇬","Algeria":"🇩🇿",
  "Ghana":"🇬🇭","Ivory Coast":"🇨🇮","Tunisia":"🇹🇳","DR Congo":"🇨🇩","Cape Verde":"🇨🇻",
  "Japan":"🇯🇵","South Korea":"🇰🇷","Korea Republic":"🇰🇷","Iran":"🇮🇷","Iraq":"🇮🇶",
  "Qatar":"🇶🇦","Saudi Arabia":"🇸🇦","Australia":"🇦🇺","Uzbekistan":"🇺🇿","Jordan":"🇯🇴",
  "USA":"🇺🇸","United States":"🇺🇸","Mexico":"🇲🇽","Canada":"🇨🇦","Panama":"🇵🇦",
  "Curaçao":"🇨🇼","Haiti":"🇭🇹","New Zealand":"🇳🇿","Ecuador":"🇪🇨","Paraguay":"🇵🇾",
  "Colombia":"🇨🇴","Uruguay":"🇺🇾","Norway":"🇳🇴","South Africa":"🇿🇦",
  "Bosnia and Herzegovina":"🇧🇦","Scotland":"🏴󠁧󠁢󠁳󠁣󠁴󠁿",
  "Bolivia":"🇧🇴","Denmark":"🇩🇰","Gambia":"🇬🇲","Hungary":"🇭🇺","Israel":"🇮🇱",
  "Italy":"🇮🇹","Jamaica":"🇯🇲","Kosovo":"🇽🇰","Mali":"🇲🇱","Nigeria":"🇳🇬",
  "Peru":"🇵🇪","Poland":"🇵🇱","Romania":"🇷🇴","Ukraine":"🇺🇦","Venezuela":"🇻🇪"
};
function fl(c){return FL[c]||"--";}
function pc(p){return p>15?"var(--bl)":p>5?"var(--gr)":"var(--tx2)";}
function st(s){return s>0?"+"+s.toFixed(2)+"%":s<0?s.toFixed(2)+"%":"--";}
function sc(s){return s>0?"var(--gr)":s<0?"var(--rd)":"var(--tx2)";}
function showTab(n){document.querySelectorAll(".pg").forEach(function(p){p.classList.remove("on");});document.querySelectorAll(".tab").forEach(function(t){t.classList.remove("on");});document.getElementById("pg-"+n).classList.add("on");document.getElementById("tb-"+n).classList.add("on");if(n==="fixtures")buildFixtures();}

/* ── Leaderboard ── */
function buildLB(){
  var s = D.slice().sort(function(a,b){return b.final_prob-a.final_prob;});
  // Top 6 only
  var h = "";
  for(var i=0; i<Math.min(6, s.length); i++){
    var t=s[i], r=i+1, rc=r<=3?"t"+r:"";
    var pct=(t.final_prob*100).toFixed(2), pctCls=t.final_prob>0.15?" vh":"";
    var sh=t.shift||0;
    h+='<div class="lb-r"><div class="lb-rk '+rc+'">'+r+'</div><div class="lb-fl">'+fl(t.country)+'</div><div class="lb-inf"><div class="lb-nm">'+t.country+'</div><div class="lb-el">Elo '+(t.elo||0).toFixed(0)+'</div><div class="pb"><div class="pb-fi" style="width:'+pct+'%;background:'+pc(t.final_prob*100)+'"></div></div></div><div class="lb-pr"><div class="lb-pct'+pctCls+'">'+pct+'%</div><div class="lb-sh" style="color:'+sc(sh)+'">'+st(sh)+'</div></div></div>';
  }
  document.getElementById("lb").innerHTML=h;
  // Render dynamic sections
  buildHomeFixtures();
  buildWCResults();
  buildFriendlies();
}

/* ── Dynamic Match Sections (SofaScore-inspired cards) ── */
var GRP_COLORS={
  "Group A":"#E63946","Group B":"#457B9D","Group C":"#2A9D8F","Group D":"#E9C46A",
  "Group E":"#F4A261","Group F":"#E76F51","Group G":"#6A4C93","Group H":"#1982C4",
  "Group I":"#8AC926","Group J":"#FF595E","Group K":"#FF924C","Group L":"#00B4D8",
  "Round of 16":"#FF6B35","Quarter-final":"#FF9F1C","Semi-final":"#E71D36",
  "Third place":"#FF9900","Final":"#FFD700",
  "Group stage":"#E63946","Knockout":"#FF6B35","Play-off":"#9B59B6",
  "Qualification":"#3498DB","Friendlies":"#95A5A6"
};

function grpColor(r){
  if(!r)return"var(--tx2)";
  for(var k in GRP_COLORS)if(r.indexOf(k)>-1)return GRP_COLORS[k];
  return"var(--tx2)";
}

function grpBadge(r){
  var c=grpColor(r);
  return'<span class="mc-badge" style="background:'+c+'">'+r+'</span>';
}

function fmtDate(d){
  if(!d)return"";
  var parts=d.split(".");
  if(parts.length===2)return parts[0]+"<span class='dyn-mo'>月</span>"+parts[1]+"<span class='dyn-mo'>日</span>";
  return d;
}

function buildHomeFixtures(){
  var matches=(MF||[]).filter(function(m){return m.score_a===null&&m.score_b===null;});
  if(!matches.length){document.getElementById("sec-fix").innerHTML='<div class="dyn-empty">暂无赛程数据</div>';return;}
  var h='<div class="dyn-list">';
  var lastDate="";
  matches.slice(0,20).forEach(function(m){
    if(m.date!==lastDate){
      h+='<div class="dyn-date-hd">'+fmtDate(m.date)+'</div>';
      lastDate=m.date;
    }
    h+='<div class="mc-row">\
      <div class="mc-rd">'+grpBadge(m.round)+'</div>\
      <div class="mc-team mc-tl">\
        <span class="mc-fl">'+fl(m.team_a)+'</span>\
        <span class="mc-nm">'+m.team_a+'</span>\
      </div>\
      <div class="mc-ct">\
        <span class="mc-vs">VS</span>\
        <span class="mc-kick">'+m.time+'</span>\
      </div>\
      <div class="mc-team mc-tr">\
        <span class="mc-nm">'+m.team_b+'</span>\
        <span class="mc-fl">'+fl(m.team_b)+'</span>\
      </div>\
    </div>';
  });
  h+='</div>';
  document.getElementById("sec-fix").innerHTML=h;
}

function buildWCResults(){
  var matches=(WCR||[]).filter(function(m){return m.score_a!==null||m.score_b!==null;});
  if(!matches.length){document.getElementById("sec-wcr").innerHTML='<div class="dyn-empty">暂无赛果数据</div>';return;}
  var h='<div class="dyn-list">';
  var lastDate="";
  matches.slice(0,20).forEach(function(m){
    if(m.date!==lastDate){
      h+='<div class="dyn-date-hd">'+fmtDate(m.date)+'</div>';
      lastDate=m.date;
    }
    var sa=m.score_a!=null?m.score_a:"-",sb=m.score_b!=null?m.score_b:"-";
    var winA=m.score_a!=null&&m.score_b!=null&&m.score_a>m.score_b;
    var winB=m.score_a!=null&&m.score_b!=null&&m.score_b>m.score_a;
    h+='<div class="mc-row">\
      <div class="mc-rd">'+grpBadge(m.round)+'</div>\
      <div class="mc-team mc-tl'+(winA?' mc-winner':'')+'">\
        <span class="mc-fl">'+fl(m.team_a)+'</span>\
        <span class="mc-nm">'+(winA?'<b>':'<b>')+m.team_a+'</b></span>\
      </div>\
      <div class="mc-ct mc-score-ct">\
        <span class="mc-scr '+(winA?'mc-sc-awin':winB?'mc-sc-bwin':'')+'">'+sa+'</span>\
        <span class="mc-scr-sep">-</span>\
        <span class="mc-scr '+(winB?'mc-sc-bwin':winA?'mc-sc-awin':'')+'">'+sb+'</span>\
        <span class="mc-ft">FT</span>\
      </div>\
      <div class="mc-team mc-tr'+(winB?' mc-winner':'')+'">\
        <span class="mc-nm"><b>'+m.team_b+'</b></span>\
        <span class="mc-fl">'+fl(m.team_b)+'</span>\
      </div>\
    </div>';
  });
  h+='</div>';
  document.getElementById("sec-wcr").innerHTML=h;
}

function buildFriendlies(){
  var matches=(FRN||[]).filter(function(m){return m.score_a!==null||m.score_b!==null;});
  if(!matches.length){document.getElementById("sec-frn").innerHTML='<div class="dyn-empty">暂无热身赛数据</div>';return;}
  var h='<div class="dyn-list">';
  var lastDate="";
  matches.slice(0,20).forEach(function(m){
    if(m.date!==lastDate){
      h+='<div class="dyn-date-hd">'+fmtDate(m.date)+'</div>';
      lastDate=m.date;
    }
    var sa=m.score_a!=null?m.score_a:"-",sb=m.score_b!=null?m.score_b:"-";
    var winA=m.score_a!=null&&m.score_b!=null&&m.score_a>m.score_b;
    var winB=m.score_a!=null&&m.score_b!=null&&m.score_b>m.score_a;
    h+='<div class="mc-row mc-frn">\
      <div class="mc-rd">'+grpBadge("热身赛")+'</div>\
      <div class="mc-team mc-tl'+(winA?' mc-winner':'')+'">\
        <span class="mc-fl">'+fl(m.team_a)+'</span>\
        <span class="mc-nm"><b>'+m.team_a+'</b></span>\
      </div>\
      <div class="mc-ct mc-score-ct">\
        <span class="mc-scr '+(winA?'mc-sc-awin':winB?'mc-sc-bwin':'')+'">'+sa+'</span>\
        <span class="mc-scr-sep">-</span>\
        <span class="mc-scr '+(winB?'mc-sc-bwin':winA?'mc-sc-awin':'')+'">'+sb+'</span>\
      </div>\
      <div class="mc-team mc-tr'+(winB?' mc-winner':'')+'">\
        <span class="mc-nm"><b>'+m.team_b+'</b></span>\
        <span class="mc-fl">'+fl(m.team_b)+'</span>\
      </div>\
    </div>';
  });
  h+='</div>';
  document.getElementById("sec-frn").innerHTML=h;
}

/* ── 最精准预测（市场校准 + grok 实时动态）── */
function buildFinal(){
  var card=document.getElementById("final-card");
  var rt=RT,teams,summary="",updated="",isRT=false,isSchedule=false;
  if(rt&&rt.teams&&rt.teams.length){teams=rt.teams;summary=rt.summary||"";updated=rt.updated||"";isRT=true;}
  else if(SP&&SP.teams&&SP.teams.length){teams=SP.teams;summary="真实赛程路径模拟";updated=SP.as_of||"";isSchedule=true;}
  else{var f=(FN&&FN.teams)||[];if(f.length===0){if(card)card.style.display="none";return;}
    teams=f.map(function(t){return{country:t.country,champion:t.champion,base:(t.market!=null?t.market:t.model),delta:0,factors:[]};});}
  var h="";
  for(var i=0;i<Math.min(14,teams.length);i++){
    var t=teams[i],r=i+1,rc=r<=3?"t"+r:"";
    var pct=(t.champion*100).toFixed(1),dv=t.delta||0;
    var dstr=Math.abs(dv)>=0.0005?((dv>0?"▲":"▼")+(Math.abs(dv)*100).toFixed(1)):"";
    var dcol=dv>0?"var(--gr)":dv<0?"var(--rd)":"var(--tx2)";
    var facts=(t.factors||[]);
    var ff=facts.length?('<div class="fin-fact">'+facts.map(function(x){return "· "+x;}).join("<br>")+'</div>'):"";
    h+='<div class="fin-row" onclick="var d=this.querySelector(\'.fin-fact\');if(d)d.classList.toggle(\'on\')">';
    h+='<div class="lb-rk '+rc+'">'+r+'</div><div class="lb-fl">'+fl(t.country)+'</div>';
    h+='<div class="lb-inf"><div class="lb-nm">'+t.country+(facts.length?' <span style="color:var(--tx3);font-size:10px">▾</span>':'')+'</div>';
    h+='<div class="pb"><div class="pb-fi" style="width:'+Math.min(100,pct*4.5)+'%;background:var(--gd)"></div></div>'+ff+'</div>';
    h+='<div class="lb-pr"><div class="lb-pct vh" style="color:var(--gd)">'+pct+'%</div><div class="lb-sh" style="color:'+dcol+'">'+dstr+'</div></div></div>';
  }
  document.getElementById("finlb").innerHTML=h;
  var note=document.getElementById("fin-note");
  if(note){
    if(isRT)note.innerHTML='<b style="color:var(--gd)">🔴 实时研判</b>（grok 联网综合伤病/状态/赔率）: '+summary+'<br>更新 '+updated+' · 点击球队展开实时因子 · 数据模型+市场+grok实时 三层融合';
    else if(isSchedule)note.innerHTML='<b style="color:var(--gd)">赛程驱动</b>：按真实小组赛程、晋级路径和淘汰赛编号模拟。更新 '+updated+' · 基线不使用高比分boost';
    else note.innerHTML="市场共识 + 数据模型融合（实时层加载中，几分钟后自动刷新）。截至 "+((FN&&FN.as_of)||"2026-06-08");
  }
}
function setUpdateTime(value){
  var dt=value?new Date(String(value).replace(" ","T")):new Date();
  if(isNaN(dt.getTime()))dt=new Date();
  var parts=new Intl.DateTimeFormat("zh-CN",{
    timeZone:'Asia/Shanghai',year:'numeric',month:'2-digit',day:'2-digit',
    hour:'2-digit',minute:'2-digit',hour12:false
  }).formatToParts(dt).reduce(function(acc,p){acc[p.type]=p.value;return acc;},{});
  var text=parts.year+"-"+parts.month+"-"+parts.day+" "+parts.hour+":"+parts.minute;
  var upd=document.getElementById("upd");
  var inf=document.getElementById("infTime");
  if(upd)upd.textContent=text;
  if(inf)inf.textContent=text;
}
function refreshRealtime(){fetch("/api/realtime",{cache:"no-store"}).then(function(r){return r.json();}).then(function(d){if(d&&d.teams&&d.teams.length){RT=d;setUpdateTime();buildFinal();}}).catch(function(){});}

function buildUCLInfo(){
  var el=document.getElementById("ucl-info");
  if(!el)return;
  var countries=Object.keys(U||{}).sort();
  if(countries.length===0){
    el.innerHTML='<div class="calibration"><div class="cal-bd">当前没有启用欧冠心态信号。</div></div>';
    return;
  }
  var h="";
  for(var i=0;i<countries.length;i++){
    var c=countries[i],row=U[c]||{};
    var total=row.total_bonus||0;
    var cls=total>=0?"var(--gr)":"var(--rd)";
    h+='<div class="calibration"><div class="cal-hd"><span>'+fl(c)+'</span><span>'+c+'</span><span style="margin-left:auto;color:'+cls+'">'+st(total/100)+'</span></div>';
    h+='<div class="cal-bd">'+(row.description||'UCL mentality signal')+'<br>';
    var players=row.players||[];
    for(var j=0;j<players.length;j++){
      var p=players[j];
      var m=p.mentality_signal||0,adj=p.wc_adjustment||0;
      var mcls=m>=0?"var(--gr)":"var(--rd)";
      h+='<b>'+p.name+'</b> · '+(p.club||'')+' · <span style="color:'+mcls+'">心态 '+m.toFixed(2)+'</span>';
      h+=' · WC调整 '+st(adj/100)+'<br><span style="color:var(--tx3)">'+(p.framework||'')+' · '+(p.tier||'')+'</span>';
      if(j<players.length-1)h+='<br><br>';
    }
    h+='</div></div>';
  }
  el.innerHTML=h;
}

/* ── Factor Breakdown ── */
function toggleFB(el){var d=el.querySelector(".fb-expanded");if(d)d.classList.toggle("on");}
function buildFB(){var s=D.slice().sort(function(a,b){return b.final_prob-a.final_prob;});var factors=[{k:"elo_score",l:"Elo锚点"},{k:"age_score",l:"年龄结构"},{k:"exp_score",l:"大赛经验"},{k:"form_score",l:"近期状态"},{k:"coach_score",l:"教练因素"},{k:"mystic_score",l:"玄学因子"}];var fc=["var(--bl)","var(--gr)","var(--gd)","var(--sl)","var(--br)","var(--rd)"];var uncBadge=function(lvl){var m={low:'<span class="unc-badge unc-low">低不确定</span>',medium:'<span class="unc-badge unc-med">中不确定</span>',high:'<span class="unc-badge unc-high">高不确定</span>'};return m[lvl]||m.medium;};var h="";for(var i=0;i<Math.min(s.length,25);i++){var t=s[i];var uncLvl=t.conformal_uncertainty||"medium";var ciLo=(t.conformal_ci_low||0)*100,ciHi=(t.conformal_ci_high||0)*100;h+='<div class="fb-r" onclick="toggleFB(this)"><div class="fb-hd"><span class="fb-fl">'+fl(t.country)+'</span><span class="fb-nm">'+t.country+'</span><div class="fb-pct-wrap"><span class="fb-pct">'+(t.final_prob*100).toFixed(1)+'%</span>'+uncBadge(uncLvl)+'</div></div><div class="fb-bars">';for(var j=0;j<factors.length;j++){var f=factors[j];var v=Math.max(0,t[f.k]||0);var max_v=0.15;var w=Math.min(100,(v/max_v*100)).toFixed(1);var val_str=(v>=0?"+":"")+(v*100).toFixed(1)+"%";h+='<div class="fb-bar"><span class="fb-lbl">'+f.l+'</span><div class="fb-track"><div class="fb-fill" style="width:'+w+'%;background:'+fc[j]+'"></div></div><span class="fb-val" style="color:'+fc[j]+'">'+val_str+'</span></div>';}h+='</div><div class="fb-expanded"><div class="fb-unc-range">置信区间 '+(ciLo).toFixed(2)+'% ~ '+(ciHi).toFixed(2)+'% &nbsp;|&nbsp; <span class="unc-text-'+uncLvl+'">'+(uncLvl==="low"?"模型高度确定":uncLvl==="medium"?"有一定不确定性":"不确定性较高")+'</span></div>';var attr=t.attribution;if(attr&&attr.attributions){h+='<div class="fb-attr"><div class="fb-attr-hd">📊 概率归因</div><div class="fb-elo-base">Elo基准概率: '+(attr.elo_baseline*100).toFixed(2)+'% &rarr; 最终概率: '+(attr.final_probability*100).toFixed(2)+'%</div>';for(var k=0;k<attr.attributions.length;k++){var a=attr.attributions[k];if(Math.abs(a.contribution)<0.0001)continue;var isPos=a.contribution>0;var cls=isPos?"attr-pos":"attr-neg";var sign=isPos?"+":"";var pct=a.contribution_pct.toFixed(1);h+='<div class="fb-attr-row"><span class="fb-attr-lbl">'+a.factor_label+'</span><div class="fb-attr-bar-wrap"><div class="fb-attr-bar" style="width:'+Math.min(100,Math.abs(a.contribution)*2000).toFixed(1)+'%;background:'+(isPos?"var(--gr)":"var(--rd)")+'"></div></div><span class="'+cls+'">'+sign+(a.contribution*100).toFixed(3)+'% ('+pct+'%)</span></div>';}h+='</div>';}h+='<div class="fb-narrative">'+(t.narrative||"")+'</div></div></div>';}document.getElementById("fb").innerHTML=h;}

/* ── Mystic ── */
function toggleMC(el){var d=el.nextElementSibling;if(d.classList.contains("on")){d.classList.remove("on");}else{d.classList.add("on");}}
function buildML(){var s=D.slice().sort(function(a,b){return b.final_prob-a.final_prob;});var h="";for(var i=0;i<s.length;i++){var t=s[i],ver=t.verdict||"--";var tc=ver.indexOf("推荐")>-1?"pos":ver.indexOf("谨慎")>-1?"neg":"neu";var mtag=t.iching?'<span class="tag mystic">易:'+t.iching+"</span>":"";var contr=t.contrarian||0,favc=t.fav_curse||0,conf=t.confidence||0.5;var sh=t.shift||0,shcls=sh>0?"pos":sh<0?"neg":"";h+='<div class="mc-r" onclick="toggleMC(this)"><div class="mc-fl">'+fl(t.country)+'</div><div><div class="mc-nm">'+t.country+'</div><div class="mc-mt">'+ver+" | "+(t.final_prob*100).toFixed(2)+"%</div></div></div>";h+='<div class="mc-dt"><div class="tags">';h+='<span class="tag '+tc+'">'+ver+"</span>";if(mtag)h+=mtag;if(t.zen&&t.zen!=="--")h+='<span class="tag neu">道:'+t.zen+"</span>";if(t.tao&&t.tao!=="--")h+='<span class="tag neu">老:'+t.tao+"</span>";h+="</div><div class='mtrics'>";h+="<div class='mtric'><div class='mtric-lbl'>偏移 / Shift</div><div class='mtric-val "+shcls+"'>"+st(sh)+"</div></div>";h+="<div class='mtric'><div class='mtric-lbl'>悖论 / Paradox</div><div class='mtric-val'>"+contr.toFixed(3)+"</div></div>";h+="<div class='mtric'><div class='mtric-lbl'>热门诅咒 / FavCurse</div><div class='mtric-val'>"+favc.toFixed(3)+"</div></div>";h+="<div class='mtric'><div class='mtric-lbl'>置信度 / Confidence</div><div class='mtric-val'>"+(conf*100).toFixed(0)+"%</div></div>";h+="</div></div>";}document.getElementById("ml").innerHTML=h;}

/* ── H2H ── */
var H2H_RECORDS={
"Argentina|Brazil":{wA:41,d:26,wB:47,t:114,note:"南美经典对决，巴西总体占优"},
"Argentina|France":{wA:5,d:3,wB:4,t:12,note:"2022决赛重演，阿根廷点球险胜"},
"Brazil|France":{wA:6,d:4,wB:8,t:18,note:"2006决赛，法国加时胜"},
"France|Germany":{wA:13,d:4,wB:14,t:31,note:"欧洲强强对话，大赛多次相遇"},
"England|Germany":{wA:13,d:5,wB:14,t:32,note:"经典大战，英格兰点球3战3败"},
"England|France":{wA:7,d:7,wB:17,t:31,note:"法国近期大赛占优"},
"Germany|Spain":{wA:8,d:6,wB:11,t:25,note:"传控vs力量，各有胜负"},
"Portugal|Spain":{wA:18,d:8,wB:11,t:37,note:"伊比利亚德比，葡萄牙总胜多"},
"Brazil|Germany":{wA:9,d:5,wB:9,t:23,note:"2014半决赛1-7成为经典"},
"Argentina|Germany":{wA:8,d:4,wB:8,t:20,note:"3次决赛，2022马拉多纳主场夺冠"},
"Croatia|England":{wA:2,d:3,wB:3,t:8,note:"2018世界杯半决赛，克罗地亚加时胜"},
"Uruguay|Brazil":{wA:31,d:18,wB:27,t:76,note:"南美最激烈对决之一"},
"Netherlands|Germany":{wA:14,d:15,wB:16,t:45,note:"欧洲老牌劲旅对抗"},
"Italy|Germany":{wA:15,d:13,wB:9,t:37,note:"欧洲杯决赛多次交锋"},
"Spain|France":{wA:16,d:7,wB:13,t:36,note:"2012欧洲杯决赛，西班牙大胜"},
"Belgium|France":{wA:5,d:4,wB:9,t:18,note:"法国近期杯赛表现更佳"},
"England|Brazil":{wA:9,d:5,wB:13,t:27,note:"2002小组赛后未在大赛相遇"},
"Portugal|Argentina":{wA:2,d:1,wB:4,t:7,note:"2014世界杯小组赛，最近一次2018"}
};
var H2H_TACTICAL={
"Brazil|France":"桑巴艺术 vs 法式精密",
"Argentina|France":"潘帕斯激情 vs 欧洲铁军",
"Argentina|Brazil":"南美双雄巅峰对话",
"France|Germany":"个人能力 vs 整体执行",
"England|Germany":"边路传中 vs 德国坦克",
"Portugal|Spain":"C罗单打 vs 整体传控",
"Brazil|Germany":"进攻艺术 vs 纪律铁军"
};

function h2hCalc(ta,tb){
  var eloA=ta.mod_elo||ta.elo||1700,eloB=tb.mod_elo||tb.elo||1700;
  var eDiff=eloA-eloB;
  // Elo-based win probability (no draw)
  var eloWinA=1/(1+Math.pow(10,-eDiff/400));
  // Draw probability 10-35%, closer teams draw more
  var drawP=Math.max(0.10,Math.min(0.35,0.30-Math.abs(eDiff)/1500));
  // Allocate remaining probability to wins, preserving Elo ratio
  var winTotal=1-drawP;
  var rawA=eloWinA*winTotal+0.03;
  var rawB=(1-eloWinA)*winTotal+0.03;
  var rawTotal=rawA+rawB;
  // Normalize wins so winA+winB = winTotal (and winA+winB+drawP=1)
  return{winA:rawA/rawTotal*winTotal,winB:rawB/rawTotal*winTotal,draw:drawP,eloDiff:eDiff};
}

function selectedSchedulePrediction(){
  var sel=document.getElementById("h2h-match");
  if(!sel||sel.value==="manual")return null;
  var rows=scheduleH2HMatches();
  return rows[parseInt(sel.value,10)]||_selectedScheduleMatch||null;
}

function scheduleMatchPrediction(){
  var base=selectedSchedulePrediction();
  if(!base)return null;
  var live=_liveSchedulePredictions[scheduleMatchKey(base)];
  if(live&&live.live_available)return live;
  return base;
}

function scheduleMatchKey(match){
  if(!match)return "";
  if(match.match_id)return String(match.match_id);
  if(match.num!==undefined&&match.num!==null&&match.num!=="")return "num:"+String(match.num);
  return "teams:"+(match.team1||"")+"|"+(match.team2||"")+"|"+(match.date||"");
}

function scheduleH2HCalc(match){
  return{
    winA:match.team1_win||0,
    draw:match.draw||0,
    winB:match.team2_win||0,
    eloDiff:null,
    source:"schedule"
  };
}

function getFactorDiff(ta,tb){
  var fs=[{k:"elo_score",l:"Elo锚点"},{k:"age_score",l:"年龄结构"},{k:"exp_score",l:"大赛经验"},{k:"form_score",l:"近期状态"},{k:"coach_score",l:"教练因素"},{k:"mystic_score",l:"玄学因子"}];
  var h="";
  for(var i=0;i<fs.length;i++){
    var f=fs[i],va=ta[f.k]||0,vb=tb[f.k]||0;
    var maxV=Math.max(va,vb,0.01);
    var pctA=(va/maxV*100).toFixed(0),pctB=(vb/maxV*100).toFixed(0);
    var wcls=va>vb?"var(--gr)":vb>va?"var(--rd)":"var(--tx2)";
    h+='<div class="h2h-fr"><span class="h2h-fr-lbl">'+f.l+'</span><div class="h2h-fr-bar"><div class="h2h-fr-a" style="width:'+pctA+'%;background:var(--bl)"></div><div class="h2h-fr-b" style="width:'+pctB+'%;background:var(--gd)"></div></div><span class="h2h-fr-val" style="color:'+wcls+'">'+(va>vb?"A":vb>va?"B":"=")+'</span></div>';
  }
  return h;
}

function getPlayerMatchups(ta,tb){
  var posC={GK:"#8e8e93",DF:"#0a84ff",MF:"#30d158",FW:"#ff453a"};
  var posN={GK:"Goalkeeper",DF:"Defender",MF:"Midfielder",FW:"Forward"};
  function topByPos(players,pos){return(players||[]).filter(function(p){return p.position===pos;}).slice(0,3);}
  var h="";
  var posCodes=["GK","DF","MF","FW"];
  for(var pi=0;pi<posCodes.length;pi++){
    var pc=posCodes[pi],posName=posN[pc];
    var aTop=(ta.players||[]).filter(function(p){return p.position===pc;}).slice(0,3);
    var bTop=(tb.players||[]).filter(function(p){return p.position===pc;}).slice(0,3);
    var maxLen=Math.max(aTop.length,bTop.length);
    if(maxLen===0)continue;
    h+='<div class="h2h-mu-pos">'+posName+'</div>';
    for(var mi=0;mi<maxLen;mi++){
      var pa=aTop[mi]||null,pb=bTop[mi]||null;
      var sa=pa?(pa.market_value||0):0,sb=pb?(pb.market_value||0):0;
      var wcls=sa>sb?"w":sb>sa?"l":"d";
      h+='<div class="h2h-mu-row">';
      h+='<div class="h2h-mu-p">'+(pa?pa.name:"—")+' <span class="h2h-wl '+wcls+'">'+(pa?(sa>0?sa.toFixed(1)+"M":"✓"):"—")+'</span></div>';
      h+='<div class="h2h-mu-s">vs</div>';
      h+='<div class="h2h-mu-r">'+(pb?'<span class="h2h-wl '+(sb>sa?"w":sb<sa?"l":"d")+'">'+(sb>0?sb.toFixed(1)+"M":"✓")+'</span> '+pb.name:"—")+'</div>';
      h+='</div>';
    }
  }
  return h;
}



function buildScorePred(ta, tb, r) {
    var eloA = ta.mod_elo || ta.elo || 1700;
    var eloB = tb.mod_elo || tb.elo || 1700;
    var lambdaA = 1.3 + (eloA - 1700) / 500 * 1.0;
    var lambdaB = 1.3 + (eloB - 1700) / 500 * 1.0;
    var shiftA = ta.shift || 0;
    var shiftB = tb.shift || 0;
    lambdaA = lambdaA * (1 + shiftA * 3.0);
    lambdaB = lambdaB * (1 + shiftB * 3.0);
    lambdaA = Math.max(0.3, Math.min(4.0, lambdaA));
    lambdaB = Math.max(0.3, Math.min(4.0, lambdaB));

    function pois(k, lam) {
        if (lam <= 0) return k === 0 ? 1 : 0;
        var p = Math.exp(-lam);
        for (var i = 1; i <= k; i++) p *= lam / i;
        return p;
    }

    var EXTREME_THRESH = 5;
    var BOOST_FACTOR = 3.0;

    var raw = [];
    for (var ga = 0; ga <= 5; ga++) {
        for (var gb = 0; gb <= 5; gb++) {
            raw.push({ga: ga, gb: gb, pois: pois(ga, lambdaA) * pois(gb, lambdaB), total: ga + gb});
        }
    }

    var sumBase = 0;
    var sumBoosted = 0;
    for (var i = 0; i < raw.length; i++) {
        raw[i].boosted = raw[i].total >= EXTREME_THRESH ? raw[i].pois * BOOST_FACTOR : raw[i].pois;
        sumBase += raw[i].pois;
        sumBoosted += raw[i].boosted;
    }
    for (var i = 0; i < raw.length; i++) {
        raw[i].baseProb = raw[i].pois / sumBase;
        raw[i].boostedProb = raw[i].boosted / sumBoosted;
        raw[i].prob = raw[i].baseProb;
    }

    // Sort by prob for grid display
    var sorted = raw.slice().sort(function(a, b){ return b.prob - a.prob; });
    var top6 = sorted.slice(0, 6);
    var totalShown = top6.reduce(function(s, x){ return s + x.prob; }, 0);

    var hiAll = raw.filter(function(x){ return x.total >= EXTREME_THRESH; });
    hiAll.sort(function(a, b){ return b.boostedProb - a.boostedProb; });
    var topHi = hiAll.slice(0, 8);
    var hiTotal = topHi.reduce(function(s, x){ return s + x.boostedProb; }, 0);

    // Featured prediction: deterministic weighted random from top 3
    // Uses matchup string hash so same matchup always picks same result (reproducible)
    // but different matchups naturally get different featured predictions
    function hashStr(s) {
        var h = 0;
        for (var i = 0; i < s.length; i++) {
            h = ((h << 5) - h) + s.charCodeAt(i);
            h = h & h;
        }
        return Math.abs(h) / 2147483647;
    }
    var matchupKey = (ta.country || "") + " vs " + (tb.country || "");
    var top3 = sorted.slice(0, 3);
    var top3Total = top3.reduce(function(s, x){ return s + x.prob; }, 0);
    var seed = hashStr(matchupKey);
    var cumsum = 0;
    for (var i = 0; i < top3.length; i++) {
        cumsum += top3[i].prob / top3Total;
        if (seed < cumsum) { var featured = top3[i]; break; }
    }
    if (!featured) featured = top3[0];

    var h = '<div class="sc-pred">';

    // Expected goals header with featured prediction highlighted
    h += '<div class="sc-pred-r">';
    h += '<div class="sc-team"><div class="sc-team-nm">' + ta.country + '</div><div class="sc-goals"><span class="sc-gl">' + lambdaA.toFixed(1) + '</span></div></div>';
    h += '<div class="sc-sep">:</div>';
    h += '<div class="sc-team"><div class="sc-team-nm">' + tb.country + '</div><div class="sc-goals"><span class="sc-gl">' + lambdaB.toFixed(1) + '</span></div></div>';
    h += '</div>';

    // Top 6 grid: always shows mathematically most probable
    h += '<div class="sc-hd"><span>最可能 / Most Likely</span><span class="sc-hd-sub">+' + (totalShown * 100).toFixed(0) + '%</span></div>';
    h += '<div class="sc-grid">';
    for (var i = 0; i < top6.length; i++) {
        var s2 = top6[i];
        var isTop = i === 0;
        var isFeatured = s2.ga === featured.ga && s2.gb === featured.gb;
        var pct = (s2.prob * 100).toFixed(1);
        h += '<div class="sc-cell' + (isTop || isFeatured ? ' top' : '') + '">';
        h += '<div class="sc-s">' + s2.ga + ' - ' + s2.gb + (isFeatured ? ' &#9733;' : '') + '</div>';
        h += '<div class="sc-p">' + pct + '%</div></div>';
    }
    h += '</div>';

    // High-scoring section
    if (topHi.length > 0) {
        h += '<div class="sc-hd sc-hd-hi"><span>&#9888;&#65039; 大比分博弈 / High-Score (&#215;3 boost for total&#8805;5)</span><span class="sc-hd-sub">' + (hiTotal * 100).toFixed(0) + '%</span></div>';
        h += '<div class="sc-grid sc-grid-hi">';
        for (var j = 0; j < topHi.length; j++) {
            var s3 = topHi[j];
            var pct2 = (s3.boostedProb * 100).toFixed(1);
            var isExtreme = s3.total >= EXTREME_THRESH;
            h += '<div class="sc-cell sc-cell-hi' + (isExtreme ? '" style="border-color:var(--gd)"' : '') + '">';
            h += '<div class="sc-s">' + s3.ga + ' - ' + s3.gb + (isExtreme ? ' &#10023;' : '') + '</div>';
            h += '<div class="sc-p">' + pct2 + '%</div></div>';
        }
        h += '</div>';
        h += '<div class="sc-note">&#128293; &#215;3 boost for extreme outcomes (total&#8805;5: 5-0,4-1,5-1,3-2...). Renormalized from base Poisson | Elo &#955;: ' + lambdaA.toFixed(2) + ' vs ' + lambdaB.toFixed(2) + '</div>';
    }

    // Most likely scoreline list — featured prediction shown first with highlight
    h += '<div class="sc-most-likely">';
    h += '<div class="sc-ml-hd">Scoreline Probability / 比分预测 ' + (featured ? '&#9733; ' + featured.ga + '-' + featured.gb : '') + '</div>';
    for (var k = 0; k < top6.length; k++) {
        var row = top6[k];
        var isFeatured = featured && row.ga === featured.ga && row.gb === featured.gb;
        var outcome = row.ga > row.gb ? 'A' : (row.ga < row.gb ? 'B' : 'D');
        var outcomeColor = outcome === 'A' ? 'var(--bl)' : 'var(--gd)';
        var pct3 = (row.prob * 100).toFixed(1);
        h += '<div class="sc-ml-row' + (isFeatured ? '" style="background:rgba(255,214,10,0.08)"' : '') + '">';
        h += '<span class="sc-ml-sc">' + fl(ta.country) + ' ' + row.ga + ' : ' + row.gb + ' ' + fl(tb.country) + (isFeatured ? ' &#9733;' : '') + '</span>';
        h += '<span class="sc-ml-od">' + outcome + '</span>';
        h += '<span class="sc-ml-d" style="color:' + outcomeColor + '">' + pct3 + '%</span></div>';
    }
    h += '</div>';
    h += '<div class="sc-note">Poisson xG baseline + separate extreme-tail boost view + mystical weighted random | Elo &#955;: ' + lambdaA.toFixed(2) + ' vs ' + lambdaB.toFixed(2) + '</div></div>';
    return h;
}

function buildScheduleScorePred(ta,tb,match){
  var lambdaA=match.lambda_team1||0;
  var lambdaB=match.lambda_team2||0;
  var scores=(match.top_scores||[]).slice(0,6);
  var totalShown=scores.reduce(function(s,x){return s+(x.prob||0);},0);
  var featured=scores[0]||null;
  var h='<div class="sc-pred">';
  h+='<div class="sc-pred-r">';
  h+='<div class="sc-team"><div class="sc-team-nm">'+ta.country+'</div><div class="sc-goals"><span class="sc-gl">'+lambdaA.toFixed(1)+'</span></div></div>';
  h+='<div class="sc-sep">:</div>';
  h+='<div class="sc-team"><div class="sc-team-nm">'+tb.country+'</div><div class="sc-goals"><span class="sc-gl">'+lambdaB.toFixed(1)+'</span></div></div>';
  h+='</div>';
  h+='<div class="sc-hd"><span>最可能 / Most Likely</span><span class="sc-hd-sub">+'+(totalShown*100).toFixed(0)+'%</span></div>';
  h+='<div class="sc-grid">';
  for(var i=0;i<scores.length;i++){
    var s=scores[i];
    var ga=s.team1_goals,gb=s.team2_goals;
    var isFeatured=featured&&ga===featured.team1_goals&&gb===featured.team2_goals;
    h+='<div class="sc-cell'+(i===0?' top':'')+'">';
    h+='<div class="sc-s">'+ga+' - '+gb+(isFeatured?' &#9733;':'')+'</div>';
    h+='<div class="sc-p">'+((s.prob||0)*100).toFixed(1)+'%</div></div>';
  }
  h+='</div>';
  h+='<div class="sc-most-likely">';
  h+='<div class="sc-ml-hd">Scoreline Probability / 比分预测 '+(featured?'&#9733; '+featured.score:'')+'</div>';
  for(var k=0;k<scores.length;k++){
    var row=scores[k];
    var outcome=row.team1_goals>row.team2_goals?'A':(row.team1_goals<row.team2_goals?'B':'D');
    var outcomeColor=outcome==='A'?'var(--bl)':(outcome==='B'?'var(--gd)':'var(--tx2)');
    var star=featured&&row.team1_goals===featured.team1_goals&&row.team2_goals===featured.team2_goals;
    h+='<div class="sc-ml-row'+(star?'" style="background:rgba(255,214,10,0.08)"':'')+'">';
    h+='<span class="sc-ml-sc">'+fl(ta.country)+' '+row.team1_goals+' : '+row.team2_goals+' '+fl(tb.country)+(star?' &#9733;':'')+'</span>';
    h+='<span class="sc-ml-od">'+outcome+'</span>';
    h+='<span class="sc-ml-d" style="color:'+outcomeColor+'">'+((row.prob||0)*100).toFixed(1)+'%</span></div>';
  }
  h+='</div>';
  var note=match.source==="live-state"?'Live-state model / 实时模型：赛前赛程预测 + 当前比分/分钟/红牌/xG/裁判可用项，重新积分最终比分概率':'Schedule model / 赛程模型：直接使用生成 artifact 的单场预测，不使用前端 H2H 临时公式或高比分 boost';
  if(match.expected_final_score&&match.expected_final_score.display)note+=' | Expected final / 预期终场: '+match.expected_final_score.display;
  h+='<div class="sc-note">'+note+' | Elo &#955;: '+lambdaA.toFixed(2)+' vs '+lambdaB.toFixed(2)+'</div></div>';
  return h;
}

function h2hChange(){
  var ta=D.find(function(x){return x.country===document.getElementById("h2h-a").value;});
  var tb=D.find(function(x){return x.country===document.getElementById("h2h-b").value;});
  if(!ta||!tb){return;}
  var sched=scheduleMatchPrediction();
  var r=sched?scheduleH2HCalc(sched):h2hCalc(ta,tb);
  var barA=(r.winA*100).toFixed(1),barB=(r.winB*100).toFixed(1),barD=(r.draw*100).toFixed(1);
  document.getElementById("h2h-bar-a").style.width=barA+"%";
  document.getElementById("h2h-bar-a").textContent=barA+"%";
  document.getElementById("h2h-bar-b").style.width=barB+"%";
  document.getElementById("h2h-bar-b").textContent=barB+"%";
  document.getElementById("h2h-bar-d").style.width=barD+"%";
  document.getElementById("h2h-bar-d").textContent=barD+"%";
  // factor diff
  var h='<div class="h2h-fc">'+getFactorDiff(ta,tb)+'</div>';

  // Conformal Prediction Set
  var cpData=(HC&&HC[ta.country]&&HC[ta.country][tb.country])?HC[ta.country][tb.country]:null;
  if(cpData){
    var setLbl=cpData.prediction_set.join("/");
    var setColor=cpData.set_size===1?"var(--gr)":cpData.set_size===2?"var(--gd)":"var(--rd)";
    var setBg=cpData.set_size===1?"rgba(48,209,88,0.12)":cpData.set_size===2?"rgba(255,214,10,0.10)":"rgba(255,69,58,0.10)";
    h+='<div class="cp-set-box" style="background:'+setBg+';border:1px solid '+setColor+';border-radius:12px;padding:12px 14px;margin-bottom:14px">';
    h+='<div class="cp-set-hd"><span class="cp-set-lbl">Conformal 预测集</span><span class="cp-set-badge" style="background:'+setColor+';color:var(--bg)">'+setLbl+'</span></div>';
    h+='<div class="cp-set-exp">'+cpData.explanation+'</div>';
    h+='<div class="cp-set-conf">置信度: '+(cpData.confidence*100).toFixed(0)+'%</div>';
    h+='</div>';
  }

  h += sched?buildScheduleScorePred(ta,tb,sched):buildScorePred(ta, tb, r);
  // historical record
  var recKey=ta.country+"|"+tb.country,recKeyRev=tb.country+"|"+ta.country;
  var rec=H2H_RECORDS[recKey]||H2H_RECORDS[recKeyRev];
  var isRev=!!H2H_RECORDS[recKeyRev];
  if(rec){
    var wA=isRev?rec.wB:rec.wA,wB=isRev?rec.wA:rec.wB;
    h+='<div class="h2h-matchup"><div class="h2h-mu-pos">历史交锋 / Historical H2H</div>';
    h+='<div style="display:flex;gap:8px;margin-top:8px">';
    h+='<div class="h2h-3m-it" style="flex:2"><div class="h2h-3m-v">'+wA+'</div><div class="h2h-3m-l">'+ta.country.slice(0,6)+' Wins</div></div>';
    h+='<div class="h2h-3m-it"><div class="h2h-3m-v">'+rec.d+'</div><div class="h2h-3m-l">Draws</div></div>';
    h+='<div class="h2h-3m-it" style="flex:2"><div class="h2h-3m-v">'+wB+'</div><div class="h2h-3m-l">'+tb.country.slice(0,6)+' Wins</div></div>';
    h+='</div>';
    h+='<div class="h2h-note">'+rec.note+' <span style="color:var(--tx2)">('+rec.t+'场 / '+rec.t+' matches)</span></div></div>';
  }
  // tactical note
  var tacKey=ta.country+"|"+tb.country,tacKeyRev=tb.country+"|"+ta.country;
  var tac=H2H_TACTICAL[tacKey]||H2H_TACTICAL[tacKeyRev];
  if(tac){
    h+='<div class="h2h-matchup"><div class="h2h-mu-pos">战术风格 / Tactical</div>';
    h+='<div class="h2h-note" style="margin-top:8px"><strong>'+tac+'</strong></div></div>';
  }
  // player matchups
  if((ta.players||[]).length>0&&(tb.players||[]).length>0){
    h+='<div class="h2h-matchup">'+getPlayerMatchups(ta,tb)+'</div>';
  }
  document.getElementById("h2h-content").innerHTML=h;
}

/* ── Squad ── */
function sqChange(){var sel=document.getElementById("sq-sel");var c=sel.value;var t=D.find(function(x){return x.country===c;});if(!t){document.getElementById("sq-content").innerHTML="<p style='color:var(--tx2);font-size:14px;padding:20px 0'>No data</p>";return;}var h='<div class="sq-card"><div class="sq-ph"><span class="sq-ph-fl">'+fl(t.country)+'</span><div><div class="sq-ph-nm">'+t.country+'</div><div class="sq-ph-elo">Elo '+(t.elo||0).toFixed(0)+' · '+(t.players?t.players.length:0)+' players</div></div></div>';if(t.players&&t.players.length>0){h+='<table class="sq-table"><thead><tr><th class="sq-th" style="width:32px">Pos</th><th class="sq-th">Name / Club</th><th class="sq-th" style="text-align:right">Caps</th><th class="sq-th" style="text-align:right">Goals</th><th class="sq-th" style="text-align:right">MV</th></tr></thead><tbody>';var pos_c={GK:"#8e8e93",DF:"#0a84ff",MF:"#30d158",FW:"#ff453a"};for(var k=0;k<t.players.length;k++){var p=t.players[k];var pc2=pos_c[p.position]||"var(--tx2)";h+='<tr><td class="sq-td"><span class="sq-pos" style="color:'+pc2+'">'+p.position+'</span></td>';h+='<td class="sq-td"><div class="sq-name">'+p.name+'</div><div class="sq-club">'+(p.club||"")+"</div></td>";h+='<td class="sq-td sq-caps">'+p.national_caps+"</td>";h+='<td class="sq-td sq-goals">'+p.national_goals+"</td>";h+='<td class="sq-td"><span class="sq-mv">'+(p.market_value||0).toFixed(1)+"M</span></td></tr>";}h+="</tbody></table>";}else{h+='<div style="padding:20px;color:var(--tx2);font-size:13px">Sample squad (no Wikipedia data) / 样本阵容（无维基数据）</div>';}h+="</div>";document.getElementById("sq-content").innerHTML=h;}

/* Polymarket comparison data */
var POLY_WINNER={
"France":{price:0.18},"Spain":{price:0.17},"England":{price:0.11},"Portugal":{price:0.10},
"Brazil":{price:0.09},"Argentina":{price:0.08},"Germany":{price:0.05},"Netherlands":{price:0.03},
"Norway":{price:0.02},"Japan":{price:0.02},"Colombia":{price:0.018},"Belgium":{price:0.018},
"Morocco":{price:0.015},"USA":{price:0.012},"Uruguay":{price:0.011},"Mexico":{price:0.011},
"Switzerland":{price:0.010},"Croatia":{price:0.009},"Ecuador":{price:0.008},"Turkiye":{price:0.007},
"Senegal":{price:0.007},"Austria":{price:0.006},"Sweden":{price:0.006},"Canada":{price:0.004},
"South Korea":{price:0.003},"Ghana":{price:0.003},"Bosnia-Herzegovina":{price:0.003},
"Italy":{price:0.003},"Australia":{price:0.002},"Nigeria":{price:0.002},"Ivory Coast":{price:0.002},
"Algeria":{price:0.002},"Serbia":{price:0.002},"Poland":{price:0.001},"Ukraine":{price:0.001},
"Cameroon":{price:0.001},"Chile":{price:0.001},"Egypt":{price:0.001},"Greece":{price:0.001},
"Mali":{price:0.001},"Paraguay":{price:0.001},"Peru":{price:0.001},"Qatar":{price:0.001},
"Romania":{price:0.001},"Saudi Arabia":{price:0.001},"Tunisia":{price:0.001},"Uzbekistan":{price:0.001},
"Venezuela":{price:0.001},"Albania":{price:0.001},"Bulgaria":{price:0.001},"Burkina Faso":{price:0.001},
"China":{price:0.001},"Czech Republic":{price:0.001},"Denmark":{price:0.001},"Finland":{price:0.001},
"Gabon":{price:0.001},"Ghana":{price:0.001},"Hungary":{price:0.001},"Iceland":{price:0.001}
};

function buildPoly(){
  var el=document.getElementById("poly-winner");
  if(!el)return;
  var rows=[];
  var valueRows=[];var overRows=[];
  for(var i=0;i<D.length;i++){
    var t=D[i];
    var market=POLY_WINNER[t.country];
    if(!market)continue;
    var modelPct=(t.final_prob*100).toFixed(1);
    var mktPct=(market.price*100).toFixed(1);
    var dev=(t.final_prob-market.price)*100;
    var devStr=(dev>=0?"+":"")+dev.toFixed(1)+"%";
    var cls=dev>1?"pos":dev<-1?"neg":"neu";
    var maxP=Math.max(t.final_prob,market.price);
    rows.push({country:t.country,modelPct:modelPct,mktPct:mktPct,dev:dev,devStr:devStr,cls:cls,maxP:maxP,barW:(maxP*100).toFixed(1)});
    if(dev>1)valueRows.push({country:t.country,dev:dev,devStr:devStr,modelPct:modelPct,mktPct:mktPct,finalProb:t.final_prob});
    if(dev<-1)overRows.push({country:t.country,dev:dev,devStr:devStr,modelPct:modelPct,mktPct:mktPct});
  }
  rows.sort(function(a,b){return b.dev-a.dev;});
  var html="";
  for(var j=0;j<rows.length;j++){
    var r=rows[j];
    var modelBar=(parseFloat(r.modelPct)/parseFloat(r.barW)*100).toFixed(0);
    var mktBar=(parseFloat(r.mktPct)/parseFloat(r.barW)*100).toFixed(0);
    html+='<div class="pm-row">';
    html+='<span class="pm-fl">'+fl(r.country)+'</span>';
    html+='<span class="pm-nm">'+r.country+'</span>';
    html+='<div class="pm-bar"><div class="pm-bar-in" style="width:'+modelBar+'%;background:var(--bl);opacity:0.7;border-radius:3px"></div></div>';
    html+='<div class="pm-bar"><div class="pm-bar-in" style="width:'+mktBar+'%;background:var(--gd);opacity:0.7;border-radius:3px"></div></div>';
    html+='<span class="pm-val '+r.cls+'">'+r.devStr+'</span>';
    html+='</div>';
  }
  el.innerHTML=html||'<div style="color:var(--tx2);font-size:13px;padding:16px 0">No matching market data</div>';

  // Build summary
  var sumEl=document.getElementById("poly-sum");
  if(sumEl){
    var sumHtml='<div class="pm-sum-tl">📊 博弈结论 / Summary</div>';
    if(valueRows.length>0){
      valueRows.sort(function(a,b){return b.dev-a.dev;});
      sumHtml+='<div class="pm-sum-grp">';
      for(var vi=0;vi<valueRows.length;vi++){
        var v=valueRows[vi];
        sumHtml+='<div class="pm-sum-row"><span class="pm-sum-dot" style="background:var(--gr)"></span><span class="pm-sum-val" style="color:var(--gr)">'+v.devStr+'</span><span class="pm-sum-lbl">'+fl(v.country)+' '+v.country+'</span><span style="color:var(--tx2);font-size:11px">'+v.modelPct+'% vs 市价'+v.mktPct+'%</span></div>';
      }
      sumHtml+='<div style="font-size:11px;color:var(--tx2);margin-top:4px">市场对你低估，可考虑买入</div>';
      sumHtml+='</div>';
    }
    if(overRows.length>0){
      overRows.sort(function(a,b){return a.dev-b.dev;});
      sumHtml+='<div class="pm-sum-grp">';
      for(var oi=0;oi<overRows.length;oi++){
        var o=overRows[oi];
        sumHtml+='<div class="pm-sum-row"><span class="pm-sum-dot" style="background:var(--rd)"></span><span class="pm-sum-val" style="color:var(--rd)">'+o.devStr+'</span><span class="pm-sum-lbl">'+fl(o.country)+' '+o.country+'</span><span style="color:var(--tx2);font-size:11px">'+o.modelPct+'% vs 市价'+o.mktPct+'%</span></div>';
      }
      sumHtml+='<div style="font-size:11px;color:var(--tx2);margin-top:4px">市场对你高估，追高需谨慎</div>';
      sumHtml+='</div>';
    }
    if(valueRows.length===0&&overRows.length===0){
      sumHtml+='<div style="font-size:11px;color:var(--tx2);margin-top:4px">当前无显著偏离（|差|&le;1%），无明显博弈机会</div>';
    }
    // Top 3 by model confidence (always show top 3 strongest model predictions)
    var allScored=[];
    for(var ai=0;ai<D.length;ai++){
      var t=D[ai];
      var mkt=POLY_WINNER[t.country];
      if(!mkt)continue;
      var dev2=(t.final_prob-mkt.price)*100;
      allScored.push({country:t.country,dev:dev2,modelPct:(t.final_prob*100).toFixed(1),mktPct:(mkt.price*100).toFixed(1),finalProb:t.final_prob});
    }
    allScored.sort(function(a,b){return b.finalProb-a.finalProb;});
    var top3Model=allScored.slice(0,3);
    var medals=["🥇","🥈","🥉"];
    sumHtml+='<div style="margin-top:14px;padding-top:12px;border-top:1px solid var(--bg3)">';
    sumHtml+='<div style="font-size:12px;font-weight:600;color:var(--bl);margin-bottom:8px">🏆 模型预测 / Model Top 3</div>';
    for(var ti=0;ti<top3Model.length;ti++){
      var t3=top3Model[ti];
      var badge="";
      var valColor="var(--tx2)";
      if(t3.dev>1){badge='<span style="background:var(--gr);color:#000;font-size:10px;padding:1px 5px;border-radius:4px;margin-left:4px">低估</span>';valColor="var(--gr)"}
      else if(t3.dev<-1){badge='<span style="background:var(--rd);color:#fff;font-size:10px;padding:1px 5px;border-radius:4px;margin-left:4px">高估</span>';valColor="var(--rd)"}
      sumHtml+='<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">';
      sumHtml+='<span style="font-size:16px">'+medals[ti]+'</span>';
      sumHtml+='<span style="font-size:14px;font-weight:600;color:var(--tx)">'+fl(t3.country)+'</span>';
      sumHtml+='<span style="font-size:12px;color:var(--tx2)">'+t3.modelPct+'%</span>';
      sumHtml+='<span style="margin-left:auto;font-size:11px;color:'+valColor+'">'+(t3.dev>=0?"+":"")+t3.dev.toFixed(1)+'%</span>'+badge;
      sumHtml+='</div>';
    }

    // Top 3 by value score: model_prob × deviation
    if(valueRows.length>0){
      var scored=valueRows.map(function(v){return {country:v.country,dev:v.dev,devStr:v.devStr,modelPct:v.modelPct,mktPct:v.mktPct,finalProb:v.finalProb,score:(v.finalProb*100)*(v.dev)}});
      scored.sort(function(a,b){return b.score-a.score;});
      var top3val=scored.slice(0,3);
      sumHtml+='<div style="font-size:12px;font-weight:600;color:var(--gr);margin-top:14px;margin-bottom:8px">💰 价值机会 / Value Picks</div>';
      for(var vi=0;vi<top3val.length;vi++){
        var v=top3val[vi];
        sumHtml+='<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px">';
        sumHtml+='<span style="font-size:14px">'+medals[vi]+'</span>';
        sumHtml+='<span style="font-size:13px;font-weight:600;color:var(--tx)">'+fl(v.country)+'</span>';
        sumHtml+='<span style="font-size:11px;color:var(--tx2)">'+v.modelPct+'%→市价'+v.mktPct+'%</span>';
        sumHtml+='<span style="margin-left:auto;font-size:12px;font-weight:700;color:var(--gr)">'+v.devStr+'</span>';
        sumHtml+='</div>';
      }
      sumHtml+='<div style="font-size:11px;color:var(--tx2);margin-top:4px">模型概率×偏差综合排名 | 仅供参考</div>';
    }
    sumHtml+='</div>';
    sumEl.innerHTML=sumHtml;
  }

  document.getElementById("poly-upd").textContent=new Date().toLocaleString("zh-CN",{month:"numeric",day:"numeric",hour:"2-digit",minute:"2-digit"});
}

var _pickerSide=null;
var _selectedScheduleMatch=null;
var _liveSchedulePredictions={};
function openPicker(side){_pickerSide=side;var t=side==="a"?"Team A / 球队A":"Team B / 球队B";document.getElementById("pick-title").textContent=t;document.getElementById("pick-search").value="";filterPickList();document.getElementById("pick-overlay").classList.add("on");document.body.style.overflow="hidden"}
function closePicker(e){if(e&&e.target!==document.getElementById("pick-overlay"))return;document.getElementById("pick-overlay").classList.remove("on");document.body.style.overflow=""}
function filterPickList(){var q=document.getElementById("pick-search").value.toLowerCase();var list=document.getElementById("pick-list");var curVal=_pickerSide==="a"?document.getElementById("h2h-a").value:document.getElementById("h2h-b").value;var html="";for(var i=0;i<D.length;i++){var t=D[i];if(t.country.toLowerCase().indexOf(q)===-1&&fl(t.country).toLowerCase().indexOf(q)===-1)continue;var isSel=t.country===curVal;html+="<div class=\"pick-item"+(isSel?" sel":"")+"\" onclick=\"selectPick(\'"+t.country+"\')\">";html+="<span class=\"pick-item-fl\">"+fl(t.country)+"</span>";html+="<span class=\"pick-item-info\"><span class=\"pick-item-nm\">"+t.country+"</span>";html+="<span class=\"pick-item-pr\">"+(t.final_prob*100).toFixed(2)+"%</span></span>";html+="<span class=\"pick-item-chk\">&#10003;</span></div>"}list.innerHTML=html||"<div style=\"padding:24px;text-align:center;color:var(--tx2);font-size:14px\">No result</div>"}
function selectPick(country){var matchSel=document.getElementById("h2h-match");if(matchSel)matchSel.value="manual";_selectedScheduleMatch=null;if(_pickerSide==="a"){document.getElementById("h2h-a").value=country;updatePickCard("a",country)}else{document.getElementById("h2h-b").value=country;updatePickCard("b",country)}closePicker();h2hChange()}
function updatePickCard(side,country){var t=D.find(function(x){return x.country===country;});if(!t)return;document.getElementById("h2h-pick-fl-"+side).textContent=fl(t.country);document.getElementById("h2h-pick-nm-"+side).textContent=t.country;document.getElementById("h2h-pick-pr-"+side).textContent="Champion "+(t.final_prob*100).toFixed(2)+"%"}
function updateSchedulePickCards(match){
  document.getElementById("h2h-pick-pr-a").textContent="Match win "+((match.team1_win||0)*100).toFixed(1)+"%";
  document.getElementById("h2h-pick-pr-b").textContent="Match win "+((match.team2_win||0)*100).toFixed(1)+"%";
}

var H2H_TEAM_ALIAS={"Bosnia & Herzegovina":"Bosnia and Herzegovina","Korea Republic":"South Korea","IR Iran":"Iran","Côte d'Ivoire":"Ivory Coast","DR of the Congo":"DR Congo"};
function h2hTeamName(c){return H2H_TEAM_ALIAS[c]||c;}
function scheduleH2HMatches(){
  var rows=(SP&&SP.matches)||[];
  return rows.filter(function(m){return m.team1&&m.team2&&m.date;}).sort(function(a,b){
    var ak=(a.date||"")+" "+(a.time||"");
    var bk=(b.date||"")+" "+(b.time||"");
    return ak<bk?-1:ak>bk?1:0;
  });
}
function populateScheduleH2H(){
  var sel=document.getElementById("h2h-match");
  if(!sel)return;
  var rows=scheduleH2HMatches();
  sel.innerHTML="";
  var manual=document.createElement("option");
  manual.value="manual";
  manual.textContent="Manual matchup / 手动选择球队";
  sel.appendChild(manual);
  for(var i=0;i<rows.length;i++){
    var m=rows[i];
    var opt=document.createElement("option");
    opt.value=String(i);
    opt.textContent=(m.date||"")+" · "+(m.round||"")+" · "+m.team1+" vs "+m.team2;
    sel.appendChild(opt);
  }
  if(rows.length>0)sel.value="0";
}
function applyScheduleMatch(){
  var sel=document.getElementById("h2h-match");
  if(!sel||sel.value==="manual")return;
  var rows=scheduleH2HMatches();
  var m=rows[parseInt(sel.value,10)];
  if(!m)return;
  _selectedScheduleMatch=m;
  var teamA=h2hTeamName(m.team1);
  var teamB=h2hTeamName(m.team2);
  document.getElementById("h2h-a").value=teamA;
  document.getElementById("h2h-b").value=teamB;
  updatePickCard("a",teamA);
  updatePickCard("b",teamB);
  updateSchedulePickCards(m);
  h2hChange();
  refreshLiveMatchPrediction(m);
}

function refreshLiveMatchPrediction(match){
  if(!match)return Promise.resolve(null);
  var key=scheduleMatchKey(match);
  var url;
  if(match.match_id)url="/api/live_match_prediction?match_id="+encodeURIComponent(match.match_id);
  else if(match.num!==undefined&&match.num!==null&&match.num!=="")url="/api/live_match_prediction?match_num="+encodeURIComponent(match.num);
  else url="/api/live_match_prediction?home="+encodeURIComponent(match.team1||"")+"&away="+encodeURIComponent(match.team2||"");
  return fetch(url,{cache:"no-store"}).then(function(r){return r.json();}).then(function(d){
    if(!d||d.error)return null;
    _liveSchedulePredictions[key]=d;
    var cur=selectedSchedulePrediction();
    if(cur&&scheduleMatchKey(cur)===key){
      updateSchedulePickCards(d);
      h2hChange();
    }
    return d;
  }).catch(function(e){console.warn("live match prediction refresh failed",e);return null;});
}

/* ── Fixtures ── */
var FL_ALIAS={"Bosnia & Herzegovina":"Bosnia and Herzegovina","Korea Republic":"South Korea","IR Iran":"Iran","Côte d'Ivoire":"Ivory Coast","DR of the Congo":"DR Congo"};
function flFx(c){if(!c)return"--";if(FL[c])return FL[c];var a=FL_ALIAS[c];if(a&&FL[a])return FL[a];return"--";}
function _isPlaceholderTeam(n){return /^[WL]\d+$/.test(n||"");}
function _normTeam(s){return (s||"").toLowerCase().replace(/&/g,"and").replace(/\s+/g," ").trim();}
function _todayStr(){var d=new Date();var y=d.getFullYear(),m=String(d.getMonth()+1).padStart(2,"0"),da=String(d.getDate()).padStart(2,"0");return y+"-"+m+"-"+da;}
function _addDays(s,n){var d=new Date(s+"T00:00:00");d.setDate(d.getDate()+n);var y=d.getFullYear(),mo=String(d.getMonth()+1).padStart(2,"0"),da=String(d.getDate()).padStart(2,"0");return y+"-"+mo+"-"+da;}
function _isLive(st){if(!st)return false;var s=st.toUpperCase();return s.indexOf("LIVE")>=0||s==="1H"||s==="2H"||s==="HT"||s==="ET"||s==="P"||s==="IN_PLAY"||s==="PAUSED";}
function _isFinished(st){if(!st)return false;var s=st.toUpperCase();return s.indexOf("FINISH")>=0||s==="FT"||s==="AET"||s==="PEN";}

var _liveScores=[];
var _fxView="today";

function _matchScore(m){
  if(!_liveScores||_liveScores.length===0)return null;
  var t1=_normTeam(m.team1),t2=_normTeam(m.team2);
  for(var i=0;i<_liveScores.length;i++){
    var s=_liveScores[i];if(s.date!==m.date)continue;
    var h=_normTeam(s.team_home),a=_normTeam(s.team_away);
    if((h===t1&&a===t2)||(h===t2&&a===t1))return s;
  }
  return null;
}

function renderMatch(m){
  var s=_matchScore(m);
  var live=s&&_isLive(s.status);
  var finished=s&&_isFinished(s.status);
  var sh=s?s.score_home:null,sa=s?s.score_away:null;
  var sc;
  if(sh!=null&&sa!=null){sc='<div class="fx-score'+(live?" live":"")+'">'+sh+' – '+sa+'</div>';}
  else{sc='<div class="fx-score ns">'+(m.time?m.time.substring(0,5):"-")+'</div>';}
  var fl1=_isPlaceholderTeam(m.team1)?"?":flFx(m.team1);
  var fl2=_isPlaceholderTeam(m.team2)?"?":flFx(m.team2);
  var rightLbl;
  if(live)rightLbl='<span class="fx-status live">'+(s.minute||s.status)+'</span>';
  else if(finished)rightLbl='<span class="fx-status">FT</span>';
  else rightLbl='<span>'+(m.time||"")+'</span>';
  return '<div class="fx-match'+(live?" live":"")+'">'+
    '<div class="fx-meta"><span class="fx-meta-grp">'+(m.group||m.round||"")+'</span><span>'+(m.ground||"")+'</span></div>'+
    '<div class="fx-row">'+
    '<div class="fx-team"><span class="fx-team-fl">'+fl1+'</span><span class="fx-team-nm">'+m.team1+'</span></div>'+
    sc+
    '<div class="fx-team away"><span class="fx-team-nm">'+m.team2+'</span><span class="fx-team-fl">'+fl2+'</span></div>'+
    '</div>'+
    '<div class="fx-ft">'+rightLbl+'<span>'+(m.round||"")+'</span></div>'+
    '</div>';
}

function renderDayGroup(date,matches,cls){
  if(!matches||matches.length===0)return"";
  var lbl=date+(cls==="today"?" · 今日":"");
  var h='<div class="fx-day-grp"><div class="fx-day-hd '+(cls||"")+'">'+lbl+'</div>';
  matches.forEach(function(m){h+=renderMatch(m);});
  return h+'</div>';
}

/* ── Standings (Plan B) ── */
function _resolveResult(m){
  // 返回 [g1,g2]（对齐 team1/team2 视角）或 null
  if(m.score&&m.score.ft&&m.score.ft.length===2&&m.score.ft[0]!=null&&m.score.ft[1]!=null){
    return [m.score.ft[0],m.score.ft[1]];
  }
  var s=_matchScore(m);
  if(s&&_isFinished(s.status)&&s.score_home!=null&&s.score_away!=null){
    var t1=_normTeam(m.team1),t2=_normTeam(m.team2);
    var h=_normTeam(s.team_home),a=_normTeam(s.team_away);
    if(h===t1&&a===t2)return [s.score_home,s.score_away];
    if(h===t2&&a===t1)return [s.score_away,s.score_home];
  }
  return null;
}
function _cmpStanding(a,b){
  if(b.Pts!==a.Pts)return b.Pts-a.Pts;
  if(b.GD!==a.GD)return b.GD-a.GD;
  if(b.GF!==a.GF)return b.GF-a.GF;
  return a.team<b.team?-1:1;
}
function computeStandings(){
  var matches=(F&&F.matches)||[];
  var groups={},order=[];
  matches.forEach(function(m){
    if(!m.group)return; // 仅小组赛
    var g=m.group;
    if(!groups[g]){groups[g]={};order.push(g);}
    [m.team1,m.team2].forEach(function(t){
      if(_isPlaceholderTeam(t))return;
      if(!groups[g][t])groups[g][t]={team:t,P:0,W:0,D:0,L:0,GF:0,GA:0,Pts:0};
    });
    var r=_resolveResult(m);
    if(r==null)return;
    var a=groups[g][m.team1],b=groups[g][m.team2];
    if(!a||!b)return;
    a.P++;b.P++;a.GF+=r[0];a.GA+=r[1];b.GF+=r[1];b.GA+=r[0];
    if(r[0]>r[1]){a.W++;b.L++;a.Pts+=3;}
    else if(r[0]<r[1]){b.W++;a.L++;b.Pts+=3;}
    else{a.D++;b.D++;a.Pts++;b.Pts++;}
  });
  order.sort();
  var out=[];
  order.forEach(function(g){
    var arr=Object.keys(groups[g]).map(function(k){var s=groups[g][k];s.GD=s.GF-s.GA;return s;});
    arr.sort(_cmpStanding);
    out.push({group:g,rows:arr});
  });
  return out;
}
function _bestThirds(standings){
  var thirds=[];
  standings.forEach(function(grp){if(grp.rows.length>=3)thirds.push({group:grp.group,s:grp.rows[2]});});
  thirds.sort(function(x,y){return _cmpStanding(x.s,y.s);});
  return thirds;
}
function renderStandings(){
  var standings=computeStandings();
  if(standings.length===0)return '<div class="fx-empty">暂无小组赛程数据</div>';
  var totalPlayed=0;
  standings.forEach(function(g){g.rows.forEach(function(r){totalPlayed+=r.P;});});
  totalPlayed=Math.floor(totalPlayed/2);
  var thirds=_bestThirds(standings);
  var bestSet={};for(var i=0;i<Math.min(8,thirds.length);i++){bestSet[thirds[i].s.team]=true;}
  var h="";
  h+='<div class="st-legend"><span><span class="st-dot" style="background:rgba(48,209,88,.6)"></span>前2出线</span><span><span class="st-dot" style="background:rgba(255,214,10,.6)"></span>第3名待定</span><span><span class="st-dot" style="background:rgba(255,69,58,.5)"></span>第4淘汰</span></div>';
  standings.forEach(function(grp){
    var played=0;grp.rows.forEach(function(r){played+=r.P;});played=Math.floor(played/2);
    h+='<div class="st-grp"><div class="st-hd"><span>'+grp.group+'</span><span class="st-prog">'+played+'/6 场</span></div>';
    h+='<div class="st-tbl"><div class="st-thr"><span class="st-pos">#</span><span>队</span><span class="st-n">场</span><span class="st-n">胜</span><span class="st-n">平</span><span class="st-n">负</span><span class="st-n">净</span><span class="st-n">分</span></div>';
    grp.rows.forEach(function(r,idx){
      var pos=idx+1,zone="z"+Math.min(4,pos);
      var q="";
      if(pos<=2)q='<span class="st-q" style="color:var(--gr)">✓</span>';
      else if(pos===3)q=bestSet[r.team]?'<span class="st-q" style="color:var(--gr)">✓③</span>':'<span class="st-q" style="color:var(--tx3)">③</span>';
      var gd=(r.GD>0?"+":"")+r.GD;
      h+='<div class="st-row '+zone+'"><span class="st-pos">'+pos+'</span><span class="st-tm"><span class="st-tm-fl">'+flFx(r.team)+'</span><span class="st-tm-nm">'+r.team+'</span>'+q+'</span><span class="st-n">'+r.P+'</span><span class="st-n">'+r.W+'</span><span class="st-n">'+r.D+'</span><span class="st-n">'+r.L+'</span><span class="st-n">'+gd+'</span><span class="st-pts">'+r.Pts+'</span></div>';
    });
    h+='</div></div>';
  });
  if(totalPlayed>0&&thirds.length>0){
    h+='<div class="st-grp"><div class="st-hd"><span style="color:var(--gd)">最佳第三名 / Best 3rd</span><span class="st-prog">前 8 出线</span></div><div class="st-tbl">';
    thirds.forEach(function(t,i){
      var inq=i<8;
      h+='<div class="st-third-row'+(inq?" in":"")+'"><span class="st-third-rk">'+(i+1)+'</span><span class="st-tm-fl">'+flFx(t.s.team)+'</span><span class="st-third-nm">'+t.s.team+' <span style="color:var(--tx2);font-size:10px">('+t.group+')</span></span><span class="st-third-st" style="color:'+(inq?"var(--gr)":"var(--tx3)")+'">'+(inq?"✓ 出线":"出局")+' · '+t.s.Pts+'分</span></div>';
    });
    h+='</div></div>';
  }else{
    h+='<div class="fx-empty" style="padding:18px">最佳第三名排名将在小组赛开始后显示</div>';
  }
  return h;
}

function buildFixtures(){
  var body=document.getElementById("fx-body");
  var updEl=document.getElementById("fx-upd");
  if(updEl)updEl.textContent="数据: openfootball + TheSportsDB · 进行中比赛每 2s 刷新";
  if(_fxView==="standings"){body.innerHTML=renderStandings();return;}
  var matches=(F&&F.matches)||[];
  var today=_todayStr();
  var html="";
  if(_fxView==="today"){
    var todayM=matches.filter(function(m){return m.date===today;});
    if(todayM.length>0){html=renderDayGroup(today,todayM,"today");}
    else{
      var future=matches.filter(function(m){return m.date>today;}).sort(function(a,b){return a.date<b.date?-1:1;});
      var nextDate=future.length>0?future[0].date:null;
      html='<div class="fx-empty">📅 今日无比赛'+(nextDate?'<br><br>下一比赛日<br><b style="color:var(--gd);font-size:15px">'+nextDate+'</b>':"")+'</div>';
      if(nextDate){var nextM=matches.filter(function(m){return m.date===nextDate;});html+=renderDayGroup(nextDate,nextM);}
    }
  }else if(_fxView==="upcoming"){
    var endDate=_addDays(today,7);
    var upcoming=matches.filter(function(m){return m.date>=today&&m.date<=endDate;});
    var byDate={};
    upcoming.forEach(function(m){if(!byDate[m.date])byDate[m.date]=[];byDate[m.date].push(m);});
    var dates=Object.keys(byDate).sort();
    if(dates.length===0)html='<div class="fx-empty">未来 7 天无比赛</div>';
    else dates.forEach(function(d){html+=renderDayGroup(d,byDate[d],d===today?"today":"");});
  }else{
    var byGroup={};var groupOrder=[];
    matches.forEach(function(m){
      var k=m.group||m.round||"Other";
      if(!byGroup[k]){byGroup[k]=[];groupOrder.push(k);}
      byGroup[k].push(m);
    });
    groupOrder.sort();
    groupOrder.forEach(function(k){
      byGroup[k].sort(function(a,b){return a.date<b.date?-1:1;});
      html+='<div class="fx-day-grp"><div class="fx-day-hd group">'+k+'</div>';
      byGroup[k].forEach(function(m){html+=renderMatch(m);});
      html+='</div>';
    });
  }
  body.innerHTML=html;
}

function switchFxView(v){_fxView=v;document.querySelectorAll(".fx-tab").forEach(function(t){t.classList.toggle("on",t.dataset.fxv===v);});buildFixtures();}

function fetchLiveScores(){
  return fetch("/api/live_scores",{cache:"no-store"}).then(function(r){return r.json();}).then(function(d){
    _liveScores=(d&&d.scores)||[];
    if(document.getElementById("pg-fixtures").classList.contains("on"))buildFixtures();
    var sched=scheduleMatchPrediction();
    if(sched)refreshLiveMatchPrediction(sched);
  }).catch(function(e){console.warn("live scores fetch failed",e);});
}

function refreshFixtures(){
  return fetch("/api/fixtures",{cache:"no-store"}).then(function(r){return r.json();}).then(function(d){
    if(d&&d.matches){F=d;if(document.getElementById("pg-fixtures").classList.contains("on"))buildFixtures();}
  }).catch(function(e){console.warn("fixtures refresh failed",e);});
}

function refreshSchedulePredictions(){
  return fetch("/api/schedule_predictions",{cache:"no-store"}).then(function(r){return r.json();}).then(function(d){
    if(!d||!d.matches)return;
    var sel=document.getElementById("h2h-match");
    var oldValue=sel?sel.value:"manual";
    SP=d;
    setUpdateTime();
    populateScheduleH2H();
    if(sel&&oldValue!=="manual"&&Number(oldValue)<sel.options.length){sel.value=oldValue;applyScheduleMatch();}
    else if(sel&&sel.value!=="manual"){applyScheduleMatch();}
    var sched=scheduleMatchPrediction();
    if(sched)refreshLiveMatchPrediction(sched);
    buildFinal();
  }).catch(function(e){console.warn("schedule predictions refresh failed",e);});
}

function refreshTeamAnalysis(){
  return fetch("/api/team_analysis",{cache:"no-store"}).then(function(r){return r.json();}).then(function(d){
    if(!d||!d.teams)return;
    D=d.teams;
    U=d.ucl||{};
    setUpdateTime();
    buildFinal();
    buildLB();
    buildFB();
    buildML();
    buildPoly();
    buildUCLInfo();
    h2hChange();
  }).catch(function(e){console.warn("team analysis refresh failed",e);});
}

/* 自适应轮询节奏：有 live 比赛 2s / 当日有未开赛 30s / 空闲 300s */
var _pollTimer=null;
function _nextPollDelay(){
  for(var i=0;i<_liveScores.length;i++){if(_isLive(_liveScores[i].status))return 2000;}
  var today=_todayStr();
  var matches=(F&&F.matches)||[];
  for(var j=0;j<matches.length;j++){if(matches[j].date===today)return 30000;}
  return 300000;
}
function pollLoop(){
  fetchLiveScores().then(function(){
    clearTimeout(_pollTimer);
    _pollTimer=setTimeout(pollLoop,_nextPollDelay());
  });
}

/* ── Init ── */
setUpdateTime("__UPDATE_TIME__");
setInterval(setUpdateTime,60000);
buildFinal();
buildLB();
buildFB();
buildML();
buildPoly();
buildUCLInfo();
// H2H: populate team selectors
var teams=D.slice().sort(function(a,b){return b.final_prob-a.final_prob;});
var selA=document.getElementById("h2h-a");
var selB=document.getElementById("h2h-b");
for(var i=0;i<teams.length;i++){
  var t=teams[i];
  var optA=document.createElement("option");optA.value=t.country;
  optA.textContent=fl(t.country)+" "+t.country+" "+(t.final_prob*100).toFixed(1)+"%";
  selA.appendChild(optA);
  var optB=document.createElement("option");optB.value=t.country;
  optB.textContent=fl(t.country)+" "+t.country+" "+(t.final_prob*100).toFixed(1)+"%";
  selB.appendChild(optB);
}
populateScheduleH2H();
if(document.getElementById("h2h-match")&&document.getElementById("h2h-match").value!=="manual"){
  applyScheduleMatch();
}else if(teams.length>1){
  selA.value=teams[0].country;
  selB.value=teams[1].country;
  updatePickCard("a",selA.value);
  updatePickCard("b",selB.value);
  h2hChange();
}
// Squad selector
var sel=document.getElementById("sq-sel");
for(var i=0;i<teams.length;i++){var opt=document.createElement("option");opt.value=teams[i].country;opt.textContent=fl(teams[i].country)+" "+teams[i].country+" "+(teams[i].final_prob*100).toFixed(1)+"%";sel.appendChild(opt);}
if(teams.length>0){sel.value=teams[0].country;sqChange();}
// Fixtures: 首屏渲染 + 自适应实时比分轮询 + 赛程定时刷新
buildFixtures();
pollLoop();
setInterval(refreshFixtures,300000);
refreshSchedulePredictions();
setInterval(refreshSchedulePredictions,60000);
setInterval(refreshTeamAnalysis,300000);
// 实时冠军调整：每 10 分钟拉一次 grok 实时层
refreshRealtime();
setInterval(refreshRealtime,600000);
</script>
</body>
</html>
'''


def _load_fixtures():
    """加载赛程缓存；不存在则自动拉取 openfootball"""
    cached = load_fixtures(FIXTURES_CACHE)
    if cached is None:
        print("📥 首次启动，拉取 openfootball 2026 赛程...")
        cached = fetch_and_save(FIXTURES_CACHE)
    if cached is None:
        return {"name": "World Cup 2026", "matches": []}
    return cached


def _load_final_pred():
    """加载数据驱动+市场校准的最终预测（若存在）"""
    if not os.path.exists(FINAL_PRED):
        return {"teams": [], "as_of": "", "market_weight": 0}
    with open(FINAL_PRED, encoding="utf-8") as f:
        return json.load(f)


def _load_schedule_predictions(path=SCHEDULE_PRED):
    """Load schedule-driven predictions generated by scripts/generate_schedule_predictions.py."""
    if not os.path.exists(path):
        return {"source": "fallback", "teams": [], "matches": [], "next_match": None}
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    payload.setdefault("source", "schedule")
    payload.setdefault("teams", [])
    payload.setdefault("matches", [])
    payload.setdefault("next_match", None)
    return payload


def _load_realtime():
    """加载 grok 实时调整的冠军概率（若存在）"""
    if not os.path.exists(REALTIME_CACHE):
        return None
    try:
        with open(REALTIME_CACHE, encoding="utf-8") as f:
            return json.load(f).get("champion")
    except Exception:
        return None


def _same_match(schedule_match, live_score):
    if not schedule_match or not live_score:
        return False
    teams = {schedule_match.get("team1"), schedule_match.get("team2")}
    live_teams = {live_score.get("team_home"), live_score.get("team_away")}
    return teams == live_teams


def _find_schedule_match(schedule_pred, match_num=None, home=None, away=None, match_id=None):
    matches = (schedule_pred or {}).get("matches") or []
    if match_id not in (None, ""):
        for match in matches:
            if str(match.get("match_id")) == str(match_id):
                return match
        return None
    if match_num not in (None, ""):
        for match in matches:
            if str(match.get("num")) == str(match_num):
                return match
        return None
    if home and away:
        wanted = {home, away}
        for match in matches:
            if {match.get("team1"), match.get("team2")} == wanted:
                return match
        return None
    return (schedule_pred or {}).get("next_match")


def _find_live_score(live_scores, schedule_match):
    for score in live_scores or []:
        if _same_match(schedule_match, score):
            return score
    return None


def _start_background_task(name, target, *args, **kwargs):
    """Run a best-effort background task without leaking thread tracebacks."""
    def run():
        try:
            target(*args, **kwargs)
        except Exception as e:
            print(f"⚠️ {name}: {e}")

    thread = threading.Thread(target=run, name=name, daemon=True)
    thread.start()
    return thread


def _refresh_analysis_state(state, loader=_load_analysis):
    global _cached_results
    _cached_results = None
    loaded = loader()
    if len(loaded) == 2:
        results, ucl_data = loaded
        h2h_conformal_map, match_fixtures, match_results, friendly_results = {}, [], [], []
    else:
        results, ucl_data, h2h_conformal_map, match_fixtures, match_results, friendly_results = loaded
    state["analysis"] = results
    state["ucl_data"] = ucl_data
    state["h2h_conformal_map"] = h2h_conformal_map
    state["match_fixtures"] = match_fixtures
    state["match_results"] = match_results
    state["friendly_results"] = friendly_results
    state["data_json"] = json.dumps(results, ensure_ascii=False)
    state["ucl_json"] = json.dumps(ucl_data, ensure_ascii=False)
    state["h2h_conf_json"] = json.dumps(h2h_conformal_map, ensure_ascii=False)
    state["match_fixtures_json"] = json.dumps(match_fixtures, ensure_ascii=False)
    state["wc_results_json"] = json.dumps(match_results, ensure_ascii=False)
    state["friendlies_json"] = json.dumps(friendly_results, ensure_ascii=False)
    state["analysis_updated_at"] = datetime.now().isoformat(timespec="seconds")
    return results, ucl_data, h2h_conformal_map, match_fixtures, match_results, friendly_results


def _start_analysis_refresh_daemon(state):
    """Refresh team/player analysis in memory; source JSON files remain untouched."""
    def loop():
        while True:
            try:
                _refresh_analysis_state(state)
            except Exception as e:
                print(f"⚠️ analysis refresh daemon: {e}")
            time.sleep(ANALYSIS_REFRESH_SECONDS)

    _start_background_task("analysis-refresh", loop)


def _start_realtime_daemon():
    """后台定期用 grok 刷新实时冠军调整（带 TTL，过期才真调）。grok 慢，独立线程不阻塞 HTTP。"""
    if not _RT_AVAILABLE:
        return

    def loop():
        while True:
            try:
                adjust_champion_probs()  # 内部 CHAMPION_TTL fresh 则跳过，不重复烧 grok
            except Exception as e:
                print(f"⚠️ realtime daemon: {e}")
            time.sleep(CHAMPION_TTL)

    _start_background_task("realtime-pred", loop)


def _start_schedule_prediction_daemon(state):
    """Periodically refresh schedule-driven predictions for API consumers."""
    def loop():
        while True:
            try:
                payload = generate_schedule_predictions(n_sim=SCHEDULE_PRED_SIMULATIONS)
                state["schedule_pred"] = payload
                state["schedule_json"] = json.dumps(payload, ensure_ascii=False)
            except Exception as e:
                print(f"⚠️ schedule prediction daemon: {e}")
            time.sleep(SCHEDULE_PRED_REFRESH_SECONDS)

    _start_background_task("schedule-pred", loop)


def run_server(port=7862):
    """启动 HTTP 服务器 — 纯 HTML/CSS/JS，无 Gradio 依赖"""
    results, ucl_data, h2h_conformal_map, match_fixtures, match_results, friendly_results = _load_analysis()
    fixtures = _load_fixtures()
    final_pred = _load_final_pred()
    schedule_pred = _load_schedule_predictions()
    realtime = _load_realtime()
    update_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    data_json = json.dumps(results, ensure_ascii=False)
    ucl_json = json.dumps(ucl_data, ensure_ascii=False)
    fixtures_json = json.dumps(fixtures, ensure_ascii=False)
    final_json = json.dumps(final_pred, ensure_ascii=False)
    schedule_json = json.dumps(schedule_pred, ensure_ascii=False)
    realtime_json = json.dumps(realtime, ensure_ascii=False)
    h2h_conf_json = json.dumps(h2h_conformal_map, ensure_ascii=False)
    match_fixtures_json = json.dumps(match_fixtures, ensure_ascii=False)
    wc_results_json = json.dumps(match_results, ensure_ascii=False)
    friendlies_json = json.dumps(friendly_results, ensure_ascii=False)

    html = HTML_BODY
    html = html.replace("__DATA__", data_json)
    html = html.replace("__UCL__", ucl_json)
    html = html.replace("__FIXTURES__", fixtures_json)
    html = html.replace("__FINAL__", final_json)
    html = html.replace("__SCHEDULE_PRED__", schedule_json)
    html = html.replace("__REALTIME__", realtime_json)
    html = html.replace("__H2H_CONF__", h2h_conf_json)
    html = html.replace("__MATCH_FIXTURES__", match_fixtures_json)
    html = html.replace("__WC_RESULTS__", wc_results_json)
    html = html.replace("__FRIENDLIES__", friendlies_json)
    html = html.replace("__UPDATE_TIME__", update_time)

    live_provider = LiveScoresProvider()
    _start_realtime_daemon()

    # daemon 刷新赛程、预测和基础分析时，原子替换内存中的最新 JSON。
    state = {
        "analysis": results,
        "ucl_data": ucl_data,
        "h2h_conformal_map": h2h_conformal_map,
        "match_fixtures": match_fixtures,
        "match_results": match_results,
        "friendly_results": friendly_results,
        "data_json": data_json,
        "ucl_json": ucl_json,
        "h2h_conf_json": h2h_conf_json,
        "match_fixtures_json": match_fixtures_json,
        "wc_results_json": wc_results_json,
        "friendlies_json": friendlies_json,
        "analysis_updated_at": datetime.now().isoformat(timespec="seconds"),
        "fixtures_json": fixtures_json,
        "schedule_pred": schedule_pred,
        "schedule_json": schedule_json,
    }

    def _on_fixtures_update(payload):
        state["fixtures_json"] = json.dumps(payload, ensure_ascii=False)

    start_refresh_daemon(FIXTURES_CACHE, on_update=_on_fixtures_update)
    _start_analysis_refresh_daemon(state)
    _start_schedule_prediction_daemon(state)

    class Handler(http.server.BaseHTTPRequestHandler):
        def _send_json(self, body: bytes, status: int = 200):
            self.send_response(status)
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            # 路由: /api/live_scores → 实时比分 JSON
            if self.path.startswith("/api/live_scores"):
                try:
                    scores = live_provider.get_today_scores()
                    self._send_json(json.dumps(
                        {"scores": scores, "fetched_at": datetime.now().isoformat(timespec="seconds")},
                        ensure_ascii=False,
                    ).encode("utf-8"))
                except Exception as e:
                    self._send_json(json.dumps({"error": str(e), "scores": []}).encode("utf-8"), 500)
                return

            # 路由: /api/team_analysis → 最新球队/球员基础分析 JSON
            if self.path.startswith("/api/team_analysis"):
                self._send_json(json.dumps({
                    "teams": state.get("analysis", []),
                    "ucl": state.get("ucl_data", {}),
                    "fetched_at": state.get("analysis_updated_at"),
                    "refresh_seconds": ANALYSIS_REFRESH_SECONDS,
                }, ensure_ascii=False).encode("utf-8"))
                return

            # 路由: /api/fixtures → 最新赛程（含 daemon 回填的 score.ft）
            if self.path.startswith("/api/fixtures"):
                self._send_json(state["fixtures_json"].encode("utf-8"))
                return

            # 路由: /api/schedule_predictions → 最新赛程驱动预测
            if self.path.startswith("/api/schedule_predictions"):
                self._send_json(state["schedule_json"].encode("utf-8"))
                return

            # 路由: /api/live_match_prediction?match_num=N → live-state 单场概率
            if self.path.startswith("/api/live_match_prediction"):
                try:
                    from urllib.parse import urlparse, parse_qs, unquote
                    q = parse_qs(urlparse(self.path).query)
                    match_id = q.get("match_id", [""])[0]
                    match_num = q.get("match_num", [""])[0]
                    home = unquote(q.get("home", [""])[0])
                    away = unquote(q.get("away", [""])[0])
                    schedule_match = _find_schedule_match(state.get("schedule_pred"), match_num, home, away, match_id)
                    if not schedule_match:
                        self._send_json(b'{"error":"match not found"}', 404)
                        return
                    scores = live_provider.get_scores(schedule_match.get("date"))
                    live_score = _find_live_score(scores, schedule_match)
                    pred = build_live_match_prediction(schedule_match, live_score)
                    pred["live_score"] = live_score
                    pred["fetched_at"] = datetime.now().isoformat(timespec="seconds")
                    self._send_json(json.dumps(pred, ensure_ascii=False).encode("utf-8"))
                except Exception as e:
                    self._send_json(json.dumps({"error": str(e)}).encode("utf-8"), 500)
                return

            # 路由: /api/realtime → grok 实时调整的冠军概率（前端定期刷新体现动态）
            if self.path.startswith("/api/realtime"):
                rt = _load_realtime()
                if _RT_AVAILABLE and not _fresh(rt, CHAMPION_TTL):
                    _start_background_task("realtime-refresh", adjust_champion_probs)
                self._send_json(json.dumps(rt, ensure_ascii=False).encode("utf-8"))
                return

            # 路由: /api/match_pred?home=X&away=Y → grok 单场实时预测（带缓存）
            if self.path.startswith("/api/match_pred"):
                try:
                    from urllib.parse import urlparse, parse_qs, unquote
                    q = parse_qs(urlparse(self.path).query)
                    home = unquote(q.get("home", [""])[0])
                    away = unquote(q.get("away", [""])[0])
                    if not _RT_AVAILABLE or not home or not away:
                        self._send_json(b'{"error":"unavailable"}', 503)
                        return
                    # 仅返回缓存；未命中则后台异步生成，前端稍后再取（不阻塞）
                    key = f"{home}|{away}"
                    cached = _load_cache().get("matches", {}).get(key)
                    if _fresh(cached, MATCH_TTL):
                        self._send_json(json.dumps(cached, ensure_ascii=False).encode("utf-8"))
                    else:
                        _start_background_task("match-pred-refresh", predict_match, home, away)
                        self._send_json(b'{"status":"analyzing"}', 202)
                except Exception as e:
                    self._send_json(json.dumps({"error": str(e)}).encode("utf-8"), 500)
                return

            # 默认: 返回 HTML
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))

        def log_message(self, fmt, *args):
            pass

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"Mobile UI: http://localhost:{port}")
        print(f"Champion | Factor | Mystic | H2H | Squad | Poly | Fixtures | Info")
        httpd.serve_forever()

def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the mobile World Cup dashboard.")
    parser.add_argument("--port", type=int, default=7862, help="HTTP port to listen on.")
    args = parser.parse_args(argv)
    run_server(port=args.port)


if __name__ == "__main__":
    main()
