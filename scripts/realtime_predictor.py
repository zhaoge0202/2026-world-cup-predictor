"""
实时因素预测层 — grok-4.3-high 联网综合实时信息，动态调整预测

三层架构：
  数据模型(Elo+MC) → 市场校准(博彩/预测市场) → 【本层】grok 实时动态调整
  grok 联网获取：关键球员伤病/停赛、近期状态、首发预期、临场赔率变动、士气

设计原则：
  - grok 只做它擅长的（综合实时信息判断利好/利空），不做精确概率计算
  - 以 final（市场校准）概率为基线，grok 给"实时增量调整"，克制为主
  - 文件缓存 + TTL，绝不让前端同步等待 grok（~30s 延迟）
"""

import os
import json
import time
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_client import chat_json, is_configured

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL_PRED = os.path.join(ROOT, "data", "wc2026_prediction_final.json")
CACHE = os.path.join(ROOT, "data", "realtime_cache.json")

CHAMPION_TTL = 6 * 3600    # 冠军概率实时调整：6 小时
MATCH_TTL = 3 * 3600       # 单场预测：3 小时


def _load_cache():
    if os.path.exists(CACHE):
        try:
            return json.load(open(CACHE, encoding="utf-8"))
        except Exception:
            pass
    return {"champion": None, "matches": {}}


def _save_cache(c):
    json.dump(c, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def _fresh(entry, ttl):
    if not entry or "updated_ts" not in entry:
        return False
    return (time.time() - entry["updated_ts"]) < ttl


def adjust_champion_probs(top_n=14, force=False):
    """grok 联网综合实时因素，调整冠军概率。返回 {teams, summary, updated}。"""
    cache = _load_cache()
    if not force and _fresh(cache.get("champion"), CHAMPION_TTL):
        return cache["champion"]
    if not is_configured():
        return cache.get("champion")

    base = json.load(open(FINAL_PRED, encoding="utf-8"))["teams"]
    top = base[:top_n]
    lines = "\n".join(f"  {t['country']}: {t['champion']*100:.1f}%" for t in top)
    prompt = (
        "你是顶级足球数据分析师。以下是基于历史数据模型+博彩市场共识的 2026 世界杯夺冠基线概率"
        f"（截至今天 {datetime.now():%Y-%m-%d}）：\n{lines}\n\n"
        "请联网获取这些球队【最新的实时因素】：核心球员伤病/停赛、近期状态与热身赛结果、"
        "预期首发强弱、临场赔率变动、球队士气。基于这些实时信息，给出每队【调整后的夺冠概率】。\n"
        "要求：1) 基线已含市场信息，调整要克制，只在有明确实时利好/利空时才显著调整；"
        "2) 概率之和不必精确为100；3) 每队列出最多2条最关键的实时因素（简体中文，具体）。\n"
        '返回 JSON：{"teams":[{"country":"英文名","adjusted":数字百分比,"factors":["因子1","因子2"]}],'
        '"summary":"一句话总体实时研判"}'
    )
    j = chat_json([{"role": "user", "content": prompt}], max_tokens=3500, timeout=200, temperature=0.2)

    adj = {}
    for t in j.get("teams", []):
        c = t.get("country") or t.get("team") or t.get("name")
        if c:
            adj[c] = t
    base_top_sum = sum(t["champion"] for t in top)
    raw = {}
    for t in top:
        a = adj.get(t["country"], {}).get("adjusted")
        raw[t["country"]] = (a / 100.0) if isinstance(a, (int, float)) else t["champion"]
    rsum = sum(raw.values()) or 1.0
    # top 内按 grok 调整重分配 base_top_sum；长尾保持 base
    out_teams = []
    for t in top:
        newp = raw[t["country"]] / rsum * base_top_sum
        out_teams.append({
            "country": t["country"], "champion": newp,
            "base": t["champion"], "delta": newp - t["champion"],
            "factors": adj.get(t["country"], {}).get("factors", []),
        })
    for t in base[top_n:]:
        out_teams.append({"country": t["country"], "champion": t["champion"],
                          "base": t["champion"], "delta": 0.0, "factors": []})
    tot = sum(t["champion"] for t in out_teams) or 1.0
    for t in out_teams:
        t["champion"] /= tot
    out_teams.sort(key=lambda x: -x["champion"])

    result = {
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "updated_ts": time.time(),
        "summary": j.get("summary", ""),
        "teams": out_teams,
    }
    cache["champion"] = result
    _save_cache(cache)
    return result


def predict_match(home, away, elo_home=None, elo_away=None, base_wdl=None, force=False):
    """grok 联网对单场综合实时因素，输出胜平负+最可能比分+因子。"""
    key = f"{home}|{away}"
    cache = _load_cache()
    if not force and _fresh(cache.get("matches", {}).get(key), MATCH_TTL):
        return cache["matches"][key]
    if not is_configured():
        return cache.get("matches", {}).get(key)

    base_txt = ""
    if base_wdl:
        base_txt = (f"\n数据模型基线（{home} 视角）：胜 {base_wdl[0]*100:.0f}% / "
                    f"平 {base_wdl[1]*100:.0f}% / 负 {base_wdl[2]*100:.0f}%。")
    elo_txt = f"\n双方 Elo：{home}={elo_home:.0f}, {away}={elo_away:.0f}。" if elo_home else ""
    prompt = (
        f"2026 世界杯比赛预测：{home} (主) vs {away} (客)。{elo_txt}{base_txt}\n"
        f"请联网获取截至今天（{datetime.now():%Y-%m-%d}）双方最新实时因素："
        "伤病/停赛、首发预期、近期状态、交锋历史、临场赔率。综合实时信息给出预测。\n"
        '返回 JSON：{"home_win":0~1,"draw":0~1,"away_win":0~1,'
        '"likely_scores":[{"score":"2-1","prob":0~1},...最多4个],'
        '"key_factors":["最多3条关键实时因素，简体中文"],"summary":"一句话研判"}'
        "（胜平负概率之和应≈1）"
    )
    j = chat_json([{"role": "user", "content": prompt}], max_tokens=1500, timeout=150, temperature=0.2)
    s = (j.get("home_win", 0) + j.get("draw", 0) + j.get("away_win", 0)) or 1.0
    result = {
        "home": home, "away": away,
        "home_win": j.get("home_win", 0) / s, "draw": j.get("draw", 0) / s,
        "away_win": j.get("away_win", 0) / s,
        "likely_scores": j.get("likely_scores", []),
        "key_factors": j.get("key_factors", []),
        "summary": j.get("summary", ""),
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "updated_ts": time.time(),
    }
    cache.setdefault("matches", {})[key] = result
    _save_cache(cache)
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion", action="store_true")
    ap.add_argument("--match", nargs=2, metavar=("HOME", "AWAY"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.champion:
        r = adjust_champion_probs(force=args.force)
        print(f"\n实时研判: {r['summary']}\n更新: {r['updated']}\n")
        print(f"{'球队':<16}{'实时':>8}{'基线':>8}{'变动':>8}  关键因子")
        for t in r["teams"][:14]:
            f = "；".join(t["factors"][:2])
            print(f"{t['country']:<16}{t['champion']*100:>7.1f}%{t['base']*100:>7.1f}%"
                  f"{t['delta']*100:>+7.1f}  {f}")
    elif args.match:
        r = predict_match(args.match[0], args.match[1], force=args.force)
        print(f"\n{r['home']} vs {r['away']}  (更新 {r['updated']})")
        print(f"胜 {r['home_win']*100:.0f}% / 平 {r['draw']*100:.0f}% / 负 {r['away_win']*100:.0f}%")
        print("可能比分:", r["likely_scores"])
        print("关键因子:", r["key_factors"])
        print("研判:", r["summary"])
    else:
        print("用法: --champion | --match HOME AWAY [--force]")
