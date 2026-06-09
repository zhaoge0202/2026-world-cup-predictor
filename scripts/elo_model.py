"""
数据驱动的国家队 Elo 模型 — 精准预测重建的地基

数据：data/intl_results.csv（martj42，49445 场国际比赛 1872-2026）
方法：World Football Elo（eloratings.net 风格）
  - 主场优势（neutral 字段区分中立场）
  - 赛事重要性分 K（世界杯/大洲赛/预选/友谊）
  - 进球差乘数 G

本模块被 backtest / 拟合 / 2026 预测复用。超参集中在 EloParams，便于网格搜索。
"""

import csv
import math
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional

DATA = "data/intl_results.csv"

# 项目 2026 队名 → 数据集队名（只有 USA 不一致）
TEAM_MAP = {"USA": "United States"}


@dataclass
class EloParams:
    home_adv: float = 100.0      # 主场优势（Elo 点）
    k_wc_finals: float = 60.0    # 世界杯决赛圈
    k_continental: float = 50.0  # 大洲杯决赛圈/洲际正赛
    k_quali: float = 40.0        # 各类预选赛
    k_friendly: float = 20.0     # 友谊赛
    k_other: float = 30.0        # 其它（Nations League/Confed 等）
    # ── 增强项（默认关闭，回测验证后再开）──
    mean_revert: float = 1.0     # 按年均值回归基数 φ：跨年时 elo=1500+φ^Δ年·(elo-1500)。
                                 # <1 启用，抑制弱区互刷的虚高分 + 久疏战不确定性回归。1.0=关闭
    k_new_boost: float = 0.0     # 新队自适应K：经验少时 K 放大，K_eff=K·(1+boost·exp(-games/k_tau))。0=关闭
    k_tau: float = 40.0          # 自适应K 的出场数衰减尺度


def classify_k(tournament: str, p: EloParams) -> float:
    t = tournament or ""
    if "qualification" in t:
        return p.k_quali
    if t == "Friendly":
        return p.k_friendly
    if t == "FIFA World Cup":
        return p.k_wc_finals
    # 大洲决赛圈正赛
    cont = ("Copa América", "African Cup of Nations", "UEFA Euro",
            "AFC Asian Cup", "Gold Cup", "CONCACAF Championship",
            "Oceania Nations Cup", "Confederations Cup")
    if any(c in t for c in cont):
        return p.k_continental
    return p.k_other


def _gd_mult(gd: int) -> float:
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    if gd == 3:
        return 1.75
    return 1.75 + (gd - 3) / 8.0


def load_matches(path: str = DATA, since_year: Optional[int] = None,
                 until_date: Optional[str] = None):
    out = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            hs, as_ = r["home_score"], r["away_score"]
            if hs == "" or as_ == "":
                continue
            try:
                hs, as_ = int(hs), int(as_)
            except ValueError:
                continue
            d = r["date"]
            if since_year and d[:4] < str(since_year):
                continue
            if until_date and d > until_date:
                continue
            out.append({
                "date": d,
                "home": r["home_team"], "away": r["away_team"],
                "hs": hs, "as": as_,
                "tour": r["tournament"],
                "neutral": r["neutral"].strip().upper() == "TRUE",
            })
    out.sort(key=lambda m: m["date"])
    return out


def expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def train_elo(matches, p: EloParams):
    """遍历全部比赛训练 Elo，返回最终 elo dict"""
    elo = defaultdict(lambda: 1500.0)
    for m in matches:
        ra, rb = elo[m["home"]], elo[m["away"]]
        ha = 0.0 if m["neutral"] else p.home_adv
        we = expected(ra + ha, rb)
        if m["hs"] > m["as"]:
            w = 1.0
        elif m["hs"] == m["as"]:
            w = 0.5
        else:
            w = 0.0
        k = classify_k(m["tour"], p) * _gd_mult(abs(m["hs"] - m["as"]))
        delta = k * (w - we)
        elo[m["home"]] = ra + delta
        elo[m["away"]] = rb - delta
    return elo


if __name__ == "__main__":
    matches = load_matches(until_date="2026-06-08")
    print(f"训练样本: {len(matches)} 场（截至 2026-06-08）")
    p = EloParams()
    elo = train_elo(matches, p)

    print("\n全球 Elo Top 20（截至 2026-06-08）:")
    top = sorted(elo.items(), key=lambda kv: -kv[1])[:20]
    for i, (t, r) in enumerate(top, 1):
        print(f"  {i:>2}. {t:<22} {r:.0f}")

    WC2026 = ["Argentina","Brazil","Uruguay","Colombia","Ecuador","Paraguay","France","Germany",
        "Spain","England","Portugal","Netherlands","Belgium","Croatia","Switzerland","Austria",
        "Czech Republic","Turkey","Sweden","Morocco","Senegal","Algeria","Egypt","Ghana",
        "Ivory Coast","Tunisia","DR Congo","Cape Verde","Japan","South Korea","Iran","Iraq",
        "Qatar","Saudi Arabia","Australia","Uzbekistan","Jordan","USA","Mexico","Canada",
        "Panama","Curaçao","Haiti","New Zealand","Norway","South Africa","Bosnia and Herzegovina","Scotland"]
    print("\n2026 参赛队 Elo（数据驱动）:")
    rows = []
    for t in WC2026:
        name = TEAM_MAP.get(t, t)
        rows.append((t, elo.get(name, 1500.0)))
    rows.sort(key=lambda x: -x[1])
    for i, (t, r) in enumerate(rows, 1):
        print(f"  {i:>2}. {t:<22} {r:.0f}")
