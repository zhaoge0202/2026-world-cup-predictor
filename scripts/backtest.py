"""
模型回测体检 — 用 openfootball 历史世界杯逐场比分评估引擎预测力

目的：在改任何参数之前，先量化"现引擎到底准不准"，建立基准标尺。
方法：
  1. 拉取 1930-2022 全部历史世界杯逐场比分
  2. 按时间顺序跑一套自洽的"在线 Elo"（标准 Elo + 进球差加权，仅用世界杯比赛训练）
  3. 每场比赛在"赛前 Elo"状态下做预测，与实际结果比对
  4. 评估三个东西，各与基准对比：
       - Poisson 比分模型（复刻 team_scoring._sim_group_match 的 λ 公式）
       - Elo 胜平负预测
       - 与基准：纯 Elo+固定平局 / 历史边际频率（zero-rule）

不修改任何模型，只复用其预测公式。评估只统计 EVAL_FROM 之后的比赛（之前当 Elo burn-in）。

用法：
    python scripts/backtest.py
"""

import requests
import math
from collections import defaultdict

HEADERS = {"User-Agent": "WorldCupPredictorBot/1.0 (backtest)"}
BASE = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/{y}/worldcup.json"

# 全部历史世界杯用于训练 Elo；评估只从 EVAL_FROM 起（之前为 burn-in）
ALL_YEARS = [1930, 1934, 1938, 1950, 1954, 1958, 1962, 1966, 1970, 1974,
             1978, 1982, 1986, 1990, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022]
EVAL_FROM = 1998

# ── Elo 参数（eloratings.net 世界杯惯例）──────────────────────────
ELO_INIT = 1500.0
ELO_K = 60.0  # 世界杯决赛圈基础 K


def _elo_expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def _gd_multiplier(gd: int) -> float:
    """进球差对 K 的放大（eloratings 风格）"""
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return 1.75 + (gd - 3) / 8.0


# ── Poisson 比分模型（复刻 team_scoring._sim_group_match 的 λ 公式）──
def _lambdas(ea: float, eb: float):
    d = (ea - eb) / 400.0
    la = max(0.15, 1.35 + d * 0.85)
    lb = max(0.15, 1.35 - d * 0.85)
    return la, lb


def _pois_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * lam ** k / math.factorial(k)


def _wdl_from_lambdas(la: float, lb: float, maxg: int = 12):
    """由两队 λ 解析算 (P胜, P平, P负)，team1 视角"""
    pw = pd = pl = 0.0
    pa = [_pois_pmf(i, la) for i in range(maxg + 1)]
    pb = [_pois_pmf(j, lb) for j in range(maxg + 1)]
    for i in range(maxg + 1):
        for j in range(maxg + 1):
            p = pa[i] * pb[j]
            if i > j:
                pw += p
            elif i == j:
                pd += p
            else:
                pl += p
    s = pw + pd + pl
    return pw / s, pd / s, pl / s


def _exact_score_logprob(la: float, lb: float, g1: int, g2: int, maxg: int = 12) -> float:
    """实际比分 (g1,g2) 在 Poisson 模型下的 log 概率（截断+归一化）"""
    g1c, g2c = min(g1, maxg), min(g2, maxg)
    pa = [_pois_pmf(i, la) for i in range(maxg + 1)]
    pb = [_pois_pmf(j, lb) for j in range(maxg + 1)]
    total = sum(pa) * sum(pb)
    p = (pa[g1c] * pb[g2c]) / total
    return math.log(max(p, 1e-12))


# ── 数据加载 ──────────────────────────────────────────────
def load_matches():
    matches = []
    for y in ALL_YEARS:
        try:
            d = requests.get(BASE.format(y=y), headers=HEADERS, timeout=20).json()
        except Exception as e:
            print(f"⚠️ {y} 拉取失败: {e}")
            continue
        for m in d.get("matches", []):
            ft = (m.get("score") or {}).get("ft")
            if not ft or len(ft) != 2 or ft[0] is None or ft[1] is None:
                continue
            matches.append({
                "year": y,
                "date": m.get("date") or "",
                "t1": m["team1"], "t2": m["team2"],
                "g1": int(ft[0]), "g2": int(ft[1]),
            })
    matches.sort(key=lambda m: (m["year"], m["date"]))
    return matches


# ── 评估累加器 ────────────────────────────────────────────
class Metric:
    def __init__(self, name):
        self.name = name
        self.ll = 0.0      # 多分类 log loss 累加
        self.brier = 0.0   # 多分类 Brier 累加
        self.correct = 0   # argmax 命中
        self.n = 0

    def add(self, probs, outcome):
        # probs=(pw,pd,pl), outcome∈{0:胜,1:平,2:负}
        self.n += 1
        self.ll += -math.log(max(probs[outcome], 1e-12))
        y = [0, 0, 0]; y[outcome] = 1
        self.brier += sum((probs[k] - y[k]) ** 2 for k in range(3))
        if max(range(3), key=lambda k: probs[k]) == outcome:
            self.correct += 1

    def row(self):
        n = max(1, self.n)
        return (self.name, self.ll / n, self.brier / n, self.correct / n * 100, self.n)


def run():
    print("📥 拉取历史世界杯比分（1930-2022）...")
    matches = load_matches()
    print(f"   共 {len(matches)} 场带比分；评估区 {EVAL_FROM}-2022\n")

    elo = defaultdict(lambda: ELO_INIT)

    m_poisson = Metric("Poisson比分模型(现引擎)")
    m_elo = Metric("纯Elo+固定平局(基准A)")
    m_base = Metric("历史边际频率(基准B/zero)")

    # 先求历史 W/D/L 边际频率（用全样本，作为最弱基准的常数预测）
    cw = cd = cl = 0
    for m in matches:
        if m["g1"] > m["g2"]: cw += 1
        elif m["g1"] == m["g2"]: cd += 1
        else: cl += 1
    tot = max(1, cw + cd + cl)
    base_prior = (cw / tot, cd / tot, cl / tot)
    DRAW_FIXED = cd / tot  # 基准A 用历史平局率当固定平局概率

    score_ll = 0.0   # Poisson 比分 log loss
    score_n = 0

    for m in matches:
        ea, eb = elo[m["t1"]], elo[m["t2"]]
        # 赛前预测
        la, lb = _lambdas(ea, eb)
        # 实际结果
        if m["g1"] > m["g2"]: outcome = 0
        elif m["g1"] == m["g2"]: outcome = 1
        else: outcome = 2

        if m["year"] >= EVAL_FROM:
            # 现引擎 Poisson 比分模型 → W/D/L
            m_poisson.add(_wdl_from_lambdas(la, lb), outcome)
            # 基准A：纯 Elo 胜率 + 固定平局
            e = _elo_expected(ea, eb)
            pa = (1 - DRAW_FIXED) * e
            pc = (1 - DRAW_FIXED) * (1 - e)
            m_elo.add((pa, DRAW_FIXED, pc), outcome)
            # 基准B：历史边际频率常数
            m_base.add(base_prior, outcome)
            # Poisson 比分 log loss
            score_ll += -_exact_score_logprob(la, lb, m["g1"], m["g2"])
            score_n += 1

        # 在线 Elo 更新（无论是否评估都更新，训练信号）
        sa = 1.0 if outcome == 0 else (0.5 if outcome == 1 else 0.0)
        k = ELO_K * _gd_multiplier(abs(m["g1"] - m["g2"]))
        e_a = _elo_expected(ea, eb)
        elo[m["t1"]] = ea + k * (sa - e_a)
        elo[m["t2"]] = eb + k * ((1 - sa) - (1 - e_a))

    # ── 报告 ──────────────────────────────────────────────
    print("=" * 68)
    print("胜平负预测（越低越好: LogLoss/Brier；越高越好: Acc）")
    print("-" * 68)
    print(f"{'模型':<28}{'LogLoss':>9}{'Brier':>8}{'Acc%':>8}{'N':>6}")
    for met in (m_poisson, m_elo, m_base):
        name, ll, br, acc, n = met.row()
        print(f"{name:<28}{ll:>9.4f}{br:>8.4f}{acc:>7.1f}{n:>7}")
    print("-" * 68)
    # 参照：完全随机三分类 log loss = ln(3) ≈ 1.0986
    print(f"{'(参照) 完全随机 1/3':<28}{math.log(3):>9.4f}{'0.6667':>8}{33.3:>7.1f}")
    print()
    print(f"Poisson 比分模型 — 实际比分 LogLoss: {score_ll/max(1,score_n):.4f}  (N={score_n})")
    print("=" * 68)

    # Elo 收敛后的 Top 队（截至 2022 末）
    top = sorted(elo.items(), key=lambda kv: -kv[1])[:12]
    print("\n训练后 Elo Top 12（截至 2022，仅世界杯比赛训练）:")
    for i, (t, r) in enumerate(top, 1):
        print(f"  {i:>2}. {t:<16} {r:.0f}")

    return {"poisson": m_poisson.row(), "elo": m_elo.row(),
            "base": m_base.row(), "score_ll": score_ll / max(1, score_n)}


if __name__ == "__main__":
    run()
