"""
数据驱动模型 — 拟合 + 回测对比（核心证据脚本）

流程（无未来泄露）：
  1. walk-forward 遍历全部国际比赛，记录每场"赛前 Elo 差"+ 实际进球
  2. 用 date < FIT_END 的数据，泊松回归（numpy IRLS）拟合 λ = exp(β0 + β1·Δelo/400)
  3. 在 held-out 测试集（2018+2022 世界杯决赛圈）上对比：
       M1 新Elo + 拟合λ Poisson         （完整新模型）
       M2 新Elo + 手工λ(1.35±0.85)       （隔离"λ拟合"贡献）
       M3 新Elo + 纯Elo胜率 + 固定平局    （隔离"Poisson"贡献）
       M4 仅世界杯训练的劣质Elo + 手工λ   （隔离"数据扩充"贡献）
       M5 zero-rule 历史频率              （最弱基准）
  指标：胜平负 LogLoss / Brier / Acc；以及实际比分 LogLoss
"""

import math
import numpy as np
from collections import defaultdict

from elo_model import load_matches, EloParams, classify_k, _gd_mult, expected

FIT_END = "2018-01-01"
TEST_YEARS = {"2018", "2022"}
MAXG = 10
_LOGFACT = np.array([math.lgamma(k + 1) for k in range(MAXG + 1)])
_KS = np.arange(MAXG + 1)


def _pmf(lam: float) -> np.ndarray:
    lam = max(lam, 1e-6)
    return np.exp(-lam + _KS * math.log(lam) - _LOGFACT)


def walk_forward(matches, p: EloParams, wc_only=False):
    elo = defaultdict(lambda: 1500.0)
    recs = []
    for m in matches:
        ra, rb = elo[m["home"]], elo[m["away"]]
        ha = 0.0 if m["neutral"] else p.home_adv
        recs.append({
            "date": m["date"], "year": m["date"][:4],
            "is_wcf": m["tour"] == "FIFA World Cup",
            "diff": ra + ha - rb, "hs": m["hs"], "as": m["as"],
        })
        if wc_only and m["tour"] != "FIFA World Cup":
            continue
        we = expected(ra + ha, rb)
        w = 1.0 if m["hs"] > m["as"] else (0.5 if m["hs"] == m["as"] else 0.0)
        k = classify_k(m["tour"], p) * _gd_mult(abs(m["hs"] - m["as"]))
        d = k * (w - we)
        elo[m["home"]] = ra + d
        elo[m["away"]] = rb - d
    return recs, elo


def fit_lambda(recs, fit_end=FIT_END):
    """泊松回归 log λ = β0 + β1·x，numpy IRLS。每场拆 home/away 两行（对称）
    fit_end: 只用此日期前的数据（评估用 FIT_END held-out；生产用全部数据传更晚日期）"""
    xs, ys = [], []
    for r in recs:
        if r["date"] >= fit_end:
            continue
        x = r["diff"] / 400.0
        xs += [x, -x]
        ys += [r["hs"], r["as"]]
    x = np.array(xs)
    y = np.array(ys, dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(100):
        mu = np.exp(np.clip(X @ beta, -8, 8))
        W = mu
        XtWX = X.T @ (W[:, None] * X)
        z = (X @ beta) + (y - mu) / mu
        beta_new = np.linalg.solve(XtWX, X.T @ (W * z))
        if np.max(np.abs(beta_new - beta)) < 1e-9:
            beta = beta_new
            break
        beta = beta_new
    return float(beta[0]), float(beta[1])


def _wdl(la, lb):
    pa, pb = _pmf(la), _pmf(lb)
    M = np.outer(pa, pb)
    pw = np.tril(M, -1).sum()
    pd = np.trace(M)
    pl = np.triu(M, 1).sum()
    s = pw + pd + pl
    return pw / s, pd / s, pl / s


def pred_fitted(diff, b0, b1):
    x = diff / 400.0
    return _wdl(math.exp(b0 + b1 * x), math.exp(b0 - b1 * x))


def pred_manual(diff):
    x = diff / 400.0
    return _wdl(max(0.15, 1.35 + 0.85 * x), max(0.15, 1.35 - 0.85 * x))


def pred_elo_fixeddraw(diff, draw):
    e = 1.0 / (1.0 + 10 ** (-diff / 400.0))
    return (1 - draw) * e, draw, (1 - draw) * (1 - e)


def score_ll(diff, hs, as_, b0, b1):
    x = diff / 400.0
    la, lb = math.exp(b0 + b1 * x), math.exp(b0 - b1 * x)
    hk, ak = min(hs, MAXG), min(as_, MAXG)
    p = _pmf(la)[hk] * _pmf(lb)[ak]
    return -math.log(max(p, 1e-12))


class Metric:
    def __init__(s, name):
        s.name, s.ll, s.br, s.ok, s.n = name, 0.0, 0.0, 0, 0
    def add(s, probs, oc):
        s.n += 1
        s.ll += -math.log(max(probs[oc], 1e-12))
        y = [0, 0, 0]; y[oc] = 1
        s.br += sum((probs[k] - y[k]) ** 2 for k in range(3))
        if max(range(3), key=lambda k: probs[k]) == oc:
            s.ok += 1
    def row(s):
        n = max(1, s.n)
        return (s.name, s.ll / n, s.br / n, s.ok / n * 100, s.n)


def outcome(hs, as_):
    return 0 if hs > as_ else (1 if hs == as_ else 2)


def run():
    p = EloParams()
    matches = load_matches(until_date="2026-06-08")
    print(f"样本 {len(matches)} 场；λ 拟合用 <{FIT_END}，测试 = {sorted(TEST_YEARS)} 世界杯\n")

    recs, _ = walk_forward(matches, p)
    recs_wc, _ = walk_forward(matches, p, wc_only=True)
    diff_wc = {(r["date"], r["hs"], r["as"]): r["diff"] for r in recs_wc}

    b0, b1 = fit_lambda(recs)
    print(f"拟合: λ = exp({b0:.4f} + {b1:.4f}·Δelo/400)")
    print(f"  Δelo=0   → λ={math.exp(b0):.3f}")
    print(f"  Δelo=+200→ 强队λ={math.exp(b0+b1*0.5):.3f} / 弱队λ={math.exp(b0-b1*0.5):.3f}")
    print(f"  手工对比 → 强队1.78 / 弱队0.93\n")

    cw = cd = cl = 0
    for r in recs:
        if r["date"] >= FIT_END:
            continue
        oc = outcome(r["hs"], r["as"])
        cw += oc == 0; cd += oc == 1; cl += oc == 2
    tot = cw + cd + cl
    prior = (cw / tot, cd / tot, cl / tot)
    draw_fixed = cd / tot

    def evaluate(label, selector):
        Ms = [Metric("M1 新Elo+拟合λ(新模型)"), Metric("M2 新Elo+手工λ"),
              Metric("M3 新Elo+纯Elo+固定平局"), Metric("M4 仅WC-Elo+手工λ"),
              Metric("M5 历史频率(zero)")]
        sll = sn = 0.0
        for r in recs:
            if not selector(r):
                continue
            oc = outcome(r["hs"], r["as"])
            Ms[0].add(pred_fitted(r["diff"], b0, b1), oc)
            Ms[1].add(pred_manual(r["diff"]), oc)
            Ms[2].add(pred_elo_fixeddraw(r["diff"], draw_fixed), oc)
            Ms[3].add(pred_manual(diff_wc.get((r["date"], r["hs"], r["as"]), 0.0)), oc)
            Ms[4].add(prior, oc)
            sll += score_ll(r["diff"], r["hs"], r["as"], b0, b1); sn += 1
        print(f"\n【{label}】 N={Ms[0].n}")
        print(f"{'模型':<24}{'LogLoss':>9}{'Brier':>8}{'Acc%':>7}")
        for M in Ms:
            nm, ll, br, acc, n = M.row()
            print(f"{nm:<24}{ll:>9.4f}{br:>8.4f}{acc:>6.1f}")
        print(f"{'(参照)完全随机':<24}{math.log(3):>9.4f}{0.6667:>8.4f}{33.3:>6.1f}")
        print(f"M1 比分 LogLoss: {sll/max(1,sn):.4f}")

    evaluate("held-out: 2018+2022 世界杯决赛圈",
             lambda r: r["is_wcf"] and r["year"] in TEST_YEARS)
    evaluate("大样本参照: 所有国际比赛 date>=2018",
             lambda r: r["date"] >= "2018-01-01")
    return b0, b1


if __name__ == "__main__":
    run()
