"""
Dixon-Coles 时间加权双泊松模型 — 模型优化（学术界足球比分预测金标准）

相比当前"Elo差→单λ"模型的升级：
  - 每队独立的 attack（进攻力）/ defense（防守力）参数，从历史比分直接学
    （捕捉"进攻强但防守弱"这类单一 Elo 丢失的信息）
  - 主场优势项
  - 时间衰减：近期比赛权重高（exp(-ξ·年龄)）
  - 赛事权重：友谊赛权重低、正赛高
拟合：sklearn PoissonRegressor（稀疏 one-hot 设计矩阵 + sample_weight）

回测：2018 / 2022 世界杯各用"其之前所有数据"拟合（无未来泄露），
对比 DC vs 当前 Elo-Poisson 的 LogLoss / Brier / Acc。
"""

import os
import sys
import math
from datetime import date

import numpy as np
from scipy import sparse
from scipy.optimize import minimize_scalar
from sklearn.linear_model import PoissonRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elo_model import load_matches, EloParams
from backtest_v2 import walk_forward, fit_lambda, _wdl, _pmf, MAXG

XI = 0.30          # 时间衰减率（半衰期≈2.3年）
MIN_GAMES = 15     # 出现少于此次数的队不单独建参（罕见队）
ALPHA = 1e-3       # 泊松回归正则


def _tour_weight(t):
    if t == "FIFA World Cup":
        return 1.0
    if "qualification" in t:
        return 0.85
    if any(c in t for c in ("Copa América", "African Cup", "UEFA Euro",
                            "AFC Asian Cup", "Gold Cup", "Confederations",
                            "Nations League")):
        return 0.95
    if t == "Friendly":
        return 0.45
    return 0.7


def _days(d1, d2):
    a = date(int(d1[:4]), int(d1[5:7]), int(d1[8:10]))
    b = date(int(d2[:4]), int(d2[5:7]), int(d2[8:10]))
    return (b - a).days


def fit_dc(matches, until_date, xi=XI):
    """用 until_date 之前的比赛拟合 Dixon-Coles 双泊松，返回参数 dict"""
    data = [m for m in matches if m["date"] < until_date]
    # 队出现次数
    cnt = {}
    for m in data:
        cnt[m["home"]] = cnt.get(m["home"], 0) + 1
        cnt[m["away"]] = cnt.get(m["away"], 0) + 1
    teams = sorted(t for t, c in cnt.items() if c >= MIN_GAMES)
    idx = {t: i for i, t in enumerate(teams)}
    K = len(teams)
    ncol = 2 * K + 1  # attack[K] + defense[K] + home

    rows, cols, vals, ys, ws = [], [], [], [], []
    ri = 0
    for m in data:
        if m["home"] not in idx or m["away"] not in idx:
            continue
        w = math.exp(-xi * _days(m["date"], until_date) / 365.0) * _tour_weight(m["tour"])
        if w < 1e-4:
            continue
        h, a = idx[m["home"]], idx[m["away"]]
        hp = 0.0 if m["neutral"] else 1.0
        # 行A：home 进攻 → y=hs
        rows += [ri, ri, ri]; cols += [h, K + a, 2 * K]; vals += [1.0, 1.0, hp]
        ys.append(m["hs"]); ws.append(w); ri += 1
        # 行B：away 进攻 → y=as
        rows += [ri, ri]; cols += [a, K + h]; vals += [1.0, 1.0]
        ys.append(m["as"]); ws.append(w); ri += 1

    X = sparse.csr_matrix((vals, (rows, cols)), shape=(ri, ncol))
    y = np.array(ys, dtype=float)
    sw = np.array(ws, dtype=float)
    model = PoissonRegressor(alpha=ALPHA, fit_intercept=True, max_iter=400)
    model.fit(X, y, sample_weight=sw)
    coef = model.coef_
    return {
        "b0": float(model.intercept_),
        "attack": {t: float(coef[idx[t]]) for t in teams},
        "defense": {t: float(coef[K + idx[t]]) for t in teams},
        "home": float(coef[2 * K]),
        "teams": set(teams),
    }


def dc_lambdas(P, home, away, neutral=True):
    att, dfn = P["attack"], P["defense"]
    ah = att.get(home, 0.0); aa = att.get(away, 0.0)
    dh = dfn.get(home, 0.0); da = dfn.get(away, 0.0)
    hp = 0.0 if neutral else P["home"]
    lh = math.exp(P["b0"] + ah + da + hp)
    la = math.exp(P["b0"] + aa + dh)
    return min(lh, 6.0), min(la, 6.0)


def dc_wdl(P, home, away, neutral=True):
    lh, la = dc_lambdas(P, home, away, neutral)
    return _wdl(lh, la)


def dc_score_ll(P, home, away, hs, as_, neutral=True):
    lh, la = dc_lambdas(P, home, away, neutral)
    p = _pmf(lh)[min(hs, MAXG)] * _pmf(la)[min(as_, MAXG)]
    return -math.log(max(p, 1e-12))


# ── 回测对比 ──────────────────────────────────────────────
def _outcome(hs, as_):
    return 0 if hs > as_ else (1 if hs == as_ else 2)


def _ll_brier_acc(preds_outcomes):
    ll = br = ok = 0.0
    n = len(preds_outcomes)
    for probs, oc in preds_outcomes:
        ll += -math.log(max(probs[oc], 1e-12))
        yv = [0, 0, 0]; yv[oc] = 1
        br += sum((probs[k] - yv[k]) ** 2 for k in range(3))
        if max(range(3), key=lambda k: probs[k]) == oc:
            ok += 1
    return ll / n, br / n, ok / n * 100, n


def run():
    print("📥 加载全量国际比赛...")
    matches = load_matches(until_date="2026-06-09")
    p = EloParams()
    recs, _ = walk_forward(matches, p)            # recs[i] 与 matches[i] 一一对应
    b0e, b1e = fit_lambda(recs, fit_end="2018-01-01")

    print("🔧 预拟合 Dixon-Coles 年度切片 (2018..2026)...")
    slices = {yr: fit_dc(matches, f"{yr}-01-01") for yr in range(2018, 2027)}
    print(f"   DC 队数≈{len(slices[2026]['teams'])}  home={slices[2026]['home']:.3f}")

    def dc_for(d):
        return slices[min(max(int(d[:4]), 2018), 2026)]

    def eval_set(idxs, label):
        dcL, eloL, ocs = [], [], []
        dc_sll = elo_sll = 0.0
        for i in idxs:
            m = matches[i]
            oc = _outcome(m["hs"], m["as"]); ocs.append(oc)
            P = dc_for(m["date"])
            dcL.append(dc_wdl(P, m["home"], m["away"], neutral=m["neutral"]))
            x = recs[i]["diff"] / 400.0
            lh, la = math.exp(b0e + b1e * x), math.exp(b0e - b1e * x)
            eloL.append(_wdl(lh, la))
            dc_sll += dc_score_ll(P, m["home"], m["away"], m["hs"], m["as"], neutral=m["neutral"])
            elo_sll += -math.log(max(_pmf(lh)[min(m["hs"], MAXG)] * _pmf(la)[min(m["as"], MAXG)], 1e-12))
        n = len(idxs)
        best = (0.5, 9e9)
        for wi in range(21):
            w = wi / 20.0
            tot = 0.0
            for dp, ep, oc in zip(dcL, eloL, ocs):
                g = [(ep[k] ** w) * (dp[k] ** (1 - w)) for k in range(3)]
                s = sum(g) or 1.0
                tot += -math.log(max(g[oc] / s, 1e-12))
            if tot / n < best[1]:
                best = (w, tot / n)
        dll, dbr, dacc, _ = _ll_brier_acc(list(zip(dcL, ocs)))
        ell, ebr, eacc, _ = _ll_brier_acc(list(zip(eloL, ocs)))
        print(f"\n{'='*64}\n{label}（N={n}）\n{'='*64}")
        print(f"{'模型':<26}{'LogLoss':>9}{'Brier':>8}{'Acc%':>7}")
        print(f"{'Dixon-Coles 双泊松':<26}{dll:>9.4f}{dbr:>8.4f}{dacc:>6.1f}")
        print(f"{'当前 Elo-Poisson':<26}{ell:>9.4f}{ebr:>8.4f}{eacc:>6.1f}")
        print(f"{'集成 DC×Elo(w*='+f'{best[0]:.2f}'+')':<26}{best[1]:>9.4f}")
        # 温度校准：对最优集成做 p^(1/T) 归一，找最小 LogLoss 的 T
        wbest = best[0]
        ens = []
        for dp, ep, oc in zip(dcL, eloL, ocs):
            g = [(ep[k] ** wbest) * (dp[k] ** (1 - wbest)) for k in range(3)]
            s = sum(g) or 1.0
            ens.append(([x / s for x in g], oc))

        def _nllT(T):
            t = 0.0
            for pr, oc in ens:
                q = [pp ** (1.0 / T) for pp in pr]; z = sum(q) or 1.0
                t += -math.log(max(q[oc] / z, 1e-12))
            return t
        rT = minimize_scalar(_nllT, bounds=(0.5, 3.0), method="bounded")
        print(f"{'集成+温度校准(T='+f'{rT.x:.2f}'+')':<26}{rT.fun/len(ens):>9.4f}")
        print(f"比分 LogLoss: DC={dc_sll/n:.4f}  Elo={elo_sll/n:.4f}")
        return best[0]

    wc_idx = [i for i, m in enumerate(matches)
              if m["tour"] == "FIFA World Cup" and m["date"][:4] in ("2018", "2022")]
    big_idx = [i for i, m in enumerate(matches) if m["date"] >= "2018-01-01"]
    eval_set(wc_idx, "held-out: 2018+2022 世界杯（目标域）")
    w_star = eval_set(big_idx, "大样本: 所有国际比赛 date>=2018（统计稳）")

    print(f"\n✅ 最优集成权重（大样本）: Elo {w_star:.2f} / DC {1-w_star:.2f}")
    return slices[2026], w_star


if __name__ == "__main__":
    run()
