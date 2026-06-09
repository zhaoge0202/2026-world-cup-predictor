"""
模型实验台 — 严格 walk-forward 无泄露回测，对比候选精度改进

目的：在不引入过拟合的前提下，逐个验证以下改进对 held-out 世界杯 + 大样本的增益，
只有在两个域都不退化（大样本为主指标）的改进才会被固化进生产模型。

候选改进：
  A. Dixon-Coles ρ 低分修正   — 独立双泊松低估平局/低比分相关性，τ(ρ) 修正
  B. 集成权重网格            — Elo-Poisson × DC-ρ 的最优混合
  C. 温度缩放校准            — p^(1/T) 归一，直接压 LogLoss
  D. Logistic stacking 元学习器 — 用多基模型的 log 概率做特征，元学习器融合

无泄露设计：
  - Elo：walk-forward，recs[i].diff 是赛前评分差
  - DC ：年度切片 slices[yr]=fit(<yr 数据)，每场用其年份对应切片
  - λ  ：用 date<FIT_END 拟合
  - 集成权重 w / 温度 T / stacking 元学习器：只在 date<FIT_END 训练期 fit
  - 测试：held-out 2018+2022 世界杯（目标域）+ 大样本 date>=2018（统计稳）

用法: PYTHONPATH=scripts python scripts/model_lab.py
"""

import os
import sys
import math
import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elo_model import load_matches, EloParams
from backtest_v2 import walk_forward, fit_lambda, _pmf, MAXG
from dixon_coles import fit_dc, _tour_weight, _days

FIT_END = "2018-01-01"          # λ/集成/校准/stacking 的训练截止
TEST_YEARS = {"2018", "2022"}
EPS = 1e-12


# ════════════════════════════════════════════════════════════════
#  Dixon-Coles ρ 低分修正
# ════════════════════════════════════════════════════════════════
def _matrix_indep(lh, la):
    """独立双泊松比分矩阵 (MAXG+1)×(MAXG+1)。"""
    return np.outer(_pmf(lh), _pmf(la))


def _apply_tau(M, lh, la, rho):
    """对低比分 2×2 角应用 Dixon-Coles τ(ρ) 修正（in-place 复制后返回）。"""
    M = M.copy()
    M[0, 0] *= (1.0 - lh * la * rho)
    M[0, 1] *= (1.0 + lh * rho)
    M[1, 0] *= (1.0 + la * rho)
    M[1, 1] *= (1.0 - rho)
    np.clip(M, EPS, None, out=M)
    return M


def dc_lambdas(P, home, away, neutral=True):
    att, dfn = P["attack"], P["defense"]
    ah = att.get(home, 0.0); aa = att.get(away, 0.0)
    dh = dfn.get(home, 0.0); da = dfn.get(away, 0.0)
    hp = 0.0 if neutral else P["home"]
    lh = math.exp(P["b0"] + ah + da + hp)
    la = math.exp(P["b0"] + aa + dh)
    return min(lh, 6.0), min(la, 6.0)


def fit_rho(matches, until_date, P, xi=0.18, years_back=18):
    """固定 attack/defense，时间加权 MLE 拟合全局 ρ（一维优化）。
    只用 until_date 前、且双队都有 DC 参数的比赛；近 years_back 年（ρ 是低分相关性，近期足够）。"""
    teams = P["teams"]
    lo = f"{int(until_date[:4]) - years_back}-01-01"
    rows = []  # (lh, la, hs, as_, w)
    for m in matches:
        if m["date"] >= until_date or m["date"] < lo:
            continue
        if m["home"] not in teams or m["away"] not in teams:
            continue
        # 只有低比分才受 τ 影响，但权重仍按全场算
        lh, la = dc_lambdas(P, m["home"], m["away"], neutral=m["neutral"])
        w = math.exp(-xi * _days(m["date"], until_date) / 365.0) * _tour_weight(m["tour"])
        if w < 1e-4:
            continue
        rows.append((lh, la, m["hs"], m["as"], w))
    if not rows:
        return 0.0
    arr = np.array(rows, dtype=float)
    lh, la, hs, as_, w = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
    m00 = (hs == 0) & (as_ == 0)
    m01 = (hs == 0) & (as_ == 1)
    m10 = (hs == 1) & (as_ == 0)
    m11 = (hs == 1) & (as_ == 1)

    def nll(rho):
        tau = np.ones_like(lh)
        tau[m00] = 1.0 - lh[m00] * la[m00] * rho
        tau[m01] = 1.0 + lh[m01] * rho
        tau[m10] = 1.0 + la[m10] * rho
        tau[m11] = 1.0 - rho
        return -np.sum(w * np.log(np.clip(tau, EPS, None)))

    r = minimize_scalar(nll, bounds=(-0.25, 0.25), method="bounded")
    return float(r.x)


# ════════════════════════════════════════════════════════════════
#  胜平负从比分矩阵
# ════════════════════════════════════════════════════════════════
def wdl_from_matrix(M):
    s = M.sum()
    pw = np.tril(M, -1).sum() / s
    pd = np.trace(M) / s
    pl = np.triu(M, 1).sum() / s
    return np.array([pw, pd, pl])


def elo_poisson_matrix(diff, b0, b1):
    x = diff / 400.0
    lh, la = math.exp(b0 + b1 * x), math.exp(b0 - b1 * x)
    return _matrix_indep(min(lh, 6), min(la, 6))


# ════════════════════════════════════════════════════════════════
#  指标
# ════════════════════════════════════════════════════════════════
def _oc(hs, as_):
    return 0 if hs > as_ else (1 if hs == as_ else 2)


def metrics(probs_list, ocs):
    ll = br = ok = 0.0
    n = len(ocs)
    for p, oc in zip(probs_list, ocs):
        p = np.clip(np.asarray(p, float), EPS, None)
        p = p / p.sum()
        ll += -math.log(p[oc])
        y = np.zeros(3); y[oc] = 1
        br += float(np.sum((p - y) ** 2))
        if int(np.argmax(p)) == oc:
            ok += 1
    return ll / n, br / n, ok / n * 100


def fit_temperature(probs_arr, ocs):
    """温度缩放：找最小化 NLL 的 T（p^(1/T) 归一）。"""
    P = np.clip(np.asarray(probs_arr, float), EPS, None)
    ocs = np.asarray(ocs)

    def nll(T):
        Q = P ** (1.0 / T)
        Q = Q / Q.sum(axis=1, keepdims=True)
        return -np.sum(np.log(np.clip(Q[np.arange(len(ocs)), ocs], EPS, None)))

    r = minimize_scalar(nll, bounds=(0.5, 3.0), method="bounded")
    return float(r.x)


def apply_temperature(probs_arr, T):
    Q = np.clip(np.asarray(probs_arr, float), EPS, None) ** (1.0 / T)
    return Q / Q.sum(axis=1, keepdims=True)


# ════════════════════════════════════════════════════════════════
#  主流程
# ════════════════════════════════════════════════════════════════
def run():
    print("📥 加载全量国际比赛 + walk-forward Elo...")
    matches = load_matches(until_date="2026-06-09")
    p = EloParams()
    recs, _ = walk_forward(matches, p)
    b0, b1 = fit_lambda(recs, fit_end=FIT_END)
    print(f"   Elo-Poisson λ=exp({b0:.4f}+{b1:.4f}·Δ/400)")

    print("🔧 预拟合 DC 年度切片 + ρ (2018..2026)...")
    slices, rhos = {}, {}
    for yr in range(2018, 2027):
        P = fit_dc(matches, f"{yr}-01-01")
        slices[yr] = P
        rhos[yr] = fit_rho(matches, f"{yr}-01-01", P)
    print(f"   DC队数≈{len(slices[2026]['teams'])}  "
          f"ρ(2018)={rhos[2018]:+.4f}  ρ(2022)={rhos[2022]:+.4f}  ρ(2026)={rhos[2026]:+.4f}")

    def dc_for(d):
        return min(max(int(d[:4]), 2018), 2026)

    # ── 为每场预计算各基模型的胜平负概率 + 比分矩阵索引 ──
    def base_preds(i):
        m = matches[i]
        yr = dc_for(m["date"])
        P = slices[yr]; rho = rhos[yr]
        # Elo-Poisson
        Me = elo_poisson_matrix(recs[i]["diff"], b0, b1)
        # DC 独立
        lh, la = dc_lambdas(P, m["home"], m["away"], neutral=m["neutral"])
        Md = _matrix_indep(lh, la)
        # DC + ρ
        Mr = _apply_tau(Md, lh, la, rho)
        return Me, Md, Mr

    # 索引集合
    train_idx = [i for i, m in enumerate(matches) if m["date"] < FIT_END
                 and m["date"] >= "2006-01-01"]  # 训练元学习器/权重/温度用近代数据
    wc_idx = [i for i, m in enumerate(matches)
              if m["tour"] == "FIFA World Cup" and m["date"][:4] in TEST_YEARS]
    big_idx = [i for i, m in enumerate(matches) if m["date"] >= "2018-01-01"]

    # 预计算所有需要的场次
    need = sorted(set(train_idx) | set(wc_idx) | set(big_idx))
    cache = {}
    for i in need:
        cache[i] = base_preds(i)

    def wdl(M):
        return wdl_from_matrix(M)

    # ── 在训练集选最优集成权重 w（Elo-Poisson × DC-ρ，log 线性池化）──
    def ens_logpool(Me, Mr, w):
        pe, pr = wdl(Me), wdl(Mr)
        g = (pe ** w) * (pr ** (1 - w))
        return g / g.sum()

    best_w, best_ll = 0.7, 9e9
    tr_ocs = [_oc(matches[i]["hs"], matches[i]["as"]) for i in train_idx]
    for wi in range(21):
        w = wi / 20.0
        preds = [ens_logpool(cache[i][0], cache[i][2], w) for i in train_idx]
        ll, _, _ = metrics(preds, tr_ocs)
        if ll < best_ll:
            best_ll, best_w = ll, w
    print(f"   训练集最优集成权重 w*(Elo)={best_w:.2f}  (训练LL={best_ll:.4f})")

    # ── 温度：在训练集集成输出上拟合 T ──
    tr_ens = np.array([ens_logpool(cache[i][0], cache[i][2], best_w) for i in train_idx])
    T = fit_temperature(tr_ens, tr_ocs)
    print(f"   训练集最优温度 T={T:.3f}")

    # ── Logistic stacking：特征 = [logElo_wdl, logDCρ_wdl]，元学习器 ──
    from sklearn.linear_model import LogisticRegression
    def feats(i):
        pe, pr = wdl(cache[i][0]), wdl(cache[i][2])
        return np.concatenate([np.log(np.clip(pe, EPS, None)),
                               np.log(np.clip(pr, EPS, None))])
    Xtr = np.array([feats(i) for i in train_idx])
    ytr = np.array(tr_ocs)
    stack = LogisticRegression(max_iter=2000, C=1.0, multi_class="multinomial")
    stack.fit(Xtr, ytr)

    # ════════ 评估 ════════
    def evaluate(idxs, label):
        ocs = [_oc(matches[i]["hs"], matches[i]["as"]) for i in idxs]
        rows = {}
        rows["M0 Elo-Poisson(基线)"] = [wdl(cache[i][0]) for i in idxs]
        rows["DC 独立双泊松"]        = [wdl(cache[i][1]) for i in idxs]
        rows["DC + ρ 低分修正"]      = [wdl(cache[i][2]) for i in idxs]
        rows[f"集成 Elo×DCρ(w={best_w:.2f})"] = [ens_logpool(cache[i][0], cache[i][2], best_w) for i in idxs]
        ens = np.array(rows[f"集成 Elo×DCρ(w={best_w:.2f})"])
        rows[f"集成+温度校准(T={T:.2f})"] = list(apply_temperature(ens, T))
        Xte = np.array([feats(i) for i in idxs])
        rows["Logistic stacking"] = list(stack.predict_proba(Xte))

        print(f"\n{'='*60}\n【{label}】 N={len(idxs)}\n{'='*60}")
        print(f"{'模型':<26}{'LogLoss':>9}{'Brier':>8}{'Acc%':>7}")
        base = None
        for nm, preds in rows.items():
            ll, br, acc = metrics(preds, ocs)
            if base is None:
                base = ll
            tag = "" if nm.startswith("M0") else f"  ({(base-ll)/base*100:+.2f}%)"
            print(f"{nm:<26}{ll:>9.4f}{br:>8.4f}{acc:>6.1f}{tag}")

    evaluate(wc_idx, "held-out: 2018+2022 世界杯（目标域）")
    evaluate(big_idx, "大样本: 所有国际比赛 date>=2018（统计稳）")
    return slices, rhos, best_w, T


if __name__ == "__main__":
    run()
