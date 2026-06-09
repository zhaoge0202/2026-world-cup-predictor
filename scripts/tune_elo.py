"""
Elo 超参网格优化 — 模型优化（Elo 是集成主力，权重 0.7）

网格搜索 home_adv（主场优势）× K 缩放（赛事权重整体强度），
walk-forward 重训 Elo + 拟合 λ + 在大样本(date>=2018)上算胜平负 LogLoss。
用大样本调（标量超参，低过拟合风险），世界杯子集确认不退化。
"""

import os
import sys
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elo_model import load_matches, EloParams
from backtest_v2 import walk_forward, fit_lambda, _wdl, _pmf, MAXG


def _oc(hs, as_):
    return 0 if hs > as_ else (1 if hs == as_ else 2)


def eval_params(matches, params, fit_end="2018-01-01", test_from="2018-01-01"):
    recs, _ = walk_forward(matches, params)
    b0, b1 = fit_lambda(recs, fit_end=fit_end)
    ll = n = 0
    ll_wc = n_wc = 0
    for i, m in enumerate(matches):
        if m["date"] < test_from:
            continue
        x = recs[i]["diff"] / 400.0
        lh, la = math.exp(b0 + b1 * x), math.exp(b0 - b1 * x)
        wdl = _wdl(lh, la)
        oc = _oc(m["hs"], m["as"])
        l = -math.log(max(wdl[oc], 1e-12))
        ll += l; n += 1
        if m["tour"] == "FIFA World Cup":
            ll_wc += l; n_wc += 1
    return ll / n, (ll_wc / n_wc if n_wc else 0), b0, b1


def scaled_params(home_adv, s):
    return EloParams(
        home_adv=home_adv,
        k_wc_finals=60 * s, k_continental=50 * s, k_quali=40 * s,
        k_friendly=20 * s, k_other=30 * s,
    )


def run():
    matches = load_matches(until_date="2026-06-09")
    base_ll, base_wc, _, _ = eval_params(matches, EloParams())
    print(f"基线 EloParams(home=100,s=1.0): 大样本LL={base_ll:.4f}  世界杯LL={base_wc:.4f}\n")

    grid_h = [50, 75, 100, 125]
    grid_s = [0.7, 0.85, 1.0, 1.2]
    print(f"{'home_adv':>9}{'K_scale':>9}{'大样本LL':>11}{'世界杯LL':>11}")
    best = None
    for h in grid_h:
        for s in grid_s:
            ll, wc, b0, b1 = eval_params(matches, scaled_params(h, s))
            mark = ""
            if best is None or ll < best[0]:
                best = (ll, wc, h, s, b0, b1); mark = " *"
            print(f"{h:>9}{s:>9.2f}{ll:>11.4f}{wc:>11.4f}{mark}")

    print(f"\n✅ 最优: home_adv={best[2]}, K_scale={best[3]:.2f}  "
          f"→ 大样本LL={best[0]:.4f}（基线 {base_ll:.4f}）, 世界杯LL={best[1]:.4f}（基线 {base_wc:.4f}）")
    print(f"   提升: 大样本 {(base_ll-best[0])/base_ll*100:+.2f}%  世界杯 {(base_wc-best[1])/base_wc*100:+.2f}%")
    print(f"   λ = exp({best[4]:.3f} + {best[5]:.3f}·Δelo/400)")
    return best


if __name__ == "__main__":
    run()
