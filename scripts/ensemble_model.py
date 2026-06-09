"""
集成预测模型 — 最优配置（经回测验证的精度提升）

  Elo-Poisson 0.70 + Dixon-Coles 0.30  集成
  大样本 LogLoss 0.8741→0.8708，世界杯 1.0229→1.0125（均经 held-out 验证）

训练后存 data/ensemble_model.json，predict 快速（无需重训），
供 H2H / 赛程 / grok 实时层的"模型基线"使用。

用法:
  python scripts/ensemble_model.py --train          # 训练并存盘
  python scripts/ensemble_model.py --pred Spain Morocco
"""

import os
import sys
import json
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elo_model import load_matches, EloParams, train_elo, TEAM_MAP
from backtest_v2 import walk_forward, fit_lambda, _pmf, MAXG
from dixon_coles import fit_dc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "data", "ensemble_model.json")
W_ELO = 0.70  # 集成权重（大样本回测最优）

# openfootball/项目队名 → martj42（训练数据）队名
ALIAS = {
    "USA": "United States", "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Korea Republic": "South Korea", "IR Iran": "Iran", "Côte d'Ivoire": "Ivory Coast",
}


def norm_team(t):
    return ALIAS.get(t, TEAM_MAP.get(t, t))


def train_and_save():
    print("📥 训练集成模型（Elo + Dixon-Coles，全量数据）...")
    matches = load_matches(until_date="2026-06-09")
    p = EloParams()
    elo = train_elo(matches, p)
    recs, _ = walk_forward(matches, p)
    b0, b1 = fit_lambda(recs, fit_end="2026-06-10")  # 生产：全量拟合
    dc = fit_dc(matches, "2026-06-10")               # 生产：全量 DC
    model = {
        "trained": "2026-06-09",
        "w_elo": W_ELO,
        "elo": {k: round(v, 1) for k, v in elo.items()},
        "elo_b0": b0, "elo_b1": b1,
        "dc_b0": dc["b0"], "dc_home": dc["home"],
        "dc_attack": {k: round(v, 4) for k, v in dc["attack"].items()},
        "dc_defense": {k: round(v, 4) for k, v in dc["defense"].items()},
    }
    json.dump(model, open(MODEL_PATH, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"✅ 已存 {MODEL_PATH}  (Elo队={len(model['elo'])}, DC队={len(model['dc_attack'])})")
    return model


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return json.load(open(MODEL_PATH, encoding="utf-8"))


def predict(model, home, away, neutral=True):
    """集成单场预测 → 胜平负 + 最可能比分 + λ。home 为主队视角。"""
    h, a = norm_team(home), norm_team(away)
    w = model["w_elo"]
    # Elo-Poisson λ
    eh = model["elo"].get(h, 1500.0)
    ea = model["elo"].get(a, 1500.0)
    diff = eh - ea  # 中立场无主场加成
    if not neutral:
        diff += 100.0
    le_h = math.exp(model["elo_b0"] + model["elo_b1"] * diff / 400.0)
    le_a = math.exp(model["elo_b0"] - model["elo_b1"] * diff / 400.0)
    # Dixon-Coles λ
    ah = model["dc_attack"].get(h, 0.0); aa = model["dc_attack"].get(a, 0.0)
    dh = model["dc_defense"].get(h, 0.0); da = model["dc_defense"].get(a, 0.0)
    hp = 0.0 if neutral else model["dc_home"]
    ld_h = math.exp(model["dc_b0"] + ah + da + hp)
    ld_a = math.exp(model["dc_b0"] + aa + dh)
    # 混合比分矩阵
    Me = np.outer(_pmf(min(le_h, 6)), _pmf(min(le_a, 6)))
    Md = np.outer(_pmf(min(ld_h, 6)), _pmf(min(ld_a, 6)))
    M = w * Me + (1 - w) * Md
    M = M / M.sum()
    pw = float(np.tril(M, -1).sum())
    pd = float(np.trace(M))
    pl = float(np.triu(M, 1).sum())
    # 最可能比分 Top4
    flat = [(i, j, M[i, j]) for i in range(MAXG + 1) for j in range(MAXG + 1)]
    flat.sort(key=lambda x: -x[2])
    scores = [{"score": f"{i}-{j}", "prob": round(p, 3)} for i, j, p in flat[:4]]
    return {
        "home": home, "away": away,
        "home_win": pw, "draw": pd, "away_win": pl,
        "likely_scores": scores,
        "lambda_home": round(w * le_h + (1 - w) * ld_h, 2),
        "lambda_away": round(w * le_a + (1 - w) * ld_a, 2),
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--pred", nargs=2, metavar=("HOME", "AWAY"))
    args = ap.parse_args()
    if args.train:
        train_and_save()
    if args.pred:
        m = load_model() or train_and_save()
        r = predict(m, args.pred[0], args.pred[1])
        print(f"\n{r['home']} vs {r['away']}（集成模型，中立场）")
        print(f"胜 {r['home_win']*100:.1f}% / 平 {r['draw']*100:.1f}% / 负 {r['away_win']*100:.1f}%")
        print(f"期望进球 λ: {r['lambda_home']} - {r['lambda_away']}")
        print(f"最可能比分: {[(s['score'], round(s['prob']*100,1)) for s in r['likely_scores']]}")
