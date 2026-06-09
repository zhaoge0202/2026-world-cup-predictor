"""
2026 世界杯最终最精准预测 — 市场锚定 + 数据模型融合

依据：对冠军这种长尾事件，博彩/预测市场的去 vig 共识是单一最优估计
（半强有效市场，聚合了伤病/状态/阵容等模型看不到的信息）。
单一模型难以稳定超越市场，故：
  - 主锚：博彩 + 预测市场去 vig 隐含概率（2026-06-08 开赛前快照）
  - 修正：v2 数据模型（全量 Elo + 真实分组 MC，经回测验证单场更准）
  - 长尾：市场只覆盖 Top 队，其余 40+ 队用 v2 模型细粒度填充
方法：对数线性池化（log-linear opinion pool），市场权重 A=0.70
"""

import os
import json
import math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2 = os.path.join(ROOT, "data", "wc2026_prediction_v2.json")

# 博彩 + 预测市场去 vig 隐含概率（2026-06-08，开赛前 48h；缩放到 Top6≈70%）
# 来源：DraftKings/FanDuel/ESPN 博彩 + Polymarket/Kalshi 预测市场共识
MARKET = {
    "Spain": 0.162, "France": 0.155, "England": 0.111,
    "Brazil": 0.089, "Argentina": 0.089, "Portugal": 0.093,
}
A = 0.70  # 市场权重（市场为主锚，模型修正）


def main():
    v2 = json.load(open(V2, encoding="utf-8"))["teams"]
    model = {t["country"]: t["champion"] for t in v2}

    S = [t for t in MARKET if t in model]
    target_top = A * sum(MARKET[t] for t in S) + (1 - A) * sum(model[t] for t in S)

    # Top 队：log-linear 池化
    raw = {t: math.exp(A * math.log(MARKET[t]) + (1 - A) * math.log(max(model[t], 1e-6))) for t in S}
    ssum = sum(raw.values())
    final = {t: raw[t] / ssum * target_top for t in S}

    # 长尾：市场未覆盖的队，用模型按比例填充剩余概率
    rest = [t for t in model if t not in MARKET]
    rsum = sum(model[t] for t in rest)
    for t in rest:
        final[t] = model[t] / rsum * (1 - target_top)

    rows = sorted(final.items(), key=lambda x: -x[1])

    print("=" * 64)
    print("2026 世界杯 — 最终最精准冠军概率（市场锚定 + 数据模型融合）")
    print("=" * 64)
    print(f"{'#':>2} {'球队':<16}{'最终':>8}{'市场':>8}{'v2模型':>9}")
    for i, (t, p) in enumerate(rows[:16], 1):
        mk = f"{MARKET[t]*100:.1f}%" if t in MARKET else "—"
        print(f"{i:>2} {t:<16}{p*100:>7.1f}%{mk:>8}{model[t]*100:>8.1f}%")
    print("-" * 64)
    print(f"概率总和: {sum(final.values())*100:.1f}%  | Top6 占比: {sum(final[t] for t in S)*100:.1f}%")

    out = os.path.join(ROOT, "data", "wc2026_prediction_final.json")
    payload = {
        "as_of": "2026-06-08",
        "method": "log-linear pool: market(A=0.70) + v2 model; tail filled by model",
        "market_weight": A,
        "teams": [{"country": t, "champion": p,
                   "market": MARKET.get(t), "model": model[t]} for t, p in rows],
    }
    json.dump(payload, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n✅ 已保存 {out}")


if __name__ == "__main__":
    main()
