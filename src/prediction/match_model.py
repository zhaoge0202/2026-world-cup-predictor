from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from typing import Dict, Iterable, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL_PATH = os.path.join(ROOT, "data", "ensemble_model.json")
DEFAULT_ELO = 1500.0
MAX_GOALS = 10

ALIAS = {
    "USA": "United States",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
}


@dataclass(frozen=True)
class MatchPrediction:
    team1: str
    team2: str
    team1_win: float
    draw: float
    team2_win: float
    lambda_team1: float
    lambda_team2: float
    top_scores: List[dict]
    score_matrix: List[dict]


def normalize_team_name(team: str) -> str:
    return ALIAS.get(team, team)


def load_ensemble_model(path: str = DEFAULT_MODEL_PATH) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ratings_from_model(model: Optional[dict]) -> Dict[str, float]:
    if not model:
        return {}
    return {k: float(v) for k, v in model.get("elo", {}).items()}


def poisson_pmf(lam: float, max_goals: int = MAX_GOALS) -> List[float]:
    lam = max(float(lam), 1e-9)
    probs = [math.exp(-lam)]
    for k in range(1, max_goals + 1):
        probs.append(probs[-1] * lam / k)
    total = sum(probs)
    return [p / total for p in probs]


def lambdas_from_elo(
    team1: str,
    team2: str,
    ratings: Dict[str, float],
    model: Optional[dict] = None,
    neutral: bool = True,
) -> tuple[float, float]:
    t1 = normalize_team_name(team1)
    t2 = normalize_team_name(team2)
    elo1 = float(ratings.get(t1, ratings.get(team1, DEFAULT_ELO)))
    elo2 = float(ratings.get(t2, ratings.get(team2, DEFAULT_ELO)))
    diff = elo1 - elo2
    if not neutral:
        diff += 100.0
    if model and "elo_b0" in model and "elo_b1" in model:
        b0 = float(model["elo_b0"])
        b1 = float(model["elo_b1"])
        x = diff / 400.0
        return math.exp(b0 + b1 * x), math.exp(b0 - b1 * x)
    x = diff / 400.0
    return max(0.2, 1.35 + 0.75 * x), max(0.2, 1.35 - 0.75 * x)


def build_score_matrix(lambda_team1: float, lambda_team2: float, max_goals: int = MAX_GOALS) -> List[dict]:
    p1 = poisson_pmf(lambda_team1, max_goals=max_goals)
    p2 = poisson_pmf(lambda_team2, max_goals=max_goals)
    rows = []
    for g1, prob1 in enumerate(p1):
        for g2, prob2 in enumerate(p2):
            rows.append({
                "team1_goals": g1,
                "team2_goals": g2,
                "score": f"{g1}-{g2}",
                "prob": prob1 * prob2,
            })
    total = sum(r["prob"] for r in rows) or 1.0
    for row in rows:
        row["prob"] /= total
    return rows


def top_scorelines(score_matrix: Iterable[dict], limit: int = 6) -> List[dict]:
    rows = sorted(score_matrix, key=lambda r: r["prob"], reverse=True)[:limit]
    return [
        {
            "score": row.get("score", f"{row['team1_goals']}-{row['team2_goals']}"),
            "team1_goals": row["team1_goals"],
            "team2_goals": row["team2_goals"],
            "prob": row["prob"],
        }
        for row in rows
    ]


def predict_match(
    team1: str,
    team2: str,
    ratings: Optional[Dict[str, float]] = None,
    model: Optional[dict] = None,
    neutral: bool = True,
    max_goals: int = MAX_GOALS,
) -> MatchPrediction:
    if model is None:
        model = load_ensemble_model()
    if ratings is None:
        ratings = ratings_from_model(model)
    lam1, lam2 = lambdas_from_elo(team1, team2, ratings, model=model, neutral=neutral)
    matrix = build_score_matrix(lam1, lam2, max_goals=max_goals)
    team1_win = sum(r["prob"] for r in matrix if r["team1_goals"] > r["team2_goals"])
    draw = sum(r["prob"] for r in matrix if r["team1_goals"] == r["team2_goals"])
    team2_win = sum(r["prob"] for r in matrix if r["team1_goals"] < r["team2_goals"])
    total = team1_win + draw + team2_win or 1.0
    return MatchPrediction(
        team1=team1,
        team2=team2,
        team1_win=team1_win / total,
        draw=draw / total,
        team2_win=team2_win / total,
        lambda_team1=lam1,
        lambda_team2=lam2,
        top_scores=top_scorelines(matrix, limit=6),
        score_matrix=matrix,
    )
