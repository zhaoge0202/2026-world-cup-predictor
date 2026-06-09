from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .bracket import Standing, best_thirds, group_letter, rank_group, resolve_group_slot, resolve_match_ref
from .match_model import load_ensemble_model, predict_match, ratings_from_model

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURES_PATH = os.path.join(ROOT, "data", "wc2026_fixtures.json")
OUTPUT_PATH = os.path.join(ROOT, "data", "wc2026_schedule_predictions.json")


def load_fixtures(path: str = FIXTURES_PATH) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("matches", [])


def fixture_sort_key(match: dict) -> tuple:
    return (match.get("date", ""), match.get("time", ""), int(match.get("num", 0) or 0), match.get("round", ""))


def completed_score(match: dict) -> Optional[Tuple[int, int]]:
    score = match.get("score") or {}
    ft = score.get("ft")
    if isinstance(ft, list) and len(ft) == 2 and ft[0] is not None and ft[1] is not None:
        return int(ft[0]), int(ft[1])
    return None


def is_concrete_team(team: str) -> bool:
    return bool(team) and not any(ch.isdigit() for ch in team) and "/" not in team


def group_matches(fixtures: Iterable[dict]) -> List[dict]:
    return [
        m for m in fixtures
        if m.get("group") and is_concrete_team(m.get("team1", "")) and is_concrete_team(m.get("team2", ""))
    ]


def next_scheduled_match(fixtures: Iterable[dict], today: Optional[str] = None) -> Optional[dict]:
    today = today or date.today().isoformat()
    candidates = [
        m for m in fixtures
        if is_concrete_team(m.get("team1", ""))
        and is_concrete_team(m.get("team2", ""))
        and completed_score(m) is None
    ]
    future = [m for m in candidates if m.get("date", "") >= today]
    selected = sorted(future or candidates, key=fixture_sort_key)
    return selected[0] if selected else None


def fixture_match_id(match: dict) -> str:
    num = match.get("num")
    if num not in (None, ""):
        return f"num:{num}"
    return "fixture:{date}|{time}|{team1}|{team2}".format(
        date=match.get("date", ""),
        time=match.get("time", ""),
        team1=match.get("team1", ""),
        team2=match.get("team2", ""),
    )


def predict_fixture(match: dict, ratings: Dict[str, float], model: Optional[dict]) -> dict:
    pred = predict_match(match["team1"], match["team2"], ratings=ratings, model=model)
    return {
        "match_id": fixture_match_id(match),
        "num": match.get("num"),
        "round": match.get("round"),
        "date": match.get("date"),
        "time": match.get("time"),
        "group": match.get("group"),
        "ground": match.get("ground"),
        "team1": match["team1"],
        "team2": match["team2"],
        "team1_win": pred.team1_win,
        "draw": pred.draw,
        "team2_win": pred.team2_win,
        "lambda_team1": pred.lambda_team1,
        "lambda_team2": pred.lambda_team2,
        "top_scores": pred.top_scores,
    }


def _sample_score(match: dict, ratings: Dict[str, float], model: Optional[dict], rng: np.random.Generator) -> tuple[int, int]:
    fixed = completed_score(match)
    if fixed is not None:
        return fixed
    pred = predict_match(match["team1"], match["team2"], ratings=ratings, model=model)
    return int(rng.poisson(pred.lambda_team1)), int(rng.poisson(pred.lambda_team2))


def _simulate_groups(
    fixtures: List[dict],
    ratings: Dict[str, float],
    model: Optional[dict],
    rng: np.random.Generator,
) -> Dict[str, List[Standing]]:
    by_group: Dict[str, Dict[str, Standing]] = defaultdict(dict)
    for match in sorted(group_matches(fixtures), key=fixture_sort_key):
        group = group_letter(match["group"])
        for team in (match["team1"], match["team2"]):
            if team not in by_group[group]:
                by_group[group][team] = Standing(group, team, elo=ratings.get(team, 1500.0))
        g1, g2 = _sample_score(match, ratings, model, rng)
        a = by_group[group][match["team1"]]
        b = by_group[group][match["team2"]]
        a.goals_for += g1
        a.goals_against += g2
        b.goals_for += g2
        b.goals_against += g1
        if g1 > g2:
            a.points += 3
        elif g2 > g1:
            b.points += 3
        else:
            a.points += 1
            b.points += 1
    return {g: rank_group(rows.values()) for g, rows in by_group.items()}


def _resolve_team_ref(
    ref: str,
    ranked_groups: Dict[str, List[Standing]],
    selected_thirds: List[Standing],
    winners: Dict[int, str],
    losers: Dict[int, str],
) -> Optional[str]:
    direct = resolve_group_slot(ref, ranked_groups, selected_thirds)
    if direct:
        return direct
    ref_match = resolve_match_ref(ref, winners, losers)
    if ref_match:
        return ref_match
    return ref if is_concrete_team(ref) else None


def _record_reach(reached: Dict[str, set], round_name: str, team1: str, team2: str) -> None:
    if round_name == "Round of 32":
        reached["round_of_32"].update([team1, team2])
    elif round_name == "Round of 16":
        reached["round_of_16"].update([team1, team2])
    elif round_name == "Quarter-final":
        reached["quarter"].update([team1, team2])
    elif round_name == "Semi-final":
        reached["semi"].update([team1, team2])
    elif round_name == "Final":
        reached["final"].update([team1, team2])


def _play_knockout(
    fixtures: List[dict],
    ranked_groups: Dict[str, List[Standing]],
    ratings: Dict[str, float],
    model: Optional[dict],
    rng: np.random.Generator,
) -> Dict[str, set]:
    thirds = best_thirds([rows[2] for rows in ranked_groups.values() if len(rows) >= 3])
    winners: Dict[int, str] = {}
    losers: Dict[int, str] = {}
    reached = {"round_of_32": set(), "round_of_16": set(), "quarter": set(), "semi": set(), "final": set(), "champion": set()}
    knockout = [m for m in sorted(fixtures, key=fixture_sort_key) if not m.get("group")]
    for match in knockout:
        num = match.get("num")
        team1 = _resolve_team_ref(match.get("team1", ""), ranked_groups, thirds, winners, losers)
        team2 = _resolve_team_ref(match.get("team2", ""), ranked_groups, thirds, winners, losers)
        if not team1 or not team2:
            continue
        round_name = match.get("round", "")
        _record_reach(reached, round_name, team1, team2)
        g1, g2 = _sample_score({"team1": team1, "team2": team2, "score": match.get("score")}, ratings, model, rng)
        if g1 == g2:
            pred = predict_match(team1, team2, ratings=ratings, model=model)
            decisive = max(pred.team1_win + pred.team2_win, 1e-9)
            winner = team1 if rng.random() < pred.team1_win / decisive else team2
        else:
            winner = team1 if g1 > g2 else team2
        loser = team2 if winner == team1 else team1
        if num is not None:
            winners[int(num)] = winner
            losers[int(num)] = loser
        if round_name == "Final":
            reached["champion"].add(winner)
    return reached


def generate_schedule_predictions(
    fixtures: Optional[List[dict]] = None,
    ratings: Optional[Dict[str, float]] = None,
    model: Optional[dict] = None,
    n_sim: int = 2000,
    seed: int = 42,
    today: Optional[str] = None,
) -> dict:
    model = model if model is not None else load_ensemble_model()
    ratings = ratings if ratings is not None else ratings_from_model(model)
    fixtures = fixtures if fixtures is not None else load_fixtures()
    concrete_matches = [
        m for m in fixtures
        if is_concrete_team(m.get("team1", "")) and is_concrete_team(m.get("team2", ""))
    ]
    match_predictions = [predict_fixture(m, ratings, model) for m in sorted(concrete_matches, key=fixture_sort_key)]
    next_match = next_scheduled_match(fixtures, today=today)
    teams = sorted({t for m in group_matches(fixtures) for t in (m["team1"], m["team2"])})
    counts = {team: defaultdict(int) for team in teams}
    rng = np.random.default_rng(seed)
    for _ in range(max(1, n_sim)):
        ranked_groups = _simulate_groups(fixtures, ratings, model, rng)
        for rows in ranked_groups.values():
            for idx, row in enumerate(rows):
                if idx == 0:
                    counts[row.team]["group_winner"] += 1
                if idx < 2:
                    counts[row.team]["qualified"] += 1
        reached = _play_knockout(fixtures, ranked_groups, ratings, model, rng)
        for key, values in reached.items():
            for team in values:
                counts.setdefault(team, defaultdict(int))[key] += 1
    denom = float(max(1, n_sim))
    team_rows = []
    for team, c in counts.items():
        team_rows.append({
            "country": team,
            "champion": c["champion"] / denom,
            "final": c["final"] / denom,
            "semi": c["semi"] / denom,
            "quarter": c["quarter"] / denom,
            "round_of_16": c["round_of_16"] / denom,
            "round_of_32": c["round_of_32"] / denom,
            "qualified": c["qualified"] / denom,
            "group_winner": c["group_winner"] / denom,
            "elo": ratings.get(team, 1500.0),
            "source": "schedule",
        })
    team_rows.sort(key=lambda r: r["champion"], reverse=True)
    return {
        "source": "schedule",
        "as_of": datetime.now().isoformat(timespec="seconds"),
        "n_sim": n_sim,
        "seed": seed,
        "teams": team_rows,
        "matches": match_predictions,
        "next_match": predict_fixture(next_match, ratings, model) if next_match else None,
    }


def save_schedule_predictions(output: dict, path: str = OUTPUT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
