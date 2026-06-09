# Schedule-Driven World Cup Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make H2H and champion probabilities follow the real 2026 World Cup fixture order.

**Architecture:** Add a focused `src/prediction` package that produces one schedule-driven JSON artifact. The mobile dashboard loads that artifact for the champion card and for the default H2H matchup, while preserving manual H2H comparison and the existing fallback data sources.

**Tech Stack:** Python standard library, NumPy already in requirements, existing fixture JSON, existing ensemble model JSON, mobile dashboard HTML/CSS/JavaScript embedded in `src/dashboard/mobile_ui.py`, `unittest` for tests.

---

## File Structure

- Create `src/prediction/__init__.py`
  - Marks the prediction package and exports public helpers.

- Create `src/prediction/match_model.py`
  - Pure one-match prediction helpers.
  - Loads `data/ensemble_model.json`.
  - Produces W/D/L, expected goals, score matrix, and unboosted top scorelines.

- Create `src/prediction/bracket.py`
  - Group standings, best-third selection, slot resolution, match reference parsing.
  - No random behavior.

- Create `src/prediction/schedule_model.py`
  - Loads fixtures and ratings.
  - Runs seeded tournament simulations in fixture order.
  - Writes schedule-level prediction output.

- Create `scripts/generate_schedule_predictions.py`
  - CLI wrapper around `schedule_model.generate_schedule_predictions`.

- Modify `src/dashboard/mobile_ui.py`
  - Inject `__SCHEDULE_PRED__`.
  - Champion card uses schedule predictions before old final predictions.
  - H2H defaults to next scheduled match.
  - Most-likely scorelines use unboosted probabilities.

- Create `tests/test_match_model.py`
- Create `tests/test_bracket.py`
- Create `tests/test_schedule_model.py`
- Create `tests/test_mobile_schedule_integration.py`

Use `python -m unittest ...`; this repo does not currently include pytest.

---

### Task 1: Add Match Prediction Core

**Files:**
- Create: `src/prediction/__init__.py`
- Create: `src/prediction/match_model.py`
- Test: `tests/test_match_model.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_match_model.py`:

```python
import unittest

from src.prediction.match_model import (
    MatchPrediction,
    poisson_pmf,
    predict_match,
    top_scorelines,
)


class MatchModelTest(unittest.TestCase):
    def test_poisson_pmf_is_normalized(self):
        probs = poisson_pmf(1.7, max_goals=10)
        self.assertAlmostEqual(sum(probs), 1.0, places=6)
        self.assertGreater(probs[1], probs[6])

    def test_top_scorelines_are_unboosted_and_sorted(self):
        rows = [
            {"team1_goals": 3, "team2_goals": 2, "prob": 0.05},
            {"team1_goals": 1, "team2_goals": 1, "prob": 0.10},
            {"team1_goals": 2, "team2_goals": 1, "prob": 0.08},
        ]
        scores = top_scorelines(rows, limit=2)
        self.assertEqual([s["score"] for s in scores], ["1-1", "2-1"])

    def test_predict_match_returns_normalized_probabilities(self):
        pred = predict_match("France", "Brazil", {"France": 1887.4, "Brazil": 1912.7})
        self.assertIsInstance(pred, MatchPrediction)
        self.assertAlmostEqual(pred.team1_win + pred.draw + pred.team2_win, 1.0, places=6)
        self.assertGreater(pred.lambda_team1, 0)
        self.assertGreater(pred.lambda_team2, 0)
        self.assertEqual(len(pred.top_scores), 6)
        self.assertIn("score", pred.top_scores[0])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python -m unittest tests.test_match_model -v
```

Expected: FAIL because `src.prediction.match_model` does not exist.

- [ ] **Step 3: Implement match model**

Create `src/prediction/__init__.py`:

```python
"""Schedule-driven prediction helpers for the World Cup dashboard."""
```

Create `src/prediction/match_model.py` with:

```python
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
            "score": row["score"],
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
```

- [ ] **Step 4: Run the tests and verify GREEN**

Run:

```bash
python -m unittest tests.test_match_model -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/prediction/__init__.py src/prediction/match_model.py tests/test_match_model.py
git commit -m "Add schedule match prediction core"
```

---

### Task 2: Add Bracket and Standings Helpers

**Files:**
- Create: `src/prediction/bracket.py`
- Test: `tests/test_bracket.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bracket.py`:

```python
import unittest

from src.prediction.bracket import (
    Standing,
    best_thirds,
    rank_group,
    resolve_group_slot,
    resolve_match_ref,
)


class BracketTest(unittest.TestCase):
    def test_rank_group_orders_by_points_goal_difference_goals_for_then_elo(self):
        rows = [
            Standing("A", "Mexico", points=4, goals_for=4, goals_against=2, elo=1800),
            Standing("A", "South Africa", points=4, goals_for=5, goals_against=3, elo=1700),
            Standing("A", "South Korea", points=3, goals_for=2, goals_against=2, elo=1750),
        ]
        ranked = rank_group(rows)
        self.assertEqual([r.team for r in ranked], ["South Africa", "Mexico", "South Korea"])

    def test_best_thirds_selects_top_eight(self):
        thirds = [Standing(chr(65 + i), f"T{i}", points=i, goals_for=i, goals_against=0, elo=1500 + i) for i in range(12)]
        selected = best_thirds(thirds, limit=8)
        self.assertEqual(len(selected), 8)
        self.assertEqual(selected[0].team, "T11")
        self.assertEqual(selected[-1].team, "T4")

    def test_resolve_group_slot_for_winner_runner_up_and_allowed_third(self):
        groups = {
            "A": [
                Standing("A", "Mexico", 6, 5, 2, 1800),
                Standing("A", "South Korea", 4, 4, 2, 1750),
                Standing("A", "South Africa", 3, 3, 4, 1650),
            ],
            "B": [
                Standing("B", "Canada", 6, 5, 2, 1760),
                Standing("B", "Qatar", 4, 4, 2, 1700),
                Standing("B", "Switzerland", 3, 3, 4, 1900),
            ],
        }
        thirds = [groups["B"][2], groups["A"][2]]
        self.assertEqual(resolve_group_slot("1A", groups, thirds), "Mexico")
        self.assertEqual(resolve_group_slot("2A", groups, thirds), "South Korea")
        self.assertEqual(resolve_group_slot("3A/B/C/D/F", groups, thirds), "Switzerland")

    def test_resolve_match_ref(self):
        winners = {73: "Mexico"}
        losers = {101: "Brazil"}
        self.assertEqual(resolve_match_ref("W73", winners, losers), "Mexico")
        self.assertEqual(resolve_match_ref("L101", winners, losers), "Brazil")
        self.assertIsNone(resolve_match_ref("W99", winners, losers))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python -m unittest tests.test_bracket -v
```

Expected: FAIL because `src.prediction.bracket` does not exist.

- [ ] **Step 3: Implement bracket helpers**

Create `src/prediction/bracket.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Optional


@dataclass
class Standing:
    group: str
    team: str
    points: int = 0
    goals_for: int = 0
    goals_against: int = 0
    elo: float = 1500.0

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against


def group_letter(group_name: str) -> str:
    return group_name.replace("Group", "").strip()


def rank_key(row: Standing) -> tuple:
    return (row.points, row.goal_difference, row.goals_for, row.elo, row.team)


def rank_group(rows: Iterable[Standing]) -> List[Standing]:
    return sorted(rows, key=rank_key, reverse=True)


def best_thirds(rows: Iterable[Standing], limit: int = 8) -> List[Standing]:
    return rank_group(rows)[:limit]


def parse_group_slot(slot: str) -> Optional[tuple[int, List[str]]]:
    m = re.fullmatch(r"([123])([A-L](?:/[A-L])*)", slot or "")
    if not m:
        return None
    return int(m.group(1)), m.group(2).split("/")


def resolve_group_slot(
    slot: str,
    ranked_groups: Dict[str, List[Standing]],
    selected_thirds: List[Standing],
) -> Optional[str]:
    parsed = parse_group_slot(slot)
    if parsed is None:
        return None
    position, groups = parsed
    if position in (1, 2) and len(groups) == 1:
        rows = ranked_groups.get(groups[0], [])
        return rows[position - 1].team if len(rows) >= position else None
    if position == 3:
        for third in selected_thirds:
            if third.group in groups:
                return third.team
    return None


def resolve_match_ref(ref: str, winners: Dict[int, str], losers: Dict[int, str]) -> Optional[str]:
    m = re.fullmatch(r"([WL])(\d+)", ref or "")
    if not m:
        return None
    num = int(m.group(2))
    return winners.get(num) if m.group(1) == "W" else losers.get(num)
```

- [ ] **Step 4: Run the tests and verify GREEN**

Run:

```bash
python -m unittest tests.test_bracket -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/prediction/bracket.py tests/test_bracket.py
git commit -m "Add schedule bracket helpers"
```

---

### Task 3: Add Schedule Simulation Model

**Files:**
- Create: `src/prediction/schedule_model.py`
- Test: `tests/test_schedule_model.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_schedule_model.py`:

```python
import unittest

from src.prediction.schedule_model import (
    completed_score,
    generate_schedule_predictions,
    next_scheduled_match,
)


class ScheduleModelTest(unittest.TestCase):
    def test_completed_score_reads_fixture_score_ft(self):
        self.assertEqual(completed_score({"score": {"ft": [2, 1]}}), (2, 1))
        self.assertIsNone(completed_score({"score": {"ft": [None, None]}}))
        self.assertIsNone(completed_score({}))

    def test_next_scheduled_match_uses_date_order(self):
        fixtures = [
            {"date": "2026-06-12", "time": "15:00 UTC-4", "team1": "Canada", "team2": "Qatar", "round": "Matchday 2", "group": "Group B"},
            {"date": "2026-06-11", "time": "13:00 UTC-6", "team1": "Mexico", "team2": "South Africa", "round": "Matchday 1", "group": "Group A"},
        ]
        match = next_scheduled_match(fixtures, today="2026-06-09")
        self.assertEqual(match["team1"], "Mexico")
        self.assertEqual(match["team2"], "South Africa")

    def test_generate_schedule_predictions_shape(self):
        fixtures = [
            {"date": "2026-06-11", "time": "13:00 UTC-6", "team1": "Mexico", "team2": "South Africa", "round": "Matchday 1", "group": "Group A"},
            {"date": "2026-06-11", "time": "20:00 UTC-6", "team1": "South Korea", "team2": "Czech Republic", "round": "Matchday 1", "group": "Group A"},
            {"date": "2026-06-18", "time": "12:00 UTC-4", "team1": "Czech Republic", "team2": "South Africa", "round": "Matchday 8", "group": "Group A"},
            {"date": "2026-06-18", "time": "19:00 UTC-6", "team1": "Mexico", "team2": "South Korea", "round": "Matchday 8", "group": "Group A"},
            {"date": "2026-06-24", "time": "19:00 UTC-6", "team1": "Czech Republic", "team2": "Mexico", "round": "Matchday 14", "group": "Group A"},
            {"date": "2026-06-24", "time": "19:00 UTC-6", "team1": "South Africa", "team2": "South Korea", "round": "Matchday 14", "group": "Group A"},
        ]
        ratings = {"Mexico": 1800, "South Africa": 1650, "South Korea": 1750, "Czech Republic": 1700}
        out = generate_schedule_predictions(fixtures=fixtures, ratings=ratings, n_sim=20, seed=7)
        self.assertEqual(out["source"], "schedule")
        self.assertEqual(out["next_match"]["team1"], "Mexico")
        self.assertEqual(len(out["matches"]), 6)
        self.assertGreater(out["matches"][0]["team1_win"], 0)
        self.assertIn("champion", out["teams"][0])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python -m unittest tests.test_schedule_model -v
```

Expected: FAIL because `src.prediction.schedule_model` does not exist.

- [ ] **Step 3: Implement schedule model**

Create `src/prediction/schedule_model.py` with these public functions:

```python
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
    return [m for m in fixtures if m.get("group") and is_concrete_team(m.get("team1", "")) and is_concrete_team(m.get("team2", ""))]


def next_scheduled_match(fixtures: Iterable[dict], today: Optional[str] = None) -> Optional[dict]:
    today = today or date.today().isoformat()
    candidates = [m for m in fixtures if is_concrete_team(m.get("team1", "")) and is_concrete_team(m.get("team2", "")) and completed_score(m) is None]
    future = [m for m in candidates if m.get("date", "") >= today]
    selected = sorted(future or candidates, key=fixture_sort_key)
    return selected[0] if selected else None


def predict_fixture(match: dict, ratings: Dict[str, float], model: Optional[dict]) -> dict:
    pred = predict_match(match["team1"], match["team2"], ratings=ratings, model=model)
    return {
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
```

Then add the simulation functions in the same file:

```python
def _sample_score(match: dict, ratings: Dict[str, float], model: Optional[dict], rng: np.random.Generator) -> tuple[int, int]:
    fixed = completed_score(match)
    if fixed is not None:
        return fixed
    pred = predict_match(match["team1"], match["team2"], ratings=ratings, model=model)
    return int(rng.poisson(pred.lambda_team1)), int(rng.poisson(pred.lambda_team2))


def _simulate_groups(fixtures: List[dict], ratings: Dict[str, float], model: Optional[dict], rng: np.random.Generator) -> Dict[str, List[Standing]]:
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


def _resolve_team_ref(ref: str, ranked_groups: Dict[str, List[Standing]], selected_thirds: List[Standing], winners: Dict[int, str], losers: Dict[int, str]) -> Optional[str]:
    direct = resolve_group_slot(ref, ranked_groups, selected_thirds)
    if direct:
        return direct
    ref_match = resolve_match_ref(ref, winners, losers)
    if ref_match:
        return ref_match
    return ref if is_concrete_team(ref) else None


def _play_knockout(fixtures: List[dict], ranked_groups: Dict[str, List[Standing]], ratings: Dict[str, float], model: Optional[dict], rng: np.random.Generator) -> Dict[str, set]:
    thirds = best_thirds([rows[2] for rows in ranked_groups.values() if len(rows) >= 3])
    winners: Dict[int, str] = {}
    losers: Dict[int, str] = {}
    reached = {"round_of_32": set(), "round_of_16": set(), "quarter": set(), "semi": set(), "final": set(), "champion": set()}
    knockout = [m for m in sorted(fixtures, key=fixture_sort_key) if not m.get("group")]
    for match in knockout:
        num = match.get("num")
        team1 = _resolve_team_ref(match.get("team1", ""), ranked_groups, thirds, winners, losers)
        team2 = _resolve_team_ref(match.get("team2", ""), ranked_groups, thirds, winners, losers)
        if not team1 or not team2 or num is None:
            continue
        round_name = match.get("round", "")
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
        g1, g2 = _sample_score({"team1": team1, "team2": team2, "score": match.get("score")}, ratings, model, rng)
        if g1 == g2:
            pred = predict_match(team1, team2, ratings=ratings, model=model)
            winner = team1 if rng.random() < pred.team1_win / max(pred.team1_win + pred.team2_win, 1e-9) else team2
        else:
            winner = team1 if g1 > g2 else team2
        loser = team2 if winner == team1 else team1
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
    concrete_matches = [m for m in fixtures if is_concrete_team(m.get("team1", "")) and is_concrete_team(m.get("team2", ""))]
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
```

- [ ] **Step 4: Run the tests and verify GREEN**

Run:

```bash
python -m unittest tests.test_schedule_model -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/prediction/schedule_model.py tests/test_schedule_model.py
git commit -m "Add schedule-driven tournament simulation"
```

---

### Task 4: Add Schedule Prediction Generator

**Files:**
- Create: `scripts/generate_schedule_predictions.py`
- Modify: `data/wc2026_schedule_predictions.json`
- Test: `tests/test_schedule_model.py`

- [ ] **Step 1: Add CLI smoke test**

Append to `tests/test_schedule_model.py`:

```python
    def test_save_schedule_predictions_writes_json(self):
        import json
        import tempfile
        from src.prediction.schedule_model import save_schedule_predictions

        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            payload = {"source": "schedule", "teams": [], "matches": [], "next_match": None}
            save_schedule_predictions(payload, tmp.name)
            loaded = json.load(open(tmp.name, encoding="utf-8"))
            self.assertEqual(loaded["source"], "schedule")
```

- [ ] **Step 2: Run the test and verify RED or targeted failure**

Run:

```bash
python -m unittest tests.test_schedule_model.ScheduleModelTest.test_save_schedule_predictions_writes_json -v
```

Expected: PASS if `save_schedule_predictions` from Task 3 is already present. If it fails, fix Task 3 before continuing.

- [ ] **Step 3: Add generator script**

Create `scripts/generate_schedule_predictions.py`:

```python
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.prediction.schedule_model import OUTPUT_PATH, generate_schedule_predictions, save_schedule_predictions


def main():
    parser = argparse.ArgumentParser(description="Generate schedule-driven 2026 World Cup predictions")
    parser.add_argument("--n-sim", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()
    payload = generate_schedule_predictions(n_sim=args.n_sim, seed=args.seed)
    save_schedule_predictions(payload, args.output)
    print(f"saved {args.output} teams={len(payload['teams'])} matches={len(payload['matches'])}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run generator**

Run:

```bash
python scripts/generate_schedule_predictions.py --n-sim 1000
```

Expected: `data/wc2026_schedule_predictions.json` is written and the command prints `saved ... teams=48 matches=72`.

- [ ] **Step 5: Commit**

Run:

```bash
git add scripts/generate_schedule_predictions.py data/wc2026_schedule_predictions.json tests/test_schedule_model.py
git commit -m "Generate schedule-driven prediction artifact"
```

---

### Task 5: Wire Schedule Predictions Into Mobile Dashboard

**Files:**
- Modify: `src/dashboard/mobile_ui.py`
- Test: `tests/test_mobile_schedule_integration.py`

- [ ] **Step 1: Write failing integration tests**

Create `tests/test_mobile_schedule_integration.py`:

```python
import unittest

import src.dashboard.mobile_ui as mobile_ui


class MobileScheduleIntegrationTest(unittest.TestCase):
    def test_schedule_prediction_loader_has_fallback_shape(self):
        payload = mobile_ui._load_schedule_predictions(path="/tmp/missing-wc-schedule-predictions.json")
        self.assertEqual(payload["source"], "fallback")
        self.assertEqual(payload["teams"], [])
        self.assertEqual(payload["matches"], [])
        self.assertIsNone(payload["next_match"])

    def test_html_has_schedule_prediction_placeholder(self):
        self.assertIn("__SCHEDULE_PRED__", mobile_ui.HTML_BODY)
        self.assertIn("var SP=__SCHEDULE_PRED__;", mobile_ui.HTML_BODY)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python -m unittest tests.test_mobile_schedule_integration -v
```

Expected: FAIL because `_load_schedule_predictions` and `__SCHEDULE_PRED__` are not present.

- [ ] **Step 3: Add Python loader and HTML injection**

Modify `src/dashboard/mobile_ui.py`:

```python
SCHEDULE_PRED = os.path.join(ROOT, "data", "wc2026_schedule_predictions.json")
```

Add near `_load_final_pred()`:

```python
def _load_schedule_predictions(path=SCHEDULE_PRED):
    """Load schedule-driven predictions generated by scripts/generate_schedule_predictions.py."""
    if not os.path.exists(path):
        return {"source": "fallback", "teams": [], "matches": [], "next_match": None}
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    payload.setdefault("source", "schedule")
    payload.setdefault("teams", [])
    payload.setdefault("matches", [])
    payload.setdefault("next_match", None)
    return payload
```

In the script header, change:

```javascript
var FN=__FINAL__;
var RT=__REALTIME__;
```

to:

```javascript
var FN=__FINAL__;
var SP=__SCHEDULE_PRED__;
var RT=__REALTIME__;
```

In `run_server`, add:

```python
schedule_pred = _load_schedule_predictions()
schedule_json = json.dumps(schedule_pred, ensure_ascii=False)
html = html.replace("__SCHEDULE_PRED__", schedule_json)
```

- [ ] **Step 4: Run tests and verify partial GREEN**

Run:

```bash
python -m unittest tests.test_mobile_schedule_integration -v
```

Expected: PASS.

- [ ] **Step 5: Wire champion card fallback order**

Modify `buildFinal()` JavaScript so the data source priority is:

1. `RT.teams`
2. `SP.teams`
3. `FN.teams`

Use this replacement structure inside `buildFinal()`:

```javascript
var rt=RT,teams,summary="",updated="",isRT=false,isSchedule=false;
if(rt&&rt.teams&&rt.teams.length){teams=rt.teams;summary=rt.summary||"";updated=rt.updated||"";isRT=true;}
else if(SP&&SP.teams&&SP.teams.length){teams=SP.teams;summary="真实赛程路径模拟";updated=SP.as_of||"";isSchedule=true;}
else{var f=(FN&&FN.teams)||[];if(f.length===0){if(card)card.style.display="none";return;}
  teams=f.map(function(t){return{country:t.country,champion:t.champion,base:(t.market!=null?t.market:t.model),delta:0,factors:[]};});}
```

Update the note block:

```javascript
if(isRT)note.innerHTML='<b style="color:var(--gd)">🔴 实时研判</b>（grok 联网综合伤病/状态/赔率）: '+summary+'<br>更新 '+updated+' · 点击球队展开实时因子 · 数据模型+市场+grok实时 三层融合';
else if(isSchedule)note.innerHTML='<b style="color:var(--gd)">赛程驱动</b>：按真实小组赛程、晋级路径和淘汰赛编号模拟。更新 '+updated+' · 基线不使用高比分boost';
else note.innerHTML="市场共识 + 数据模型融合（实时层加载中，几分钟后自动刷新）。截至 "+((FN&&FN.as_of)||"2026-06-08");
```

- [ ] **Step 6: Commit**

Run:

```bash
git add src/dashboard/mobile_ui.py tests/test_mobile_schedule_integration.py
git commit -m "Load schedule predictions in mobile dashboard"
```

---

### Task 6: Make H2H Default Schedule-Aware and Keep Manual Mode

**Files:**
- Modify: `src/dashboard/mobile_ui.py`
- Test: `tests/test_mobile_schedule_integration.py`

- [ ] **Step 1: Add failing H2H HTML checks**

Append to `tests/test_mobile_schedule_integration.py`:

```python
    def test_h2h_schedule_controls_exist(self):
        self.assertIn('id="h2h-match"', mobile_ui.HTML_BODY)
        self.assertIn("function populateScheduleH2H", mobile_ui.HTML_BODY)
        self.assertIn("function applyScheduleMatch", mobile_ui.HTML_BODY)
```

- [ ] **Step 2: Run test and verify RED**

Run:

```bash
python -m unittest tests.test_mobile_schedule_integration.MobileScheduleIntegrationTest.test_h2h_schedule_controls_exist -v
```

Expected: FAIL because controls/functions are not present.

- [ ] **Step 3: Add H2H schedule selector markup**

In `src/dashboard/mobile_ui.py`, insert above `<div class="h2h-teams">`:

```html
    <div class="sel-wrap" style="margin-bottom:12px">
      <select class="sel" id="h2h-match" onchange="applyScheduleMatch()"></select>
    </div>
```

- [ ] **Step 4: Add schedule H2H JavaScript**

Add before the init block:

```javascript
function scheduleH2HMatches(){
  var rows=(SP&&SP.matches)||[];
  return rows.filter(function(m){return m.team1&&m.team2&&m.date;}).sort(function(a,b){
    var ak=(a.date||"")+" "+(a.time||"");
    var bk=(b.date||"")+" "+(b.time||"");
    return ak<bk?-1:ak>bk?1:0;
  });
}
function populateScheduleH2H(){
  var sel=document.getElementById("h2h-match");
  if(!sel)return;
  var rows=scheduleH2HMatches();
  sel.innerHTML="";
  var manual=document.createElement("option");
  manual.value="manual";
  manual.textContent="Manual matchup / 手动选择球队";
  sel.appendChild(manual);
  for(var i=0;i<rows.length;i++){
    var m=rows[i];
    var opt=document.createElement("option");
    opt.value=String(i);
    opt.textContent=(m.date||"")+" · "+(m.round||"")+" · "+m.team1+" vs "+m.team2;
    sel.appendChild(opt);
  }
  if(rows.length>0)sel.value="0";
}
function applyScheduleMatch(){
  var sel=document.getElementById("h2h-match");
  if(!sel||sel.value==="manual")return;
  var rows=scheduleH2HMatches();
  var m=rows[parseInt(sel.value,10)];
  if(!m)return;
  document.getElementById("h2h-a").value=m.team1;
  document.getElementById("h2h-b").value=m.team2;
  updatePickCard("a",m.team1);
  updatePickCard("b",m.team2);
  h2hChange();
}
```

In the init block, after team selector population and before `h2hChange()`, call:

```javascript
populateScheduleH2H();
if(document.getElementById("h2h-match")&&document.getElementById("h2h-match").value!=="manual"){applyScheduleMatch();}
else if(teams.length>1){selA.value=teams[0].country;selB.value=teams[1].country;h2hChange();}
```

Remove the earlier unconditional:

```javascript
if(teams.length>1){selA.value=teams[0].country;selB.value=teams[1].country;}
h2hChange();
```

- [ ] **Step 5: Run test and verify GREEN**

Run:

```bash
python -m unittest tests.test_mobile_schedule_integration -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/dashboard/mobile_ui.py tests/test_mobile_schedule_integration.py
git commit -m "Default H2H to scheduled matches"
```

---

### Task 7: Remove High-Score Boost From Most-Likely Scorelines

**Files:**
- Modify: `src/dashboard/mobile_ui.py`
- Test: `tests/test_mobile_schedule_integration.py`

- [ ] **Step 1: Add failing source check**

Append to `tests/test_mobile_schedule_integration.py`:

```python
    def test_most_likely_uses_base_probability(self):
        body = mobile_ui.HTML_BODY
        self.assertIn("baseProb", body)
        self.assertIn("boostedProb", body)
        self.assertNotIn("raw[i].prob = raw[i].boosted / sumBoosted", body)
```

- [ ] **Step 2: Run test and verify RED**

Run:

```bash
python -m unittest tests.test_mobile_schedule_integration.MobileScheduleIntegrationTest.test_most_likely_uses_base_probability -v
```

Expected: FAIL because current JS assigns `raw[i].prob` from boosted probability.

- [ ] **Step 3: Modify `buildScorePred` probabilities**

In `buildScorePred`, keep `pois`, but compute two probabilities:

```javascript
var sumBase = 0;
var sumBoosted = 0;
for (var i = 0; i < raw.length; i++) {
    raw[i].boosted = raw[i].total >= EXTREME_THRESH ? raw[i].pois * BOOST_FACTOR : raw[i].pois;
    sumBase += raw[i].pois;
    sumBoosted += raw[i].boosted;
}
for (var i = 0; i < raw.length; i++) {
    raw[i].baseProb = raw[i].pois / sumBase;
    raw[i].boostedProb = raw[i].boosted / sumBoosted;
    raw[i].prob = raw[i].baseProb;
}
```

Change the high-score section to sort and display `boostedProb`:

```javascript
var hiAll = raw.filter(function(x){ return x.total >= EXTREME_THRESH; });
hiAll.sort(function(a, b){ return b.boostedProb - a.boostedProb; });
var topHi = hiAll.slice(0, 8);
var hiTotal = topHi.reduce(function(s, x){ return s + x.boostedProb; }, 0);
```

In high-score rendering, change:

```javascript
var pct2 = (s3.prob * 100).toFixed(1);
```

to:

```javascript
var pct2 = (s3.boostedProb * 100).toFixed(1);
```

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
python -m unittest tests.test_mobile_schedule_integration -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/dashboard/mobile_ui.py tests/test_mobile_schedule_integration.py
git commit -m "Keep H2H most-likely scorelines unboosted"
```

---

### Task 8: Full Verification

**Files:**
- No new files.

- [ ] **Step 1: Run all unit tests**

Run:

```bash
python -m unittest discover -s tests -v
```

Expected: PASS.

- [ ] **Step 2: Regenerate schedule predictions**

Run:

```bash
python scripts/generate_schedule_predictions.py --n-sim 3000
```

Expected: writes `data/wc2026_schedule_predictions.json` and prints nonzero team/match counts.

- [ ] **Step 3: Smoke import dashboard**

Run:

```bash
python - <<'PY'
import src.dashboard.mobile_ui as m
print(bool(m._load_schedule_predictions()["teams"]))
print("__SCHEDULE_PRED__" in m.HTML_BODY)
PY
```

Expected first line `True`; second line `True` before runtime replacement.

- [ ] **Step 4: Commit regenerated artifact if changed**

Run:

```bash
git status --short
git add data/wc2026_schedule_predictions.json
git commit -m "Refresh schedule-driven prediction artifact"
```

If `git status --short` shows no change to `data/wc2026_schedule_predictions.json`, skip this commit and note that the artifact was already current.

---

## Plan Self-Review

- Spec coverage:
  - H2H defaults to real scheduled match: Task 6.
  - Champion probabilities from fixture-order simulation: Tasks 3, 4, 5.
  - Fixtures, H2H, champion card share one source: Tasks 4, 5, 6.
  - France vs Brazil default only if scheduled: Task 6.
  - Deterministic output: Tasks 3 and 8 use seed.
  - Most-likely scorelines unboosted: Task 7.

- Placeholder scan:
  - No deferred work markers are used.
  - Fixture slot terms refer to real bracket strings like `1A`, `W73`, and `3A/B/C/D/F`.

- Type consistency:
  - `MatchPrediction` fields are consumed by `schedule_model.predict_fixture`.
  - `Standing.group` uses group letters, matching `resolve_group_slot`.
  - `wc2026_schedule_predictions.json` fields match dashboard `SP.teams`, `SP.matches`, and `SP.next_match`.
