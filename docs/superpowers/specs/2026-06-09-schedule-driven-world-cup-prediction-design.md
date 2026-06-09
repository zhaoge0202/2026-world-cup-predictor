# Schedule-Driven World Cup Prediction Design

## Goal

Make the predictor follow the real 2026 World Cup schedule instead of treating
head-to-head matchups, champion probabilities, and fixtures as separate views.

The prediction chain should be:

`fixtures -> group standings -> knockout path -> champion probabilities -> match predictions`

This covers both requested surfaces:

- H2H defaults to real scheduled matches while keeping manual matchup analysis.
- Champion probabilities are generated from real group fixtures and a stable
  knockout path, not from shuffled groups or arbitrary top-team matchups.

## Current Problem

The mobile dashboard currently injects several independent data sources:

- `D=__DATA__`: model-derived team list from `_load_analysis()`.
- `F=__FIXTURES__`: fixture cache used by the fixtures tab and standings view.
- `FN=__FINAL__`: final champion prediction JSON.

The H2H tab sorts `D` by `final_prob` and defaults to the top two teams. This
can produce France vs Brazil even when that matchup is not the next scheduled
match. The champion model also has code paths that shuffle groups or knockout
participants, so it is not a transparent fixture-order forecast.

## Proposed Architecture

Add a schedule-driven prediction module under `src/prediction/`:

- `schedule_model.py`
  - Loads fixtures from `data/wc2026_fixtures.json`.
  - Loads team strength inputs from existing Elo/team scoring sources.
  - Produces fixture-level predictions, simulated standings, knockout results,
    and champion probabilities.

- `match_model.py`
  - Provides a small, reusable Elo-Poisson API for one match.
  - Returns win/draw/loss probabilities, expected goals, and unboosted scoreline
    probabilities.

- `bracket.py`
  - Encodes progression rules: group standings, best thirds, and knockout
    pairing.
  - Uses fixture bracket metadata when available.
  - Falls back to a deterministic 32-team pairing rule when fixture data only
    contains placeholders.

The dashboard reads one combined output:

- `data/wc2026_schedule_predictions.json`

This file becomes the source of truth for schedule-based match predictions and
champion probabilities.

## Data Flow

1. Load fixtures in chronological order.
2. For completed matches, use recorded scores from fixture/live score data.
3. For unplayed group matches, predict with Elo-Poisson and sample outcomes for
   simulation.
4. Build group standings after every group match.
5. Select 32 knockout teams:
   - 12 group winners
   - 12 group runners-up
   - 8 best third-place teams
6. Resolve knockout matches:
   - If fixture team names are concrete, use them.
   - If fixture uses placeholders, fill from deterministic progression slots.
7. Repeat the tournament simulation many times.
8. Write:
   - per-match expected goals, W/D/L, and top scorelines
   - group qualification probabilities
   - round reach probabilities
   - champion probabilities

## H2H Behavior

The H2H tab changes from "top two teams by probability" to schedule-aware mode:

- Default card: next scheduled match based on current date and fixture order.
- Match selector: list real scheduled matches grouped by date/round.
- Manual mode: still allows arbitrary Team A vs Team B comparison.

The H2H scoreline display uses unboosted Poisson probabilities for "Most
Likely". The high-score section remains a separate speculative area and must not
change the most-likely list.

## Champion Probability Behavior

The champion leaderboard uses `wc2026_schedule_predictions.json` when available.
If that file is missing or stale, the dashboard can fall back to the existing
model, but the UI should label the fallback clearly.

The generated champion rows include:

- `country`
- `champion`
- `final`
- `semi`
- `quarter`
- `round_of_16` or `round_of_32`
- `source: "schedule"`
- `as_of`

Existing market/realtime overlays can continue to apply on top of this base,
but they should treat schedule-driven probabilities as the baseline.

## Determinism

All simulations use a fixed seed by default so refreshes are reproducible.

No knockout-stage random shuffle is allowed. If the real bracket cannot be
fully inferred from the fixture source, the fallback pairing rule must be stable
and documented in code.

## Error Handling

- Missing fixtures: dashboard falls back to existing model and shows a source
  note.
- Missing Elo for a team: use a documented default Elo and record the missing
  team in metadata.
- Unknown placeholders: keep the match unresolved until bracket fill can infer
  teams.
- Completed match score conflicts between cache and live score provider: prefer
  fixture cache for persisted tournament state; live scores are display-only
  unless explicitly written back.

## Testing

Add focused tests for:

- group standings ordering
- best-third selection
- deterministic knockout pairing
- match prediction normalization
- champion probability output shape
- H2H default selecting the next scheduled match

Use the repo convention from `AGENTS.md`: run Python tests with `python`, not
`python3`.

## Non-Goals

- Do not add new external APIs.
- Do not remove manual H2H.
- Do not redesign the whole mobile UI.
- Do not make high-score boosted probabilities drive the most-likely scoreline.
- Do not rely on live news or market data for the base schedule simulation.

## Acceptance Criteria

- The H2H default matchup is a real scheduled match.
- Champion probabilities come from a fixture-order simulation output.
- The fixtures tab, H2H tab, and champion card can all point back to the same
  schedule prediction source.
- France vs Brazil appears as a default only if it is the next scheduled match
  or selected manually.
- Re-running the generator with the same data and seed produces the same output.
