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
