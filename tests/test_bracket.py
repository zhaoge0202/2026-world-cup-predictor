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
        thirds = [
            Standing(chr(65 + i), f"T{i}", points=i, goals_for=i, goals_against=0, elo=1500 + i)
            for i in range(12)
        ]

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
