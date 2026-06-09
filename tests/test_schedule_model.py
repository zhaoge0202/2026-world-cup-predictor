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
            {
                "date": "2026-06-12",
                "time": "15:00 UTC-4",
                "team1": "Canada",
                "team2": "Qatar",
                "round": "Matchday 2",
                "group": "Group B",
            },
            {
                "date": "2026-06-11",
                "time": "13:00 UTC-6",
                "team1": "Mexico",
                "team2": "South Africa",
                "round": "Matchday 1",
                "group": "Group A",
            },
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
