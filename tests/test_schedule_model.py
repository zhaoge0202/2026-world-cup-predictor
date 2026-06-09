import unittest

from src.prediction.schedule_model import (
    completed_score,
    fixture_match_id,
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
        self.assertEqual(out["next_match"]["match_id"], fixture_match_id(fixtures[0]))
        self.assertEqual(len(out["matches"]), 6)
        self.assertTrue(all(m.get("match_id") for m in out["matches"]))
        self.assertGreater(out["matches"][0]["team1_win"], 0)
        self.assertIn("champion", out["teams"][0])

    def test_fixture_match_id_uses_num_when_available(self):
        fixture = {"num": 12, "date": "2026-06-11", "team1": "Mexico", "team2": "South Africa"}

        self.assertEqual(fixture_match_id(fixture), "num:12")

    def test_fixture_match_id_falls_back_to_fixture_fields(self):
        fixture = {"date": "2026-06-11", "time": "13:00 UTC-6", "team1": "Mexico", "team2": "South Africa"}

        self.assertEqual(fixture_match_id(fixture), "fixture:2026-06-11|13:00 UTC-6|Mexico|South Africa")

    def test_unnumbered_final_still_counts_champion(self):
        fixtures = [
            {"date": "2026-06-11", "time": "13:00 UTC-6", "team1": "Mexico", "team2": "South Africa", "round": "Matchday 1", "group": "Group A"},
            {"date": "2026-07-19", "team1": "Mexico", "team2": "South Africa", "round": "Final"},
        ]
        ratings = {"Mexico": 1800, "South Africa": 1650}

        out = generate_schedule_predictions(fixtures=fixtures, ratings=ratings, n_sim=20, seed=9)

        champion_total = sum(row["champion"] for row in out["teams"])
        self.assertGreater(champion_total, 0.9)

    def test_save_schedule_predictions_writes_json(self):
        import json
        import tempfile

        from src.prediction.schedule_model import save_schedule_predictions

        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            payload = {"source": "schedule", "teams": [], "matches": [], "next_match": None}
            save_schedule_predictions(payload, tmp.name)
            with open(tmp.name, encoding="utf-8") as f:
                loaded = json.load(f)

        self.assertEqual(loaded["source"], "schedule")


if __name__ == "__main__":
    unittest.main()
