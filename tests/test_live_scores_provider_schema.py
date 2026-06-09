import unittest

from scripts.live_scores_provider import _normalize_football_data, _normalize_thesportsdb


class LiveScoresProviderSchemaTest(unittest.TestCase):
    def test_thesportsdb_normalization_includes_optional_live_model_fields(self):
        row = _normalize_thesportsdb({"strHomeTeam": "Mexico", "strAwayTeam": "South Africa"})

        for key in ("red_home", "red_away", "yellow_home", "yellow_away", "xg_home", "xg_away", "referee"):
            self.assertIn(key, row)

        self.assertIsNone(row["xg_home"])
        self.assertIsNone(row["red_home"])

    def test_football_data_normalization_extracts_referee_when_available(self):
        row = _normalize_football_data({
            "homeTeam": {"name": "Mexico"},
            "awayTeam": {"name": "South Africa"},
            "score": {"fullTime": {"home": 0, "away": 0}},
            "referees": [{"name": "Test Referee"}],
        })

        self.assertEqual(row["referee"], "Test Referee")
        self.assertIsNone(row["xg_away"])
        self.assertIn("red_away", row)


if __name__ == "__main__":
    unittest.main()
