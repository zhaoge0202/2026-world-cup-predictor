import unittest

from src.prediction.live_state_model import build_live_match_prediction, load_referee_profiles


BASE_MATCH = {
    "num": 1,
    "team1": "Mexico",
    "team2": "South Africa",
    "team1_win": 0.737,
    "draw": 0.165,
    "team2_win": 0.098,
    "lambda_team1": 2.37,
    "lambda_team2": 0.74,
    "top_scores": [{"score": "2-0", "team1_goals": 2, "team2_goals": 0, "prob": 0.126}],
}


class LiveStateModelTest(unittest.TestCase):
    def test_leading_late_increases_live_win_probability(self):
        live = {
            "team_home": "Mexico",
            "team_away": "South Africa",
            "score_home": 1,
            "score_away": 0,
            "minute": "75",
            "status": "LIVE",
        }

        pred = build_live_match_prediction(BASE_MATCH, live)

        self.assertEqual(pred["source"], "live-state")
        self.assertGreater(pred["team1_win"], BASE_MATCH["team1_win"])
        self.assertAlmostEqual(pred["team1_win"] + pred["draw"] + pred["team2_win"], 1.0, places=6)
        self.assertGreaterEqual(pred["expected_final_score"]["team1"], 1)
        self.assertEqual(pred["current_score"], {"team1": 1, "team2": 0})

    def test_red_card_penalizes_the_shorthanded_team(self):
        level_live = {
            "team_home": "Mexico",
            "team_away": "South Africa",
            "score_home": 0,
            "score_away": 0,
            "minute": 35,
            "status": "IN_PLAY",
        }
        no_red = build_live_match_prediction(BASE_MATCH, level_live)
        home_red = build_live_match_prediction(BASE_MATCH, {**level_live, "red_home": 1, "red_away": 0})

        self.assertLess(home_red["team1_win"], no_red["team1_win"])
        self.assertGreater(home_red["team2_win"], no_red["team2_win"])
        self.assertIn("red_card", home_red["adjustments"])

    def test_realtime_xg_changes_remaining_goal_expectation_when_available(self):
        level_live = {
            "team_home": "Mexico",
            "team_away": "South Africa",
            "score_home": 0,
            "score_away": 0,
            "minute": 30,
            "status": "LIVE",
        }
        no_xg = build_live_match_prediction(BASE_MATCH, level_live)
        high_home_xg = build_live_match_prediction(
            BASE_MATCH,
            {**level_live, "xg_home": 1.8, "xg_away": 0.2},
        )

        self.assertGreater(high_home_xg["lambda_team1"], no_xg["lambda_team1"])
        self.assertLess(high_home_xg["lambda_team2"], no_xg["lambda_team2"])
        self.assertIn("xg", high_home_xg["adjustments"])

    def test_missing_referee_profiles_load_as_empty_data(self):
        self.assertEqual(load_referee_profiles(path="/tmp/missing-referee-profiles.json"), {})


if __name__ == "__main__":
    unittest.main()
