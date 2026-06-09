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
