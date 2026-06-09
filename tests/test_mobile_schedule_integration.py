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

    def test_h2h_schedule_controls_exist(self):
        self.assertIn('id="h2h-match"', mobile_ui.HTML_BODY)
        self.assertIn("function populateScheduleH2H", mobile_ui.HTML_BODY)
        self.assertIn("function applyScheduleMatch", mobile_ui.HTML_BODY)

    def test_most_likely_uses_base_probability(self):
        body = mobile_ui.HTML_BODY
        self.assertIn("baseProb", body)
        self.assertIn("boostedProb", body)
        self.assertNotIn("raw[i].prob = raw[i].boosted / sumBoosted", body)


if __name__ == "__main__":
    unittest.main()
