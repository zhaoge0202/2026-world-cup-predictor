import unittest
import inspect

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

    def test_h2h_keeps_single_probability_display(self):
        body = mobile_ui.HTML_BODY

        self.assertIn('id="h2h-bar-a"', body)
        self.assertIn('id="h2h-bar-d"', body)
        self.assertIn('id="h2h-bar-b"', body)
        self.assertNotIn('id="h2h-pa"', body)
        self.assertNotIn('id="h2h-pd"', body)
        self.assertNotIn('id="h2h-pb"', body)

    def test_h2h_probability_bar_widths_are_percentage_based(self):
        body = mobile_ui.HTML_BODY

        self.assertIn(".h2h-bar-a{flex:0 0 auto;", body)
        self.assertIn(".h2h-bar-d{flex:0 0 auto;", body)
        self.assertIn(".h2h-bar-b{flex:0 0 auto;", body)
        self.assertIn('style.width=barA+"%"', body)
        self.assertIn('style.width=barD+"%"', body)
        self.assertIn('style.width=barB+"%"', body)

    def test_h2h_uses_selected_schedule_prediction_values(self):
        body = mobile_ui.HTML_BODY
        self.assertIn("function selectedSchedulePrediction", body)
        self.assertIn("function scheduleH2HCalc", body)
        self.assertIn("team1_win", body)
        self.assertIn("lambda_team1", body)
        self.assertIn("top_scores", body)
        self.assertIn("Schedule model / 赛程模型", body)

    def test_most_likely_uses_base_probability(self):
        body = mobile_ui.HTML_BODY
        self.assertIn("baseProb", body)
        self.assertIn("boostedProb", body)
        self.assertNotIn("raw[i].prob = raw[i].boosted / sumBoosted", body)

    def test_h2h_uses_schedule_match_prediction_when_selected(self):
        body = mobile_ui.HTML_BODY
        self.assertIn("var _selectedScheduleMatch=null;", body)
        self.assertIn("scheduleMatchPrediction()", body)
        self.assertIn("buildScheduleScorePred", body)
        self.assertIn("team1_win", body)
        self.assertIn("lambda_team1", body)
        self.assertIn("top_scores", body)

    def test_schedule_predictions_refresh_api_is_wired(self):
        body = mobile_ui.HTML_BODY
        self.assertIn("function refreshSchedulePredictions", body)
        self.assertIn('fetch("/api/schedule_predictions"', body)
        self.assertIn("setInterval(refreshSchedulePredictions", body)
        self.assertEqual(mobile_ui.SCHEDULE_PRED_REFRESH_SECONDS, 60)
        self.assertTrue(hasattr(mobile_ui, "_start_schedule_prediction_daemon"))
        self.assertIn("/api/schedule_predictions", inspect.getsource(mobile_ui.run_server))

    def test_live_match_prediction_api_is_wired(self):
        source = inspect.getsource(mobile_ui.run_server)

        self.assertIn("/api/live_match_prediction", source)
        self.assertIn("match_id", source)
        self.assertIn("build_live_match_prediction", source)
        self.assertIn("function refreshLiveMatchPrediction", mobile_ui.HTML_BODY)
        self.assertIn('"/api/live_match_prediction?match_id="', mobile_ui.HTML_BODY)
        self.assertIn('"/api/live_match_prediction?match_num="', mobile_ui.HTML_BODY)
        self.assertIn('"/api/live_match_prediction?home="', mobile_ui.HTML_BODY)
        self.assertIn("fetch(url", mobile_ui.HTML_BODY)
        self.assertIn("_liveSchedulePredictions", mobile_ui.HTML_BODY)
        self.assertIn("function scheduleMatchKey", mobile_ui.HTML_BODY)

    def test_team_analysis_refresh_api_is_wired(self):
        source = inspect.getsource(mobile_ui.run_server)

        self.assertTrue(hasattr(mobile_ui, "_refresh_analysis_state"))
        self.assertTrue(hasattr(mobile_ui, "_start_analysis_refresh_daemon"))
        self.assertIn("/api/team_analysis", source)
        self.assertIn("function setUpdateTime", mobile_ui.HTML_BODY)
        self.assertIn("function refreshTeamAnalysis", mobile_ui.HTML_BODY)
        self.assertIn('fetch("/api/team_analysis"', mobile_ui.HTML_BODY)
        self.assertIn("setUpdateTime();", mobile_ui.HTML_BODY)
        self.assertIn("setInterval(refreshTeamAnalysis,300000)", mobile_ui.HTML_BODY)

    def test_schedule_and_realtime_refresh_update_header_time(self):
        body = mobile_ui.HTML_BODY

        self.assertIn("setUpdateTime();", body)
        self.assertIn("function setUpdateTime(value)", body)
        self.assertIn("timeZone:'Asia/Shanghai'", body)
        self.assertIn("setInterval(setUpdateTime,60000)", body)
        self.assertNotIn("toISOString()).replace", body)

    def test_realtime_refresh_backend_uses_ten_minute_ttl(self):
        source = inspect.getsource(mobile_ui._start_realtime_daemon)

        self.assertIn("CHAMPION_TTL", source)
        self.assertNotIn("time.sleep(3600)", source)
        self.assertNotIn("6h", source)

    def test_realtime_api_triggers_background_refresh_when_stale(self):
        source = inspect.getsource(mobile_ui.run_server)

        self.assertIn("_fresh", source)
        self.assertIn("CHAMPION_TTL", source)
        self.assertIn("adjust_champion_probs", source)

    def test_refresh_analysis_state_replaces_json(self):
        state = {}

        mobile_ui._refresh_analysis_state(state, loader=lambda: ([{"country": "Mexico"}], {"Mexico": {}}))

        self.assertEqual(state["analysis"][0]["country"], "Mexico")
        self.assertIn('"country": "Mexico"', state["data_json"])
        self.assertIn('"Mexico"', state["ucl_json"])

    def test_live_match_helpers_find_schedule_and_score(self):
        schedule = {
            "matches": [{"match_id": "fixture:2026-06-11|13:00 UTC-6|Mexico|South Africa", "num": 1, "team1": "Mexico", "team2": "South Africa"}],
            "next_match": {"num": 2, "team1": "Canada", "team2": "Qatar"},
        }
        scores = [{"team_home": "South Africa", "team_away": "Mexico", "status": "LIVE"}]

        match = mobile_ui._find_schedule_match(schedule, match_num="1")

        self.assertEqual(match["team1"], "Mexico")
        self.assertEqual(mobile_ui._find_live_score(scores, match), scores[0])

        by_id = mobile_ui._find_schedule_match(schedule, match_id="fixture:2026-06-11|13:00 UTC-6|Mexico|South Africa")
        self.assertEqual(by_id["team2"], "South Africa")

    def test_schedule_match_lookup_does_not_fallback_when_explicit_key_misses(self):
        schedule = {"matches": [], "next_match": {"num": 2, "team1": "Canada", "team2": "Qatar"}}

        self.assertIsNone(mobile_ui._find_schedule_match(schedule, match_id="missing"))
        self.assertIsNone(mobile_ui._find_schedule_match(schedule, match_num="1"))
        self.assertIsNone(mobile_ui._find_schedule_match(schedule, home="Mexico", away="South Africa"))
        self.assertEqual(mobile_ui._find_schedule_match(schedule)["team1"], "Canada")


if __name__ == "__main__":
    unittest.main()
