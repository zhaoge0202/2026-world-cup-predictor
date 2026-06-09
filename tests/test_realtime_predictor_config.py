import unittest

from scripts import realtime_predictor


class RealtimePredictorConfigTest(unittest.TestCase):
    def test_champion_realtime_cache_ttl_is_ten_minutes(self):
        self.assertEqual(realtime_predictor.CHAMPION_TTL, 10 * 60)


if __name__ == "__main__":
    unittest.main()
