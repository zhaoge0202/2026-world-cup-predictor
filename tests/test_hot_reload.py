import os
import sys
import tempfile
import unittest
from pathlib import Path

from src.dashboard import hot_reload


class HotReloadTest(unittest.TestCase):
    def test_watch_files_include_source_but_skip_runtime_caches(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "src" / "dashboard" / "mobile_ui.py"
            cache = root / "data" / "realtime_cache.json"
            ignored = root / "__pycache__" / "mobile_ui.pyc"
            src.parent.mkdir(parents=True)
            cache.parent.mkdir(parents=True)
            ignored.parent.mkdir(parents=True)
            src.write_text("print('ok')", encoding="utf-8")
            cache.write_text("{}", encoding="utf-8")
            ignored.write_text("compiled", encoding="utf-8")

            watched = hot_reload.discover_watch_files(root)

        self.assertIn(src, watched)
        self.assertNotIn(cache, watched)
        self.assertNotIn(ignored, watched)

    def test_snapshot_detects_modified_and_deleted_files(self):
        with tempfile.TemporaryDirectory() as td:
            watched = Path(td) / "app.py"
            watched.write_text("v1", encoding="utf-8")
            before = hot_reload.snapshot_files([watched])

            watched.write_text("v2", encoding="utf-8")
            after_write = hot_reload.snapshot_files([watched])
            watched.unlink()
            after_delete = hot_reload.snapshot_files([watched])

        self.assertTrue(hot_reload.snapshot_changed(before, after_write))
        self.assertTrue(hot_reload.snapshot_changed(before, after_delete))

    def test_describe_snapshot_changes_names_changed_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            changed = root / "src" / "dashboard" / "mobile_ui.py"
            added = root / "scripts" / "new_task.py"
            changed.parent.mkdir(parents=True)
            added.parent.mkdir(parents=True)
            changed.write_text("v1", encoding="utf-8")
            before = hot_reload.snapshot_files([changed])

            changed.write_text("v2", encoding="utf-8")
            added.write_text("print('new')", encoding="utf-8")
            after = hot_reload.snapshot_files([changed, added])

        details = hot_reload.describe_snapshot_changes(before, after, root=root)

        self.assertIn("modified src/dashboard/mobile_ui.py", details)
        self.assertIn("added scripts/new_task.py", details)

    def test_build_server_command_uses_current_python_and_port(self):
        cmd = hot_reload.build_server_command(port=9123)

        self.assertEqual(cmd[0], sys.executable)
        self.assertEqual(cmd[1:3], ["-m", "src.dashboard.mobile_ui"])
        self.assertIn("--port", cmd)
        self.assertIn("9123", cmd)

    def test_mobile_ui_has_port_cli_entrypoint(self):
        self.assertTrue(hasattr(hot_reload, "main"))
        self.assertTrue(hasattr(__import__("src.dashboard.mobile_ui", fromlist=["main"]), "main"))

    def test_should_watch_path_filters_generated_files(self):
        self.assertTrue(hot_reload.should_watch_path(Path("src/dashboard/mobile_ui.py")))
        self.assertTrue(hot_reload.should_watch_path(Path("config.py")))
        self.assertFalse(hot_reload.should_watch_path(Path("data/realtime_cache.json")))
        self.assertFalse(hot_reload.should_watch_path(Path("src/__pycache__/x.pyc")))
        self.assertFalse(hot_reload.should_watch_path(Path(".git/index")))


if __name__ == "__main__":
    unittest.main()
