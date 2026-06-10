"""Development hot-reload runner for the mobile dashboard.

The production HTTP server intentionally stays simple. This module runs it as a
child process and restarts that child when source files change.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
WATCH_SUFFIXES = {".py", ".html", ".css", ".js", ".json", ".md", ".toml", ".yaml", ".yml"}
WATCH_ROOTS = {"src", "scripts", "config.py", "requirements.txt", "README.md"}
IGNORED_DIRS = {".git", ".mypy_cache", ".pytest_cache", "__pycache__", ".venv", "venv", "node_modules"}
IGNORED_DATA_FILES = {
    "data/realtime_cache.json",
    "data/wc2026_fixtures.json",
    "data/wc2026_schedule_predictions.json",
}


def _relative(path: Path, root: Path = ROOT) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def should_watch_path(path: Path, root: Path = ROOT) -> bool:
    """Return whether a path should trigger a development server restart."""
    rel = _relative(path, root)
    parts = set(Path(rel).parts)
    if parts & IGNORED_DIRS:
        return False
    if rel in IGNORED_DATA_FILES:
        return False
    if path.suffix not in WATCH_SUFFIXES:
        return False
    return rel.split("/", 1)[0] in WATCH_ROOTS or rel in WATCH_ROOTS


def discover_watch_files(root: Path = ROOT) -> list[Path]:
    """Discover files whose edits should restart the child server."""
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and should_watch_path(path, root):
            files.append(path)
    return sorted(files)


def snapshot_files(paths: Iterable[Path]) -> dict[Path, tuple[int, int] | None]:
    """Capture mtime/size for watched files. Missing files are represented by None."""
    snap: dict[Path, tuple[int, int] | None] = {}
    for path in paths:
        try:
            st = path.stat()
            snap[path] = (st.st_mtime_ns, st.st_size)
        except FileNotFoundError:
            snap[path] = None
    return snap


def snapshot_changed(before: dict[Path, tuple[int, int] | None],
                     after: dict[Path, tuple[int, int] | None]) -> bool:
    return before != after


def describe_snapshot_changes(
    before: dict[Path, tuple[int, int] | None],
    after: dict[Path, tuple[int, int] | None],
    root: Path = ROOT,
) -> list[str]:
    """Return compact human-readable file changes between two snapshots."""
    details: list[str] = []
    for path in sorted(set(before) | set(after), key=lambda p: _relative(p, root)):
        old = before.get(path)
        new = after.get(path)
        if old == new:
            continue
        if old is None and new is not None:
            action = "added"
        elif old is not None and new is None:
            action = "removed"
        else:
            action = "modified"
        details.append(f"{action} {_relative(path, root)}")
    return details


def build_server_command(port: int) -> list[str]:
    return [sys.executable, "-m", "src.dashboard.mobile_ui", "--port", str(port)]


def _start_child(port: int, root: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    return subprocess.Popen(build_server_command(port), cwd=root, env=env)


def _stop_child(proc: subprocess.Popen, timeout: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout)


def run(port: int = 7862, interval: float = 1.0, root: Path = ROOT) -> None:
    """Run the mobile dashboard and restart it when watched files change."""
    print(f"Hot reload: http://localhost:{port}")
    print("Watching src/, scripts/, config.py, requirements.txt, README.md")
    child = _start_child(port, root)
    watched = discover_watch_files(root)
    snapshot = snapshot_files(watched)
    try:
        while True:
            time.sleep(interval)
            next_watched = discover_watch_files(root)
            next_snapshot = snapshot_files(next_watched)
            if child.poll() is not None:
                print("Server exited; restarting...")
                child = _start_child(port, root)
                watched = next_watched
                snapshot = next_snapshot
                continue
            if watched != next_watched or snapshot_changed(snapshot, next_snapshot):
                changes = describe_snapshot_changes(snapshot, next_snapshot, root=root)
                if changes:
                    print(f"Change detected ({'; '.join(changes[:5])}); restarting server...")
                else:
                    print("Change detected; restarting server...")
                _stop_child(child)
                child = _start_child(port, root)
                watched = next_watched
                snapshot = next_snapshot
    except KeyboardInterrupt:
        print("Stopping hot reload server...")
    finally:
        _stop_child(child)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run mobile UI with development hot reload.")
    parser.add_argument("--port", type=int, default=7862, help="HTTP port for the mobile UI.")
    parser.add_argument("--interval", type=float, default=1.0, help="File polling interval in seconds.")
    args = parser.parse_args(argv)
    run(port=args.port, interval=args.interval)


if __name__ == "__main__":
    main()
