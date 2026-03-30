"""
Tests for Starlette 1.0 compatibility.

Ensures no deprecated Starlette APIs are used (on_event, @app.route, etc.)
that were removed in Starlette 1.0.

Last Updated: 2026-03-30

Run with: uv run pytest tests/unit/test_starlette_compat.py -v
"""

import ast
import os
import time
from pathlib import Path

import pytest

# All router modules that get registered with the FastAPI app
_ROUTER_DIR = Path(__file__).parent.parent.parent / "web" / "routers"


class TestNoDeprecatedOnEvent:
    """Starlette 1.0 removed on_event(). Ensure no router uses it."""

    def _find_on_event_calls(self, filepath: Path) -> list[int]:
        """Return line numbers where .on_event() is called in the AST."""
        source = filepath.read_text()
        tree = ast.parse(source, filename=str(filepath))
        hits = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "on_event"
            ):
                hits.append(node.lineno)
        return hits

    def test_no_on_event_in_routers(self):
        violations = {}
        for py_file in sorted(_ROUTER_DIR.glob("*.py")):
            if py_file.name.startswith("__"):
                continue
            hits = self._find_on_event_calls(py_file)
            if hits:
                violations[py_file.name] = hits

        assert violations == {}, (
            f"Deprecated on_event() found (removed in Starlette 1.0): {violations}. "
            "Move startup logic to server.py:main() or use a lifespan handler."
        )

    def test_no_on_event_in_server(self):
        server_py = _ROUTER_DIR.parent / "server.py"
        hits = self._find_on_event_calls(server_py)
        assert hits == [], (
            f"Deprecated on_event() in server.py at lines {hits}. "
            "Use lifespan or call directly in main()."
        )


class TestCleanupOldVideos:
    """Test the video cleanup function works correctly."""

    @pytest.fixture
    def video_dir(self, tmp_path):
        video_dir = tmp_path / "videos"
        video_dir.mkdir()

        (video_dir / "new_video.mp4").write_bytes(b"fake mp4 data")

        old_video = video_dir / "old_video.mp4"
        old_video.write_bytes(b"old mp4 data")
        old_thumb = video_dir / "old_video.png"
        old_thumb.write_bytes(b"old thumb data")
        # Set mtime to 48 hours ago
        old_time = time.time() - (48 * 3600)
        os.utime(old_video, (old_time, old_time))
        os.utime(old_thumb, (old_time, old_time))

        return video_dir

    def test_cleanup_deletes_old_videos(self, video_dir, monkeypatch):
        from web.routers.ltx2 import cleanup_old_videos

        monkeypatch.setattr("web.routers.ltx2.VIDEO_OUTPUT_DIR", video_dir)

        deleted = cleanup_old_videos(max_age_hours=24)

        assert deleted == 1
        assert (video_dir / "new_video.mp4").exists()
        assert not (video_dir / "old_video.mp4").exists()
        assert not (video_dir / "old_video.png").exists()

    def test_cleanup_returns_zero_when_nothing_old(self, video_dir, monkeypatch):
        from web.routers.ltx2 import cleanup_old_videos

        monkeypatch.setattr("web.routers.ltx2.VIDEO_OUTPUT_DIR", video_dir)

        deleted = cleanup_old_videos(max_age_hours=72)
        assert deleted == 0
        assert len(list(video_dir.glob("*.mp4"))) == 2
