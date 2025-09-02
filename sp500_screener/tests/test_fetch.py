import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import time
from pathlib import Path

from core import fetch


def test_disk_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "CACHE_DIR", Path(tmp_path))
    fetch.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @fetch.disk_cache(ttl_hours=1)
    def _dummy(ticker: str):
        return {"ts": time.time()}

    first = _dummy("TEST")
    second = _dummy("TEST")
    assert first["ts"] == second["ts"]
