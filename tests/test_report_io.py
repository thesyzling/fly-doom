"""Preserve complete reports when Windows briefly locks the destination."""

import json
from pathlib import Path

import pytest

from flydoom import calibration


@pytest.mark.parametrize("release", [True, False])
def test_report_replace_retries_without_truncating_previous_report(tmp_path, monkeypatch, release):
    target = tmp_path / "report.json"
    target.write_text('{"status": "previous"}', encoding="utf-8")
    replace = Path.replace
    attempts, delays = [], []

    def locked_replace(source, destination):
        attempts.append(destination)
        assert json.loads(target.read_text(encoding="utf-8"))["status"] == "previous"
        if not release or len(attempts) < 3:
            raise PermissionError("Sharing violation fixture")
        return replace(source, destination)

    monkeypatch.setattr(Path, "replace", locked_replace)
    monkeypatch.setattr(calibration, "sleep", delays.append)
    if release:
        calibration.write_json(target, {"status": "new"})
        assert len(attempts) == 3
        assert json.loads(target.read_text(encoding="utf-8"))["status"] == "new"
    else:
        with pytest.raises(PermissionError):
            calibration.write_json(target, {"status": "new"})
        assert len(attempts) == 6
        assert json.loads(target.read_text(encoding="utf-8"))["status"] == "previous"
    assert len(delays) == len(attempts) - 1
