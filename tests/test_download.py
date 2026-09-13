"""Offline download-integrity tests using explicitly synthetic archive fixtures."""
import hashlib
import importlib.util
from pathlib import Path
import zipfile

import pytest

spec = importlib.util.spec_from_file_location("fetch_epom", Path(__file__).resolve().parents[1] / "scripts/fetch_epom.py")
fetcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetcher)


@pytest.fixture
def archive(tmp_path, monkeypatch):
    checkpoint, config = b"synthetic test checkpoint", b'{"test_fixture":true}'
    path = tmp_path / "fixture.zip"
    with zipfile.ZipFile(path, "w") as package:
        package.writestr(f"weights/epom/checkpoint_p0/{fetcher.CHECKPOINT_NAME}", checkpoint)
        package.writestr("weights/epom/cfg.json", config)
        package.writestr("../../must-not-extract.txt", "unselected member")
    monkeypatch.setattr(fetcher, "ARCHIVE_SHA256", fetcher.file_hash(path))
    monkeypatch.setattr(fetcher, "CHECKPOINT_SHA256", hashlib.sha256(checkpoint).hexdigest())
    monkeypatch.setattr(fetcher, "CONFIG_SHA256", hashlib.sha256(config).hexdigest())
    return path


def test_offline_fetch_checks_hashes_and_extracts_only_selected_members(archive, tmp_path):
    output = tmp_path / "model"
    result = fetcher.fetch(output, archive)
    assert result["checkpoint_sha256"] == fetcher.CHECKPOINT_SHA256
    assert {p.name for p in output.iterdir()} == {"cfg.json", fetcher.CHECKPOINT_NAME, "provenance.json"}
    assert not (tmp_path / "must-not-extract.txt").exists()


@pytest.mark.parametrize("hash_name", ["ARCHIVE_SHA256", "CHECKPOINT_SHA256", "CONFIG_SHA256"])
def test_bad_hash_never_promotes_partial_model(archive, tmp_path, monkeypatch, hash_name):
    monkeypatch.setattr(fetcher, hash_name, "0" * 64)
    output = tmp_path / "model"
    with pytest.raises(ValueError, match="hash mismatch"):
        fetcher.fetch(output, archive)
    assert not output.exists()
    assert not list(tmp_path.glob("epom-fetch-*"))


def test_existing_model_is_not_overwritten(archive, tmp_path):
    output = tmp_path / "model"
    output.mkdir()
    (output / "keep.txt").write_text("user data")
    with pytest.raises(FileExistsError):
        fetcher.fetch(output, archive)
    assert (output / "keep.txt").read_text() == "user data"
