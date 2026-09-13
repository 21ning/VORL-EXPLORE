"""Command-line contracts without downloading weights or running experiments."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("script", ["run_vorl", "fetch_epom", "warmstart_gate", "verify_rollout", "render_rollout", "check_policy"])
def test_help_from_another_directory(script, tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / f"{script}.py"), "--help"],
                            cwd=tmp_path, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize("option,value", [
    ("--robots", "0"), ("--robots", "-1"), ("--horizon", "0"),
    ("--threads", "0"), ("--seed", "-1"), ("--dynamic-obstacles", "-1"),
])
def test_invalid_run_arguments_do_not_create_output(option, value, tmp_path, monkeypatch):
    run = load_script("run_vorl", monkeypatch)
    output = tmp_path / "run"
    with pytest.raises(SystemExit) as error:
        run.main([option, value, "--output", str(output)])
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("size,robots,horizon", [("40", 4, 320), ("80", 16, 480)])
def test_defaults_are_checkout_relative_and_forwarded(size, robots, horizon, tmp_path, monkeypatch):
    run = load_script("run_vorl", monkeypatch)
    model = tmp_path / "model"
    model.mkdir()
    (model / "cfg.json").write_text("{}")
    (model / "checkpoint_fixture.pth").write_bytes(b"not a real policy; runner is mocked")
    monkeypatch.chdir(tmp_path)
    # Capture the function call without serializing its Path arguments.
    recorded = {}
    def runner(**kwargs):
        recorded.update(kwargs)
        return {"status": "unit-test"}
    monkeypatch.setitem(sys.modules, "vorl.runner", SimpleNamespace(run_episode=runner))
    run.main(["--size", size, "--policy-dir", str(model), "--output", "run", "--dynamic-obstacles", "0"])
    assert recorded["robots"] == robots and recorded["horizon"] == horizon
    assert recorded["dynamic_obstacles"] == 0
    assert recorded["gate"]["status"] == "fitted_from_self_supervised_rollouts"
    assert not recorded["adapt"] and not recorded["collect"]


def test_missing_policy_gives_setup_hint_and_leaves_no_output(tmp_path, monkeypatch, capsys):
    run = load_script("run_vorl", monkeypatch)
    output = tmp_path / "run"
    with pytest.raises(SystemExit):
        run.main(["--policy-dir", str(tmp_path / "missing"), "--output", str(output)])
    assert "fetch_epom.py" in capsys.readouterr().err
    assert not output.exists()


def test_existing_output_is_preserved(tmp_path, monkeypatch):
    common = load_script("_common", monkeypatch)
    marker = tmp_path / "keep.txt"
    marker.write_text("user data")
    with pytest.raises(ValueError, match="already exists"):
        common.require_new_output(tmp_path)
    assert marker.read_text() == "user data"


@pytest.mark.parametrize("content", ["{invalid", "[]", "null"])
def test_invalid_json_is_rejected(content, tmp_path, monkeypatch):
    common = load_script("_common", monkeypatch)
    path = tmp_path / "config.json"
    path.write_text(content)
    with pytest.raises(ValueError):
        common.load_json(path)


def test_warmstart_rejects_oversubscribed_workers_before_loading_policy(tmp_path, monkeypatch):
    warmstart = load_script("warmstart_gate", monkeypatch)
    output = tmp_path / "run"
    with pytest.raises(SystemExit) as error:
        warmstart.main(["--workers", "8", "--threads", "8", "--output", str(output)])
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("text", ["0", "0,0", "0,-1", "a,1", "0,"])
def test_invalid_seed_lists(text, monkeypatch):
    import argparse
    common = load_script("_common", monkeypatch)
    with pytest.raises(argparse.ArgumentTypeError):
        common.seed_list(text)


@pytest.mark.parametrize("key,value", [("assignment_interval", 0), ("history_window", -1),
                                       ("gate_learning_rate", float("nan")), ("quality_weights", [1, 2]),
                                       ("gate", {}), ("sensing_radius", True)])
def test_invalid_configuration_is_rejected(key, value, monkeypatch):
    common = load_script("_common", monkeypatch)
    config = common.load_json(common.DEFAULT_CONFIG)
    config[key] = value
    with pytest.raises(ValueError):
        common.validate_config(config)


def test_supplied_gate_matches_config_and_unchanged_method():
    gate = json.loads((ROOT / "checkpoints/fidelity-gate.json").read_text())
    config = json.loads((ROOT / "configs/reconstruction.json").read_text())
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    digest = hashlib.sha256()
    for source in sorted((ROOT / "vorl").glob("*.py")):
        digest.update(source.name.encode())
        digest.update(source.read_bytes())
    assert gate["config_sha256"] == config_hash
    assert gate["method_source_sha256"] == digest.hexdigest()
