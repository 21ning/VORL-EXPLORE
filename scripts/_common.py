"""Shared command-line validation; independent of ML dependencies."""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "reconstruction.json"
DEFAULT_GATE = ROOT / "checkpoints" / "fidelity-gate.json"
DEFAULT_POLICY = ROOT / "models" / "epom"


def nonnegative_int(value):
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def positive_int(value):
    number = nonnegative_int(value)
    if number == 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def seed_list(value):
    try:
        seeds = [nonnegative_int(item.strip()) for item in value.split(",")]
    except argparse.ArgumentTypeError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated nonnegative integers") from error
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("use at least two distinct seeds")
    return seeds


def load_json(path):
    try:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read JSON file {path}: {error}") from error
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return result


def require_new_output(path):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ValueError(f"Output already exists; choose a new directory: {path}")
    for parent in path.parents:
        if parent.exists():
            if not parent.is_dir():
                raise ValueError(f"Output parent is not a directory: {parent}")
            break


def require_policy(path):
    path = Path(path)
    checkpoints = [p for p in path.glob("checkpoint_*.pth") if p.is_file()]
    if not (path / "cfg.json").is_file() or len(checkpoints) != 1:
        raise ValueError(
            f"Expected cfg.json and one checkpoint_*.pth in {path}. "
            "Download them with: python scripts/fetch_epom.py --output models/epom"
        )


def validate_config(config):
    """Reject malformed settings before loading a policy or starting workers."""
    def number(data, key, *, positive=False, integer=False):
        value = data.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0 or (positive and value == 0)
                or (integer and not isinstance(value, int))):
            raise ValueError(f"Invalid configuration value: {key}")
        return value

    if not isinstance(config.get("profile"), str) or not config["profile"]:
        raise ValueError("Configuration requires a profile name")
    for key in ("sensing_radius", "interaction_radius", "assignment_interval", "history_window",
                "recovery_steps", "recovery_cooldown", "oscillation_switches", "update_interval",
                "gate_fit_epochs", "gate_fit_batch_size"):
        number(config, key, positive=True, integer=True)
    for key in ("quality_margin", "gate_learning_rate"):
        number(config, key, positive=True)
    number(config, "gate_l2")
    for key in ("assignment", "gate"):
        if not isinstance(config.get(key), dict):
            raise ValueError(f"Configuration requires an object: {key}")
    assignment = config["assignment"]
    for key in ("lambda0", "lambda1", "rho0", "rho1", "interaction_radius", "beta"):
        number(assignment, key)
    for key in ("sigma_position", "sigma_goal"):
        number(assignment, key, positive=True)
    if assignment["interaction_radius"] != config["interaction_radius"]:
        raise ValueError("The two interaction_radius settings must agree")
    gate = config["gate"]
    low, high = number(gate, "low"), number(gate, "high")
    if not low < high <= 1:
        raise ValueError("Gate thresholds must satisfy 0 <= low < high <= 1")
    number(gate, "dwell", positive=True, integer=True)
    if not isinstance(gate.get("initial_planner"), bool):
        raise ValueError("initial_planner must be a boolean")
    weights = config.get("quality_weights")
    if not isinstance(weights, list) or len(weights) != 4:
        raise ValueError("quality_weights requires four positive numbers")
    for value in weights:
        number({"quality_weights": value}, "quality_weights", positive=True)
