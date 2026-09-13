"""Verify strict recurrent-policy loading and deterministic reset, not exploration success."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import require_new_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        require_new_output(args.output)
    except ValueError as error:
        parser.error(str(error))
    from vorl.policy import EpomPolicy

    policy = EpomPolicy(args.config, args.checkpoint, seed=17)
    width = 2 * policy.radius + 1
    observations = np.zeros((4, 3, width, width), dtype=np.float32)
    observations[:, 2, policy.radius, -1] = 1
    coordinates = np.zeros((4, 2), dtype=np.float32)
    targets = np.tile([0, 10], (4, 1)).astype(np.float32)
    first, logits = policy.act(observations, coordinates, targets)
    second, second_logits = policy.act(observations, coordinates, targets)
    policy.reset(17)
    repeated, repeated_logits = policy.act(observations, coordinates, targets)
    assert np.array_equal(first, repeated)
    assert np.array_equal(logits, repeated_logits)
    assert logits.shape == (4, 5) and np.all(np.isfinite(second_logits))
    result = {**policy.provenance, "status": "policy_loading_check_passed", "main_method_reproduced": False,
              "first_actions": first.tolist(), "second_actions": second.tolist(),
              "logits_shape": list(logits.shape), "deterministic_reset": True,
              "recurrent_logits_changed": not np.array_equal(logits, second_logits),
              "scope": "Synthetic observations for loading/inference validation, not a rollout"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
