"""Run VORL-EXPLORE in a dynamic grid environment."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import (
    DEFAULT_CONFIG, DEFAULT_GATE, DEFAULT_POLICY, load_json,
    nonnegative_int, positive_int, require_new_output, require_policy, validate_config,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Method configuration")
    parser.add_argument("--policy-dir", type=Path, default=DEFAULT_POLICY, help="EPOM model directory")
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE, help="Fitted fidelity checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=nonnegative_int, default=1000)
    parser.add_argument("--size", type=int, choices=(40, 80), default=40)
    parser.add_argument("--robots", type=positive_int, help="Team size; 4 for size 40, 16 for size 80")
    parser.add_argument("--dynamic-obstacles", type=nonnegative_int, default=8)
    parser.add_argument("--horizon", type=positive_int, help="Step limit; 320 for size 40, 480 for size 80")
    parser.add_argument("--threads", type=positive_int, default=4)
    parser.add_argument("--adapt", action="store_true", help="Enable online gate updates")
    args = parser.parse_args(argv)
    try:
        config, gate = load_json(args.config), load_json(args.gate)
        validate_config(config)
        require_new_output(args.output)
        require_policy(args.policy_dir)
    except ValueError as error:
        parser.error(str(error))
    # Validate input before importing the ML runtime or creating output files.
    from vorl.runner import run_episode

    result = run_episode(config=config, policy_dir=args.policy_dir, gate=gate,
                         seed=args.seed, size=args.size,
                         robots=args.robots if args.robots is not None else (4 if args.size == 40 else 16),
                         dynamic_obstacles=args.dynamic_obstacles, density=0.3,
                         horizon=args.horizon if args.horizon is not None else (320 if args.size == 40 else 480),
                         output=args.output, threads=args.threads, collect=False, adapt=args.adapt)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
