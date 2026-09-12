"""Run the coupled VORL method reconstruction with a fitted gate and real EPOM."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vorl.runner import run_episode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/reconstruction.json"))
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--size", type=int, choices=(40, 80), default=40)
    parser.add_argument("--robots", type=int)
    parser.add_argument("--dynamic-obstacles", type=int, default=8)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--adapt", action="store_true", help="Separate adaptation mode; disabled for main runs")
    args = parser.parse_args()
    result = run_episode(config=json.loads(args.config.read_text(encoding="utf-8")),
                         policy_dir=args.policy_dir, gate=json.loads(args.gate.read_text(encoding="utf-8")),
                         seed=args.seed, size=args.size, robots=args.robots or (4 if args.size == 40 else 16),
                         dynamic_obstacles=args.dynamic_obstacles, density=0.3,
                         horizon=args.horizon or (320 if args.size == 40 else 480),
                         output=args.output, threads=args.threads, collect=False, adapt=args.adapt)
    print(json.dumps(result, indent=2))
