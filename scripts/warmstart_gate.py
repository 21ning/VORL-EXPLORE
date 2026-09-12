"""Collect real 64x64/64-robot rollouts, then fit the eight-feature logistic gate."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vorl.fidelity import FEATURE_NAMES, sigmoid
from vorl.runner import canonical_hash, run_episode, source_hash


def collect_one(arguments):
    with threadpool_limits(arguments.pop("threads")) as _:
        # Torch threads are explicitly passed separately through this private key.
        arguments["threads"] = arguments.pop("torch_threads")
        return run_episode(**arguments)


def fit_gate(features, quality, config):
    keep = np.abs(quality) >= config["quality_margin"]
    features = np.asarray(features[keep], dtype=np.float64)
    labels = (quality[keep] >= 0).astype(np.float64)
    if len(labels) < 1000 or np.unique(labels).size != 2:
        raise ValueError("Need at least 1000 clear-margin samples containing both quality classes")
    weights, bias = np.zeros(8), 0.0
    rng = np.random.default_rng(20260912)
    learning_rate, l2 = config["gate_learning_rate"], config["gate_l2"]
    def loss():
        logits = features @ weights + bias
        return float(np.mean(np.logaddexp(0, logits) - labels * logits) + l2 / 2 * (weights @ weights))
    initial_loss = loss()
    with threadpool_limits(1):
        for _ in range(config["gate_fit_epochs"]):
            order = rng.permutation(len(labels))
            for start in range(0, len(order), config["gate_fit_batch_size"]):
                indices = order[start:start + config["gate_fit_batch_size"]]
                batch = features[indices]
                residual = sigmoid(batch @ weights + bias) - labels[indices]
                weights -= learning_rate * ((batch.T @ residual) / len(indices) + l2 * weights)
                bias -= learning_rate * float(np.mean(residual))
        final_loss = loss()
    if not np.isfinite(final_loss) or final_loss >= initial_loss:
        raise RuntimeError("Warm-start optimization did not reduce training loss")
    return weights, bias, {"selected_samples": len(labels), "positive_samples": int(labels.sum()),
                            "negative_samples": int(len(labels) - labels.sum()),
                            "initial_training_loss": initial_loss, "final_training_loss": final_loss,
                            "metric_scope": "Optimization checks only; not held-out task performance"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/reconstruction.json"))
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=512)
    args = parser.parse_args()
    if args.workers < 1 or args.threads < 1 or args.workers * args.threads > 32:
        parser.error("Keep total requested compute at or below 32 CPU threads")
    seeds = [int(s) for s in args.seeds.split(",")]
    if len(set(seeds)) != len(seeds) or len(seeds) < 2:
        parser.error("Use at least two distinct warm-start seeds")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=False)
    cold_gate = {"status": "untrained_collection_initialization", "weights": [0.0] * 8, "bias": 0.0}
    jobs = []
    for seed in seeds:
        density = float(np.random.default_rng(seed).uniform(0.15, 0.45))
        jobs.append(dict(config=config, policy_dir=str(args.policy_dir.resolve()), gate=cold_gate,
                         seed=seed, size=64, robots=64, dynamic_obstacles=32, density=density,
                         horizon=args.horizon, output=str(args.output.resolve() / f"seed-{seed}"),
                         threads=args.threads, torch_threads=args.threads, collect=True, adapt=False))
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(collect_one, job): job["seed"] for job in jobs}
        for future in as_completed(futures):
            summary = future.result()
            print(json.dumps({"completed_seed": futures[future], "steps": summary["steps"],
                              "wall_seconds": summary["wall_seconds"]}), flush=True)
    feature_parts, quality_parts, hashes = [], [], {}
    for seed in seeds:
        path = args.output / f"seed-{seed}" / "gate-data.npz"
        with np.load(path, allow_pickle=False) as data:
            feature_parts.append(data["features"])
            quality_parts.append(data["quality"])
        hashes[f"seed-{seed}/gate-data.npz"] = hashlib.sha256(path.read_bytes()).hexdigest()
    features, quality = np.concatenate(feature_parts), np.concatenate(quality_parts)
    weights, bias, fit_metrics = fit_gate(features, quality, config)
    checkpoint = {
        "status": "fitted_from_self_supervised_rollouts" if args.horizon >= 512 else "debug_fit_not_for_evaluation",
        "profile": config["profile"], "weights": weights.tolist(), "bias": bias,
        "feature_names": list(FEATURE_NAMES), "training_seeds": seeds,
        "training_grid_size": 64, "training_robots": 64, "training_horizon": args.horizon,
        "training_dynamic_obstacles": 32,
        "training_density": "one seeded Uniform(0.15, 0.45) draw per episode",
        "initialization": "zero weights for data collection; pooled logistic BCE fitting on actual delayed outcomes",
        "transfer": "shared warm-start coefficients copied to each evaluation robot; main evaluation is frozen",
        "config_sha256": canonical_hash(config), "method_source_sha256": source_hash(),
        "data_sha256": hashes, **fit_metrics,
        "caveat": "Newly fitted reconstruction gate, not the original paper checkpoint",
    }
    (args.output / "fidelity-gate.json").write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checkpoint, indent=2), flush=True)


if __name__ == "__main__":
    main()
