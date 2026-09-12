"""Inspect a reference archive without importing it or deserializing checkpoints."""
import argparse
import ast
import hashlib
import json
import pickletools
import zipfile
from pathlib import Path


def checkpoint_strings(path):
    """Read pickle opcodes as data; never execute torch.load or pickle.load."""
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith("/data.pkl")]
        if len(names) != 1:
            raise ValueError(f"Expected one data.pkl in {path.name}")
        data = archive.read(names[0])
    return [arg for _, arg, _ in pickletools.genops(data) if isinstance(arg, str)]


def inspect_reference(root):
    root = Path(root).resolve()
    if not (root / "CoExMIX3.py").is_file():
        raise ValueError("Reference directory must contain CoExMIX3.py")
    syntax_errors = []
    sources = sorted(root.rglob("*.py"))
    for path in sources:
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            syntax_errors.append({"file": path.relative_to(root).as_posix(), "error": str(exc)})
    weights = []
    for path in sorted(root.rglob("*.pth")):
        strings = checkpoint_strings(path)
        weights.append({
            "file": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
            "has_value_head_names": any("value_head" in s for s in strings),
            "has_actor_names": any("action_parameterization" in s or "action_distribution" in s for s in strings),
            "has_recurrent_core_names": any("core.rnn" in s or "core.core.weight_hh" in s for s in strings),
        })
    policy_dir = root / "weights/epom/checkpoint_p0"
    policy_files = sorted(p.name for p in policy_dir.glob("*.pth"))
    return {
        "status": "reference_intake_only",
        "paper_main_method_verified": False,
        "python_files": len(sources), "syntax_errors": syntax_errors,
        "checkpoints": weights, "default_policy_files": policy_files,
        "missing_artifacts": [
            *([] if policy_files else ["EPOM recurrent actor policy checkpoint"]),
            "Final fidelity gate implementation, warm-start checkpoint and evaluation settings",
        ],
        "note": "Parameter names indicate roles, not checkpoint compatibility. No pickle executed.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect_reference(args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    # A failed readiness check is intentional, not a successful main-method run.
    raise SystemExit(2 if result["missing_artifacts"] or result["syntax_errors"] else 0)


if __name__ == "__main__":
    main()
