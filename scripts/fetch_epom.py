"""Retrieve the reference project's upstream EPOM checkpoint with pinned hashes.

This is not the VORL paper's independently trained checkpoint. Provenance is
recorded explicitly so users cannot confuse upstream compatibility with exact
paper-result reproduction. Requires only the Python standard library.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request
import zipfile

URL = "https://github.com/Cognitive-AI-Systems/when-to-switch/releases/download/v0/weights.zip"
ARCHIVE_SHA256 = "cbbf44bce9dc8b59ce4f6059c565c76567d024845d0378286fc862ead6f8b727"
CHECKPOINT_NAME = "checkpoint_000311682_1000002674.pth"
CHECKPOINT_SHA256 = "549feac19e21593af072677305945d7c22bd7f66cb07927a92eb59f2f8a3cce9"


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def fetch(output, archive=None):
    output = Path(output).resolve()
    if archive is not None and not Path(archive).is_file():
        raise FileNotFoundError("--archive must name an existing input file")
    # Refuse to replace existing user data, including partial previous runs.
    if output.exists():
        raise FileExistsError(f"Use a new empty output path: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="epom-fetch-", dir=output.parent) as scratch:
        archive = Path(archive).resolve() if archive else Path(scratch) / "weights.zip"
        if not archive.exists():
            request = urllib.request.Request(URL, headers={"User-Agent": "VORL-reproduction-intake"})
            with urllib.request.urlopen(request, timeout=60) as response, archive.open("xb") as destination:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    destination.write(block)
        if file_hash(archive) != ARCHIVE_SHA256:
            raise ValueError("Upstream archive hash mismatch; refusing extraction")
        extracted = Path(scratch) / "verified"
        extracted.mkdir()
        selected = {
            "weights/epom/cfg.json": "cfg.json",
            f"weights/epom/checkpoint_p0/{CHECKPOINT_NAME}": CHECKPOINT_NAME,
        }
        with zipfile.ZipFile(archive) as package:
            for name, local_name in selected.items():
                (extracted / local_name).write_bytes(package.read(name))
        if file_hash(extracted / CHECKPOINT_NAME) != CHECKPOINT_SHA256:
            raise ValueError("Policy checkpoint hash mismatch")
        provenance = {
            "source_url": URL, "archive_sha256": ARCHIVE_SHA256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "config_sha256": file_hash(extracted / "cfg.json"),
            "provenance": "When to Switch upstream EPOM; not claimed VORL paper training",
        }
        (extracted / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        # Directory promotion happens only after all artifact checks pass.
        extracted.rename(output)
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--archive", type=Path, help="Use an already downloaded, hash-checked release archive")
    args = parser.parse_args()
    print(json.dumps(fetch(args.output, args.archive), indent=2))
