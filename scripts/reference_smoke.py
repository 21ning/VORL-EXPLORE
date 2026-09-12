"""Run the supplied classical modules and record an explicitly non-VORL GIF.

This diagnostic imports the unmodified archive's A* and Voronoi functions.
It uses a small, static, fully specified harness, not the paper simulator or
the original CoExMIX3 experiment. It cannot validate EPOM or the paper gate.
"""
import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def make_map():
    grid = np.zeros((24, 24), dtype=np.uint8)
    grid[[0, -1], :] = 1
    grid[:, [0, -1]] = 1
    grid[2:22, 8] = 1
    grid[2:22, 15] = 1
    grid[6:9, 8] = 0
    grid[16:19, 15] = 0
    grid[12, 3:6] = 1
    return grid


def observe(truth, known, positions, radius=3):
    for r, c in positions:
        rows = slice(max(0, r - radius), min(truth.shape[0], r + radius + 1))
        cols = slice(max(0, c - radius), min(truth.shape[1], c + radius + 1))
        known[rows, cols] = truth[rows, cols]


def safe_step(truth, positions, proposals):
    """Conservative synchronous filter: occupied-at-start cells are blocked."""
    occupied = set(positions)
    counts = {p: proposals.count(p) for p in proposals}
    next_positions = []
    for old, proposed in zip(positions, proposals):
        r, c = proposed
        valid = (0 <= r < truth.shape[0] and 0 <= c < truth.shape[1]
                 and truth[r, c] == 0 and counts[proposed] == 1
                 and (proposed == old or proposed not in occupied)
                 and abs(r - old[0]) + abs(c - old[1]) <= 1)
        next_positions.append(proposed if valid else old)
    assert len(set(next_positions)) == len(next_positions)
    return next_positions


def render_frame(truth, known, positions, goals, trails, step):
    cell, top = 18, 78
    img = Image.new("RGB", (truth.shape[1] * cell, top + truth.shape[0] * cell + 32), "#101827")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=14)
    draw.text((12, 10), "REFERENCE DIAGNOSTIC | A* ONLY", fill="#ffffff", font=font)
    draw.text((12, 32), "Not VORL-EXPLORE main-method reproduction", fill="#f5c76a", font=ImageFont.load_default(size=12))
    coverage = float(np.mean(known != 255))
    draw.text((12, 52), f"Step {step:03d}   Observed {coverage:.1%}   Static map", fill="#bbc8da", font=ImageFont.load_default(size=12))
    for r in range(truth.shape[0]):
        for c in range(truth.shape[1]):
            color = "#687386" if known[r, c] == 255 else ("#253044" if known[r, c] else "#edf2f7")
            draw.rectangle((c * cell, top + r * cell, (c + 1) * cell - 1, top + (r + 1) * cell - 1), fill=color)
    colors = ["#168ce8", "#e88d1b", "#a343cf", "#18976c"]
    for i, (pos, goal) in enumerate(zip(positions, goals)):
        color = colors[i % len(colors)]
        points = [(c * cell + cell // 2, top + r * cell + cell // 2) for r, c in trails[i][-30:]]
        if len(points) > 1:
            draw.line(points, fill=color, width=2)
        if goal is not None:
            gr, gc = goal
            x, y = gc * cell + cell // 2, top + gr * cell + cell // 2
            draw.line((x - 5, y, x + 5, y), fill=color, width=2)
            draw.line((x, y - 5, x, y + 5), fill=color, width=2)
        r, c = pos
        x, y = c * cell + cell // 2, top + r * cell + cell // 2
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline="white", width=1)
        draw.text((x - 3, y - 6), str(i + 1), fill="white", font=ImageFont.load_default(size=10))
    draw.text((12, img.height - 23), "Gray: unknown   +: goal   Line: recent actual path", fill="#bbc8da", font=ImageFont.load_default(size=12))
    return img


def run(reference, output, max_steps=96):
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    reference = Path(reference).resolve()
    if not (reference / "planner/Astar.py").is_file():
        raise ValueError("Reference root must contain planner/Astar.py")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(reference))
    astar = importlib.import_module("planner.Astar").a_star_search
    voronoi = importlib.import_module("targetalgo.Voronoi").improved_voronoi_assignment
    truth = make_map()
    known = np.full_like(truth, 255)
    positions = [(2, 2), (21, 2), (2, 21), (21, 21)]
    trails = [[pos] for pos in positions]
    observe(truth, known, positions)
    frames, records = [], []
    goals = [None] * len(positions)
    path_length = 0
    reason = "horizon"
    for step in range(max_steps + 1):
        goals = voronoi(positions, known, current_targets=goals, agent_repulsion=1)
        goals = [tuple(int(v) for v in g) if g is not None else None for g in goals]
        records.append({"step": step, "positions": positions, "goals": goals, "known_map": known.tolist()})
        frames.append(render_frame(truth, known, positions, goals, trails, step))
        if np.all(known != 255):
            reason = "all_cells_observed"
            break
        if step == max_steps:
            break
        # The reference A* accepts unknown cells. This harness explicitly blocks
        # them so that the diagnostic does not plan through unobserved space.
        traversable = np.where(known == 0, 0, 1).astype(np.uint8)
        proposals = []
        for pos, goal in zip(positions, goals):
            path = astar(traversable, pos, goal) if goal is not None else None
            proposals.append(path[1] if path and len(path) > 1 else pos)
        next_positions = safe_step(truth, positions, proposals)
        path_length += sum(a != b for a, b in zip(positions, next_positions))
        positions = next_positions
        for trail, pos in zip(trails, positions):
            trail.append(pos)
        observe(truth, known, positions)
    trajectory = {"method": "reference_astar_diagnostic", "static_map": truth.tolist(), "frames": records}
    trajectory_path = output / "trajectory.json"
    trajectory_path.write_text(json.dumps(trajectory, separators=(",", ":")) + "\n", encoding="utf-8")
    frames[0].save(output / "reference-astar.gif", save_all=True, append_images=frames[1:], duration=110, loop=0, disposal=2)
    frames[-1].save(output / "preview.png")
    with Image.open(output / "reference-astar.gif") as gif:
        gif_frames = gif.n_frames
        for index in range(gif_frames):
            gif.seek(index)
            gif.load()
    source_hashes = {name: hashlib.sha256((reference / name).read_bytes()).hexdigest()
                     for name in ["planner/Astar.py", "targetalgo/Voronoi.py"]}
    result = {
        "status": "diagnostic_completed", "main_method_reproduced": False,
        "method": "reference_astar_diagnostic", "map_size": 24, "agents": 4,
        "dynamic_obstacles": 0, "seed": None, "map_generation": "fixed_fixture",
        "steps": records[-1]["step"], "stop_reason": reason,
        "observed_cell_fraction": float(np.mean(known != 255)),
        "total_moved_steps": path_length, "trajectory_frames": len(records), "gif_frames": gif_frames,
        "trajectory_sha256": hashlib.sha256(trajectory_path.read_bytes()).hexdigest(),
        "source_sha256": source_hashes,
        "caveat": "Harness diagnostic only: no EPOM, fidelity gate, dynamic obstacles, or paper metrics.",
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-steps", default=96, type=int)
    args = parser.parse_args()
    run(args.reference, args.output, args.max_steps)
