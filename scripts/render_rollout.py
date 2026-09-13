"""Render an actual method-evaluation trajectory, never a synthetic replacement."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vorl.grid import frontiers

COLORS = ["#249bea", "#e68a24", "#ad62d8", "#18a787", "#e160a1", "#6b79dd", "#aa9633", "#4aabaa",
          "#ef6b48", "#727ed0", "#aa7394", "#75a12b", "#bc6734", "#258a6d", "#2d73a8", "#9f5daa"]


def render_frame(data, summary, index):
    size, robots = summary["size"], summary["robots"]
    cell = 12 if size == 40 else 8
    pad, gap, top, bottom = 20, 28, 100, 112
    panel = size * cell
    width, height = 2 * panel + 2 * pad + gap, top + panel + bottom
    image = Image.new("RGB", (width, height), "#101827")
    draw = ImageDraw.Draw(image)
    title = ImageFont.load_default(size=22)
    font = ImageFont.load_default(size=14)
    small = ImageFont.load_default(size=12)
    draw.text((pad, 12), "VORL-EXPLORE", fill="white", font=title)
    draw.text((pad, 42), "Fidelity-coupled frontier allocation and hybrid motion control", fill="#b4c4d7", font=font)
    gate_text = "online adaptation" if summary["online_adaptation"] else "frozen gate"
    draw.text((pad, 65), f"{size} x {size}   Robots {robots}   Dynamic obstacles {summary['dynamic_obstacles']}   Step {index:03d}/{len(data['known'])-1}   {gate_text}", fill="#d0deed", font=font)
    draw.text((pad, top - 18), "WORLD STATE (renderer only)", fill="#95aac1", font=small)
    draw.text((pad + panel + gap, top - 18), "SHARED OBSERVATION (controller input)", fill="#95aac1", font=small)
    field = min(index, len(data["goals"]) - 1)
    goals = data["goals"][field] if field >= 0 else np.full((robots, 2), -1)
    for side in (0, 1):
        left = pad + side * (panel + gap)
        known = data["static_map"] if side == 0 else data["known"][index]
        for r in range(size):
            for c in range(size):
                value = known[r, c]
                color = "#8994a4" if value == 255 else ("#344252" if value else "#edf2f7")
                draw.rectangle((left + c * cell, top + r * cell, left + (c + 1) * cell - 1, top + (r + 1) * cell - 1), fill=color)
        if side == 0:
            for r, c in data["dynamic"][index]:
                x, y = left + c * cell, top + r * cell
                draw.rectangle((x + 1, y + 1, x + cell - 2, y + cell - 2), fill="#e45151")
        else:
            for r, c in frontiers(data["known"][index]):
                x, y = left + c * cell + cell // 2, top + r * cell + cell // 2
                draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill="#21a36b")
        for robot in range(robots):
            color = COLORS[robot % len(COLORS)]
            trail = data["positions"][max(0, index - 20):index + 1, robot]
            points = [(left + c * cell + cell // 2, top + r * cell + cell // 2) for r, c in trail]
            if len(points) > 1:
                draw.line(points, fill=color, width=2)
            gr, gc = goals[robot]
            if gr >= 0:
                x, y = left + gc * cell + cell // 2, top + gr * cell + cell // 2
                draw.line((x - 3, y - 3, x + 3, y + 3), fill=color, width=1)
                draw.line((x - 3, y + 3, x + 3, y - 3), fill=color, width=1)
            r, c = data["positions"][index, robot]
            x, y = left + c * cell + cell // 2, top + r * cell + cell // 2
            radius = max(3, cell // 2 - 1)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="white", width=1)
    base = top + panel + 12
    coverage = np.mean(data["known"][index] != 255)
    draw.text((pad, base), f"Observed cells {coverage:.1%}   Red: moving obstacles   Gray: unknown   Green dots: frontiers   x: assigned goal", fill="#c7d4e4", font=small)
    columns = 8 if robots > 8 else 4
    column_width = (width - 2 * pad) // columns
    mode_names = ["A*", "RL", "REC"]
    for robot in range(robots):
        x, y = pad + (robot % columns) * column_width, base + 24 + (robot // columns) * 21
        draw.rectangle((x, y + 3, x + 9, y + 12), fill=COLORS[robot % len(COLORS)])
        if field >= 0:
            p = data["fidelity"][field, robot]
            mode = mode_names[data["modes"][field, robot]]
            text = f"R{robot + 1}: {p:.2f} {mode}"
        else:
            text = f"R{robot + 1}"
        draw.text((x + 14, y), text, fill="#dce6f3", font=small)
    if index == len(data["known"]) - 1:
        terminal = "NO FRONTIERS REMAIN" if summary["success_no_frontiers"] else "HORIZON REACHED - FRONTIERS REMAIN"
        draw.text((pad, height - 40), terminal, fill="#f4c66b", font=small)
    draw.text((pad, height - 21), "Recorded grid trajectory | Upstream EPOM policy | Implementation details: README", fill="#9cafc4", font=small)
    return image


def render(run, output, stride=2):
    run, output = Path(run), Path(output)
    if stride < 1:
        raise ValueError("stride must be positive")
    summary = json.loads((run / "summary.json").read_text())
    if summary["phase"] != "method_evaluation":
        raise ValueError("Only fitted-gate method-evaluation rollouts may use this renderer")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    with np.load(run / "trajectory.npz", allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    indices = list(range(0, len(data["known"]), stride))
    if indices[-1] != len(data["known"]) - 1:
        indices.append(len(data["known"]) - 1)
    frames = [render_frame(data, summary, i) for i in indices]
    gif = output / "vorl-explore.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=[120] * (len(frames) - 1) + [1200], loop=0, disposal=2)
    frames[0].save(output / "first.png")
    frames[len(frames) // 2].save(output / "middle.png")
    frames[-1].save(output / "last.png")
    with Image.open(gif) as decoded:
        actual_frames = decoded.n_frames
        for i in range(actual_frames):
            decoded.seek(i)
            decoded.load()
    report = {"trajectory_states": len(data["known"]), "rendered_state_indices": indices,
              "decoded_gif_frames": actual_frames, "size": frames[0].size, "source_seed": summary["seed"],
              "source_trajectory_sha256": summary["trajectory_sha256"], "profile": summary["profile"]}
    (output / "render-check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()
    render(args.run, args.output, args.stride)
