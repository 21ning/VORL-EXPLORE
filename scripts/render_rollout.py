"""Replay recorded shared-map exploration with Pogema 1.1.1 AnimationMonitor."""
import argparse
from copy import copy
import hashlib
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vorl.grid import frontiers

POGEMA_VERSION = "1.1.1"
UNKNOWN_COLOR = "#e0e0e0"
UNKNOWN_OPACITY = 0.55
OBSERVER_UNKNOWN_COLOR = "#808080"
OBSERVER_UNKNOWN_OPACITY = 0.15
DYNAMIC_COLOR = "#737373"
DYNAMIC_OPACITY = 1
GRID_COLOR = "#8a8a8a"
# Pogema 1.1.1 AnimationSettings.time_scale is 0.28 seconds per step.
STEP_MS, INTRO_MS, FINAL_HOLD_MS = 280, 1120, 2240


def validate_recording(data, summary, radius):
    """Validate geometry and replay team sensing, without importing the policy."""
    size, robots, moving = summary["size"], summary["robots"], summary["dynamic_obstacles"]
    states = len(data["known"])
    expected = {"static_map": (size, size), "known": (states, size, size),
                "positions": (states, robots, 2), "dynamic": (states, moving, 2),
                "goals": (states - 1, robots, 2)}
    if states < 1 or robots < 1 or moving < 0 or not isinstance(radius, int) or radius < 0:
        raise ValueError("Invalid playback dimensions or sensing radius")
    for key, shape in expected.items():
        if data[key].shape != shape or not np.issubdtype(data[key].dtype, np.integer):
            raise ValueError(f"Invalid {key} shape or dtype")
    if "dynamic_active" in data and data["dynamic_active"].shape != (states, moving):
        raise ValueError("Invalid recorded dynamic trip state")
    static_history = data.get("static", np.repeat(data["static_map"][None], states, axis=0))
    if static_history.shape != (states, size, size) or not np.isin(static_history, [0, 1]).all():
        raise ValueError("Invalid static map")
    joint = np.concatenate([data["positions"], data["dynamic"]], axis=1)
    if np.any(joint < 0) or np.any(joint >= size):
        raise ValueError("Recorded position is outside the map")
    if np.any(static_history[np.arange(states)[:, None], data["positions"][..., 0], data["positions"][..., 1]]):
        raise ValueError("Recorded robot intersects a static obstacle")
    if any(len({tuple(p) for p in state}) != robots + moving for state in joint):
        raise ValueError("Recorded entities overlap")
    if np.any(np.abs(np.diff(joint, axis=0)).sum(axis=-1) > 1):
        raise ValueError("Recorded entity moves more than one grid cell")
    goals = data["goals"]
    valid = np.all((goals >= 0) & (goals < size), axis=-1)
    if not np.all(valid | np.all(goals == -1, axis=-1)):
        raise ValueError("Invalid recorded goal")
    # Match GridWorld.observe: a persistent union of square sensor windows,
    # including last-observed dynamic occupancy outside the current field of view.
    shared = np.full((size, size), 255, dtype=np.uint8)
    dynamic_visible = np.zeros((states, moving), dtype=bool)
    for index, positions in enumerate(data["positions"]):
        truth = static_history[index].copy()
        moving_positions = data["dynamic"][index]
        truth[moving_positions[:, 0], moving_positions[:, 1]] = 1
        observed = np.zeros((size, size), dtype=bool)
        for row, col in positions:
            observed[max(0, row - radius):min(size, row + radius + 1),
                     max(0, col - radius):min(size, col + radius + 1)] = True
        shared[observed] = truth[observed]
        if not np.array_equal(shared, data["known"][index]):
            raise ValueError(f"Shared map differs from team sensing at state {index}")
        dynamic_visible[index] = observed[moving_positions[:, 0], moving_positions[:, 1]]
    return dynamic_visible


def make_monitor(data, summary, dynamic_visible, *, observer=False):
    """Adapt recorded states, not simulation dynamics, to the installed Pogema."""
    try:
        import gym
        from pogema import GridConfig
        from pogema.animation import AnimationConfig, AnimationMonitor, Rectangle
    except ImportError as error:
        raise RuntimeError("Use the separate environment in requirements-animation.txt") from error
    if version("pogema") != POGEMA_VERSION:
        raise RuntimeError(f"This playback adapter requires pogema=={POGEMA_VERSION}")
    robots = summary["robots"]
    static_history = data.get("static", np.repeat(data["static_map"][None], len(data["known"]), axis=0))
    joint = np.concatenate([data["positions"], data["dynamic"]], axis=1)
    goals = np.full_like(joint, -1)
    # A decision at state t governs transition t -> t+1. There is no next
    # assignment at the terminal state, and moving obstacles have no goal rings.
    goals[:-1, :robots] = data["goals"]
    goal_visible = np.all(goals >= 0, axis=-1)
    drawable_goals = np.where(goal_visible[..., None], goals, joint)
    # A gray square denotes an assigned obstacle trip, rather than one
    # interpolated displacement. Older recordings remain renderable.
    if "dynamic_active" in data:
        moving = np.asarray(data["dynamic_active"], dtype=bool)
    else:
        moving = np.zeros_like(dynamic_visible)
        moving[:-1] = np.any(np.diff(data["dynamic"], axis=0) != 0, axis=-1)
    moving_visible = moving if observer else moving & dynamic_visible
    entity_visible = np.concatenate([np.ones((len(joint), robots), dtype=bool), moving_visible], axis=1)
    if observer:
        # Viewer-only truth: never fed to a policy or written into the shared map.
        ordinary_occupancy = static_history.astype(bool).copy()
        for state, cells in enumerate(data["dynamic"]):
            stationary = cells[~moving[state]]
            ordinary_occupancy[state, stationary[:, 0], stationary[:, 1]] = True
    else:
        ordinary_occupancy = data["known"] == 1
    for state, cells in enumerate(data["dynamic"]):
        highlighted = cells[moving_visible[state]]
        ordinary_occupancy[state, highlighted[:, 0], highlighted[:, 1]] = False

    class RecordedTrajectory(gym.Env):
        """Read-only playback; step never runs a policy or Pogema physics."""

        def __init__(self):
            self.grid_config = GridConfig(size=summary["size"], num_agents=joint.shape[1],
                                          density=0, on_target="restart")
            self.grid = SimpleNamespace(obstacles=static_history[0].copy())
            self.action_space = gym.spaces.Discrete(1)
            self.observation_space = gym.spaces.Discrete(1)
            self.seek(0)

        def seek(self, index):
            self.index = index
            self.grid.positions_xy = joint[index].tolist()
            self.grid.finishes_xy = drawable_goals[index].tolist()

        def reset(self, **kwargs):
            self.seek(0)
            return 0

        def step(self, action):
            if action is not None or self.index + 1 >= len(joint):
                raise ValueError("Playback accepts only the next recorded state")
            self.seek(self.index + 1)
            count = joint.shape[1]
            return 0, [0.0] * count, [False] * count, [{} for _ in range(count)]

    class SharedMapMonitor(AnimationMonitor):
        """Native Pogema shapes/motion with a team-shared observation layer."""

        def set_visibility(self, shape, values, static):
            shape.attributes["visibility"] = "visible" if values[0] else "hidden"
            shape.animations = [a for a in shape.animations if a.attributes["attributeName"] != "visibility"]
            if not static:
                tokens = ["visible" if value else "hidden" for value in values]
                animation = self.compressed_anim("visibility", tokens, self.svg_settings.time_scale)
                animation.attributes["calcMode"] = "discrete"
                shape.add_animation(animation)

        def create_obstacles(self, grid_holder, animation_config):
            cfg = self.svg_settings
            size = summary["size"]
            # Shared view uses recorded occupancy; observer view uses display-only
            # ground truth. Neither view changes the recorded observations.
            occupied = np.any(self.ordinary_history, axis=0).astype(np.uint8)
            holder = grid_holder.copy(update={"obstacles": occupied})
            blocks = super().create_obstacles(holder, animation_config)
            cells = [(size - j - 1, i) for i in range(size) for j in range(size)
                     if occupied[size - j - 1, i]]
            for block, (row, col) in zip(blocks, cells):
                block.attributes.update(x=col * cfg.scale_size,
                                        y=-(size - row) * cfg.scale_size,
                                        width=cfg.scale_size, height=cfg.scale_size,
                                        rx=0, data_layer="occupancy")
                self.set_visibility(block, self.ordinary_history[:, row, col], animation_config.static)
            fog = []
            for row, col in np.argwhere(np.any(self.shared_history == 255, axis=0)):
                # Use Pogema's own SVG rectangle, coordinate system and animation
                # compressor; the mask is an adapter for team exploration.
                tile = Rectangle(x=int(col) * cfg.scale_size,
                                 y=(size - int(row) - 1) * cfg.scale_size,
                                 width=cfg.scale_size, height=cfg.scale_size,
                                 fill=OBSERVER_UNKNOWN_COLOR if observer else UNKNOWN_COLOR,
                                 opacity=OBSERVER_UNKNOWN_OPACITY if observer else UNKNOWN_OPACITY,
                                 data_layer="unknown-mask")
                self.set_visibility(tile, self.shared_history[:, row, col] == 255, animation_config.static)
                fog.append(tile)
            background = Rectangle(x=0, y=0, width=size * cfg.scale_size,
                                   height=size * cfg.scale_size, fill="#ffffff")
            # Grid lines stay above the mask in both display modes. Shared view
            # hides unknown geometry; observer view tints the original geometry.
            extent, line_width = size * cfg.scale_size, 4
            grid = []
            for line in range(size + 1):
                offset = line * cfg.scale_size - line_width / 2
                grid.extend([
                    Rectangle(x=offset, y=0, width=line_width, height=extent,
                              fill=GRID_COLOR, opacity=0.65, data_layer="grid"),
                    Rectangle(x=0, y=offset, width=extent, height=line_width,
                              fill=GRID_COLOR, opacity=0.65, data_layer="grid"),
                ])
            return [background] + blocks + fog + grid

        def create_agents(self, grid_holder, animation_config):
            agents = super().create_agents(grid_holder, animation_config)
            for index, agent in enumerate(agents):
                if index >= robots:
                    scale = self.svg_settings.scale_size
                    row, col = grid_holder.agents_xy_history[0][index]
                    agent = Rectangle(x=col * scale, y=(summary["size"] - row - 1) * scale,
                                      width=scale, height=scale, rx=0, fill=DYNAMIC_COLOR,
                                      opacity=DYNAMIC_OPACITY, stroke=GRID_COLOR,
                                      stroke_width=4, data_layer="moving-obstacle")
                    agents[index] = agent
                self.set_visibility(agent, self.entity_visible[:, index], True)
            return agents

        def animate_agents(self, agents, egocentric_idx, grid_holder):
            super().animate_agents(agents, egocentric_idx, grid_holder)
            for index, agent in enumerate(agents):
                if index >= robots:
                    # Reuse Pogema's exact motion/keyTimes; convert circle centers
                    # into rectangle top-left coordinates, without rerouting motion.
                    for animation in agent.animations:
                        attribute = animation.attributes["attributeName"]
                        if attribute in ("cx", "cy"):
                            animation.attributes["attributeName"] = {"cx": "x", "cy": "y"}[attribute]
                            animation.attributes["values"] = ";".join(
                                f"{float(value) - self.svg_settings.scale_size / 2:g}"
                                for value in animation.attributes["values"].split(";"))
                self.set_visibility(agent, self.entity_visible[:, index], False)

        def create_targets(self, grid_holder, animation_config):
            holder = grid_holder.copy(update={"targets_xy": grid_holder.targets_xy_history[0]})
            targets = super().create_targets(holder, animation_config)[:robots]
            for index, target in enumerate(targets):
                self.set_visibility(target, self.goal_visible[:, index], True)
            return targets

        def animate_targets(self, targets, grid_holder, animation_config):
            super().animate_targets(targets, grid_holder, animation_config)
            for index, target in enumerate(targets):
                # Assigned frontiers jump; they do not move through intervening cells.
                for animation in target.animations:
                    animation.attributes["calcMode"] = "discrete"
                self.set_visibility(target, self.goal_visible[:, index], False)

    monitor = SharedMapMonitor(RecordedTrajectory(), AnimationConfig(save_every_idx_episode=None))
    palette = monitor.svg_settings.colors
    monitor.svg_settings.colors = ([palette[i % len(palette)] for i in range(robots)] +
                                  [DYNAMIC_COLOR] * summary["dynamic_obstacles"])
    monitor.svg_settings.time_scale = STEP_MS / 1000
    monitor.reset()
    for _ in range(1, len(joint)):
        monitor.step(None)
    if not np.array_equal(monitor.agents_xy_history, joint):
        raise ValueError("Pogema playback changed recorded positions")
    if not np.array_equal(monitor.targets_xy_history, drawable_goals):
        raise ValueError("Pogema playback changed recorded targets")
    # A labelled pre-observation intro uses initial poses and an entirely unknown
    # map, matching GridWorld initialization. It is NOT another rollout step.
    intro, hold = INTRO_MS // STEP_MS, FINAL_HOLD_MS // STEP_MS - 1
    def timeline(values, initial):
        return np.concatenate([np.repeat(initial[None], intro, axis=0), values,
                               np.repeat(values[-1:], hold, axis=0)])
    monitor.agents_xy_history = timeline(joint, joint[0]).tolist()
    monitor.targets_xy_history = timeline(drawable_goals, drawable_goals[0]).tolist()
    monitor.dones_history = np.zeros((len(monitor.agents_xy_history), joint.shape[1]), dtype=bool).tolist()
    monitor.shared_history = timeline(data["known"], np.full_like(data["known"][0], 255))
    initial_occupancy = np.zeros_like(ordinary_occupancy[0])
    if observer:
        initial_occupancy = static_history[0].astype(bool).copy()
        initial_dynamic = data["dynamic"][0]
        initial_occupancy[initial_dynamic[:, 0], initial_dynamic[:, 1]] = True
    monitor.ordinary_history = timeline(ordinary_occupancy, initial_occupancy)
    monitor.entity_visible = timeline(entity_visible, np.arange(joint.shape[1]) < robots)
    monitor.goal_visible = timeline(goal_visible, np.zeros(joint.shape[1], dtype=bool))
    monitor.recorded_offset = intro
    monitor.observer_view = observer
    return monitor


def observer_svg_layers(svg):
    """Place native moving squares beneath the fog, leaving robots above it."""
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    root = ET.fromstring(svg)
    # Grid is a fallback for a fully explored frame, which has no fog rectangles.
    overlay = [shape for shape in root if shape.get("data-layer") in ("unknown-mask", "grid")]
    squares = [shape for shape in root if shape.get("data-layer") == "moving-obstacle"]
    for square in squares:
        root.remove(square)
    insertion = list(root).index(overlay[0])
    for square in squares:
        root.insert(insertion, square)
        insertion += 1
    return ET.tostring(root, encoding="unicode")


def native_frame_svg(monitor, index):
    """Select a native Pogema frame without changing the full SVG timeline."""
    from pogema.animation import AnimationConfig
    frame = copy(monitor)
    for key in ["agents_xy_history", "targets_xy_history", "dones_history"]:
        setattr(frame, key, [getattr(monitor, key)[index]])
    for key in ["shared_history", "ordinary_history", "entity_visible", "goal_visible"]:
        setattr(frame, key, getattr(monitor, key)[index:index + 1])
    svg = frame.create_animation(AnimationConfig(static=True)).render()
    return observer_svg_layers(svg) if monitor.observer_view else svg


def render(run, output, stride=1, *, require_success=False, svg_only=False, width=720, observer=False):
    run, output = Path(run), Path(output)
    if stride < 1 or width < 160:
        raise ValueError("stride must be positive and width must be at least 160")
    summary = json.loads((run / "summary.json").read_text())
    if summary["phase"] != "method_evaluation":
        raise ValueError("Only fitted-gate method-evaluation rollouts may use this renderer")
    if output.exists():
        raise FileExistsError(output)
    if hashlib.sha256((run / "trajectory.npz").read_bytes()).hexdigest() != summary["trajectory_sha256"]:
        raise ValueError("Trajectory hash does not match the run summary")
    with np.load(run / "trajectory.npz", allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    completed = not frontiers(data["known"][-1])
    if completed != summary["success_no_frontiers"]:
        raise ValueError("Completion flag does not match the final recorded map")
    if require_success and not completed:
        raise ValueError("A successful demo requires no remaining frontiers")
    config = json.loads((run / "config.json").read_text())
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if config_hash != summary["config_sha256"]:
        raise ValueError("Config hash does not match the run summary")
    dynamic_visible = validate_recording(data, summary, config["sensing_radius"])
    monitor = make_monitor(data, summary, dynamic_visible, observer=observer)
    if not svg_only:
        try:
            import cairosvg
            from PIL import Image
        except (ImportError, OSError) as error:
            raise RuntimeError("GIF needs CairoSVG and libcairo2; see README, or use --svg-only") from error
    output.mkdir(parents=True)
    svg = output / "vorl-explore.svg"
    monitor.save_animation(str(svg))
    if observer:
        svg.write_text(observer_svg_layers(svg.read_text()))
    report = {"backend": "pogema.animation.AnimationMonitor", "pogema_version": version("pogema"),
              "playback": "recorded VORL states; no Pogema simulation or policy rerun",
              "view": "persistent team-shared map; moving entities visible only in current team sensing",
              "map_style": "cell-aligned occupancy, translucent near-white unknown mask with a slight gray tint, grid lines above mask",
              "unknown_mask_color": UNKNOWN_COLOR, "unknown_mask_opacity": UNKNOWN_OPACITY,
              "unobserved_occupancy": "hidden independently of mask opacity; no ground-truth map beneath the mask",
              "dynamic_obstacle_style": "gray square for an assigned obstacle trip; ordinary occupancy after arrival",
              "moving_obstacle_color": DYNAMIC_COLOR, "moving_obstacle_opacity": DYNAMIC_OPACITY,
              "motion_indicator": "recorded dynamic trip state; legacy recordings use position changes",
              "playback_speed": "1x Pogema 1.1.1 default: 0.28 seconds per step",
              "pre_observation_intro": "initial robot poses on an all-unknown map; not a rollout step",
              "trajectory_states": len(data["known"]), "native_history_matches_recording": True,
              "shared_map_replay_matches_recording": True, "native_timeline_states": len(monitor.dones_history),
              "recorded_state_offset": monitor.recorded_offset,
              "initial_observed_cell_fraction": float(np.mean(data["known"][0] != 255)),
              "source_seed": summary["seed"], "source_trajectory_sha256": summary["trajectory_sha256"],
              "profile": summary["profile"], "success_no_frontiers": completed,
              "final_frontier_count": len(frontiers(data["known"][-1])),
              "svg_sha256": hashlib.sha256(svg.read_bytes()).hexdigest()}
    if observer:
        report.update(view="observer ground truth with gray overlay on team-unexplored cells",
                      map_style="original geometry under 15% gray overlay; grid lines above overlay",
                      unknown_mask_color=OBSERVER_UNKNOWN_COLOR, unknown_mask_opacity=OBSERVER_UNKNOWN_OPACITY,
                      unobserved_occupancy="visible to viewer under gray overlay; not supplied to agents",
                      recorded_shared_map_unchanged=True)
    if not svg_only:
        indices = list(range(0, len(data["known"]), stride))
        if indices[-1] != len(data["known"]) - 1:
            indices.append(len(data["known"]) - 1)
        display_indices = [0] + [i + monitor.recorded_offset for i in indices]
        frames = []
        for frame_number, index in enumerate(display_indices, 1):
            png = cairosvg.svg2png(bytestring=native_frame_svg(monitor, index).encode(),
                                  output_width=width, output_height=width, background_color="white")
            with Image.open(BytesIO(png)) as source:
                frames.append(source.convert("RGB"))
            if frame_number % 50 == 0 or frame_number == len(display_indices):
                print(f"Rendered Pogema frame {frame_number}/{len(display_indices)}", flush=True)
        durations = [INTRO_MS] + [(b - a) * STEP_MS for a, b in zip(indices, indices[1:])] + [FINAL_HOLD_MS]
        gif = output / "vorl-explore.gif"
        frames[0].save(gif, save_all=True, append_images=frames[1:], duration=durations, loop=0, disposal=2)
        for name, index in [("intro", 0), ("first", 1), ("middle", len(frames) // 2), ("last", len(frames) - 1)]:
            frames[index].save(output / f"{name}.png")
        with Image.open(gif) as decoded:
            actual_frames, duration = decoded.n_frames, 0
            for index in range(actual_frames):
                decoded.seek(index)
                decoded.load()
                duration += decoded.info["duration"]
        report.update(gif_source="Pogema static SVG frames rasterized with CairoSVG; Pillow encodes GIF only",
                      rendered_state_indices=indices, decoded_gif_frames=actual_frames,
                      size=frames[0].size, gif_duration_ms=duration, step_duration_ms=STEP_MS,
                      intro_ms=INTRO_MS, final_hold_ms=FINAL_HOLD_MS, cairosvg_version=version("CairoSVG"),
                      pillow_version=version("Pillow"), gif_sha256=hashlib.sha256(gif.read_bytes()).hexdigest())
    (output / "render-check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rendered_state_indices"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=1, help="GIF sampling stride; SVG keeps every state")
    parser.add_argument("--width", type=int, default=720, help="GIF width in pixels")
    parser.add_argument("--svg-only", action="store_true", help="Export native SVG without CairoSVG")
    parser.add_argument("--require-success", action="store_true", help="Reject runs with remaining frontiers")
    parser.add_argument("--observer", action="store_true", help="Show full terrain under a 15%% gray unknown overlay; display only")
    args = parser.parse_args()
    render(args.run, args.output, args.stride, require_success=args.require_success,
           svg_only=args.svg_only, width=args.width, observer=args.observer)
