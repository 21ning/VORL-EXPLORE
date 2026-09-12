"""Strict EPOM checkpoint inference, using Sample Factory's original model core.

Unlike the legacy entry point, this does not start an environment just to get
space shapes, does not hard-code the checkpoint directory, and never falls back
to random weights when a file or parameter is missing. No training is claimed.
"""
import hashlib
import json
from pathlib import Path

import gym
import numpy as np
import torch
from torch import nn

from sample_factory.algorithms.appo.model import create_actor_critic
from sample_factory.algorithms.appo.model_utils import (
    EncoderBase, ResBlock, get_hidden_size, get_obs_shape, nonlinearity, register_custom_encoder,
)
from sample_factory.algorithms.utils.pytorch_utils import calc_num_elements
from sample_factory.utils.utils import AttrDict


class EpomEncoder(EncoderBase):
    """EPOM residual encoder topology from the provided appo/utils/encoder.py.

Sample Factory retains ownership of residual blocks and fully-connected naming,
so strict state-dict loading checks the original architecture without remapping.
"""
    def __init__(self, cfg, obs_space, timing):
        super().__init__(cfg, timing)
        shape = get_obs_shape(obs_space)
        settings = cfg.full_config["experiment_settings"]
        filters = settings["pogema_encoder_num_filters"]
        blocks = settings["pogema_encoder_num_res_blocks"]
        self.conv_head = nn.Sequential(
            nn.Conv2d(shape.obs[0], filters, kernel_size=3, stride=1, padding=1),
            *[ResBlock(cfg, filters, filters, timing) for _ in range(blocks)],
            nonlinearity(cfg),
        )
        self.conv_head_out_size = calc_num_elements(self.conv_head, shape.obs)
        self.coordinates_mlp = nn.Sequential(
            nn.Linear(4, cfg.hidden_size), nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size), nn.ReLU(),
        )
        self.init_fc_blocks(self.conv_head_out_size + cfg.hidden_size)

    def forward(self, observations):
        coordinates = torch.cat([observations["xy"], observations["target_xy"]], dim=-1)
        coordinates = coordinates / torch.clamp(torch.abs(coordinates), min=64.0)
        coordinates = self.coordinates_mlp(coordinates)
        grid = self.conv_head(observations["obs"]).contiguous().view(-1, self.conv_head_out_size)
        return self.forward_fc_blocks(torch.cat([grid, coordinates], dim=-1))


def safe_checkpoint(path):
    version = tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:2])
    if version < (2, 6):
        raise RuntimeError("PyTorch >= 2.6 is required for the patched weights-only loader")
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("Expected a full EPOM checkpoint with a model state dictionary")
    return checkpoint


class EpomPolicy:
    def __init__(self, config_path, checkpoint_path, *, seed, threads=4):
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        if not self.config_path.is_file() or not self.checkpoint_path.is_file():
            raise FileNotFoundError("Both EPOM cfg.json and the recurrent policy checkpoint are required")
        cfg = AttrDict(json.loads(self.config_path.read_text(encoding="utf-8")))
        if cfg.encoder_custom != "pogema_residual" or not cfg.actor_critic_share_weights:
            raise ValueError("This adapter supports the reference shared EPOM residual architecture")
        self.cfg = cfg
        env_cfg = cfg.full_config["environment"]
        self.radius = env_cfg.get("grid_memory_obs_radius") or env_cfg["grid_config"]["obs_radius"]
        width = self.radius * 2 + 1
        observation_space = gym.spaces.Dict({
            "obs": gym.spaces.Box(0.0, 1.0, shape=(3, width, width), dtype=np.float32),
            "xy": gym.spaces.Box(-1024, 1024, shape=(2,), dtype=np.float32),
            "target_xy": gym.spaces.Box(-1024, 1024, shape=(2,), dtype=np.float32),
        })
        if not isinstance(threads, int) or threads < 1:
            raise ValueError("threads must be a positive integer")
        torch.set_num_threads(threads)
        register_custom_encoder("pogema_residual", EpomEncoder)
        # The original initializer uses RNG; isolate it from callers and then
        # replace every learned parameter via strict checkpoint loading.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = create_actor_critic(cfg, observation_space, gym.spaces.Discrete(5))
        checkpoint = safe_checkpoint(self.checkpoint_path)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.eval()
        self.hidden = None
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.provenance = {
            "checkpoint_sha256": hashlib.sha256(self.checkpoint_path.read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256(self.config_path.read_bytes()).hexdigest(),
            "strict_state_dict_loaded": True,
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
            "torch": torch.__version__, "device": "cpu", "seed": seed,
            "training_provenance": "external checkpoint; not asserted to be VORL paper training",
        }

    def reset(self, seed):
        self.hidden = None
        self.generator.manual_seed(seed)

    @torch.inference_mode()
    def act(self, grid_observations, coordinates, targets):
        """Infer from caller-provided partial observations only; never a truth map.

Grid channels are local obstacle memory, visible agents, and clamped goal marker.
Coordinates are relative to each robot's episode start, as in the reference.
"""
        grid = np.asarray(grid_observations, dtype=np.float32)
        coords = np.asarray(coordinates, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        width = 2 * self.radius + 1
        n = len(grid)
        if grid.shape != (n, 3, width, width) or coords.shape != (n, 2) or targets.shape != (n, 2) or not n:
            raise ValueError("Unexpected EPOM observation dimensions")
        if not all(np.all(np.isfinite(a)) for a in (grid, coords, targets)):
            raise ValueError("EPOM observations must be finite")
        if np.any(grid < 0) or np.any(grid > 1):
            raise ValueError("EPOM grid channels must be in [0, 1]")
        if self.hidden is None or self.hidden.shape[0] != n:
            self.hidden = torch.zeros((n, get_hidden_size(self.cfg)), dtype=torch.float32)
        observations = AttrDict({"obs": torch.from_numpy(grid.copy()), "xy": torch.from_numpy(coords.copy()),
                                 "target_xy": torch.from_numpy(targets.copy())})
        head = self.model.forward_head(observations)
        core, self.hidden = self.model.forward_core(head, self.hidden)
        logits, _ = self.model.action_parameterization(core)
        if not torch.isfinite(logits).all():
            raise RuntimeError("Policy returned nonfinite logits")
        probabilities = torch.softmax(logits, dim=-1)
        actions = torch.multinomial(probabilities, num_samples=1, generator=self.generator).squeeze(-1)
        return actions.numpy(), logits.numpy()
