from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# Shared constants with tinyphysics/train_ppo
CONTEXT_LENGTH = 20
CONTROL_START_IDX = 100
COST_END_IDX = 500
VOCAB_SIZE = 1024
LATACCEL_RANGE = (-5.0, 5.0)
STEER_RANGE = (-2.0, 2.0)
MAX_ACC_DELTA = 0.5
DEL_T = 0.1
ACC_G = 9.81
OBS_DIM = 22

# Default SB3 training setup (adjust for hardware)
TOTAL_TIMESTEPS = 5_000_000
NUM_ENVS = 32
N_STEPS = 2048
BATCH_SIZE = 16_384
LEARNING_RATE = 3e-4
N_EPOCHS = 10
OUTPUT_PATH = "ppo_sb3"


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class EnvConfig:
    model_path: str = "./models/tinyphysics.onnx"
    data_path: str = "./data"
    reward_scale: float = 10000.0
    batch_truncation_length: int = 550
    seed: int = 0


class TinyPhysicsEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, config: EnvConfig) -> None:
        super().__init__()
        self.config = config
        self.device = _device()
        self.rng = np.random.default_rng(config.seed)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([STEER_RANGE[0]], dtype=np.float32),
            high=np.array([STEER_RANGE[1]], dtype=np.float32),
            dtype=np.float32,
        )

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers: list[Any]
        if torch.cuda.is_available():
            providers = [
                ("CUDAExecutionProvider", {"device_id": 0}),
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(
            config.model_path, opts, providers=providers
        )
        self.bins = torch.linspace(
            LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE, device=self.device
        )

        self._load_data()
        self.reset()

    # Gymnasium seeding API
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        idx = self.rng.integers(0, self.num_segments)
        self.segment = self.all_data_full[idx].clone().to(self.device)

        self.max_steps = self.segment.shape[0]
        self.step_idx = CONTEXT_LENGTH

        self.action_history = self.segment[:, 4].clone()
        self.lataccel_history = self.segment[:, 3].clone()
        self.current_lataccel = self.lataccel_history[self.step_idx - 1].clone()
        self.prev_error = torch.zeros(1, device=self.device)
        self.error_integral = torch.zeros(1, device=self.device)

        obs = self._get_obs()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action_tensor = (
            torch.as_tensor(action, dtype=torch.float32, device=self.device)
            .flatten()
            .clamp(*STEER_RANGE)
        )

        self.action_history[self.step_idx] = action_tensor[0]
        new_lataccel = self._physics_step()

        if self.step_idx >= CONTROL_START_IDX:
            self.current_lataccel = new_lataccel
        else:
            self.current_lataccel = self.lataccel_history[self.step_idx]

        self.lataccel_history[self.step_idx] = self.current_lataccel

        target = self.segment[self.step_idx, 3]
        error = target - self.current_lataccel
        prev_lataccel = self.lataccel_history[self.step_idx - 1]
        jerk = (self.current_lataccel - prev_lataccel) / DEL_T

        lataccel_cost = (error**2) * 100.0
        jerk_cost = (jerk**2) * 100.0
        reward = -((lataccel_cost * 50.0 + jerk_cost) / self.config.reward_scale)

        self.prev_error = error.unsqueeze(0)
        self.error_integral = torch.clamp(self.error_integral + error, -5, 5)

        self.step_idx += 1
        terminated = self.step_idx >= COST_END_IDX
        truncated = False

        obs = self._get_obs()
        info = {
            "lataccel_cost": lataccel_cost.item(),
            "jerk_cost": jerk_cost.item(),
            "target_lataccel": target.item(),
            "current_lataccel": self.current_lataccel.item(),
        }
        return obs, float(reward.item()), terminated, truncated, info

    def _load_data(self) -> None:
        data_files = sorted(Path(self.config.data_path).glob("*.csv"))[:5000]
        segments: list[np.ndarray] = []
        skipped = 0
        for file in data_files:
            df = pd.read_csv(file)
            if len(df) < self.config.batch_truncation_length:
                skipped += 1
                continue
            arr = np.column_stack(
                (
                    np.sin(df["roll"].values[: self.config.batch_truncation_length])
                    * ACC_G,
                    df["vEgo"].values[: self.config.batch_truncation_length],
                    df["aEgo"].values[: self.config.batch_truncation_length],
                    df["targetLateralAcceleration"].values[
                        : self.config.batch_truncation_length
                    ],
                    -df["steerCommand"].values[: self.config.batch_truncation_length],
                )
            )
            segments.append(arr.astype(np.float32))

        if not segments:
            raise RuntimeError("No data segments loaded for TinyPhysicsEnv")

        self.all_data_full = torch.tensor(np.stack(segments), dtype=torch.float32)
        self.num_segments = self.all_data_full.shape[0]
        print(
            f"TinyPhysicsEnv loaded {self.num_segments} segments (skipped {skipped})."
        )

    def _physics_step(self) -> torch.Tensor:
        idx = self.step_idx
        ctx_start = idx - CONTEXT_LENGTH

        ctx_actions = self.action_history[ctx_start:idx]
        ctx_states = self.segment[ctx_start:idx, :3]
        states_input = torch.cat([ctx_actions.unsqueeze(-1), ctx_states], dim=-1)
        ctx_lataccel = self.lataccel_history[ctx_start:idx].clamp(*LATACCEL_RANGE)
        tokens_input = torch.searchsorted(self.bins, ctx_lataccel.contiguous())

        states_np = states_input.unsqueeze(0).cpu().numpy().astype(np.float32)
        tokens_np = tokens_input.unsqueeze(0).cpu().numpy().astype(np.int64)

        logits = self.session.run(None, {"states": states_np, "tokens": tokens_np})[0]
        logits_tensor = torch.from_numpy(logits).to(self.device)

        probs = torch.softmax(logits_tensor[:, -1, :] / 0.8, dim=-1)
        samples = torch.multinomial(probs, 1).squeeze(-1)
        preds = self.bins[samples]
        preds = torch.clamp(
            preds,
            self.current_lataccel - MAX_ACC_DELTA,
            self.current_lataccel + MAX_ACC_DELTA,
        )
        return preds.squeeze(0)

    def _window_stats(
        self, start: int, end: int, fallback: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if end > start:
            window = self.segment[start:end, 3]
            mean = window.mean()
            std = window.std(unbiased=False)
        else:
            mean = fallback
            std = torch.zeros(1, device=self.device)
        return mean, std

    def _get_obs(self) -> np.ndarray:
        idx = self.step_idx
        roll = self.segment[idx, 0]
        v_ego = self.segment[idx, 1]
        a_ego = self.segment[idx, 2]
        target = self.segment[idx, 3]

        error = target - self.current_lataccel
        error_deriv = error - self.prev_error.squeeze(0)

        prev_action = self.action_history[idx - 1]
        prev_prev_action = self.action_history[idx - 2] if idx >= 2 else prev_action
        action_delta = prev_action - prev_prev_action

        end1 = min(idx + 6, self.max_steps)
        end2 = min(idx + 16, self.max_steps)

        future_near, future_near_std = self._window_stats(idx + 1, end1, target)
        future_mid, future_mid_std = self._window_stats(idx + 6, end2, target)

        prev_lataccel = self.lataccel_history[idx - 1]
        lataccel_delta = self.current_lataccel - prev_lataccel
        lat_hist = [self.lataccel_history[max(idx - k, 0)] for k in range(1, 5)]

        progress = float(
            max(
                0.0,
                min(
                    1.0, (idx - CONTROL_START_IDX) / (COST_END_IDX - CONTROL_START_IDX)
                ),
            )
        )

        obs = torch.stack(
            [
                error / 5.0,
                error_deriv / 2.0,
                self.error_integral.squeeze(0) / 5.0,
                target / 5.0,
                v_ego / 30.0,
                a_ego / 4.0,
                roll / 2.0,
                prev_action / 2.0,
                prev_prev_action / 2.0,
                self.prev_error.squeeze(0) / 5.0,
                future_near / 5.0,
                future_mid / 5.0,
                self.current_lataccel / 5.0,
                lataccel_delta / 5.0,
                action_delta / 2.0,
                future_near_std / 5.0,
                future_mid_std / 5.0,
                torch.tensor(progress, device=self.device),
                *[hist / 5.0 for hist in lat_hist],
            ]
        )
        return obs.detach().cpu().numpy().astype(np.float32)

    def render(self) -> None:  # pragma: no cover - visualization not required
        return


def make_vec_env(config: EnvConfig, num_envs: int, seed: int) -> SubprocVecEnv:
    def _init_env(env_rank: int):
        def _thunk():
            env_config = EnvConfig(
                model_path=config.model_path,
                data_path=config.data_path,
                reward_scale=config.reward_scale,
                batch_truncation_length=config.batch_truncation_length,
                seed=seed + env_rank,
            )
            env = TinyPhysicsEnv(env_config)
            env.reset(seed=seed + env_rank)
            return env

        return _thunk

    return SubprocVecEnv([_init_env(i) for i in range(num_envs)])


def train_sb3() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 0
    config = EnvConfig(seed=seed)
    vec_env = make_vec_env(config, num_envs=NUM_ENVS, seed=seed)

    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 256, 256], vf=[256, 256, 256])],
        activation_fn=torch.nn.Tanh,
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        device=device,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(OUTPUT_PATH)
    vec_env.close()
    print(f"Saved SB3 PPO model to {OUTPUT_PATH}")


if __name__ == "__main__":
    train_sb3()
