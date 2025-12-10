from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from torch import Tensor, nn
from torch.types import Number

# Constants coppied from tinyphysics.py
CONTEXT_LENGTH = 20
CONTROL_START_IDX = 100
COST_END_IDX = 500
VOCAB_SIZE = 1024
LATACCEL_RANGE = (-5.0, 5.0)
STEER_RANGE = (-2.0, 2.0)
MAX_ACC_DELTA = 0.5
DEL_T = 0.1
ACC_G = 9.81


@dataclass
class Config:
    # Environment
    batch_size: int = 1024
    rollout_steps: int = COST_END_IDX - CONTEXT_LENGTH
    obs_dim: int = 12
    batch_truncation_length: int = 550

    # Network
    hidden_size: int = 128

    # PPO
    lr: float = 2e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.005
    value_coef: float = 0.5
    value_clip_eps: float = 0.2
    max_grad_norm: float = 0.5
    update_epochs: int = 3
    minibatch_size: int = 4096
    reward_scale: float = 10000.0

    # Training
    total_iterations: int = 2000

    # Paths
    model_path: str = "./models/tinyphysics.onnx"
    data_path: str = "./data"


# Batched physics sim
class BatchedSimulator:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.n = config.batch_size

        # Load ONNX physics model
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if torch.cuda.is_available():
            self.device = "cuda"
            providers = [
                ("CUDAExecutionProvider", {"device_id": 0}),
                "CPUExecutionProvider",
            ]
        else:
            self.device = "cpu"
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            config.model_path, opts, providers=providers
        )

        # Tokenizer bins
        self.bins = torch.linspace(
            LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE, device=self.device
        )

        # Load CSV data (first 5000 segments for training pool)
        all_files = sorted(Path(config.data_path).glob("*.csv"))[:5000]
        print(f"Loading {len(all_files)} CSV files (train pool)")
        self._load_data(all_files)
        print(f"Train pool shape: {self.all_data_full.shape}")

        # Actual batch size per rollout (may be smaller if pool < batch size)
        self.n = min(config.batch_size, self.num_segments)

    def _load_data(self, data_files: list[Path]) -> None:
        all_data = []
        skipped = 0
        for f in data_files:
            df = pd.read_csv(f)

            if len(df) < self.config.batch_truncation_length:
                skipped += 1
                continue

            # csvs vary in length a little bit
            # we just truncate them all to batch_truncation_length (550)
            data = np.column_stack(
                (
                    np.asarray(
                        np.sin(df["roll"].values[: self.config.batch_truncation_length])
                        * ACC_G
                    ),
                    np.asarray(
                        df["vEgo"].values[: self.config.batch_truncation_length]
                    ),
                    np.asarray(
                        df["aEgo"].values[: self.config.batch_truncation_length]
                    ),
                    np.asarray(
                        df["targetLateralAcceleration"].values[
                            : self.config.batch_truncation_length
                        ]
                    ),
                    np.asarray(
                        -df["steerCommand"].values[
                            : self.config.batch_truncation_length
                        ]  # type: ignore[reportOperatorIssue]
                    ),
                )
            )
            all_data.append(data)

        print(f"Loaded {len(all_data)} segments (skipped {skipped} short files)")
        self.all_data_full = torch.tensor(np.stack(all_data), dtype=torch.float32)
        self.num_segments = self.all_data_full.shape[0]
        self.max_steps = self.config.batch_truncation_length

    def reset(self) -> Tensor:
        self.step_idx = CONTEXT_LENGTH

        # Sample a random batch of segments from the train pool
        batch_size = min(self.config.batch_size, self.num_segments)
        indices = torch.randperm(self.num_segments)[:batch_size]
        self.all_data = self.all_data_full[indices].to(self.device)
        self.n = batch_size

        # Init histories
        self.action_history = self.all_data[:, :, 4].clone()
        self.lataccel_history = self.all_data[:, :, 3].clone()
        self.current_lataccel = self.lataccel_history[:, self.step_idx - 1].clone()

        # Error integral
        self.error_integral = torch.zeros(self.n, device=self.device)

        # Prev error for deriv calc
        self.prev_error = torch.zeros(self.n, device=self.device)

        return self._get_obs()

    def _get_obs(self) -> Tensor:
        idx = self.step_idx

        # State from all data
        roll = self.all_data[:, idx, 0]
        v_ego = self.all_data[:, idx, 1]
        a_ego = self.all_data[:, idx, 2]
        target = self.all_data[:, idx, 3]

        # Error terms
        error = target - self.current_lataccel
        error_deriv = error - self.prev_error

        # Previous actions
        prev_action = self.action_history[:, idx - 1]
        prev_prev_action = self.action_history[:, idx - 2] if idx >= 2 else prev_action

        # Future
        end1 = min(idx + 6, self.max_steps)
        end2 = min(idx + 16, self.max_steps)
        future_near = (
            self.all_data[:, idx + 1 : end1, 3].mean(dim=1)
            if end1 > idx + 1
            else target
        )
        future_mid = (
            self.all_data[:, idx + 6 : end2, 3].mean(dim=1)
            if end2 > idx + 6
            else target
        )

        return torch.stack(
            [
                error / 5.0,
                error_deriv / 2.0,
                self.error_integral / 5.0,
                target / 5.0,
                v_ego / 30.0,
                a_ego / 4.0,
                roll / 2.0,
                prev_action / 2.0,
                prev_prev_action / 2.0,
                self.prev_error / 5.0,
                future_near / 5.0,
                future_mid / 5.0,
            ],
            dim=1,
        )

    def step(self, actions: Tensor) -> tuple[Tensor, Tensor, bool]:
        actions = torch.clamp(actions, STEER_RANGE[0], STEER_RANGE[1])
        self.action_history[:, self.step_idx] = actions

        # Physics step
        new_lataccel = self._physics_step()

        # Only use model after we have control
        if self.step_idx >= CONTROL_START_IDX:
            self.current_lataccel = new_lataccel
        else:
            self.current_lataccel = self.all_data[:, self.step_idx, 3]

        self.lataccel_history[:, self.step_idx] = self.current_lataccel

        # Reward
        target = self.all_data[:, self.step_idx, 3]
        error = target - self.current_lataccel

        prev_lataccel = self.lataccel_history[:, self.step_idx - 1]
        jerk = (self.current_lataccel - prev_lataccel) / DEL_T

        lataccel_cost = error**2 * 100.0
        jerk_cost = jerk**2 * 100.0
        rewards = -(lataccel_cost * 50.0 + jerk_cost) / self.config.reward_scale

        self.prev_error = error
        self.error_integral = torch.clamp(self.error_integral + error, -5, 5)

        self.step_idx += 1
        dones = self.step_idx >= COST_END_IDX

        return self._get_obs(), rewards, dones

    def _physics_step(self) -> Tensor:
        idx = self.step_idx

        # Context
        ctx_actions = self.action_history[:, idx - CONTEXT_LENGTH : idx]
        ctx_states = self.all_data[:, idx - CONTEXT_LENGTH : idx, :3]
        states_input = torch.cat([ctx_actions.unsqueeze(-1), ctx_states], dim=-1)

        # Tokens
        ctx_lataccel = self.lataccel_history[:, idx - CONTEXT_LENGTH : idx]
        tokens_input = torch.searchsorted(
            self.bins, ctx_lataccel.contiguous().clamp(*LATACCEL_RANGE)
        )

        # Run ONNX (on CPU for now)
        states_np = states_input.cpu().numpy().astype(np.float32)
        tokens_np = tokens_input.cpu().numpy().astype(np.int64)

        logits = self.session.run(None, {"states": states_np, "tokens": tokens_np})[0]
        logits = torch.tensor(logits, device=self.device)

        probs = torch.softmax(logits[:, -1, :] / 0.8, dim=-1)
        samples = torch.multinomial(probs, 1).squeeze(-1)
        preds = self.bins[samples]

        preds = torch.clamp(
            preds,
            self.current_lataccel - MAX_ACC_DELTA,
            self.current_lataccel + MAX_ACC_DELTA,
        )

        return preds

    def compute_cost(self) -> tuple[Number, Number, Number]:
        target = self.all_data[:, CONTROL_START_IDX:COST_END_IDX, 3]
        pred = self.lataccel_history[:, CONTROL_START_IDX:COST_END_IDX]

        lataccel_cost = ((target - pred) ** 2).mean(dim=1) * 100
        jerk = (pred[:, 1:] - pred[:, :-1]) / DEL_T
        jerk_cost = (jerk**2).mean(dim=1) * 100
        total_cost = lataccel_cost * 50.0 + jerk_cost

        return (
            total_cost.mean().item(),
            lataccel_cost.mean().item(),
            jerk_cost.mean().item(),
        )


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int) -> None:
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.actor_mean = nn.Linear(hidden, 1)
        self.actor_logstd = nn.Linear(hidden, 1)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs):
        h = self.shared(obs)
        mean = self.actor_mean(h)
        logstd = self.actor_logstd(h)
        logstd = torch.clamp(logstd, min=-2.0, max=0.5)
        return mean, logstd, self.critic(h)

    def get_action(self, obs, deterministic=False):
        mean, logstd, value = self.forward(obs)
        std = logstd.exp()

        if deterministic:
            action = mean
        else:
            action = mean + std * torch.randn_like(mean)

        action = torch.clamp(action, STEER_RANGE[0], STEER_RANGE[1])

        log_prob = -0.5 * (
            ((action - mean) / (std + 1e-8)) ** 2 + 2 * logstd + np.log(2 * np.pi)
        )

        return action.squeeze(-1), log_prob.squeeze(-1), value.squeeze(-1)

    def evaluate(self, obs, actions):
        mean, logstd, value = self.forward(obs)
        std = logstd.exp()

        log_prob = -0.5 * (
            ((actions.unsqueeze(-1) - mean) / (std + 1e-8)) ** 2
            + 2 * logstd
            + np.log(2 * np.pi)
        )
        entropy = 0.5 * (1 + np.log(2 * np.pi) + 2 * logstd)

        return log_prob.squeeze(-1), value.squeeze(-1), entropy.mean()


def train() -> None:
    config = Config()

    sim = BatchedSimulator(config)
    print(f"Using device: {sim.device}")

    model = ActorCritic(config.obs_dim, config.hidden_size).to(sim.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_cost = float("inf")

    for iter in range(config.total_iterations):
        # Entropy decay
        ent_coef = config.entropy_coef * (1 - iter / config.total_iterations)
        ent_coef = max(ent_coef, 0.01)

        obs_buf = []
        act_buf = []
        rew_buf = []
        logp_buf = []
        val_buf = []

        # Rollout
        obs = sim.reset()
        for _ in range(config.rollout_steps):
            with torch.no_grad():
                action, log_prob, value = model.get_action(obs)

            next_obs, reward, done = sim.step(action)

            obs_buf.append(obs)
            act_buf.append(action)
            rew_buf.append(reward)
            logp_buf.append(log_prob)
            val_buf.append(value)

            obs = next_obs

            if done:
                break

        # Stack buffers
        obs_t = torch.stack(obs_buf)
        act_t = torch.stack(act_buf)
        rew_t = torch.stack(rew_buf)
        logp_t = torch.stack(logp_buf)
        val_t = torch.stack(val_buf)

        # GAE
        with torch.no_grad():
            _, _, last_val = model.get_action(obs)

        advantages = torch.zeros_like(rew_t)
        last_gae = 0
        for t in reversed(range(len(rew_buf))):
            if t == len(rew_buf) - 1:
                nextval = last_val
            else:
                nextval = val_t[t + 1]
            delta = rew_t[t] + config.gamma * nextval - val_t[t]
            advantages[t] = last_gae = (
                delta + config.gamma * config.gae_lambda * last_gae
            )

        returns = advantages + val_t

        # Flatten
        T, N = obs_t.shape[:2]
        obs_flat = obs_t.reshape(T * N, -1)
        act_flat = act_t.reshape(T * N)
        logp_flat = logp_t.reshape(T * N)
        adv_flat = advantages.reshape(T * N)
        ret_flat = returns.reshape(T * N)
        val_flat = val_t.reshape(T * N)

        # Normalize advantages
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        # PPO update
        total_samples = T * N
        for _ in range(config.update_epochs):
            perm = torch.randperm(total_samples, device=sim.device)

            for start in range(0, total_samples, config.minibatch_size):
                idx = perm[start : start + config.minibatch_size]

                new_logp, new_val, entropy = model.evaluate(
                    obs_flat[idx], act_flat[idx]
                )

                ratio = torch.exp(new_logp - logp_flat[idx])
                surr1 = ratio * adv_flat[idx]
                surr2 = (
                    torch.clamp(ratio, 1 - config.clip_eps, 1 + config.clip_eps)
                    * adv_flat[idx]
                )

                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss clipping
                v_loss_unclipped = (new_val - ret_flat[idx]) ** 2
                v_clipped = val_flat[idx] + torch.clamp(
                    new_val - val_flat[idx],
                    -config.value_clip_eps,
                    config.value_clip_eps,
                )
                v_loss_clipped = (v_clipped - ret_flat[idx]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                value_loss = 0.5 * v_loss_max.mean()

                loss = policy_loss + config.value_coef * value_loss - ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

        # Evaluate
        total_cost, lat_cost, jerk_cost = sim.compute_cost()

        if total_cost < best_cost:
            best_cost = total_cost
            torch.save(model.state_dict(), "best_ppo.pt")

        print(
            f"Iter {iter:3d} | cost: {total_cost:7.2f} | best_cost: {best_cost:7.2f} | lat: {lat_cost:6.2f} | jerk: {jerk_cost:6.2f} | entropy: {entropy.item():.3f}"  # type: ignore[reportPossiblyUnboundVariable]
        )

        if iter % 10 == 0:
            print(
                f"  policy_loss: {policy_loss.item():.4f} | value_loss: {value_loss.item():.4f}"  # type: ignore[reportPossiblyUnboundVariable]
            )

        if iter % 200 == 0 and iter > 0:
            torch.save(model.state_dict(), f"checkpoint_{iter}.pt")


if __name__ == "__main__":
    train()
