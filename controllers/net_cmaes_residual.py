from pathlib import Path

import numpy as np

from tinyphysics import FuturePlan, State

from . import BaseController

# Architecture Constants (Must match training!)
INPUT_SIZE = 15
HIDDEN_SIZE = 18

# PID Constants (From training script)
PID_KP = 0.195
PID_KI = 0.100
PID_KD = -0.053


class Controller(BaseController):
    def __init__(self):
        # Weights
        self.w1 = np.zeros((INPUT_SIZE, HIDDEN_SIZE))
        self.b1 = np.zeros(HIDDEN_SIZE)
        self.w2 = np.zeros((HIDDEN_SIZE, 1))
        self.b2 = np.zeros(1)

        # State
        self.prev_error = 0.0
        self.error_integral = 0.0
        self.prev_action = 0.0
        self.prev_prev_action = 0.0  # t-2

        # Auto-load parameters
        params_path = Path("residual_best_params.npy")
        if params_path.exists():
            self.set_params(np.load(params_path))
        else:
            print(
                "WARNING: residual_best_params.npy not found! Using zero weights (Pure PID)."
            )

    def set_params(self, params: np.ndarray) -> None:
        # Unpack flat params back into weights
        idx = 0

        # W1: (15, 18)
        w1_size = INPUT_SIZE * HIDDEN_SIZE
        self.w1 = params[idx : idx + w1_size].reshape(INPUT_SIZE, HIDDEN_SIZE)
        idx += w1_size

        # B1: (18,)
        b1_size = HIDDEN_SIZE
        self.b1 = params[idx : idx + b1_size]
        idx += b1_size

        # W2: (18, 1)
        w2_size = HIDDEN_SIZE * 1
        self.w2 = params[idx : idx + w2_size].reshape(HIDDEN_SIZE, 1)
        idx += w2_size

        # B2: (1,)
        b2_size = 1
        self.b2 = params[idx : idx + b2_size]

    def reset(self) -> None:
        self.prev_error = 0.0
        self.error_integral = 0.0
        self.prev_action = 0.0
        self.prev_prev_action = 0.0

    def update(
        self,
        target_lataccel: np.float64,
        current_lataccel: np.float64,
        state: State,
        future_plan: FuturePlan,
    ) -> float:
        # --- 1. Feature Extraction (Match net_cmaes_residual_4gpu.py logic) ---

        # Error Terms
        error = target_lataccel - current_lataccel
        error_deriv = error - self.prev_error
        self.error_integral = np.clip(self.error_integral + error, -5.0, 5.0)

        # Future Window (Raw Points - 6 points)
        fp = future_plan.lataccel if future_plan.lataccel else []
        # Pad with target if short
        if len(fp) < 6:
            fp = list(fp) + [target_lataccel] * (6 - len(fp))
        raw_future = np.array(fp[:6])

        # Input Vector Construction (15 Dim)
        # Order: [Error, Deriv, Int, Target, V, A, Roll, U_t1, U_t2, Future(6)]
        x = np.array(
            [
                error / 5.0,
                error_deriv / 2.0,
                self.error_integral / 5.0,
                target_lataccel / 5.0,
                state.v_ego / 30.0,
                state.a_ego / 4.0,
                state.roll_lataccel / 2.0,
                self.prev_action / 2.0,  # t-1
                self.prev_prev_action / 2.0,  # t-2
                *(raw_future / 5.0),  # 6 Future points
            ]
        )

        # --- 2. Neural Network (Residual) ---
        # Layer 1
        h1 = np.tanh(x @ self.w1 + self.b1)
        # Layer 2
        out = np.tanh(h1 @ self.w2 + self.b2)

        # Scale (1.0 matching training script)
        nn_residual = float(out[0]) * 1.0

        # --- 3. PID Controller (Base) ---
        p_term = error * PID_KP
        i_term = self.error_integral * PID_KI  # Already clamped
        d_term = error_deriv * PID_KD

        pid_action = p_term + i_term + d_term

        # --- 4. Combine ---
        final_action = pid_action + nn_residual

        # --- 5. Update State ---
        self.prev_error = error
        self.prev_prev_action = self.prev_action
        self.prev_action = final_action

        return final_action
