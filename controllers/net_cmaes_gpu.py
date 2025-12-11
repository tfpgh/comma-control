import argparse
import pickle
import time
from pathlib import Path

import cma
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch

# Configuration
INPUT_SIZE = 10
HIDDEN_SIZE = 12  # Matches Run 2 (Small & Efficient)
POPULATION_SIZE = 256  # Huge population for GPU
NUM_SEGMENTS = 64  # Routes per candidate
MAX_GENERATIONS = 10000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Physics Constants
CONTEXT_LENGTH = 20
CONTROL_START_IDX = 100
COST_END_IDX = 500
LATACCEL_RANGE = (-5.0, 5.0)
STEER_RANGE = (-2.0, 2.0)
MAX_ACC_DELTA = 0.5
DEL_T = 0.1
ACC_G = 9.81
VOCAB_SIZE = 1024


class GPURunner:
    def __init__(self, model_path: str, data_path: str):
        print(f"Initializing GPU Runner on {DEVICE}...")
        
        # 1. Load Data to GPU
        self.data_path = Path(data_path)
        self.all_data = self._load_data()
        self.num_total_segments = self.all_data.shape[0]
        
        # 2. Load Physics Model (ONNX)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if DEVICE.type == "cpu":
            providers = ["CPUExecutionProvider"]
            
        self.ort_session = ort.InferenceSession(model_path, opts, providers=providers)
        
        # Pre-compute token bins
        self.bins = torch.linspace(
            LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE, device=DEVICE
        )

    def _load_data(self) -> torch.Tensor:
        """Loads all CSVs into a single (N, Steps, 5) tensor on GPU."""
        csv_files = sorted(self.data_path.glob("*.csv"))[:5000]
        print(f"Loading {len(csv_files)} data files...")
        
        segments = []
        for f in csv_files:
            df = pd.read_csv(f)
            if len(df) < 550:
                continue
            
            # Extract columns: roll, vEgo, aEgo, targetLat, steerCommand
            # Shape: (Steps, 5)
            data = np.column_stack([
                np.sin(df["roll"].values[:550]) * ACC_G,
                df["vEgo"].values[:550],
                df["aEgo"].values[:550],
                df["targetLateralAcceleration"].values[:550],
                -df["steerCommand"].values[:550] # Negate steer to match internal sign convention
            ])
            segments.append(data)
            
        tensor = torch.tensor(np.stack(segments), dtype=torch.float32, device=DEVICE)
        print(f"Data loaded: {tensor.shape} (Segments, Steps, Channels)")
        return tensor

    def rollout_batch(self, population_params: np.ndarray) -> np.ndarray:
        """
        Runs the entire population against random segments in parallel.
        population_params: (PopSize, NumParams)
        Returns: (PopSize,) cost array
        """
        pop_size = population_params.shape[0]
        n_segs = NUM_SEGMENTS
        batch_size = pop_size * n_segs
        
        # 1. Select random segments for this generation
        seg_indices = torch.randint(
            0, self.num_total_segments, (n_segs,), device=DEVICE
        )
        # Repeat segments for each population member: (PopSize * NumSegs)
        # [Seg1, Seg2, ..., Seg1, Seg2, ...]
        batch_data = self.all_data[seg_indices].repeat(pop_size, 1, 1)
        
        # 2. Unpack Weights for parallel execution
        # Weights shape: (PopSize, Input, Hidden), (PopSize, Hidden), ...
        # We need to repeat them for each segment: (PopSize * NumSegs, ...)
        
        # Parse flat params into tensors
        # Structure: W1(10x12), B1(12), W2(12x1), B2(1)
        # Total: 120 + 12 + 12 + 1 = 145 params
        
        params = torch.tensor(population_params, dtype=torch.float32, device=DEVICE)
        
        idx = 0
        # W1
        w1_size = INPUT_SIZE * HIDDEN_SIZE
        w1 = params[:, idx:idx+w1_size].reshape(pop_size, INPUT_SIZE, HIDDEN_SIZE)
        idx += w1_size
        
        # B1
        b1_size = HIDDEN_SIZE
        b1 = params[:, idx:idx+b1_size].reshape(pop_size, 1, HIDDEN_SIZE)
        idx += b1_size
        
        # W2
        w2_size = HIDDEN_SIZE * 1
        w2 = params[:, idx:idx+w2_size].reshape(pop_size, HIDDEN_SIZE, 1)
        idx += w2_size
        
        # B2
        b2_size = 1
        b2 = params[:, idx:idx+b2_size].reshape(pop_size, 1, 1)
        
        # Expand weights to match batch size (PopSize * NumSegs)
        # w1 becomes (PopSize, 1, In, Hid) -> broadcast over segments
        
        # 3. Simulation Loop
        # Init State
        lataccel_history = batch_data[:, :, 3].clone() # Fill with target initially
        # Initialize action_history with zeros to match eval controller behavior
        # Only pre-fill context window (0-19) from data for physics simulation
        action_history = torch.zeros_like(batch_data[:, :, 4])
        action_history[:, :CONTEXT_LENGTH] = batch_data[:, :CONTEXT_LENGTH, 4].clone()
        current_lataccel = lataccel_history[:, CONTEXT_LENGTH-1]
        
        # Controller State
        prev_error = torch.zeros(batch_size, 1, device=DEVICE)
        error_integral = torch.zeros(batch_size, 1, device=DEVICE)
        prev_action = torch.zeros(batch_size, 1, device=DEVICE)
        
        # Pre-calculate future windows
        targets = batch_data[:, :, 3]
        
        for step in range(CONTEXT_LENGTH, COST_END_IDX):
            # --- A. Feature Extraction ---
            # Indices
            idx_now = step
            
            # State vars
            roll = batch_data[:, idx_now, 0:1]
            v_ego = batch_data[:, idx_now, 1:2]
            a_ego = batch_data[:, idx_now, 2:3]
            target = batch_data[:, idx_now, 3:4]
            
            # Error
            error = target - current_lataccel.unsqueeze(1)
            error_deriv = error - prev_error
            error_integral = torch.clamp(error_integral + error, -5.0, 5.0)
            
            # Future (Vectorized window mean)
            # Slicing with clamp to handle end of episode
            end_near = min(step + 5, 550)
            future_near = targets[:, step+1:end_near].mean(dim=1, keepdim=True)
            if step + 1 >= end_near: future_near = target
                
            end_mid = min(step + 15, 550)
            future_mid = targets[:, step+5:end_mid].mean(dim=1, keepdim=True)
            if step + 5 >= end_mid: future_mid = target

            # Stack Inputs (Batch, 10)
            # Scaling matched to Runs 1-3
            x = torch.cat([
                error / 5.0,
                error_deriv / 2.0,
                error_integral / 5.0,
                target / 5.0,
                v_ego / 30.0,
                a_ego / 4.0,
                roll / 2.0,
                prev_action / 2.0,
                future_near / 5.0,
                future_mid / 5.0,
            ], dim=1)
            
            # --- B. Neural Network Pass (The "Trick") ---
            # We need to apply 256 different networks to 256 groups of 64 cars.
            # Reshape input: (Pop, Segs, Input)
            x_grouped = x.view(pop_size, n_segs, INPUT_SIZE)
            
            # Layer 1: (Pop, Segs, In) @ (Pop, In, Hid) -> (Pop, Segs, Hid)
            # Torch matmul broadcasts automatically on the first dim!
            h1 = torch.tanh(torch.matmul(x_grouped, w1) + b1)
            
            # Layer 2: (Pop, Segs, Hid) @ (Pop, Hid, 1) -> (Pop, Segs, 1)
            out = torch.tanh(torch.matmul(h1, w2) + b2)
            
            # Action scale
            action_grouped = out * 2.0
            
            # Flatten back to (Batch, 1)
            action = action_grouped.view(batch_size, 1)
            
            # Update Controller State
            prev_error = error
            prev_action = action
            
            # --- C. Physics Step ---
            # Update History
            action_history[:, step] = action.squeeze()
            
            # Prepare ONNX Inputs
            # Context: (Batch, 20, 4) -> [Actions, Roll, VEgo, AEgo]
            start_ctx = step - CONTEXT_LENGTH
            ctx_actions = action_history[:, start_ctx:step].unsqueeze(2)
            ctx_states = batch_data[:, start_ctx:step, 0:3]
            onnx_states = torch.cat([ctx_actions, ctx_states], dim=2)
            
            ctx_lataccel = lataccel_history[:, start_ctx:step].clamp(*LATACCEL_RANGE)
            onnx_tokens = torch.searchsorted(self.bins, ctx_lataccel.contiguous())
            
            # IO Binding for Speed (Keep tensors on GPU)
            # Currently ORT python API with GPU tensors is tricky, we fallback to numpy transfer for just the IO
            # This is the bottleneck, but still faster than CPU multiprocessing
            # Ideally we use IO Binding here, but standard run() is safer for now.
            
            # Note: We can't batch 6400 into one ONNX call easily if ORT limits batch size
            # But TinyPhysics is small, it should handle it.
            
            states_np = onnx_states.cpu().numpy().astype(np.float32)
            tokens_np = onnx_tokens.cpu().numpy().astype(np.int64)
            
            logits = self.ort_session.run(None, {"states": states_np, "tokens": tokens_np})[0]
            
            # Sampling (Argmax or Multinomial? Simulator uses Multinomial with temp 0.8)
            # For CMA-ES evaluation, deterministic (Argmax) is often better/stable, 
            # but we should match the challenge physics.
            logits_t = torch.from_numpy(logits).to(DEVICE)
            probs = torch.softmax(logits_t[:, -1, :] / 0.8, dim=-1)
            
            # Fast multinomial? Or just take expectation/argmax for stability?
            # Let's use expectation (weighted average) for smooth gradients? 
            # No, simulation is discrete. Let's sample.
            samples = torch.multinomial(probs, 1).squeeze(-1)
            pred_lataccel = self.bins[samples]
            
            # Clamp physics limits
            pred_lataccel = torch.clamp(
                pred_lataccel,
                current_lataccel - MAX_ACC_DELTA,
                current_lataccel + MAX_ACC_DELTA
            )
            
            # Update Physics State
            if step >= CONTROL_START_IDX:
                current_lataccel = pred_lataccel
            else:
                current_lataccel = targets[:, step] # Data fallback
                
            lataccel_history[:, step] = current_lataccel

        # 4. Calculate Costs
        # Slice relevant range
        targets = targets[:, CONTROL_START_IDX:COST_END_IDX]
        actuals = lataccel_history[:, CONTROL_START_IDX:COST_END_IDX]
        
        # Lataccel Cost
        lat_cost = torch.mean((actuals - targets) ** 2, dim=1) * 100.0
        
        # Jerk Cost
        jerk = (actuals[:, 1:] - actuals[:, :-1]) / DEL_T
        jerk_cost = torch.mean(jerk ** 2, dim=1) * 100.0
        
        total_cost = (lat_cost * 50.0) + jerk_cost
        
        # Reshape to (Pop, Segs) and average over segments
        total_cost_grouped = total_cost.view(pop_size, n_segs)
        mean_cost = torch.mean(total_cost_grouped, dim=1)
        
        return mean_cost.cpu().numpy()


def main():
    print(f"Starting GPU CMA-ES with Population {POPULATION_SIZE} on {NUM_SEGMENTS} segments.")
    runner = GPURunner(
        model_path="./models/tinyphysics.onnx",
        data_path="./data"
    )
    
    # Calculate Params Size
    num_params = (INPUT_SIZE * HIDDEN_SIZE) + HIDDEN_SIZE + (HIDDEN_SIZE * 1) + 1
    print(f"Parameter Vector Size: {num_params}")
    
    # Init CMA
    x0 = np.random.randn(num_params) * 0.1
    es = cma.CMAEvolutionStrategy(x0, 0.5, {
        'popsize': POPULATION_SIZE,
        'maxiter': MAX_GENERATIONS
    })
    
    best_ever = float("inf")
    
    while not es.stop():
        start_time = time.time()
        
        solutions = es.ask()
        
        # Convert list of arrays to single (Pop, Params) array
        pop_params = np.stack(solutions)
        
        # GPU Evaluation
        costs = runner.rollout_batch(pop_params)
        
        es.tell(solutions, costs)
        
        # Stats
        gen_best = np.min(costs)
        gen_mean = np.mean(costs)
        duration = time.time() - start_time
        
        if gen_best < best_ever:
            best_ever = gen_best
            np.save("gpu_best_params.npy", es.result.xbest)
            
        print(
            f"Gen {es.countiter:4d} | Best: {gen_best:6.2f} | Mean: {gen_mean:6.2f} | "
            f"AllTime: {best_ever:6.2f} | Time: {duration:.2f}s"
        )
        
        if es.countiter % 50 == 0:
             np.save(f"gpu_checkpoint_{es.countiter}.npy", es.result.xbest)


if __name__ == "__main__":
    main()
