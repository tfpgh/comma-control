import time
import torch
import numpy as np
import pandas as pd
import onnxruntime as ort
import cma
import torch.multiprocessing as mp
from pathlib import Path
from queue import Empty

# --- Configuration ---
NUM_GPUS = 4  # You have 4 A6000s
POPULATION_SIZE = 1024  # Total population (must be divisible by NUM_GPUS)
NUM_SEGMENTS = 64  # Routes per candidate
MAX_GENERATIONS = 10000
INPUT_SIZE = 10
HIDDEN_SIZE = 12

# Physics Constants
CONTEXT_LENGTH = 20
CONTROL_START_IDX = 100
COST_END_IDX = 500
LATACCEL_RANGE = (-5.0, 5.0)
MAX_ACC_DELTA = 0.5
DEL_T = 0.1
ACC_G = 9.81
VOCAB_SIZE = 1024


class GPUWorker(mp.Process):
    def __init__(self, device_id, task_queue, result_queue, data_path, model_path):
        super().__init__()
        self.device_id = device_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.data_path = Path(data_path)
        self.model_path = model_path
        self.device = torch.device(f"cuda:{device_id}")

    def run(self):
        print(f"[GPU {self.device_id}] Initializing...")
        
        # 1. Load Data (Pinned to this GPU)
        self.all_data = self._load_data()
        self.num_total_segments = self.all_data.shape[0]
        
        # 2. Load ONNX Model
        # Note: We prioritize CUDAExecutionProvider for this specific device
        providers = [
            ("CUDAExecutionProvider", {"device_id": self.device_id}),
            "CPUExecutionProvider"
        ]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.ort_session = ort.InferenceSession(self.model_path, opts, providers=providers)
        
        self.bins = torch.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE, device=self.device)
        
        print(f"[GPU {self.device_id}] Ready.")

        while True:
            try:
                # Wait for a batch of parameters
                # task is a tuple: (population_chunk_params, seed)
                task = self.task_queue.get()
                if task is None: # Sentinel for shutdown
                    break
                
                params, seed = task
                costs = self.rollout_batch(params, seed)
                self.result_queue.put((self.device_id, costs))
                
            except Exception as e:
                print(f"[GPU {self.device_id}] Error: {e}")
                break

    def _load_data(self):
        csv_files = sorted(self.data_path.glob("*.csv"))[:5000]
        segments = []
        for f in csv_files:
            df = pd.read_csv(f)
            if len(df) < 550: continue
            data = np.column_stack([
                np.sin(df["roll"].values[:550]) * ACC_G,
                df["vEgo"].values[:550],
                df["aEgo"].values[:550],
                df["targetLateralAcceleration"].values[:550],
                -df["steerCommand"].values[:550]
            ])
            segments.append(data)
        return torch.tensor(np.stack(segments), dtype=torch.float32, device=self.device)

    def rollout_batch(self, population_params, seed):
        # Seed logic for deterministic runs per generation if needed
        # We use a random subset of segments, but synced across GPUs? 
        # No, simpler to let each GPU pick random segments.
        
        pop_size = population_params.shape[0]
        n_segs = NUM_SEGMENTS
        batch_size = pop_size * n_segs
        
        # 1. Random Segments
        seg_indices = torch.randint(0, self.num_total_segments, (n_segs,), device=self.device)
        batch_data = self.all_data[seg_indices].repeat(pop_size, 1, 1)
        
        # 2. Unpack Params
        params = torch.tensor(population_params, dtype=torch.float32, device=self.device)
        
        idx = 0
        w1 = params[:, idx : idx + INPUT_SIZE * HIDDEN_SIZE].reshape(pop_size, INPUT_SIZE, HIDDEN_SIZE)
        idx += INPUT_SIZE * HIDDEN_SIZE
        b1 = params[:, idx : idx + HIDDEN_SIZE].reshape(pop_size, 1, HIDDEN_SIZE)
        idx += HIDDEN_SIZE
        w2 = params[:, idx : idx + HIDDEN_SIZE].reshape(pop_size, HIDDEN_SIZE, 1)
        idx += HIDDEN_SIZE
        b2 = params[:, idx : idx + 1].reshape(pop_size, 1, 1)
        
        # 3. Sim State
        lataccel_history = batch_data[:, :, 3].clone()
        action_history = batch_data[:, :, 4].clone()
        current_lataccel = lataccel_history[:, CONTEXT_LENGTH-1]
        
        prev_error = torch.zeros(batch_size, 1, device=self.device)
        error_integral = torch.zeros(batch_size, 1, device=self.device)
        prev_action = torch.zeros(batch_size, 1, device=self.device)
        
        targets = batch_data[:, :, 3]

        # 4. Step Loop
        for step in range(CONTEXT_LENGTH, COST_END_IDX):
            # Features
            idx_now = step
            error = targets[:, idx_now:idx_now+1] - current_lataccel.unsqueeze(1)
            error_deriv = error - prev_error
            error_integral = torch.clamp(error_integral + error, -5.0, 5.0)
            
            end_near = min(step + 5, 550)
            future_near = targets[:, step+1:end_near].mean(dim=1, keepdim=True)
            if step + 1 >= end_near: future_near = targets[:, idx_now:idx_now+1]
                
            end_mid = min(step + 15, 550)
            future_mid = targets[:, step+5:end_mid].mean(dim=1, keepdim=True)
            if step + 5 >= end_mid: future_mid = targets[:, idx_now:idx_now+1]

            x = torch.cat([
                error / 5.0,
                error_deriv / 2.0,
                error_integral / 5.0,
                targets[:, idx_now:idx_now+1] / 5.0,
                batch_data[:, idx_now, 1:2] / 30.0,
                batch_data[:, idx_now, 2:3] / 4.0,
                batch_data[:, idx_now, 0:1] / 2.0,
                prev_action / 2.0,
                future_near / 5.0,
                future_mid / 5.0,
            ], dim=1)
            
            # Controller (Batch Matrix Mul)
            x_grouped = x.view(pop_size, n_segs, INPUT_SIZE)
            h1 = torch.tanh(torch.matmul(x_grouped, w1) + b1)
            out = torch.tanh(torch.matmul(h1, w2) + b2)
            action_grouped = out * 2.0
            action = action_grouped.view(batch_size, 1)
            
            prev_error = error
            prev_action = action
            action_history[:, step] = action.squeeze()
            
            # Physics (ONNX)
            start_ctx = step - CONTEXT_LENGTH
            ctx_actions = action_history[:, start_ctx:step].unsqueeze(2)
            ctx_states = batch_data[:, start_ctx:step, 0:3]
            onnx_states = torch.cat([ctx_actions, ctx_states], dim=2)
            ctx_lataccel = lataccel_history[:, start_ctx:step].clamp(*LATACCEL_RANGE)
            onnx_tokens = torch.searchsorted(self.bins, ctx_lataccel.contiguous())
            
            # Sync Point (CPU Transfer)
            states_np = onnx_states.cpu().numpy().astype(np.float32)
            tokens_np = onnx_tokens.cpu().numpy().astype(np.int64)
            logits = self.ort_session.run(None, {"states": states_np, "tokens": tokens_np})[0]
            
            logits_t = torch.from_numpy(logits).to(self.device)
            probs = torch.softmax(logits_t[:, -1, :] / 0.8, dim=-1)
            samples = torch.multinomial(probs, 1).squeeze(-1)
            pred_lataccel = self.bins[samples]
            
            pred_lataccel = torch.clamp(
                pred_lataccel,
                current_lataccel - MAX_ACC_DELTA,
                current_lataccel + MAX_ACC_DELTA
            )
            
            if step >= CONTROL_START_IDX:
                current_lataccel = pred_lataccel
            else:
                current_lataccel = targets[:, step]
            lataccel_history[:, step] = current_lataccel

        # Cost Calc
        targets_slice = targets[:, CONTROL_START_IDX:COST_END_IDX]
        actuals_slice = lataccel_history[:, CONTROL_START_IDX:COST_END_IDX]
        lat_cost = torch.mean((actuals_slice - targets_slice) ** 2, dim=1) * 100.0
        jerk = (actuals_slice[:, 1:] - actuals_slice[:, :-1]) / DEL_T
        jerk_cost = torch.mean(jerk ** 2, dim=1) * 100.0
        total_cost = (lat_cost * 50.0) + jerk_cost
        
        return torch.mean(total_cost.view(pop_size, n_segs), dim=1).cpu().numpy()


def main():
    mp.set_start_method('spawn')
    
    print(f"Starting 4-GPU CMA-ES. Total Pop: {POPULATION_SIZE} ({POPULATION_SIZE//NUM_GPUS} per GPU)")
    
    task_queues = [mp.Queue() for _ in range(NUM_GPUS)]
    result_queue = mp.Queue()
    
    workers = []
    for i in range(NUM_GPUS):
        p = GPUWorker(i, task_queues[i], result_queue, "./data", "./models/tinyphysics.onnx")
        p.start()
        workers.append(p)
        
    # CMA Setup
    num_params = (INPUT_SIZE * HIDDEN_SIZE) + HIDDEN_SIZE + HIDDEN_SIZE + 1
    x0 = np.random.randn(num_params) * 0.1
    es = cma.CMAEvolutionStrategy(x0, 0.5, {
        'popsize': POPULATION_SIZE,
        'maxiter': MAX_GENERATIONS
    })
    
    best_ever = float("inf")
    
    try:
        while not es.stop():
            start_time = time.time()
            solutions = es.ask()
            
            # Split Params for GPUs
            chunk_size = len(solutions) // NUM_GPUS
            chunks = [solutions[i:i + chunk_size] for i in range(0, len(solutions), chunk_size)]
            
            # Dispatch
            for i in range(NUM_GPUS):
                task_queues[i].put((np.stack(chunks[i]), 0)) # 0 is dummy seed
                
            # Collect
            all_costs = [None] * NUM_GPUS
            for _ in range(NUM_GPUS):
                device_id, costs = result_queue.get()
                all_costs[device_id] = costs
                
            # Flatten results in correct order
            flat_costs = np.concatenate(all_costs)
            
            es.tell(solutions, flat_costs)
            
            gen_best = np.min(flat_costs)
            gen_mean = np.mean(flat_costs)
            duration = time.time() - start_time
            
            if gen_best < best_ever:
                best_ever = gen_best
                np.save("4gpu_best_params.npy", es.result.xbest)
                
            print(f"Gen {es.countiter:4d} | Best: {gen_best:6.2f} | Mean: {gen_mean:6.2f} | Time: {duration:.2f}s")
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        for q in task_queues:
            q.put(None)
        for w in workers:
            w.join()

if __name__ == "__main__":
    main()
