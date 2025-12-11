import time
from pathlib import Path

import cma
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import torch.multiprocessing as mp

# --- Configuration ---
NUM_GPUS = 4
POPULATION_SIZE = 512
NUM_SEGMENTS = 64
MAX_GENERATIONS = 10000
INPUT_SIZE = 15
HIDDEN_SIZE = 18

# Physics Constants
CONTEXT_LENGTH = 20
CONTROL_START_IDX = 100
COST_END_IDX = 500
LATACCEL_RANGE = (-5.0, 5.0)
MAX_ACC_DELTA = 0.5
DEL_T = 0.1
ACC_G = 9.81
VOCAB_SIZE = 1024

# PID
PID_KP = 0.195
PID_KI = 0.100
PID_KD = -0.053


class GPUWorker(mp.Process):
    def __init__(self, device_id, task_queue, result_queue, data_path, model_path):
        super().__init__()
        self.device_id = device_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.data_path = Path(data_path)
        self.model_path = model_path

    def run(self):
        # 1. Setup Device
        self.device = torch.device(f"cuda:{self.device_id}")
        torch.cuda.set_device(self.device)
        print(f"[GPU {self.device_id}] Initializing I/O Binding Worker...")

        # 2. Load Data (Pinned to VRAM)
        self.all_data = self._load_data()
        self.num_total_segments = self.all_data.shape[0]

        # 3. Load ONNX Model & Verify CUDA
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            self.session = ort.InferenceSession(
                self.model_path,
                opts,
                providers=[("CUDAExecutionProvider", {"device_id": self.device_id})],
            )
        except Exception as e:
            print(f"[GPU {self.device_id}] FAILED to load CUDA provider: {e}")
            return

        active_providers = self.session.get_providers()
        if "CUDAExecutionProvider" not in active_providers:
            print(
                f"[GPU {self.device_id}] WARNING: ORT fallback to {active_providers}. Performance will be terrible."
            )

        # 4. Introspect Model for Names
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.name_states = next(
            (n for n in self.input_names if "state" in n), self.input_names[0]
        )
        self.name_tokens = next(
            (n for n in self.input_names if "token" in n), self.input_names[1]
        )
        self.name_output = self.output_names[0]

        print(
            f"[GPU {self.device_id}] Bindings: {self.name_states}, {self.name_tokens} -> {self.name_output}"
        )

        # 5. Pre-allocate IO Buffers
        self.local_pop_size = POPULATION_SIZE // NUM_GPUS
        self.batch_size = self.local_pop_size * NUM_SEGMENTS

        # Input Buffers (Reusable memory)
        self.onnx_state_buffer = torch.zeros(
            (self.batch_size, 20, 4), dtype=torch.float32, device=self.device
        )
        self.onnx_token_buffer = torch.zeros(
            (self.batch_size, 20), dtype=torch.int64, device=self.device
        )

        # Output Buffer (Batch, 20, 1024)
        self.onnx_output_buffer = torch.zeros(
            (self.batch_size, 20, VOCAB_SIZE), dtype=torch.float32, device=self.device
        )

        # Setup Binding
        self.binding = self.session.io_binding()

        # Bind Output (Persistent)
        self.binding.bind_output(
            name=self.name_output,
            device_type="cuda",
            device_id=self.device_id,
            element_type=np.float32,
            shape=tuple(self.onnx_output_buffer.shape),
            buffer_ptr=self.onnx_output_buffer.data_ptr(),
        )

        self.bins = torch.linspace(
            LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE, device=self.device
        )
        print(f"[GPU {self.device_id}] Ready. Batch Size: {self.batch_size}")

        # Task Loop
        while True:
            try:
                task = self.task_queue.get()
                if task is None:
                    break

                params, seed = task
                costs, lat, jerk = self.rollout_batch(params)
                self.result_queue.put((self.device_id, costs, lat, jerk))

            except Exception as e:
                print(f"[GPU {self.device_id}] CRITICAL ERROR: {e}")
                import traceback

                traceback.print_exc()
                break

    def _load_data(self):
        csv_files = sorted(self.data_path.glob("*.csv"))[:5000]
        segments = []
        for f in csv_files:
            df = pd.read_csv(f)
            if len(df) < 550:
                continue
            data = np.column_stack(
                [
                    np.sin(df["roll"].values[:550]) * ACC_G,
                    df["vEgo"].values[:550],
                    df["aEgo"].values[:550],
                    df["targetLateralAcceleration"].values[:550],
                    -df["steerCommand"].values[:550],
                ]
            )
            segments.append(data)
        return torch.tensor(np.stack(segments), dtype=torch.float32, device=self.device)

    def rollout_batch(self, population_params):
        pop_size = population_params.shape[0]
        n_segs = NUM_SEGMENTS
        batch_size = self.batch_size

        # 1. Select Random Segments
        seg_indices = torch.randint(
            0, self.num_total_segments, (n_segs,), device=self.device
        )
        batch_data = self.all_data[seg_indices].repeat(pop_size, 1, 1)

        # 2. Parse Controller Weights
        params = torch.tensor(
            population_params, dtype=torch.float32, device=self.device
        )

        idx = 0
        w1 = params[:, idx : idx + INPUT_SIZE * HIDDEN_SIZE].reshape(
            pop_size, INPUT_SIZE, HIDDEN_SIZE
        )
        idx += INPUT_SIZE * HIDDEN_SIZE
        b1 = params[:, idx : idx + HIDDEN_SIZE].reshape(pop_size, 1, HIDDEN_SIZE)
        idx += HIDDEN_SIZE
        w2 = params[:, idx : idx + HIDDEN_SIZE].reshape(pop_size, HIDDEN_SIZE, 1)
        idx += HIDDEN_SIZE
        b2 = params[:, idx : idx + 1].reshape(pop_size, 1, 1)

        # 3. Initialize Sim State
        lataccel_history = batch_data[:, :, 3].clone()
        # Initialize action_history with zeros to match eval controller behavior
        # Only pre-fill context window (0-19) from data for physics simulation
        action_history = torch.zeros_like(batch_data[:, :, 4])
        action_history[:, :CONTEXT_LENGTH] = batch_data[:, :CONTEXT_LENGTH, 4].clone()
        current_lataccel = lataccel_history[:, CONTEXT_LENGTH - 1]

        prev_error = torch.zeros(batch_size, 1, device=self.device)
        error_integral = torch.zeros(batch_size, 1, device=self.device)

        targets = batch_data[:, :, 3]

        # 4. Simulation Loop
        for step in range(CONTEXT_LENGTH, COST_END_IDX):
            # --- Feature Extraction ---
            idx_now = step
            error = targets[:, idx_now : idx_now + 1] - current_lataccel.unsqueeze(1)
            error_deriv = error - prev_error
            error_integral = torch.clamp(error_integral + error, -5.0, 5.0)

            # Future Window (Raw Points)
            end_idx = min(step + 7, 550)
            raw_future = targets[:, step + 1 : end_idx]

            # Pad if needed
            missing = 6 - raw_future.shape[1]
            if missing > 0:
                pad = targets[:, idx_now : idx_now + 1].repeat(1, missing)
                raw_future = torch.cat([raw_future, pad], dim=1)

            # Fetch past actions consistently
            u_t1 = (
                action_history[:, step - 1].unsqueeze(1)
                if step >= 1
                else torch.zeros_like(current_lataccel.unsqueeze(1))
            )
            u_t2 = (
                action_history[:, step - 2].unsqueeze(1)
                if step >= 2
                else torch.zeros_like(u_t1)
            )

            x = torch.cat(
                [
                    error / 5.0,
                    error_deriv / 2.0,
                    error_integral / 5.0,
                    targets[:, idx_now : idx_now + 1] / 5.0,
                    batch_data[:, idx_now, 1:2] / 30.0,
                    batch_data[:, idx_now, 2:3] / 4.0,
                    batch_data[:, idx_now, 0:1] / 2.0,
                    u_t1 / 2.0,
                    u_t2 / 2.0,
                    raw_future / 5.0,
                ],
                dim=1,
            )

            # --- Controller ---
            x_grouped = x.view(pop_size, n_segs, INPUT_SIZE)
            h1 = torch.tanh(torch.matmul(x_grouped, w1) + b1)
            out = torch.tanh(torch.matmul(h1, w2) + b2)

            residual = (out * 1.0).view(batch_size, 1)

            p_term = error * PID_KP
            i_term = error_integral * PID_KI
            d_term = error_deriv * PID_KD

            pid_action = p_term + i_term + d_term

            action = pid_action + residual

            prev_error = error
            action_history[:, step] = action.squeeze()

            # --- Physics (IO Binding) ---
            start_ctx = step - CONTEXT_LENGTH

            # Fill buffers
            ctx_actions = action_history[:, start_ctx:step].unsqueeze(2)
            ctx_states = batch_data[:, start_ctx:step, 0:3]
            torch.cat([ctx_actions, ctx_states], dim=2, out=self.onnx_state_buffer)

            ctx_lataccel = lataccel_history[:, start_ctx:step].clamp(*LATACCEL_RANGE)
            torch.searchsorted(
                self.bins, ctx_lataccel.contiguous(), out=self.onnx_token_buffer
            )

            # Bind
            self.binding.bind_input(
                name=self.name_states,
                device_type="cuda",
                device_id=self.device_id,
                element_type=np.float32,
                shape=tuple(self.onnx_state_buffer.shape),
                buffer_ptr=self.onnx_state_buffer.data_ptr(),
            )
            self.binding.bind_input(
                name=self.name_tokens,
                device_type="cuda",
                device_id=self.device_id,
                element_type=np.int64,
                shape=tuple(self.onnx_token_buffer.shape),
                buffer_ptr=self.onnx_token_buffer.data_ptr(),
            )

            torch.cuda.synchronize(self.device)
            self.session.run_with_iobinding(self.binding)

            # Post-process
            logits = self.onnx_output_buffer[:, -1, :].view(batch_size, VOCAB_SIZE)
            probs = torch.softmax(logits / 0.8, dim=-1)
            samples = torch.multinomial(probs, 1).squeeze(-1)
            pred_lataccel = self.bins[samples]

            current_lataccel = torch.clamp(
                pred_lataccel,
                current_lataccel - MAX_ACC_DELTA,
                current_lataccel + MAX_ACC_DELTA,
            )

            if step < CONTROL_START_IDX:
                current_lataccel = targets[:, step]
            lataccel_history[:, step] = current_lataccel

        # Cost Calc
        targets_slice = targets[:, CONTROL_START_IDX:COST_END_IDX]
        actuals_slice = lataccel_history[:, CONTROL_START_IDX:COST_END_IDX]
        lat_cost = torch.mean((actuals_slice - targets_slice) ** 2, dim=1) * 100.0
        jerk = (actuals_slice[:, 1:] - actuals_slice[:, :-1]) / DEL_T
        jerk_cost = torch.mean(jerk**2, dim=1) * 100.0
        total_cost = (lat_cost * 50.0) + jerk_cost

        # Aggregate
        costs = torch.mean(total_cost.view(pop_size, n_segs), dim=1).cpu().numpy()
        mean_lat = torch.mean(lat_cost).cpu().item()
        mean_jerk = torch.mean(jerk_cost).cpu().item()

        return costs, mean_lat, mean_jerk


def main():
    mp.set_start_method("spawn", force=True)

    print("Starting 4-GPU CMA-ES (High-Performance IO Binding).")
    print(f"Population: {POPULATION_SIZE} | Segments: {NUM_SEGMENTS}")

    task_queues = [mp.Queue() for _ in range(NUM_GPUS)]
    result_queue = mp.Queue()

    workers = []
    for i in range(NUM_GPUS):
        p = GPUWorker(
            i, task_queues[i], result_queue, "./data", "./models/tinyphysics.onnx"
        )
        p.start()
        workers.append(p)

    # num_params = (INPUT_SIZE * HIDDEN_SIZE) + HIDDEN_SIZE + HIDDEN_SIZE + 1
    # x0 = np.random.randn(num_params) * 0.01

    x0 = np.load("residual_best_params.npy")

    es = cma.CMAEvolutionStrategy(
        x0, 0.5, {"popsize": POPULATION_SIZE, "maxiter": MAX_GENERATIONS}
    )

    print("Optimization started...")
    best_ever = float("inf")

    try:
        while not es.stop():
            start_time = time.time()
            solutions = es.ask()

            chunk_size = len(solutions) // NUM_GPUS
            chunks = [
                solutions[i : i + chunk_size]
                for i in range(0, len(solutions), chunk_size)
            ]

            for i in range(NUM_GPUS):
                task_queues[i].put((np.stack(chunks[i]), 0))

            all_costs = [None] * NUM_GPUS
            total_lat = 0.0
            total_jerk = 0.0

            for _ in range(NUM_GPUS):
                device_id, costs, lat, jerk = result_queue.get()
                all_costs[device_id] = costs
                total_lat += lat
                total_jerk += jerk

            flat_costs = np.concatenate(all_costs)
            avg_lat = total_lat / NUM_GPUS
            avg_jerk = total_jerk / NUM_GPUS

            es.tell(solutions, flat_costs)

            gen_best = np.min(flat_costs)
            gen_mean = np.mean(flat_costs)
            duration = time.time() - start_time

            if gen_best < best_ever:
                best_ever = gen_best
                np.save("residual_best_params.npy", es.result.xbest)

            print(
                f"Gen {es.countiter:4d} | Best: {gen_best:6.2f} | Mean: {gen_mean:6.2f} | "
                f"BestEver: {best_ever:6.2f} | Lat: {avg_lat:5.2f} | Jerk: {avg_jerk:5.2f} | "
                f"Sigma: {es.sigma:.4f} | Time: {duration:.2f}s"
            )

            if es.countiter % 50 == 0:
                np.save(f"residual_checkpoint_{es.countiter}.npy", es.result.xbest)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        for q in task_queues:
            q.put(None)
        for w in workers:
            w.join()


if __name__ == "__main__":
    main()
