from pathlib import Path

import numpy as np

from controllers import BaseController
from tinyphysics import FuturePlan, State

INPUT_LAYER_SIZE = 10  # Changing requires adding or removing actual inputs
HIDDEN_LAYER_1_SIZE = 12
HIDDEN_LAYER_2_SIZE = 12
NUM_PARAMS = (
    INPUT_LAYER_SIZE * HIDDEN_LAYER_1_SIZE
    + HIDDEN_LAYER_1_SIZE
    + HIDDEN_LAYER_1_SIZE * HIDDEN_LAYER_2_SIZE
    + HIDDEN_LAYER_2_SIZE
    + HIDDEN_LAYER_2_SIZE
    + 1
)


class Controller(BaseController):
    def __init__(self) -> None:
        self.w1 = np.zeros((INPUT_LAYER_SIZE, HIDDEN_LAYER_1_SIZE))
        self.b1 = np.zeros(HIDDEN_LAYER_1_SIZE)
        self.w2 = np.zeros((HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE))
        self.b2 = np.zeros(HIDDEN_LAYER_2_SIZE)
        self.w3 = np.zeros((HIDDEN_LAYER_2_SIZE, 1))
        self.b3 = np.zeros(1)

        # integral/derivative state
        self.prev_error = 0.0
        self.error_integral = 0.0
        self.prev_action = 0.0

    def set_params(self, params: np.ndarray) -> None:
        idx = 0
        self.w1 = params[
            idx : (idx := idx + INPUT_LAYER_SIZE * HIDDEN_LAYER_1_SIZE)
        ].reshape(INPUT_LAYER_SIZE, HIDDEN_LAYER_1_SIZE)
        self.b1 = params[idx : (idx := idx + HIDDEN_LAYER_1_SIZE)]
        self.w2 = params[
            idx : (idx := idx + HIDDEN_LAYER_1_SIZE * HIDDEN_LAYER_2_SIZE)
        ].reshape(HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE)
        self.b2 = params[idx : (idx := idx + HIDDEN_LAYER_2_SIZE)]
        self.w3 = params[idx : (idx := idx + HIDDEN_LAYER_2_SIZE)].reshape(
            HIDDEN_LAYER_2_SIZE, 1
        )
        self.b3 = params[idx : (idx := idx + 1)]

    def reset(self) -> None:
        self.prev_error = 0.0
        self.error_integral = 0.0
        self.prev_action = 0.0

    def update(
        self,
        target_lataccel: np.float64,
        current_lataccel: np.float64,
        state: State,
        future_plan: FuturePlan,
    ) -> float:
        # calculate error terms
        error = target_lataccel - current_lataccel
        error_deriv = error - self.prev_error
        self.error_integral = np.clip(self.error_integral + error, -5, 5)

        # futures
        fp = future_plan.lataccel if future_plan.lataccel else []
        future_near = np.mean(fp[:5]) if len(fp) >= 5 else target_lataccel
        future_mid = np.mean(fp[5:15]) if len(fp) >= 15 else target_lataccel

        # build input features
        # here I normalize them to try to make learning faster
        # not sure this does much...
        x = np.array(
            [
                error / 5.0,
                error_deriv / 2.0,
                self.error_integral / 5.0,
                target_lataccel / 5.0,
                state.v_ego / 30.0,
                state.a_ego / 4.0,
                state.roll_lataccel / 2.0,
                self.prev_action / 2.0,
                future_near / 5.0,
                future_mid / 5.0,
            ]
        )

        # forward pass
        hidden_1 = np.tanh(x @ self.w1 + self.b1)
        hidden_2 = np.tanh(hidden_1 @ self.w2 + self.b2)
        output = np.tanh(hidden_2 @ self.w3 + self.b3)

        # scale to steering range
        action = float(output[0]) * 2.0

        # update state
        self.prev_error = error
        self.prev_action = action

        return action


# Training

# These are global variables set once per worker process
WORKER_MODEL = None
WORKER_DATA_FILES = None


def init_worker(model_path: Path, data_files: list[Path]) -> None:
    global WORKER_MODEL, WORKER_DATA_FILES
    from tinyphysics import TinyPhysicsModel

    WORKER_MODEL = TinyPhysicsModel(str(model_path), debug=False)
    WORKER_DATA_FILES = data_files


# Evaluate one parameter set candidate, runs in worker
def evaluate(args: tuple[np.ndarray, int, int]) -> float:
    from tinyphysics import TinyPhysicsSimulator

    global WORKER_MODEL, WORKER_DATA_FILES
    params, seed, num_segments = args

    if WORKER_MODEL is None or WORKER_DATA_FILES is None:
        raise ValueError("Worker process global values not initialized")

    controller = Controller()
    controller.set_params(params)

    rng = np.random.default_rng(seed=seed)
    indices = rng.choice(len(WORKER_DATA_FILES), size=num_segments, replace=False)

    total_cost = 0
    for idx in indices:
        controller.reset()

        sim = TinyPhysicsSimulator(
            WORKER_MODEL, str(WORKER_DATA_FILES[idx]), controller, debug=False
        )
        cost = sim.rollout()
        total_cost += cost["total_cost"]

    return total_cost / num_segments


if __name__ == "__main__":
    print("CMA-ES Run 4: 10 -> 12 -> 8 -> 1")

    import pickle
    from multiprocessing import Pool

    import cma

    NUM_CORES = 35
    POPULATION_SIZE = 35
    NUM_SEGMENTS = 100
    MAX_GENERATIONS = 3000
    MODEL_PATH = Path("./models/tinyphysics.onnx")
    DATA_PATH = Path("./data")
    RUN_NAME = "run_4"

    data_files = sorted(DATA_PATH.glob("*.csv"))[:5000]
    print(f"Found {len(data_files)} data files")

    # cmaes setup
    if Path(f"{RUN_NAME}_cmaes_state.pkl").exists():
        with open(f"{RUN_NAME}_cmaes_state.pkl", "rb") as f:
            es = pickle.load(f)

        best_ever = es.result.fbest
        gen = es.countiter

        print(f"Resumed from gen ~{es.countiter}")
    else:
        x0 = np.random.randn(NUM_PARAMS) * 0.1
        es = cma.CMAEvolutionStrategy(
            x0,
            0.5,
            {
                "popsize": POPULATION_SIZE,
                "maxiter": MAX_GENERATIONS,
            },
        )

        best_ever = float("inf")
        gen = 0

    print(
        f"Running CMA-ES: {NUM_PARAMS} params, segs={NUM_SEGMENTS}, cores={NUM_CORES}"
    )

    with Pool(
        NUM_CORES, initializer=init_worker, initargs=(MODEL_PATH, data_files)
    ) as pool:
        while not es.stop():
            solutions = es.ask()

            args = [(sol, gen, NUM_SEGMENTS) for sol in solutions]

            fitness = pool.map(evaluate, args)
            es.tell(solutions, fitness)

            gen_best = min(fitness)
            gen_mean = np.mean(fitness)

            if gen_best < best_ever:
                best_ever = gen_best
                np.save(f"{RUN_NAME}_best_params.npy", es.result.xbest)  # type: ignore[reportArgumentType]

            print(
                f"Gen {gen:4d} | best: {gen_best:6.2f} | mean: {gen_mean:6.2f} | best_ever: {best_ever:6.2f} | sigma: {es.sigma:.4f}"
            )

            if gen % 100 == 0 and gen > 0:
                np.save(f"{RUN_NAME}_checkpoint_{gen}.npy", es.result.xbest)  # type: ignore[reportArgumentType]
                with open(f"{RUN_NAME}_cmaes_state.pkl", "wb") as f:
                    pickle.dump(es, f)

            gen += 1

    np.save(f"{RUN_NAME}_final_params.npy", es.result.xbest)  # type: ignore[reportArgumentType]
    print(f"Done! Best: {es.result.fbest:.2f}")
