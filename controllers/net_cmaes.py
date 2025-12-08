import numpy as np

from tinyphysics import FuturePlan, State

from controllers import BaseController
from pathlib import Path


INPUT_LAYER_SIZE = 8  # Changing requires adding or removing actual inputs
HIDDEN_LAYER_SIZE = 12
NUM_PARAMS = (
    INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE + 1
)


class Controller(BaseController):
    def __init__(self) -> None:
        self.w1 = np.zeros(
            (INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE)
        )  # input -> hidden weights
        self.b1 = np.zeros(HIDDEN_LAYER_SIZE)  # hidden biases
        self.w2 = np.zeros((HIDDEN_LAYER_SIZE, 1))  # hidden -> output weights
        self.b2 = np.zeros(1)  # output bias

        # integral/derivative state
        self.prev_error = 0.0
        self.error_integral = 0.0
        self.prev_action = 0.0

    def set_params(self, params: np.ndarray) -> None:
        idx = 0
        self.w1 = params[
            idx : (idx := idx + INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE)
        ].reshape(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE)
        self.b1 = params[idx : (idx := idx + HIDDEN_LAYER_SIZE)]
        self.w2 = params[idx : (idx := idx + HIDDEN_LAYER_SIZE)].reshape(
            HIDDEN_LAYER_SIZE, 1
        )
        self.b2 = params[idx : (idx := idx + 1)]

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

        # build input features
        x = np.array(
            [
                error,
                error_deriv,
                self.error_integral,
                target_lataccel,
                state.v_ego,
                state.a_ego,
                state.roll_lataccel,
                self.prev_action,
            ]
        )

        # forward pass
        hidden = np.tanh(x @ self.w1 + self.b1)
        output = np.tanh(hidden @ self.w2 + self.b2)

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
    import cma
    from multiprocessing import Pool

    NUM_CORES = 63
    POPULATION_SIZE = 64
    NUM_SEGMENTS = 300
    MAX_GENERATIONS = 2000
    MODEL_PATH = Path("./models/tinyphysics.onnx")
    DATA_PATH = Path("./data")

    data_files = sorted(DATA_PATH.glob("*.csv"))
    print(f"Found {len(data_files)} data files")

    # cmaes setup
    x0 = np.random.randn(NUM_PARAMS) * 0.1
    es = cma.CMAEvolutionStrategy(
        x0,
        0.5,
        {
            "popsize": POPULATION_SIZE,
            "maxiter": MAX_GENERATIONS,
        },
    )

    print(
        f"Running CMA-ES: {NUM_PARAMS} params, segs={NUM_SEGMENTS}, cores={NUM_CORES}"
    )

    best_ever = float("inf")
    with Pool(
        NUM_CORES, initializer=init_worker, initargs=(MODEL_PATH, data_files)
    ) as pool:
        gen = 0
        while not es.stop():
            solutions = es.ask()

            args = [(sol, gen, NUM_SEGMENTS) for sol in solutions]

            fitness = pool.map(evaluate, args)
            es.tell(solutions, fitness)

            gen_best = min(fitness)
            gen_mean = np.mean(fitness)

            if gen_best < best_ever:
                best_ever = gen_best
                np.save("best_params.npy", es.result.xbest)  # type: ignore[reportArgumentType]

            print(
                f"Gen {gen:4d} | best: {gen_best:6.2f} | mean: {gen_mean:6.2f} | best_ever: {best_ever:6.2f}"
            )

            if gen % 100 == 0 and gen > 0:
                np.save(f"checkpoint_{gen}.npy", es.result.xbest)  # type: ignore[reportArgumentType]

            gen += 1

    np.save("final_params.npy", es.result.xbest)  # type: ignore[reportArgumentType]
    print(f"Done! Best: {es.result.fbest:.2f}")
