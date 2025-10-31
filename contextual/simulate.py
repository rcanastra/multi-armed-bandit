from bandit import *
from strategy import *
import numpy as np
from typing import Callable

def simulate_contextual(
    strats: List[ContextualBanditStrategy],
    bandit: ContextualBandit,
    context_generator: Callable[[], np.ndarray],
    num_rounds: int,
) -> np.ndarray:
    results = np.empty((len(strats), num_rounds))
    for i in range(num_rounds):
        context = context_generator()
        bandit.fix_probabilistic_state(context)
        for j, strat in enumerate(strats):
            arm = strat.choose_arm(context)
            result = bandit.pull_arm(arm)
            results[j, i] = result
            strat.record_result(arm, result)
                
    return results


results = simulate_contextual(
    [DiscreteEpsilonGreedyStrategy(0, 1, 4, 3, 2, 0.1)],
    FiniteArmedContextualBandit([
        GaussianLinearBanditArm(np.array([0.1, 0.2, 0.4])),
        GaussianLinearBanditArm(np.array([0.1, 0.05, 0.0])),
    ]),
    lambda: np.array([0.5, 0.5, 0.5]),
    1000
)
print(np.sum(results))
