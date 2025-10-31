from bandit import *
from strategy import *
import numpy as np
from typing import Callable

def simulate(
    strats: List[StochasticBanditStrategy],
    bandit: StochasticBandit,
    num_rounds: int,
    num_simulations: int
) -> List[List[float]]:
    results = [[0 for _ in range(num_simulations)] for _ in range(len(strats))]
    for i in range(num_simulations):
        for j in range(num_rounds):
            bandit.fix_probabilistic_state()
            for k, strat in enumerate(strats):
                arm = strat.choose_arm()
                result = bandit.pull_arm(arm)
                results[k][i] += result
                strat.record_result(arm, result)

    return results

def simulate_contextual(
    strats: List[ContextualBanditStrategy],
    bandit: ContextualBandit,
    context_generator: Callable[[], List[float]],
    num_rounds: int,
    num_simulations: int
) -> List[List[float]]:
    results = [[] for _ in range(len(strats))]
    for i in range(num_simulations):
        for j in range(num_rounds):
            context = context_generator()
            bandit.fix_probabilistic_state(context)
            for k, strat in enumerate(strats):
                arm = strat.choose_arm()
                result = bandit.pull_arm(arm)
                results[k].append(result)
                strat.record_result(context, arm, result)

    return results


# print(simulate([MyStrategy()], SimpleBandit([0.1, 0.9]), 1000, 10))
results = simulate([ThompsonSamplingBeta(2)], BernoulliBandit([0.1, 0.9]), 10, 1)
results = simulate(
    [ThompsonSamplingBeta(2), UniformExploration(2, 3)],
    BernoulliBandit([0.4, 0.6]),
    100,
    5,
)
print(results)
# print(np.sum(results))
