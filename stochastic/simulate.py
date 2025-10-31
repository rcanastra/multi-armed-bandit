import argparse

from bandit import *
from strategy import *
import numpy as np
from typing import Callable

# Arms can be made non-independent by implementing StochasticBandit
# in some appropriate way. And arm pulls can be made non stationary
# similarly. "Stochastic" is not really the right term, given this.
def simulate(
    strats: List[StochasticBanditStrategy],
    bandit: StochasticBandit,
    num_rounds: int,
) -> np.ndarray:
    results = np.empty((len(strats), num_rounds))
    for i in range(num_rounds):
        bandit.fix_probabilistic_state()
        for j, strat in enumerate(strats):
            arm = strat.choose_arm()
            result = bandit.pull_arm(arm)
            results[j, i] = result
            strat.record_result(arm, result)
                
    return results

# parser = argparse.ArgumentParser(description='Stochastic multi-armed bandit')
# # ignore these, too hard to deal with
# # parser.add_argument('bandit', type=str)
# # parser.add_argument('strats', type=str)
# parser.add_argument('--num-rounds', '-r', type=int)

# args = parser.parse_args()

# # bandit = args.bandit
# # strats = args.strats

# num_rounds = args.num_rounds

num_rounds = 10

# print(simulate([MyStrategy()], SimpleBandit([0.1, 0.9]), 1000, 10))
results = simulate([ThompsonSamplingBeta(2)], BernoulliBandit([0.1, 0.9]), num_rounds)
print(np.sum(results))
