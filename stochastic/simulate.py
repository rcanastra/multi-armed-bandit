import argparse

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
    results = [[] for _ in range(len(strats))]
    for i in range(num_simulations):
        for j in range(num_rounds):
            bandit.fix_probabilistic_state()
            for k, strat in enumerate(strats):
                arm = strat.choose_arm()
                result = bandit.pull_arm(arm)
                results[k].append(result)
                strat.record_result(arm, result)
                
    return results

parser = argparse.ArgumentParser(description='Stochastic multi-armed bandit')
# ignore these, too hard to deal with
# parser.add_argument('bandit', type=str)
# parser.add_argument('strats', type=str)
parser.add_argument('--num-rounds', '-r', type=int)
parser.add_argument('--num-simulations', '-s', type=int)

args = parser.parse_args()

# bandit = args.bandit
# strats = args.strats

num_rounds = args.num_rounds
num_simulations = args.num_simulations

# print(simulate([MyStrategy()], SimpleBandit([0.1, 0.9]), 1000, 10))
results = simulate([ThompsonSamplingBeta(2)], BernoulliBandit([0.1, 0.9]), num_rounds, num_simulations)
print(np.sum(results))
