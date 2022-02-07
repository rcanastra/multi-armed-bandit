from bandit import *
from strategy import *
import numpy as np
from typing import Callable, Optional

def simulate(
    strats: List[Strategy],
    bandit: Bandit,
    context_generator: Optional[Callable[[], np.ndarray]],
    num_rounds: int,
) -> np.ndarray:

    # maybe type checking here, eg contextual and stochastic
    # strategies/bandits should not mix
    
    results = np.empty((len(strats), num_rounds))
    
    for i in range(num_rounds):
        
        if context_generator is not None:
            context = context_generator()
        else:
            context = None
        bandit.process_context(context)

        for j, strat in enumerate(strats):
            arm = strat.choose_arm()
            bandit.process_arm(arm)
            result = bandit.pull_arm(arm)
            results[j, i] = result
            strat.record_result(arm, result)
                
    return results

results = simulate([ThompsonSamplingBeta(2)], BernoulliBandit([0.1, 0.9]), None, 10)
print(results)
