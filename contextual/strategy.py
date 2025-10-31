from abc import ABC, abstractmethod
from typing import List
import numpy as np


class ContextualBanditStrategy(ABC):
    def choose_arm(self) -> int:
        pass
    def record_result(self, context: List[float], arm: int, result: float):
      pass

# copied here to use for DiscreteEpsilonGreedyStrategy  
class EpsilonGreedy:
  def __init__(self, num_arms: int, epsilon: float) -> None:
    self.num_arms = num_arms

    assert epsilon >= 0.0 and epsilon <= 1.0
    self.epsilon = epsilon

    self.sum_results = [0.0 for _ in range(num_arms)]
    self.num_pulls = [0 for _ in range(num_arms)]

  def choose_arm(self) -> int:
    success = (self.epsilon < np.random.uniform())
    if success:
      return np.random.randint(self.num_arms)
    else:
      return np.argmax([x/y if y != 0 else 0.0 for x,y in zip(self.sum_results, self.num_pulls)])

  def record_result(self, arm: int, result: float) -> None:
    self.sum_results[arm] += result


class DiscreteEpsilonGreedyStrategy(ContextualBanditStrategy):
    # context is assumed to be in [a, b]^dim
    # n is the number of grid points we take along each dimension
    # a and b are guaranteed to be grid points
    # 
    def __init__(self, a: float, b: float, n: int, dim: int, num_arms: int, epsilon: float) -> None:
        self.a = a
        self.b = b
        self.n = n
        self.dim = dim
        self.num_arms = num_arms
        self.epsilon = epsilon

        self.grid_strategies = [EpsilonGreedy(num_arms, epsilon) for _ in range(n**dim)]

    # coef[0]*x^N + ... + coef[N]*x^0
    def poly_eval(self, coef: np.ndarray, x: int) -> int:
        val = coef[0]
        i = 1
        while i < len(coef):
            val = val*x + coef[i]
            i += 1
        return val
        
    def choose_arm(self, context: np.ndarray) -> int:
        # round to the nearest grid point
        # a + i*(b-a)/(n-1)
        # round (xi - a) / ((b - a) / (n - 1))
        gridpoint = np.round((context - self.a) * (self.n-1) / (self.b - self.a)).astype(np.int)

        self.gridpoint_index = self.poly_eval(gridpoint, self.n)

        self.arm = self.grid_strategies[self.gridpoint_index].choose_arm()
        return self.arm

    def record_result(self, arm: int, result: float) -> None:
        self.grid_strategies[self.gridpoint_index].record_result(arm, result)
