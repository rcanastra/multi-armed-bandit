from typing import List
from abc import ABC, abstractmethod
import numpy as np

class StochasticBanditArm(ABC):
    def fix_probabilistic_state(self) -> None:
        pass
    def pull(self) -> float:
        pass

class Bandit(ABC):
    @abstractmethod
    def pull_arm(self, n: int) -> float:
        pass

class FiniteArmedStochasticBandit(Bandit):
    def __init__(self, arms: List[StochasticBanditArm]) -> None:
        self.arms = arms
    def fix_probabilistic_state(self) -> None:
        for arm in self.arms:
            arm.fix_probabilistic_state()
    def pull_arm(self, arm: int) -> float:
        return self.arms[arm].pull()

class StochasticBandit(Bandit):
    @abstractmethod
    def fix_probabilistic_state(self, context: List[float]) -> None:
        pass

    @abstractmethod
    def pull_arm(self, n: int) -> float:
        pass
    
class BernoulliBandit(StochasticBandit):
    def __init__(self, probabilities: List[float]) -> None:
        self.probabilities = probabilities
    def fix_probabilistic_state(self) -> None:
        self.coin_flips = np.random.binomial(1, self.probabilities)
    def pull_arm(self, n: int) -> float:
        return self.coin_flips[n]


class GaussianStochasticBanditArm:
    def __init__(self, mu: float, sigma: float) -> None:
        assert sigma >= 0.0
        self.mu = mu
        self.sigma = sigma
    def fix_probabilistic_state(self) -> None:
        self.observed_value = np.random.normal(self.mu, self.sigma)
    def pull(self) -> float:
        return self.observed_value
    
