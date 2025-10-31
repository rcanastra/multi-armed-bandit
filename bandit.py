from typing import List
from abc import ABC, abstractmethod
import numpy as np

class Bandit(ABC):
    @abstractmethod
    def pull_arm(self, n: int) -> float:
        pass

class StochasticBandit(Bandit):
    @abstractmethod
    def fix_probabilistic_state(self, context: List[float]) -> None:
        pass

    @abstractmethod
    def pull_arm(self, n: int) -> float:
        pass


class ContextualBandit(Bandit):
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
