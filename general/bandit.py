from typing import List, NewType, Union, Optional
from abc import ABC, abstractmethod
import numpy as np

ArmType = NewType("ArmType", Union[int, float, np.ndarray])

class BanditArm(ABC):
    def fix_probabilistic_state(self) -> None:
        pass
    def pull_arm(self) -> float:
        pass

class Bandit(ABC):
    # consists of multiple arms
    def process_context(self, context: Optional[np.ndarray]) -> None:
        pass
    def process_arm(self, arm: ArmType) -> None:
        pass
    # allow arm to be from N, R, or R^n
    def pull_arm(self, arm: ArmType) -> float:
        pass

class FiniteArmedStochasticBandit(Bandit):
    def __init__(self, arms: List[BanditArm]) -> None:
        self.arms = arms
    def process_context(self, context: Optional[np.ndarray]) -> None:
        for arm in self.arms:
            arm.fix_probabilistic_state()
    def pull_arm(self, arm: int) -> float:
        return self.arms[arm].pull()

class BernoulliBandit(Bandit):
    def __init__(self, probabilities: List[float]) -> None:
        self.probabilities = probabilities
    def pull_arm(self, n: int) -> float:
        return self.coin_flips[n]


class BernoulliBanditArm:
    def __init__(self, p: float) -> None:
        assert p >= 0.0 and p <= 1.0
        self.p = p
    def fix_probabilistic_state(self) -> None:
        self.observed_value = np.random.binomial(1, self.p)
    def pull(self) -> float:
        return self.observed_value

class GaussianBanditArm:
    def __init__(self, mu: float, sigma: float) -> None:
        assert sigma >= 0.0
        self.mu = mu
        self.sigma = sigma
    def fix_probabilistic_state(self) -> None:
        self.observed_value = np.random.normal(self.mu, self.sigma)
    def pull(self) -> float:
        return self.observed_value

