from typing import List
from abc import ABC, abstractmethod
import numpy as np

class Bandit(ABC):
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

class GaussianLinearBandit(Bandit):
    # mu(a | x) = x * Theta
    # a is a vector and Theta is a matrix
    # we will also assume a is distributed gaussian with variance 1
    def __init__(self, theta: np.ndarray, sigma: float = 1) -> None:
        self.theta = theta
        self.sigma = sigma
    
    def fix_probabilistic_state(self, context: List[float]) -> None:
        means = np.dot(self.theta, context)
        self.rewards = np.random.normal(means, np.ones(len(means)))

    def pull_arm(self, n: int) -> float:
        return self.rewards[n]

class SinusoidalBandit(Bandit):
    # mu(a | x) = sin(x_1) * ... * sin(x_n)
    def __init__(self, dim: int, amplitude: float) -> None:
        self.dim = dim
        self.amplitude = amplitude

    def fix_probabilistic_state(self, context: List[float]) -> None:
        means = np.product(np.sin(context))
        self.rewards = np.random.normal(means, np.ones(len(means)))

    def pull_arm(self, n: int) -> float:
        return self.rewards[n]
    
