from typing import List
from abc import ABC, abstractmethod
import numpy as np

class ContextualBanditArm(ABC):
    def fix_probabilistic_state(self, context: np.ndarray) -> None:
        pass
    def pull(self) -> float:
        pass

class Bandit(ABC):
    @abstractmethod
    def pull_arm(self, n: int) -> float:
        pass

class ContextualBandit(Bandit):
    @abstractmethod
    def fix_probabilistic_state(self, context: np.ndarray) -> None:
        pass

    @abstractmethod
    def pull_arm(self, n: int) -> float:
        pass


class FiniteArmedContextualBandit(Bandit):
    def __init__(self, arms: List[ContextualBanditArm]) -> None:
        self.arms = arms
    def fix_probabilistic_state(self, context: np.ndarray) -> None:
        for arm in self.arms:
            arm.fix_probabilistic_state(context)
    def pull_arm(self, arm: int) -> float:
        return self.arms[arm].pull()
    
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
    

class GaussianLinearBanditArm(ContextualBanditArm):
    # mu(a | x) = x * theta
    # a, x, theta are vectors
    # we will also assume a is distributed gaussian with variance 1
    def __init__(self, theta: np.ndarray, sigma: float = 1) -> None:
        self.theta = theta
        self.sigma = sigma
    
    def fix_probabilistic_state(self, context: np.ndarray) -> None:
        mean = np.dot(self.theta, context)
        self.reward = np.random.normal(mean, 1)

    def pull(self) -> float:
        return self.reward
    
