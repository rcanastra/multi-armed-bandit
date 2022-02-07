from abc import ABC, abstractmethod
from typing import List
import numpy as np
from distributions import ConjugateDistributions


class StochasticBanditStrategy(ABC):
  def choose_arm(self) -> int:
    pass
  
  def record_result(self, arm: int, result: float) -> None:
    pass
      

class UniformExploration(StochasticBanditStrategy):
  def __init__(self, num_arms: int, num_times_explore: int) -> None:
    self.num_arms = num_arms
    self.num_times_explore = num_times_explore
    self.num_rounds_so_far = 0
    self.sum_results = [0.0 for _ in range(num_arms)]

  def choose_arm(self) -> int:
    if self.num_rounds_so_far < self.num_times_explore * self.num_arms:
      return self.num_rounds_so_far % self.num_arms
    else:
      return self.best_arm

  def record_result(self, arm: int, result: float) -> None:
    # we could stop doing this after N rounds, but the naming becomes hard
    self.num_rounds_so_far += 1
    self.sum_results[arm] += result
    if self.num_rounds_so_far == self.num_times_explore * self.num_arms:
      self.best_arm = np.argmax(self.sum_results)

class EpsilonGreedy(StochasticBanditStrategy):
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
    

class UCB(StochasticBanditStrategy):
  def __init__(self, num_arms: int) -> None:
    self.num_arms = num_arms
    self.num_rounds_so_far = 0
    self.num_pulls = np.array([0 for _ in range(num_arms)])
    self.sum_results = np.array([0 for _ in range(num_arms)])

  def choose_arm(self) -> int:
    if self.num_rounds_so_far < self.num_arms:
      return self.num_rounds_so_far
    else:
      # mu = mean, r = half the confidence interval
      mu = self.sum_results / self.num_pulls
      r = np.sqrt(2 * np.log(self.num_rounds_so_far) / self.num_pulls)
      return np.argmax(mu + r)

  def record_result(self, arm: int, result: float) -> None:
    self.num_rounds_so_far += 1
    self.num_pulls[arm] += 1
    self.sum_results[arm] += result
    
class ThompsonSamplingBeta(StochasticBanditStrategy):
    def __init__(
        self,
        num_arms: int,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        self.beta_params = ([alpha for _ in range(num_arms)], [beta for _ in range(num_arms)])
        
    def choose_arm(self) -> int:
        sampled = np.random.beta(*self.beta_params)
        return np.argmax(sampled)

    def record_result(self, arm: int, result: float) -> None:
        if result > 0.5:
            self.beta_params[0][arm] += 1.0
        else:
            self.beta_params[1][arm] += 1.0
        
class ThompsonSamplingConjugateDistributions(StochasticBanditStrategy):
    def __init__(
        self,
        num_arms: int,
        conj_dists: List[ConjugateDistributions],
    ) -> None:
      self.conj_dists = conj_dists
        
    def choose_arm(self) -> int:
      sampled = [conj_dist.likelihood_mean_sample_prior() for conj_dist in self.conj_dists]
      return np.argmax(sampled)

    def record_result(self, arm: int, result: float) -> None:
      self.conj_dists[arm].update_prior(result)
        

class ThompsonSamplingBootstrap(StochasticBanditStrategy):
  def __init__(self, num_arms: int) -> None:
    self.num_arms = num_arms
    self.num_rounds_so_far = 0
    self.results = [[] for _ in range(num_arms)]

  def choose_arm(self) -> int:
    if self.num_rounds_so_far < self.num_arms:
      return self.num_rounds_so_far
    else:
      best_arm = 0
      best_arm_sample = None
      samples = [np.random.choice(self.results[i]) for i in range(self.num_arms)]
      return np.argmax(samples)

  def record_result(self, arm: int, result: float) -> None:
    self.num_rounds_so_far += 1
    self.results[arm].append(result)
