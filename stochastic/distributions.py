from abc import ABC, abstractmethod
from typing import List
import numpy as np


class ConjugateDistributions(ABC):
  @abstractmethod
  def __init__(self, params: List[float]) -> None:
    pass

  @abstractmethod
  def update_prior(self, x: float) -> None:
    pass

  @abstractmethod
  # "mean of the likelihood" after sampling from a prior
  # eg, if the prior is pi(theta) and the likelihood is p(x|theta)
  # then we sample theta_0 from pi(theta) and compute the mean of
  # p(x|theta_0)
  def likelihood_mean_sample_prior(self) -> float:
    pass

class BetaBernoulli(ConjugateDistributions):
  def __init__(self, prior_params: List[float]) -> None:
    # prior_params[0] is alpha and prior_params[1] is beta
    self.prior_params = prior_params

  def update_prior(self, x: float) -> None:
    # x should be 0.0 or 1.0
    if x > 0.5:
      self.prior_params[0] += 1.0
    else:
      self.prior_params[1] += 1.0

  # def sample_prior(self) -> float:
  #   return np.random.beta(*self.prior_params)

  # "mean of the likelihood" after sampling from a prior
  # eg, if the prior is pi(theta) and the likelihood is p(x|theta)
  # then we sample theta_0 from pi(theta) and compute the mean of
  # p(x|theta_0)
  def likelihood_mean_sample_prior(self) -> float:
      return np.random.beta(*self.prior_params)

class NormalGammaNormal(ConjugateDistributions):
    # reference:
    # https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/lectures/lecture5.pdf
    
    def __init__(self, prior_params: List[float]) -> None:
        # params: mu_0, n_0_tau, alpha, beta
        self.prior_params = prior_params

    def update_prior(self, x: float) -> None:
        mu_0, n_0, alpha, beta = self.prior_params

        # order matters for beta, n_0, mu_0
        alpha += 0.5
        beta += n_0 * (x - mu_0)**2 / (2*(n_0 + 1))
        n_0 += 1
        mu_0 += 1/n_0 * (x - mu_0)

        self.prior_params = mu_0, n_0_tau, alpha, beta

    def likelihood_mean_sample_prior(self) -> float:
        mu_0, n_0, alpha, beta = self.prior_params

        tau = np.random.gamma(alpha, 1/beta)
        mu = np.random.normal(mu_0, np.sqrt(n_0 * tau))

        return mu
