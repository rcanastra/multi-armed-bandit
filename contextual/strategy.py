from abc import ABC, abstractmethod
from typing import List
import numpy as np


class ContextualBanditStrategy(ABC):
    def choose_arm(self) -> int:
        pass
    def record_result(self, context: List[float], arm: int, result: float):
      pass

