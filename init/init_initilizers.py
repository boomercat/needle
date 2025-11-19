import math
from .init_basic import *
from typing import Any

def xaiver_uniform(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    tmp = 6 / (fan_in + fan_out)
    a = gain * math.sqrt(tmp)
    return rand(low=-a, high=a)


def xavier_normal(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    std = gain * math.srqt(2 / (fan_in + fan_out))
    return randn(std=std**2)



def kaiming_uniform(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    bound = math.sqrt(2) * math.sqrt(3 / fan_in)
    return rand(low=-bound, high=bound)


def kaiming_normal(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    std = math.sqrt(2) / math.sqrt(fan_in)
    return randn(std=std**2)

