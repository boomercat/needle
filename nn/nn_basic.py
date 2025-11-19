from typing import Any
from needle.autograd import Tensor
from needle import ops
import  needle.init as init
import numpy as numpy

class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:

    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []
    

def _children_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_children_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _children_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _children_modules(v)
        return modules
    else:
        return []
    


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> list[Tensor]:
        return _unpack_params(self.__dict__)
    

    def _children(self) -> list["Module"]:
        return _children_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            self.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = False
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    

class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x
    
class Linear(Module):
    def __init__(self, in_feature: int, 
                 out_feature: int, 
                 bias: bool, 
                 devices: Any | None = None, 
                 dtype: str = "float32") -> None:

        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature

        raise NotImplementedError()
    
    def forward(self, X: Tensor) -> Tensor:
        raise NotImplementedError()
    

class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        raise NotImplementedError()


