from typing import Optional, List, Tuple, Union
import numpy
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import NDarry

import numpy as array_api

class EWiseAdd(TensorOp):
    def compute(self, a: NDarry, b: NDarry):
        return a + b
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad
    
def add(a, b):
    return EWiseAdd()(a, b)

#标量加
class AddScalar(TensorOp):
    def __init__(self, scalar):
            self.scalar = scalar
    def compute(self, a: NDarry):
        return a + self.scalar
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad
    

def add_scalar(a, scalar):
    return AddScalar(scalar)(a)

class EWiseMul(TensorOp):
    def compute(self, a: NDarry, b: NDarry):
        return a * b
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs
    
def multiply(a: NDarry, b: NDarry):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDarry):
        return a * self.scalar
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        return (self.scalar * out_grad,)
    
def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)
    
class PowerScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDarry):
        return a ** self.scalar
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad * self.scalar * node.inputs[0] ** (self.scalar-1)
    
def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)

class EWisePow(TensorOp):
    def compute(self, a: NDarry, b: NDarry):
        return a ** b
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        if not isinstance(node.inputs[0], NDarry) or not isinstance(node.inputs[1], NDarry):
            raise ValueError("both inputs must be tensors NDarry")
        x, y = node.inputs[0], node.inputs[1]
        grad_0 = out_grad * y * (x ** (y-1))
        grad_1 = out_grad * y**x (array_api.log(x.data))
        return grad_0, grad_1
    
def power(a, b):
    return EWisePow(a, b)

class EWiseDiv(TensorOp):
    def compute(self, a: NDarry, b: NDarry):
        return a / b
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        x, y = node.inputs[0], node.inputs[1]
        grad_0 = out_grad / y
        grad_1 = -out_grad * x / (y**2)
        return grad_0, grad_1
    
def divide(a, b):
    return EWiseDiv()(a, b)

class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDarry):
        return a / self.scalar
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad / self.scalar,)
    
def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)

class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
    
    # def compute()
        
    # def gradient()
        
def transpose(a, axes = None):
    return Transpose(axes)(a)

class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a,b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (1 / node.inputs[0])
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return  out_grad * exp(node.inputs[0])


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a,0)

    def gradient(self, out_grad, node):
        out_grad * Tensor(array_api.greater(node.inputs[0].realize_cached_data(), 0))

def relu(a):
    return ReLU()(a)