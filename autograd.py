import needle
from typing import List, Optional, NamedTuple, Tuple, Union
from .backend_numpy import Device, cpu, all_devices
from collections import namedtuple
import numpy as array_api
import numpy

from needle import init

NDarry = numpy.ndarray


LAZY_MODE = False
TENSOR_COUNTER = 0

class Op:
    """Operator definition """
    def __call__(self, *args):
        raise NotImplementedError()
    
    def compute(self, *args: Tuple[NDarry]):
        raise NotImplementedError()
    def gradient(self, out_grad:"Value", node:"Value") -> Union["Value", Tuple["Value"]]:
        raise NotImplementedError()
    
    def gradient_as_tuple(self, out_grad:"Value", node:"Value") -> Tuple["Value"]:
        output = self.gradient(out_grad, node)
        if isinstance(output, tuple):
            return output
        elif isinstance(output, list):
            return tuple(output)
        else:
            return (output,)
        
class Value:
    """a value in the computational grapg"""
    op: Optional[Op]
    inputs: List["Value"]

    cached_data: NDarry
    requires_grad: bool

    def realized_cached_data(self):
        """ run compute to realize the cache data"""
        if self.cached_data is not None:
            return self.cached_data
        self.cached_data = self.op.compute(
            *[x.realized_cached_data() for x in self.inputs]
        )
        return self.cached_data
    
    def is_lead(self):
        return self.op is None

    def __del__(self):
        global TENSOR_COUNTER
        TENSOR_COUNTER -= 1

    def _init(self,
                op: Optional[Op],
                inputs: List["Tensor"],
                *,
                num_outputs: int = 1,
                cached_data: List[object] = None,
                requires_grad: Optional[bool] = None
            ):
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1
        if requires_grad is None:
            requires_grad = any([x.requires_grad for x in inputs])
        self.op = op
        self.inputs = inputs
        self.num_outputs = num_outputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad

    
    @classmethod
    def make_const(cls, data, *, requires_data = False):
        value = cls.__new__(cls)
        value.__init__(None, [], cached_data=data, requires_grad=requires_data)
        return value
    
    @classmethod
    def make_from_op(cls, op: Op, inputs: List["Value"]):
        value = cls.__new__(cls)
        value.__init__(op, inputs)

        if not LAZY_MODE:
            if not value.requires_grad:
                return value.detach()
        return value
    



class TensorOp(Op):
    def __call__(self, *args):
        return Tensor.make_from_op(self,args)

class Tensor(Value):
    grad: "Tensor"

    def __init__(
        self,
        array,
        *,
        device: Optional[Device] = None,
        dtype=None,
        requires_grad=True,
        **kwargs
    ):
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                cached_data = array.realized_cached_data()
            else:
                # fall back, copy through numpy conversion
                cached_data = Tensor._array_from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )
        else:
            device = device if device else cpu()
            cached_data = Tensor._array_from_numpy(array, device=device, dtype=dtype)

        self._init(
            None,
            [],
            cached_data=cached_data,
            requires_grad=requires_grad,
        )

    @staticmethod
    def _array_from_numpy(numpy_array, device, dtype):
        if array_api is numpy:
            return numpy.array(numpy_array, dtype=dtype)
        return array_api.array(numpy_array, device=device, dtype=dtype)

    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"]):
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        if not LAZY_MODE:
            if not tensor.requires_grad:
                return tensor.detach()
            tensor.realized_cached_data()
        return tensor

    @staticmethod
    def make_const(data, requires_grad=False):
        tensor = Tensor.__new__(Tensor)
        tensor._init(
            None,
            [],
            cached_data=data
            if not isinstance(data, Tensor)
            else data.realized_cached_data(),
            requires_grad=requires_grad,
        )
        return tensor

    @property
    def data(self):
        return self.detach()

    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, "%s %s" % (
            value.dtype,
            self.dtype,
        )
        self.cached_data = value.realized_cached_data()

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return Tensor.make_const(self.realized_cached_data())

    @property
    def shape(self):
        return self.realized_cached_data().shape

    @property
    def dtype(self):
        return self.realized_cached_data().dtype

    @property
    def device(self):
        data = self.realized_cached_data()
        # numpy array always sits on cpu
        if array_api is numpy:
            return cpu()
        return data.device

    def backward(self, out_grad=None):
        out_grad = (
            out_grad
            if out_grad
            else init.ones(*self.shape, dtype=self.dtype, device=self.device)
        )
        compute_gradient_of_variables(self, out_grad)

    def __repr__(self):
        return "needle.Tensor(" + str(self.realized_cached_data()) + ")"

    def __str__(self):
        return self.realized_cached_data().__str__()

    def numpy(self):
        data = self.realized_cached_data()
        if array_api is numpy:
            return numpy.array(data)
        return data.numpy()

    def __add__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseAdd()(self, other)
        else:
            return needle.ops.AddScalar(other)(self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseMul()(self, other)
        else:
            return needle.ops.MulScalar(other)(self)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWisePow()(self, other)
        else:
            return needle.ops.PowerScalar(other)(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseAdd()(self, needle.ops.Negate()(other))
        else:
            return needle.ops.AddScalar(-other)(self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseDiv()(self, other)
        else:
            return needle.ops.DivScalar(other)(self)

    def __matmul__(self, other):
        return needle.ops.MatMul()(self, other)

    def matmul(self, other):
        return needle.ops.MatMul()(self, other)

    def sum(self, axes=None):
        return needle.ops.Summation(axes)(self)

    def broadcast_to(self, shape):
        return needle.ops.BroadcastTo(shape)(self)

    def reshape(self, shape):
        return needle.ops.Reshape(shape)(self)

    def __neg__(self):
        return needle.ops.Negate()(self)

    def transpose(self, axes=None):
        return needle.ops.Transpose(axes)(self)

    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__
    __rmatmul__ = __matmul__


def compute_gradient_of_variables(output_tensor, out_grad):
    """Take gradient of output node with respect to each node in node_list.

    Store the computed result in the grad field of each Variable.
    """
    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {}
    # Special note on initializing gradient of
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[output_tensor] = [out_grad]

    # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))

    for node in reverse_topo_order:
      grad_list = node_to_output_grads_list[node]
      grad_sum = sum_node_list(grad_list)
      node.grad = grad_sum

      if node.op is not None:
        grads = node.op.gradient(grad_sum, node)
        if not isinstance(grads, tuple):
          grads = [grads]
        for sub_node, grad in zip(node.inputs,grads):
          if node_to_output_grads_list.get(sub_node) is None:
            node_to_output_grads_list[sub_node] = []
          node_to_output_grads_list[sub_node].append(grad)

def find_topo_sort(node_list: List[Value]) -> List[Value]:
    """Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.
    """
    visited = set()
    topo_order = []
    topo_sort_dfs(node_list[0], visited, topo_order)
    return topo_order


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    visited.add(node)
    for sub_node in node.inputs:
      if sub_node not in visited:
        topo_sort_dfs(sub_node, visited, topo_order)
    topo_order.append(node) 


#求节点和
def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    from functools import reduce

    return reduce(add, node_list)