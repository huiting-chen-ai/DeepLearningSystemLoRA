from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        log_sum_exp_op = LogSumExp(axes=(1,))
        return Z - log_sum_exp_op.compute(Z).reshape((-1, 1))
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        log_softmax_output = node
        softmax = exp(log_softmax_output)
        sum_out_grad = summation(out_grad, axes=(1,))
        sum_out_grad = reshape(sum_out_grad, (Z.shape[0], 1))
        sum_out_grad = broadcast_to(sum_out_grad, Z.shape)
        return out_grad - multiply(softmax, sum_out_grad)
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes


    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        M = Z.max(axis=self.axes, keepdims=True)
        M_shape = list(Z.shape)
        if self.axes is not None:
            if isinstance(self.axes, int):
                M_shape[self.axes] = 1
            else:
                for ax in self.axes:
                    M_shape[ax] = 1
        else:
            M_shape = [1] * len(M_shape)
        r = array_api.log(array_api.sum(array_api.exp(Z-M.reshape(M_shape).broadcast_to(Z.shape)), axis=self.axes, keepdims=True))+M
        new_shape = list(r.shape)
        if self.axes is not None:
            if isinstance(self.axes, int):
                new_shape.pop(self.axes)
            else:
                for ax in reversed(self.axes):
                    new_shape.pop(ax)
        else:
            new_shape = [1]
        return r.reshape(new_shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        axes = self.axes
        if axes is None:
            axes = tuple(range(len(node.inputs[0].shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        z = node.inputs[0]
        shape = [1 if i in axes else z.shape[i] for i in range(len(z.shape))]
        gradient = exp(z - node.reshape(shape).broadcast_to(z.shape))
        return out_grad.reshape(shape).broadcast_to(z.shape)*gradient
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)