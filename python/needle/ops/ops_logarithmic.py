from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        # Logsoftmax is z - logsumexp(z), logsumexp done the stable way over axis 1
        max_per_row = array_api.max(Z, axis = 1, keepdims = True)
        sum_of_exps = array_api.sum(array_api.exp(Z - max_per_row), axis = 1, keepdims = True)
        log_sum_exp_per_row = array_api.log(sum_of_exps) + max_per_row
        return Z - log_sum_exp_per_row
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Grad is out_grad - softmax(z) * (row sum of out_grad)
        input_shape = node.inputs[0].shape
        num_rows = input_shape[0]

        # Node is the logsoftmax output so exp of it is the softmax
        softmax_z = exp(node)

        row_sums_of_grad = summation(out_grad, axes = (1,))
        row_sums_broadcast = broadcast_to(reshape(row_sums_of_grad, (num_rows, 1)), input_shape)

        return out_grad - softmax_z * row_sums_broadcast
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        # Pull the max out first so exp never overflows
        max_keepdims = array_api.max(Z, axis = self.axes, keepdims = True)
        max_reduced = array_api.max(Z, axis = self.axes)

        sum_of_exps = array_api.sum(array_api.exp(Z - max_keepdims), axis = self.axes)
        return array_api.log(sum_of_exps) + max_reduced
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Grad of logsumexp is softmax(z) = exp(z - logsumexp(z))
        # Both out_grad and node are the reduced shape, put the summed axes back as 1s then broadcast
        input_tensor = node.inputs[0]
        input_shape = input_tensor.shape

        if self.axes is None:
            summed_axes = tuple(range(len(input_shape)))
        elif isinstance(self.axes, int):
            summed_axes = (self.axes,)
        else:
            summed_axes = tuple(self.axes)

        shape_with_ones = []
        for axis_index, dim_size in enumerate(input_shape):
            if axis_index in summed_axes:
                shape_with_ones.append(1)
            else:
                shape_with_ones.append(dim_size)
        shape_with_ones = tuple(shape_with_ones)

        log_sum_exp_broadcast = broadcast_to(reshape(node, shape_with_ones), input_shape)
        softmax_z = exp(input_tensor - log_sum_exp_broadcast)

        out_grad_broadcast = broadcast_to(reshape(out_grad, shape_with_ones), input_shape)
        return out_grad_broadcast * softmax_z
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

