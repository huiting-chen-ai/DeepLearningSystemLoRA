"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
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


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        # Weight is (in, out) so forward is just X @ weight, no transpose
        # Weight has to be made before bias or the random stream changes
        self.weight = Parameter(
            init.kaiming_uniform(in_features, out_features, device = device, dtype = dtype)
        )

        if bias:
            # Bias uses fan_in = out_features, kaiming gives (out, 1) so flatten to (out,)
            bias_column = init.kaiming_uniform(out_features, 1, device = device, dtype = dtype)
            self.bias = Parameter(ops.reshape(bias_column, (out_features,)))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        output = ops.matmul(X, self.weight)

        # No implicit broadcasting in needle, make bias a (1, out) row then stretch it
        if self.bias is not None:
            bias_row = ops.reshape(self.bias, (1, self.out_features))
            output = output + ops.broadcast_to(bias_row, output.shape)

        return output
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        # Keep the batch dim, squash the rest into one
        batch_size = X.shape[0]
        features_per_sample = int(np.prod(X.shape[1:]))
        return ops.reshape(X, (batch_size, features_per_sample))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Each module feeds the next
        current = x
        for module in self.modules:
            current = module(current)
        return current
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        # Loss per row is logsumexp(z) - z[y], then avg over the batch
        batch_size, num_classes = logits.shape

        log_sum_exp_per_example = ops.logsumexp(logits, axes = (1,))

        # One hot mask pulls out the true class logit in each row
        y_one_hot = init.one_hot(num_classes, y, device = logits.device, dtype = logits.dtype)
        true_class_logit = ops.summation(logits * y_one_hot, axes = (1,))

        total_loss = ops.summation(log_sum_exp_per_example - true_class_logit)
        return total_loss / batch_size
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        # Learnable scale and shift, start as identity
        self.weight = Parameter(init.ones(dim, device = device, dtype = dtype))
        self.bias = Parameter(init.zeros(dim, device = device, dtype = dtype))

        # Running stats are plain tensors not Parameters so the optimizer skips them
        self.running_mean = init.zeros(dim, device = device, dtype = dtype)
        self.running_var = init.ones(dim, device = device, dtype = dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size, num_features = x.shape
        row_shape = (1, num_features)

        if self.training:
            # Stats over the batch dim, one mean and var per feature
            batch_mean = ops.summation(x, axes = (0,)) / batch_size
            centered = x - ops.broadcast_to(ops.reshape(batch_mean, row_shape), x.shape)
            batch_var = ops.summation(centered ** 2, axes = (0,)) / batch_size

            # Running avgs for eval time, .data detaches so the graph gets dropped
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.data

            variance_broadcast = ops.broadcast_to(ops.reshape(batch_var, row_shape), x.shape)
            normalized = centered / ((variance_broadcast + self.eps) ** 0.5)
        else:
            # Eval time uses the running stats instead of the batch
            mean_broadcast = ops.broadcast_to(ops.reshape(self.running_mean, row_shape), x.shape)
            variance_broadcast = ops.broadcast_to(ops.reshape(self.running_var, row_shape), x.shape)
            normalized = (x - mean_broadcast) / ((variance_broadcast + self.eps) ** 0.5)

        weight_broadcast = ops.broadcast_to(ops.reshape(self.weight, row_shape), x.shape)
        bias_broadcast = ops.broadcast_to(ops.reshape(self.bias, row_shape), x.shape)
        return weight_broadcast * normalized + bias_broadcast
        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device = device, dtype = dtype))
        self.bias = Parameter(init.zeros(dim, device = device, dtype = dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Stats over the feature dim this time, one mean and var per row
        batch_size, num_features = x.shape
        column_shape = (batch_size, 1)

        row_mean = ops.summation(x, axes = (1,)) / num_features
        centered = x - ops.broadcast_to(ops.reshape(row_mean, column_shape), x.shape)

        # Biased variance, divide by N not N-1
        row_var = ops.summation(centered ** 2, axes = (1,)) / num_features
        variance_broadcast = ops.broadcast_to(ops.reshape(row_var, column_shape), x.shape)

        normalized = centered / ((variance_broadcast + self.eps) ** 0.5)

        row_shape = (1, num_features)
        weight_broadcast = ops.broadcast_to(ops.reshape(self.weight, row_shape), x.shape)
        bias_broadcast = ops.broadcast_to(ops.reshape(self.bias, row_shape), x.shape)
        return weight_broadcast * normalized + bias_broadcast
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Only drop while training, eval is identity
        if not self.training:
            return x

        # Keep each element with prob 1-p and scale survivors by 1/(1-p) so the mean stays put
        # Mask is a constant so grad only flows through x
        keep_mask = init.randb(*x.shape, p = 1 - self.p, device = x.device, dtype = x.dtype)
        return x * keep_mask / (1 - self.p)
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Skip connection, block only learns the diff from identity
        return self.fn(x) + x
        ### END YOUR SOLUTION
