"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            if param.grad is None:
                continue

            # Everything goes through .data so no graph builds up across steps
            # Weight decay is just L2 folded into the grad
            grad = param.grad.data + self.weight_decay * param.data

            # Momentum, u = beta * u + (1 - beta) * grad
            previous_velocity = self.u.get(param, ndl.init.zeros_like(param))
            new_velocity = self.momentum * previous_velocity + (1 - self.momentum) * grad
            self.u[param] = new_velocity

            param.data = param.data - self.lr * new_velocity
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        # Step count for bias correction
        self.t += 1

        for param in self.params:
            if param.grad is None:
                continue

            grad = param.grad.data + self.weight_decay * param.data

            # Running avg of the grad
            previous_m = self.m.get(param, ndl.init.zeros_like(param))
            new_m = self.beta1 * previous_m + (1 - self.beta1) * grad
            self.m[param] = new_m

            # Running avg of the squared grad
            previous_v = self.v.get(param, ndl.init.zeros_like(param))
            new_v = self.beta2 * previous_v + (1 - self.beta2) * (grad ** 2)
            self.v[param] = new_v

            # Early on the avgs lean toward zero, divide that out
            m_hat = new_m / (1 - self.beta1 ** self.t)
            v_hat = new_v / (1 - self.beta2 ** self.t)

            param.data = param.data - self.lr * m_hat / (v_hat ** 0.5 + self.eps)
        ### END YOUR SOLUTION
