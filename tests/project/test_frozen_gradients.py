import numpy as np
import pytest

import needle as ndl
from needle import ops


def tensor(values, requires_grad=True):
    return ndl.Tensor(np.array(values, dtype=np.float32), device=ndl.cpu(),
                      requires_grad=requires_grad)


def test_new_values_have_no_grad_and_frozen_output_is_skipped():
    x = tensor([1, 2])
    frozen = tensor([3, 4], requires_grad=False)
    intermediate = x * 2
    detached = intermediate.detach()
    assert all(value.grad is None for value in (x, frozen, intermediate, detached))
    frozen.backward()
    assert frozen.grad is None


def test_frozen_linear_with_shared_input_and_adapters():
    x_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
    w_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    a_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b_np = np.array([[1, 0, 2], [0, 1, 3]], dtype=np.float32)
    layer = ndl.nn.LoRALinear(ndl.nn.Linear(2, 3, bias=True, device=ndl.cpu()), 2)
    layer.base_layer.weight.data = tensor(w_np)
    layer.base_layer.bias.data = tensor([[1, 2, 3]])
    layer.lora_A.data = tensor(a_np)
    layer.lora_B.data = tensor(b_np)
    x = tensor(x_np)
    output = layer(x)
    output.sum().backward()

    ones = np.ones((2, 3), dtype=np.float32)
    np.testing.assert_allclose(output.numpy(), x_np @ w_np + [1, 2, 3] + x_np @ a_np @ b_np / 2)
    np.testing.assert_allclose(x.grad.numpy(), ones @ (w_np + a_np @ b_np / 2).T)
    np.testing.assert_allclose(layer.lora_A.grad.numpy(), x_np.T @ ones @ b_np.T / 2)
    np.testing.assert_allclose(layer.lora_B.grad.numpy(), (x_np @ a_np).T @ ones / 2)
    assert layer.base_layer.weight.grad is None
    assert layer.base_layer.bias.grad is None


def test_shared_branches_and_tensor_tuple_gradients():
    x = tensor([1, 2])
    frozen = tensor([5, 6], requires_grad=False)
    shared = x * frozen
    pair = ops.fused_add_scalars(shared, 1, 2)
    assert pair.grad is None
    (pair[0] * 2 + pair[1] * 3 + shared).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [30, 36])
    np.testing.assert_allclose(shared.grad.numpy(), [6, 6])
    assert frozen.grad is None
    assert pair.grad is not None


def test_freezing_after_backward_preserves_stale_grad_until_reset():
    weight = tensor([2])
    (weight * 3).sum().backward()
    stale = weight.grad
    weight.requires_grad = False
    x = tensor([4])
    (x * weight).sum().backward()
    assert weight.grad is stale
    np.testing.assert_allclose(x.grad.numpy(), [2])
    ndl.optim.SGD([weight]).reset_grad()
    assert weight.grad is None


def test_frozen_intermediate_stops_its_branch():
    x = tensor([2])
    frozen_branch = x * 3
    result = frozen_branch * 4 + x * 5
    frozen_branch.requires_grad = False
    result.backward()
    assert frozen_branch.grad is None
    np.testing.assert_allclose(x.grad.numpy(), [5])


@pytest.mark.parametrize("optimizer", [ndl.optim.SGD, ndl.optim.Adam])
def test_optimizer_skips_frozen_and_missing_gradients(optimizer):
    layer = ndl.nn.LoRALinear(ndl.nn.Linear(2, 2, bias=True, device=ndl.cpu()), 1)
    frozen_weight = layer.base_layer.weight
    frozen_bias = layer.base_layer.bias
    frozen_weight.grad = tensor(np.full(frozen_weight.shape, 100))
    frozen_bias.grad = tensor(np.full(frozen_bias.shape, 100))
    unused = tensor([3])
    params = layer.parameters() + [unused]
    opt = optimizer(params, lr=0.1, weight_decay=0.2)
    before = [p.numpy().copy() for p in (frozen_weight, frozen_bias, unused)]
    x = tensor([[1, 2]])
    layer(x).sum().backward()
    adapter_before = layer.lora_B.numpy().copy()
    if isinstance(opt, ndl.optim.SGD):
        grad_before = layer.lora_B.grad.numpy().copy()
        opt.clip_grad_norm(max_norm=100)
        np.testing.assert_allclose(layer.lora_B.grad.numpy(), grad_before)
    opt.step()
    for p, expected in zip((frozen_weight, frozen_bias, unused), before):
        np.testing.assert_array_equal(p.numpy(), expected)
    assert not np.array_equal(layer.lora_B.numpy(), adapter_before)
    assert frozen_weight.grad is not None
    assert unused.grad is None
    opt.reset_grad()
    opt.step()
    for p, expected in zip((frozen_weight, frozen_bias, unused), before):
        np.testing.assert_array_equal(p.numpy(), expected)


@pytest.mark.parametrize("optimizer", [ndl.optim.SGD, ndl.optim.Adam])
def test_optimizer_skips_previously_trained_frozen_parameter(optimizer):
    p = tensor([2])
    opt = optimizer([p], lr=0.1, weight_decay=0.5)
    (p * 3).backward()
    opt.step()
    trained_value = p.numpy().copy()
    stale_grad = p.grad
    p.requires_grad = False
    opt.step()
    np.testing.assert_array_equal(p.numpy(), trained_value)
    assert p.grad is stale_grad
    opt.reset_grad()
    p.requires_grad = True
    opt.step()
    np.testing.assert_array_equal(p.numpy(), trained_value)


@pytest.mark.parametrize("optimizer", [ndl.optim.SGD, ndl.optim.Adam])
def test_optimizer_updates_trainable_gradient_with_weight_decay(optimizer):
    p = tensor([2])
    p.grad = tensor([3])
    opt = optimizer([p], lr=0.1, weight_decay=0.5)
    opt.step()
    expected = 1.6 if optimizer is ndl.optim.SGD else 1.9
    np.testing.assert_allclose(p.numpy(), [expected], rtol=1e-6)
