"""Numerical parity tests for Needle LoRA against a PyTorch reference."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import needle as ndl
from needle import ops


@pytest.mark.parametrize("rank", [1, 2, 4])
@pytest.mark.parametrize("bias", [False, True])
def test_lora_linear_matches_pytorch_forward_and_gradients(rank, bias):
    """Use identical weights/factors and compare outputs plus input/LoRA grads."""
    rng = np.random.RandomState(2025 + rank + int(bias))
    batch, in_features, out_features = 3, 6, 5

    base = ndl.nn.Linear(in_features, out_features, bias=bias, device=ndl.cpu())
    layer = ndl.nn.LoRALinear(base, rank)

    x_np = rng.randn(batch, in_features).astype(np.float32)
    weight_np = rng.randn(in_features, out_features).astype(np.float32)
    a_np = rng.randn(in_features, rank).astype(np.float32)
    b_np = rng.randn(rank, out_features).astype(np.float32)
    bias_np = rng.randn(1, out_features).astype(np.float32) if bias else None

    base.weight.data = ndl.Tensor(weight_np, device=ndl.cpu())
    if bias:
        base.bias.data = ndl.Tensor(bias_np, device=ndl.cpu())
    layer.lora_A.data = ndl.Tensor(a_np, device=ndl.cpu())
    layer.lora_B.data = ndl.Tensor(b_np, device=ndl.cpu())

    x_needle = ndl.Tensor(x_np, device=ndl.cpu(), requires_grad=True)
    output_needle = layer(x_needle)
    ops.summation(output_needle).backward()

    x_torch = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
    weight_torch = torch.tensor(weight_np, dtype=torch.float32)
    a_torch = torch.tensor(a_np, dtype=torch.float32, requires_grad=True)
    b_torch = torch.tensor(b_np, dtype=torch.float32, requires_grad=True)

    output_torch = x_torch @ weight_torch + (x_torch @ a_torch @ b_torch) / rank
    if bias:
        output_torch = output_torch + torch.tensor(bias_np, dtype=torch.float32)
    output_torch.sum().backward()

    np.testing.assert_allclose(
        output_needle.numpy(), output_torch.detach().numpy(), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        x_needle.grad.numpy(), x_torch.grad.numpy(), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        layer.lora_A.grad.numpy(), a_torch.grad.numpy(), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        layer.lora_B.grad.numpy(), b_torch.grad.numpy(), rtol=1e-5, atol=1e-5
    )

    # The base layer is frozen by LoRALinear; it should not receive parameter grads.
    assert not base.weight.requires_grad
    assert base.weight.grad is None
    if bias:
        assert not base.bias.requires_grad
        assert base.bias.grad is None


def test_lora_initially_preserves_base_layer_output():
    """The zero-initialized B factor makes the initial adapter update zero."""
    rng = np.random.RandomState(7)
    base = ndl.nn.Linear(4, 3, bias=True, device=ndl.cpu())
    x_np = rng.randn(2, 4).astype(np.float32)
    x = ndl.Tensor(x_np, device=ndl.cpu())

    expected = base(x).numpy()
    adapted = ndl.nn.LoRALinear(base, rank=2)

    np.testing.assert_allclose(adapted(x).numpy(), expected, rtol=1e-6, atol=1e-6)


def test_apply_lora_freezes_base_and_leaves_adapters_trainable():
    """Check target replacement, parameter freezing, and rank-dependent counts."""
    model = ndl.nn.Sequential(
        ndl.nn.Linear(4, 6),
        ndl.nn.ReLU(),
        ndl.nn.Linear(6, 3),
    )
    ndl.nn.apply_lora(model, targets=["modules.0"], rank=2)

    first, _, second = model.modules
    assert isinstance(first, ndl.nn.LoRALinear)
    assert isinstance(second, ndl.nn.Linear)
    assert all(not parameter.requires_grad for parameter in first.base_layer.parameters())
    assert first.lora_A.requires_grad
    assert first.lora_B.requires_grad
    assert all(not parameter.requires_grad for parameter in second.parameters())

    trainable_elements = sum(
        parameter.size for parameter in model.parameters() if parameter.requires_grad
    )
    assert trainable_elements == 2 * (4 * 2 + 2 * 6)
