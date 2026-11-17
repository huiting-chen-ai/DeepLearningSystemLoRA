"""Tests for the lightweight Needle profiler."""

import numpy as np

import needle as ndl
from needle import ops


def test_profiler_records_operation_and_phase():
    x = ndl.Tensor(np.ones((2, 3), dtype=np.float32), device=ndl.cpu())
    profiler = ndl.profiler.Profiler()

    with profiler:
        with profiler.phase("forward"):
            y = ops.exp(x + 1.0)

    summary = profiler.summary()
    assert summary["phases"]["forward"]["calls"] == 1
    assert summary["operations"]["forward"]["AddScalar"]["calls"] == 1
    assert summary["operations"]["forward"]["Exp"]["calls"] == 1
    assert summary["allocation_totals"]["count"] > 0
    assert summary["allocation_totals"]["bytes"] > 0
    assert y.shape == (2, 3)


def test_parameter_report_counts_trainable_parameters():
    model = ndl.nn.Linear(3, 2, device=ndl.cpu())
    model.weight.requires_grad = False

    report = ndl.profiler.parameter_report(model)
    assert report["total"] == 3 * 2 + 2
    assert report["trainable"] == 2
    assert report["frozen"] == 3 * 2
    assert report["trainable_percent"] == 100.0 * 2 / 8


def test_profiler_can_be_reused_after_reset():
    x = ndl.Tensor(np.ones((1, 1), dtype=np.float32), device=ndl.cpu())
    profiler = ndl.profiler.Profiler()
    with profiler:
        _ = ops.exp(x)
    profiler.reset()
    assert profiler.summary()["operations"] == {}
    with profiler:
        _ = ops.exp(x + 1.0)
    assert profiler.summary()["operations"]["unlabeled"]["Exp"]["calls"] == 1
