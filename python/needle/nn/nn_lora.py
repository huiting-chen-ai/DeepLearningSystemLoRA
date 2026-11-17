"""Low-rank adapters for Needle linear layers."""

from typing import Iterable

from needle import init, ops
from .nn_basic import Linear, Module, Parameter


class LoRALinear(Module):
    """Wrap a :class:`Linear` layer with a trainable low-rank update.

    The wrapped linear layer is frozen. The update is ``(x @ lora_A @ lora_B) / rank``;
    ``lora_B`` starts at zero, so the wrapper initially produces exactly the same
    output as its base layer. Inputs follow ``Linear``'s 2-D matrix-multiply contract.

    Args:
        base_layer: Linear layer to adapt.
        rank: Positive rank of the low-rank update.
    """

    def __init__(self, base_layer: Linear, rank: int) -> None:
        super().__init__()
        if not isinstance(base_layer, Linear):
            raise TypeError("base_layer must be a needle.nn.Linear")
        if not isinstance(rank, int) or rank <= 0:
            raise ValueError("rank must be a positive integer")

        self.base_layer = base_layer
        self.rank = rank
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.scaling = 1.0 / rank

        device = base_layer.weight.device
        dtype = base_layer.weight.dtype
        bound = 1.0 / (self.in_features ** 0.5)
        self.lora_A = Parameter(
            init.rand(
                self.in_features,
                rank,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
        )
        self.lora_B = Parameter(
            init.zeros(
                rank,
                self.out_features,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
        )

        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        base_output = self.base_layer(x)
        adapter_output = ops.matmul(ops.matmul(x, self.lora_A), self.lora_B)
        return base_output + self.scaling * adapter_output


def _named_modules(module: Module, prefix: str = ""):
    """Yield ``(path, module)`` pairs for modules nested in attributes/containers."""
    for name, value in vars(module).items():
        path = f"{prefix}.{name}" if prefix else name
        yield from _walk_named(value, path)


def _walk_named(value, path: str):
    if isinstance(value, Module):
        yield path, value
        yield from _named_modules(value, path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _walk_named(child, f"{path}.{index}")
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_named(child, f"{path}.{key}")


def _replace_targets(value, path: str, targets: set[str], matched: set[str]):
    """Recursively replace matching Linear children, preserving containers."""
    if isinstance(value, Linear):
        leaf_name = path.rsplit(".", 1)[-1]
        hits = {target for target in targets
                if target == path or target == leaf_name or path.endswith("." + target)}
        if hits:
            matched.update(hits)
            # Rank is attached by apply_lora via the closure below.
            return value
        return value

    if isinstance(value, Module):
        for name, child in list(vars(value).items()):
            child_path = f"{path}.{name}" if path else name
            vars(value)[name] = _replace_targets(child, child_path, targets, matched)
        return value

    if isinstance(value, list):
        for index, child in enumerate(value):
            value[index] = _replace_targets(child, f"{path}.{index}", targets, matched)
        return value

    if isinstance(value, tuple):
        return tuple(
            _replace_targets(child, f"{path}.{index}", targets, matched)
            for index, child in enumerate(value)
        )

    if isinstance(value, dict):
        for key, child in list(value.items()):
            value[key] = _replace_targets(child, f"{path}.{key}", targets, matched)
        return value

    return value


def apply_lora(model: Module, targets: Iterable[str], rank: int) -> Module:
    """Freeze a model and replace selected Linear modules with LoRA wrappers.

    ``targets`` may contain exact dotted module paths (for example,
    ``"transformerlayer.modules.0.layer1.modules.0.q_projection"``) or leaf
    attribute names (for example, ``"q_projection"``), in which case every
    matching Linear layer is adapted. At least one Linear must match each target.

    The original model is modified in place and returned. All pre-existing model
    parameters are frozen; only each adapter's ``lora_A`` and ``lora_B`` remain
    trainable.
    """
    if not isinstance(model, Module):
        raise TypeError("model must be a needle.nn.Module")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise ValueError("rank must be a positive integer")
    if isinstance(targets, str):
        targets = [targets]
    try:
        target_set = set(targets)
    except TypeError as exc:
        raise TypeError("targets must be a string or iterable of strings") from exc
    if not target_set or any(not isinstance(target, str) or not target for target in target_set):
        raise ValueError("targets must contain one or more non-empty strings")

    # Check target names before mutating/freezing the supplied model.
    available_paths = [path for path, child in _named_modules(model)
                       if isinstance(child, Linear)]
    matched_targets = {
        target for target in target_set
        if any(target == path or target == path.rsplit(".", 1)[-1]
               or path.endswith("." + target) for path in available_paths)
    }
    missing = target_set - matched_targets
    if missing:
        raise ValueError(
            "No Linear module matched target(s): " + ", ".join(sorted(missing))
        )

    # Freeze all existing parameters first; adapter parameters are created as trainable.
    for parameter in model.parameters():
        parameter.requires_grad = False

    matched = set()

    def replace(value, path):
        if isinstance(value, Linear):
            leaf_name = path.rsplit(".", 1)[-1]
            hits = {target for target in target_set
                    if target == path or target == leaf_name or path.endswith("." + target)}
            if hits:
                matched.update(hits)
                return LoRALinear(value, rank)
            return value
        if isinstance(value, Module):
            for name, child in list(vars(value).items()):
                child_path = f"{path}.{name}" if path else name
                vars(value)[name] = replace(child, child_path)
            return value
        if isinstance(value, list):
            for index, child in enumerate(value):
                value[index] = replace(child, f"{path}.{index}")
            return value
        if isinstance(value, tuple):
            return tuple(replace(child, f"{path}.{index}")
                         for index, child in enumerate(value))
        if isinstance(value, dict):
            for key, child in list(value.items()):
                value[key] = replace(child, f"{path}.{key}")
            return value
        return value

    replace(model, "")
    # Defensive check in case the model changed during traversal.
    if matched != matched_targets:
        raise RuntimeError("The model's module tree changed while applying LoRA")
    return model
