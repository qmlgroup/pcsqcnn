"""Adam update matching the optimizer used in the MFA reference training code."""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer


class MFAAdam(Optimizer):
    """PennyLane Adam defaults, including its epsilon placement and beta2=.99."""

    def __init__(self, params, *, lr: float = 0.01) -> None:
        if lr <= 0:
            raise ValueError("Learning rate must be positive.")
        super().__init__(params, defaults={"lr": lr, "beta1": 0.9, "beta2": 0.99, "eps": 1e-8})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("MFAAdam does not support sparse gradients.")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(parameter)
                    state["v"] = torch.zeros_like(parameter)
                state["step"] += 1
                step = state["step"]
                state["m"].mul_(beta1).add_(gradient, alpha=1 - beta1)
                state["v"].mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                corrected_lr = group["lr"] * math.sqrt(1 - beta2**step) / (1 - beta1**step)
                parameter.addcdiv_(
                    state["m"], state["v"].sqrt().add_(group["eps"]), value=-corrected_lr
                )
        return loss
