"""Regression tests for the Diffusion Policy EMA dependency boundary."""

import torch
import torch.nn as nn

from robomimic.algo.diffusion_policy import _DiffusionPolicyEMA


def _weight(module: nn.Module) -> torch.Tensor:
    return next(module.parameters()).detach().clone()


def test_diffusion_policy_ema_updates_and_round_trips_weights():
    source = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        source.weight.fill_(1.0)

    ema = _DiffusionPolicyEMA(source, power=0.75)
    assert not ema.averaged_model.training
    assert all(not parameter.requires_grad for parameter in ema.averaged_model.parameters())
    torch.testing.assert_close(_weight(ema.averaged_model), _weight(source))

    with torch.no_grad():
        source.weight.fill_(3.0)
    ema.step(source)
    assert not torch.equal(_weight(ema.averaged_model), torch.ones((1, 2)))

    saved_weights = ema.state_dict()
    restored = _DiffusionPolicyEMA(nn.Linear(2, 1, bias=False), power=0.75)
    restored.load_state_dict(saved_weights)
    torch.testing.assert_close(
        _weight(restored.averaged_model),
        _weight(ema.averaged_model),
    )

    before_step = _weight(restored.averaged_model)
    with torch.no_grad():
        source.weight.fill_(5.0)
    restored.step(source)
    assert not torch.equal(_weight(restored.averaged_model), before_step)
