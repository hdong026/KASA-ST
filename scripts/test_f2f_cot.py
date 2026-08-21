"""Structural tests for the recurrent F2F CoT model (no dataset required)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.F2FCoT_arch import F2FCoTMultiDepthNet, F2FCoTNet


def small_model(device, model_class=F2FCoTNet):
    return model_class(
        node_size=7,
        input_len=12,
        output_len=12,
        resolutions=[2, 3, 4, 6, 12],
        d_d=8,
        d_td=4,
        d_dw=4,
        d_spa=8,
        num_layer=1,
        resolution_dim=6,
        memory_dim=5,
        context_channels=2,
        condition_channels=3,
        adj_mx_path=None,
        adp_topk=3,
    ).to(device)


def sample(device):
    history = torch.randn(2, 12, 7, 4, device=device)
    history[..., 1] = torch.rand_like(history[..., 1])
    history[..., 2] = torch.randint(
        0, 7, history[..., 2].shape, device=device
    ).to(history.dtype)
    return history


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(7)
    model = small_model(device)
    history = sample(device)

    fixed = model.rollout(history, (3, 6, 12))
    assert [tuple(value.shape) for value in fixed["forecasts"]] == [
        (2, 3, 7, 1),
        (2, 6, 7, 1),
        (2, 12, 7, 1),
    ]
    assert fixed["reasoning_calls"] == 3
    assert model.parameter_breakdown()["one_shared_reasoning_core"] is True
    assert not any("edge" in name.lower() for name, _ in model.named_modules())

    extra = model.rollout(history, (3, 4, 6, 12))
    assert extra["reasoning_calls"] == 4
    assert tuple(extra["forecasts"][1].shape) == (2, 4, 7, 1)

    # Stage III partially decouples reasoning depth from resolution.  The new
    # recurrent program adds no parameters and retains repeated Z_12 states by
    # step index instead of collapsing them in a resolution-keyed dictionary.
    multidepth = small_model(device, F2FCoTMultiDepthNet)
    multidepth.load_state_dict(model.state_dict(), strict=True)
    assert sum(p.numel() for p in multidepth.parameters()) == sum(
        p.numel() for p in model.parameters()
    )
    refined = multidepth.rollout(history, (3, 6, 12, 12, 12))
    assert refined["resolutions"] == (3, 6, 12, 12, 12)
    assert refined["reasoning_calls"] == 5
    assert len(refined["forecasts"]) == 5
    assert all(tuple(value.shape) == (2, 12, 7, 1) for value in refined["forecasts"][2:])
    assert refined["steps"][3]["is_same_resolution_refinement"] is True

    loss = sum(value.abs().mean() for value in fixed["forecasts"])
    loss.backward()
    assert all(
        parameter.grad is not None for parameter in model.reasoning_core.parameters()
    )
    assert model.trace_memory.update_gate.weight.grad is not None
    destination_grad = model.resolution_conditioner.dst_embedding.weight.grad
    assert destination_grad is not None and float(destination_grad.abs().sum()) > 0

    # Shared-prefix fork: 3->12 and 3->6->12 must consume the exact same executed Z_3
    # and the same ForecastTraceMemory.  This is object identity, not a second rollout.
    forked = multidepth.rollout_shared_prefix_pair(history)
    prefix_z3 = forked["prefix"]["forecast"]
    assert forked["shared_prefix"]["z3_same_object"] is True
    assert forked["shared_prefix"]["z3_is_prefix_object"] is True
    assert forked["shared_prefix"]["prefix_memory_same_object"] is True
    assert forked["short"]["forecasts"][0] is prefix_z3
    assert forked["long"]["forecasts"][0] is prefix_z3
    assert forked["short"]["forecasts"][0] is forked["long"]["forecasts"][0]
    assert tuple(forked["short"]["resolutions"]) == (3, 12)
    assert tuple(forked["long"]["resolutions"]) == (3, 6, 12)
    assert forked["shared_prefix"]["extra_reasoning_calls"] == 1
    pair_loss = forked["short"]["pred"].abs().mean() + forked["long"]["pred"].abs().mean()
    pair_loss.backward()
    assert all(
        parameter.grad is not None
        for parameter in multidepth.reasoning_core.parameters()
    )
    before = sum(p.numel() for p in model.parameters())
    after = sum(p.numel() for p in multidepth.parameters())
    assert before == after

    multidepth.eval()
    with torch.no_grad():
        forked_eval = multidepth.rollout_shared_prefix_pair(history)
        independent_short = multidepth.rollout(history, (3, 12))
        independent_long = multidepth.rollout(history, (3, 6, 12))
        assert torch.allclose(
            independent_short["pred"], forked_eval["short"]["pred"], atol=1e-5
        )
        assert torch.allclose(
            independent_long["pred"], forked_eval["long"]["pred"], atol=1e-5
        )
        assert independent_short["forecasts"][0] is not independent_long["forecasts"][0]
        assert forked_eval["short"]["forecasts"][0] is forked_eval["long"]["forecasts"][0]

    # Context is causally used: changing an already emitted explicit forecast
    # while keeping X and the requested transition fixed changes the next state.
    model.eval()
    with torch.no_grad():
        start = model.begin_reasoning(history)
        reached, _ = model.reason_step(history, start, 3)
        altered = type(reached)(
            memory=model.trace_memory.update(
                start.memory,
                reached.latest_forecast + 0.5,
                model.resolution_conditioner(
                    0, 3, 12, len(history), history.device, history.dtype
                )[0],
                12,
            ),
            latest_forecast=reached.latest_forecast + 0.5,
            current_resolution=3,
            forecasts=(reached.latest_forecast + 0.5,),
            resolutions=(3,),
        )
        normal_next, _ = model.reason_step(history, reached, 6)
        altered_next, _ = model.reason_step(history, altered, 6)
        assert not torch.allclose(
            normal_next.latest_forecast, altered_next.latest_forecast
        )

    print("F2FCoT structural tests passed")


if __name__ == "__main__":
    main()
