"""Structural and objective checks for the isolated post-Z3 GRPO pilot."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.F2FCoTResolutionNative_arch.f2f_cot_resolution_native_v1_shared_prefix import (
    F2FCoTResolutionNativeV1SharedPrefixNet,
)
from basicts.archs.arch_zoo.F2FCoTResolutionNative_arch.post_z3_constrained_grpo import (
    BudgetDualPanel,
    FrozenPostZ3Environment,
    PostZ3BudgetRouter,
    clipped_trajectory_grpo_loss,
    leave_one_out_advantages,
)


def small_model(device: torch.device):
    return F2FCoTResolutionNativeV1SharedPrefixNet(
        node_size=7,
        d_d=8,
        d_td=4,
        d_dw=4,
        d_spa=8,
        evidence_num_layer=1,
        reasoner_num_layer=1,
        graph_rank=4,
        adp_hidden_dim=4,
        adp_topk=3,
    ).to(device)


def sample(device: torch.device):
    history = torch.randn(3, 12, 7, 4, device=device)
    history[..., 1] = torch.rand_like(history[..., 1])
    history[..., 2] = torch.randint(0, 7, history[..., 2].shape, device=device).float()
    return history


def main() -> None:
    torch.manual_seed(31)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = small_model(device)
    environment = FrozenPostZ3Environment(model)
    assert not any(parameter.requires_grad for parameter in model.parameters())
    history = sample(device)
    observation = environment.begin(history)
    assert observation.state.current_resolution == 3
    assert model.evidence_encoder.encode_count == 1

    pair = environment.forced_pair(observation)
    reference = model.rollout_shared_prefix_pair(history)
    assert pair.shape == (3, 2, 12, 7, 1)
    assert torch.equal(pair[:, 0], reference["pred_short"])
    assert torch.equal(pair[:, 1], reference["pred_long"])
    assert pair[:, 0].data_ptr() != pair[:, 1].data_ptr()

    policy = PostZ3BudgetRouter(node_size=7, node_hidden=8, hidden_dim=16).to(device)
    budgets = torch.tensor([0.25, 0.50, 0.75], device=device)
    logits = policy(observation, budgets)
    probabilities = torch.sigmoid(logits)
    assert torch.allclose(probabilities, budgets, atol=1e-6), probabilities

    returns = torch.tensor(
        [[-1.000, -1.010, -1.000, -1.010], [-4.0, -4.0, -4.0, -4.0]],
        device=device,
    )
    advantages = leave_one_out_advantages(returns)
    assert torch.allclose(advantages.mean(1), torch.zeros(2, device=device), atol=1e-7)
    assert 0.0 < float(advantages[0].abs().max()) < 0.02
    assert torch.equal(advantages[1], torch.zeros_like(advantages[1]))

    actions = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 1, 0]], device=device)
    old_probability = probabilities.detach()
    expanded = old_probability[:, None].expand_as(actions)
    behavior_log_prob = torch.where(
        actions.bool(), expanded.log(), (1.0 - expanded).log()
    )
    fake_advantage = torch.tensor(
        [[0.01, -0.01, 0.01, -0.01]] * 3, device=device
    )
    loss, details = clipped_trajectory_grpo_loss(
        logits,
        actions,
        behavior_log_prob,
        fake_advantage,
        old_probability,
        clip_ratio=0.2,
        entropy_coefficient=0.0,
        kl_coefficient=0.01,
    )
    assert torch.isfinite(loss)
    assert torch.allclose(details["ratio_mean"], torch.ones((), device=device), atol=1e-6)
    loss.backward()
    assert any(parameter.grad is not None for parameter in policy.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())

    dual = BudgetDualPanel(torch.tensor([0.25, 0.75]), learning_rate=0.5)
    dual.update(torch.tensor([0, 0, 1, 1]), torch.tensor([0.75, 0.75, 0.25, 0.25]))
    assert float(dual.lambdas[0]) > 0.0
    assert float(dual.lambdas[1]) == 0.0

    runner = (ROOT / "scripts" / "train_resolution_native_post_z3_grpo.py").read_text()
    assert 'make_loader("test"' not in runner.lower()
    print("post-Z3 constrained GRPO structural tests passed")


if __name__ == "__main__":
    main()

