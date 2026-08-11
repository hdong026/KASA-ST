#!/usr/bin/env python3
"""Train Plan A controller on temporal cross-fitted oracle (safety-locked)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _smoke_controller(device: str) -> int:
    import torch
    from basicts.archs.arch_zoo.ChainForecasting_arch.adaptive_forecast_refinement_route import (
        AdaptiveForecastRefinementRouteNet,
    )
    from basicts.archs.arch_zoo.ChainForecasting_arch.forecast_refinement_gain_loss import (
        compute_pair_imbalance_weights,
        refinement_gain_total_loss,
    )
    from basicts.archs.arch_zoo.ChainForecasting_arch.forecast_refinement_routes import (
        route_scores_from_gains,
    )
    from scripts.budget_f2f_synth_kwargs import synthetic_budget_f2f_kwargs

    dev = torch.device(
        device if ("cpu" in device or torch.cuda.is_available()) else "cpu"
    )
    model = AdaptiveForecastRefinementRouteNet(
        **synthetic_budget_f2f_kwargs(
            node_size=7, training_phase="refinement_controller"
        )
    ).to(dev)
    model.set_training_phase("refinement_controller")
    model.freeze_backbone(True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    true_gains = torch.randn(32, 3, device=dev)
    scores_true = route_scores_from_gains(
        true_gains[:, 0],
        true_gains[:, 1],
        true_gains[:, 2],
        index_map=model.index_map,
        n_routes=4,
    )
    w_pos, w_neg, _ = compute_pair_imbalance_weights(scores_true.cpu())
    for step in range(2):
        h = torch.randn(8, 12, 7, 4, device=dev)
        pred = model.estimate_refinement_gains(h)["predicted_gains"]
        loss, parts = refinement_gain_total_loss(
            pred,
            true_gains[:8],
            index_map=model.index_map,
            n_routes=4,
            pair_weights_pos=w_pos.to(dev),
            pair_weights_neg=w_neg.to(dev),
        )
        opt.zero_grad()
        loss.backward()
        for p in model.backbone.parameters():
            if p.grad is not None and float(p.grad.abs().sum()) > 0:
                raise RuntimeError("backbone received gradients")
        opt.step()
        print(f"[smoke A4] step={step} loss={float(loss):.4f} parts={parts}")
    out = Path("/tmp/kasa_planA_controller_smoke.pt")
    torch.save({"model_state_dict": model.state_dict(), "smoke": True}, out)
    print("SMOKE TEST ONLY - NOT A SCIENTIFIC RESULT")
    print(f"Wrote {out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--crossfit-oracle", default=None)
    p.add_argument("--valid-oracle", default=None)
    p.add_argument("--supernet-checkpoint", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--confirm-full-run", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--num-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--out-dir",
        default="checkpoints/PEMS04/H12/budget_f2f/crossfit_refinement_controller",
    )
    args, unknown = p.parse_known_args()

    if not args.smoke_test and not args.confirm_full_run:
        raise RuntimeError(
            "Full training requires --confirm-full-run "
            "(or pass --smoke-test for a non-scientific code path)."
        )

    if args.smoke_test:
        return _smoke_controller(args.device)

    if not args.crossfit_oracle or not args.valid_oracle or not args.supernet_checkpoint:
        raise RuntimeError(
            "Formal mode requires --crossfit-oracle --valid-oracle --supernet-checkpoint"
        )

    fwd = [
        "train_forecast_refinement_controller.py",
        "--supernet-checkpoint",
        args.supernet_checkpoint,
        "--train-oracle",
        args.crossfit_oracle,
        "--valid-oracle",
        args.valid_oracle,
        "--device",
        args.device,
        "--num-epochs",
        str(args.num_epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--out-dir",
        args.out_dir,
        "--allow-oracle-subset",
        "--confirm-full-run",
    ]
    fwd.extend(unknown)
    sys.argv = fwd
    from scripts.train_forecast_refinement_controller import main as base_main

    return base_main()


if __name__ == "__main__":
    raise SystemExit(main())
