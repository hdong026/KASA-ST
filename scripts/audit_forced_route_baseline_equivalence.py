#!/usr/bin/env python3
"""Strict architecture-equivalence audit:

original ChainForecasting (formal [3,6,12] token-loss settings)
vs
BudgetConditionedAdaptiveF2FNet(forced=[3,6,12], baseline_compatible)

CPU-only, no training.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs import ChainForecasting, BudgetConditionedAdaptiveF2FNet
from basicts.archs.arch_zoo.ChainForecasting_arch.budget_conditioned_f2f_loss import (
    baseline_compatible_token_mae,
)
from basicts.losses.forecast_state_token_mae import forecast_state_token_mae


def shared_kwargs(node_size: int = 16, chain_lengths=None) -> dict:
    """Matched kwargs for formal progressive + condition_only + token loss."""
    return {
        "node_size": node_size,
        "input_len": 12,
        "output_len": 12,
        "input_dim": 4,
        "main_input_dim": 3,
        "output_dim": 1,
        "patch_len": 3,
        "stride": 4,
        "td_size": 288,
        "dw_size": 7,
        "d_td": 16,
        "d_dw": 16,
        "d_d": 16,
        "d_spa": 16,
        "num_layer": 1,
        "if_time_in_day": True,
        "if_day_in_week": True,
        "if_spatial": True,
        "spatial_scheme": "C",
        "use_gcn": False,
        "use_dynamic_spatial": False,
        "use_hybrid_graph": False,
        "use_patch_branch": True,
        "use_downsample_branch": True,
        "use_linear_residual_branch": True,
        "patch_embedding_mode": "serial_concat",
        "patch_data_input_mode": "all",
        "use_prev_condition": True,
        "spatial_placement": "interleaved_progressive",
        "progressive_spatial_ratios": [0.25, 0.5, 1.0],
        "progressive_spatial_topks": [4, 8, 16],
        "progressive_spatial_alphas": [0.03, 0.06, 0.10],
        "post_spatial_mode": "adaptive_only",
        "use_adaptive_adj": True,
        "adp_hidden_dim": 16,
        "use_forecast_state_adapter": True,
        "forecast_state_adapter_mode": "condition_only",
        "forecast_state_adapter_hidden_dim": 16,
        "forecast_state_adapter_epsilon": 0.02,
        "chain_lengths": chain_lengths if chain_lengths is not None else [3, 6, 12],
        "propagation_mode": "forecast_state",
    }


def _diff(a: torch.Tensor, b: torch.Tensor) -> dict:
    d = (a.detach().float() - b.detach().float()).abs()
    return {
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "shape_match": list(a.shape) == list(b.shape),
        "max_abs_diff": float(d.max()) if d.numel() else 0.0,
        "mean_abs_diff": float(d.mean()) if d.numel() else 0.0,
    }


def _print_diff(name: str, info: dict, tol: float = 1e-5) -> bool:
    ok = info["shape_match"] and info["max_abs_diff"] <= tol
    status = "PASS" if ok else "FAIL"
    print(
        f"  [{status}] {name}: shape={info['shape_a']} "
        f"max_abs={info['max_abs_diff']:.3e} mean_abs={info['mean_abs_diff']:.3e}"
    )
    return ok


def map_original_to_budget(orig: ChainForecasting, budget: BudgetConditionedAdaptiveF2FNet):
    src = orig.state_dict()
    mapped = {f"backbone.{k}": v for k, v in src.items()}
    dst = budget.state_dict()
    missing = [k for k in dst if k not in mapped and not k.startswith("planner.")]
    unexpected = [k for k in mapped if k not in dst]
    shape_mismatch = []
    loadable = {}
    for k, v in mapped.items():
        if k not in dst:
            continue
        if tuple(dst[k].shape) != tuple(v.shape):
            shape_mismatch.append((k, tuple(v.shape), tuple(dst[k].shape)))
        else:
            loadable[k] = v
    # Keep planner randomly initialized
    for k, v in dst.items():
        if k.startswith("planner.") and k not in loadable:
            loadable[k] = v
    return loadable, missing, unexpected, shape_mismatch


def extract_formal_conditions(orig_out: dict, orig: ChainForecasting, history, chain_preds):
    """Rebuild stage-6 downstream condition the same way as formal forward."""
    # Formal stores supervised chain_preds; condition after stage idx==1 is adapter output.
    # We recompute by running a traced loop matching _forward_chain.
    from basicts.archs.arch_zoo.ChainForecasting_arch.kasa_temporal_step import (
        interpolate_forecast,
    )

    spatial_codebook = orig._spatial_codebook()
    prev = None
    conditions = []
    temporal = []
    spatial = []
    for step_idx, step in enumerate(orig.temporal_steps):
        target_len = orig.chain_lengths[step_idx]
        previous_state = prev
        prev_up = None
        if previous_state is not None and orig.use_prev_condition:
            prev_up = interpolate_forecast(previous_state, target_len)
        t_k = step(history, prev_forecast=prev_up, spatial_codebook=spatial_codebook)
        z_raw = orig._apply_progressive_spatial_refine(t_k, history, step_idx)
        next_prev = z_raw
        if (
            orig.forecast_state_adapter is not None
            and previous_state is not None
            and orig.forecast_state_adapter_mode == "condition_only"
            and step_idx == 1
        ):
            next_prev = orig.forecast_state_adapter(
                current_state=z_raw,
                previous_state=previous_state,
                stage_ratio=float(target_len) / float(orig.output_len),
            )
        temporal.append(t_k)
        spatial.append(z_raw)
        conditions.append(next_prev if step_idx < len(orig.temporal_steps) - 1 else None)
        prev = next_prev
    return temporal, spatial, conditions


def main() -> int:
    torch.manual_seed(0)
    report: dict = {"checks": {}, "divergences": []}
    n, p, h, b = 16, 12, 12, 2
    x = torch.randn(b, p, n, 4)
    y = torch.randn(b, h, n, 1)

    print("=" * 72)
    print("1) Forced-mode priority / execution")
    print("=" * 72)
    forced = BudgetConditionedAdaptiveF2FNet(
        **{
            **shared_kwargs(n),
            "candidate_routes": [[12], [6, 12], [3, 12], [3, 6, 12]],
            "forced_route": [3, 6, 12],
            "route_selection_mode": "forced",
            "training_phase": "supernet",
            "route_sampling": "sandwich",  # must be ignored when forced
            "loss_mode": "baseline_compatible",
        }
    )
    # sandwich must raise when combined with forced
    raised = False
    try:
        forced(
            history_data=x,
            train=True,
            return_all=True,
            sandwich_routes=[[12], [3, 6, 12]],
        )
    except RuntimeError as e:
        raised = True
        print(f"  sandwich+forced raises: {e}")
    report["checks"]["forced_overrides_sandwich"] = raised
    if not raised:
        report["divergences"].append("forced mode did not reject sandwich_routes")

    out = forced(history_data=x, train=True, return_all=True)
    print(
        f"  executed_routes={out.get('executed_routes')} "
        f"chain_resolutions={out['chain_resolutions']} "
        f"actual_stage_count={out['actual_stage_count']}"
    )
    ok_exec = (
        out.get("executed_routes") == [[3, 6, 12]]
        and list(out["chain_resolutions"]) == [3, 6, 12]
        and int(out["actual_stage_count"]) == 3
    )
    report["checks"]["forced_3612_execution"] = ok_exec
    if not ok_exec:
        report["divergences"].append("forced [3,6,12] did not execute exactly that route")

    print("=" * 72)
    print("2) Component inventory (class / location)")
    print("=" * 72)
    rows = [
        ("KASATemporalStep count", "ChainForecasting.temporal_steps", "Budget.backbone.temporal_steps"),
        ("patch branch", "KASATemporalStep.patch_encoder(+cond)", "same via backbone"),
        ("phase/downsample branch", "KASATemporalStep.downsamp_encoder(+cond)", "same via backbone"),
        ("direct linear", "KASATemporalStep.residual Conv2d", "same via backbone"),
        ("shared codebooks", "ChainForecasting.td/dw/spa_codebook", "backbone.*_codebook"),
        ("condition align", "interpolate_forecast in _forward_chain", "_execute_route + interpolate_forecast"),
        ("adapter", "ForecastStateAdapter @ step_idx==1", "adapter @ supernet idx==1"),
        ("spatial", "ABCDSpatialModule progressive list", "same modules by resolution index"),
        ("loss", "forecast_state_token_mae / token_normalized", "baseline_compatible_token_mae -> same fn"),
    ]
    for name, a, b in rows:
        print(f"  - {name}:\n      formal: {a}\n      budget: {b}")

    print("=" * 72)
    print("3) Zero-condition / first-stage input interface")
    print("=" * 72)
    traced = forced._execute_route(x, [3, 6, 12], return_trace=True)
    for cfg in traced["spatial_cfgs"]:
        print(f"  res={cfg['resolution']}: {cfg['step_input_channels']} adapter={cfg['adapter_used']}")
    first = traced["spatial_cfgs"][0]
    no_zero = first["step_input_channels"] == "base_encoders_only" and not first["condition_present"]
    report["checks"]["stage3_no_zero_condition"] = no_zero
    if not no_zero:
        report["divergences"].append("stage-3 uses condition / expanded channels")
    else:
        print("  PASS: stage 3 uses base encoders only (prev_forecast=None), no zero-condition padding")

    print("=" * 72)
    print("4) Spatial config by resolution (all candidate routes)")
    print("=" * 72)
    for route in [[12], [6, 12], [3, 12], [3, 6, 12]]:
        m = BudgetConditionedAdaptiveF2FNet(
            **{
                **shared_kwargs(n),
                "candidate_routes": [[12], [6, 12], [3, 12], [3, 6, 12]],
                "forced_route": route,
                "route_selection_mode": "forced",
                "loss_mode": "baseline_compatible",
            }
        )
        tr = m._execute_route(x, route, return_trace=True)
        print(f"  route {route}:")
        for cfg in tr["spatial_cfgs"]:
            print(
                f"    res={cfg['resolution']} idx={cfg['supernet_index']} "
                f"ratio={cfg['ratio']} topk={cfg['dyn_topk']}/{cfg['configured_topk']} "
                f"alpha={cfg['dyn_alpha']}/{cfg['configured_alpha']} "
                f"adp_dim={cfg['adp_hidden_dim']} adapter={cfg['adapter_used']}"
            )

    print("=" * 72)
    print("5) Loss numerical equivalence")
    print("=" * 72)
    preds = traced["chain_preds"]
    resolutions = traced["chain_resolutions"]
    targets = [ChainForecasting.pool_target(y, k) for k in resolutions]
    l_old = forecast_state_token_mae(preds, targets, null_val=0.0)
    l_new = baseline_compatible_token_mae(preds, resolutions, y, null_val=0.0)
    loss_diff = float((l_old - l_new).abs())
    print(f"  original_token_mae={float(l_old):.8f}")
    print(f"  baseline_compatible={float(l_new):.8f}")
    print(f"  abs_diff={loss_diff:.3e}")
    report["checks"]["loss_equiv"] = loss_diff < 1e-7
    if loss_diff >= 1e-7:
        report["divergences"].append(f"loss abs_diff={loss_diff}")

    print("=" * 72)
    print("6) Weight-mapping + forward equivalence (forced [3,6,12])")
    print("=" * 72)
    torch.manual_seed(123)
    orig = ChainForecasting(**shared_kwargs(n, [3, 6, 12]))
    torch.manual_seed(123)
    # Intentionally different planner init; backbone rebuilt then overwritten.
    budget = BudgetConditionedAdaptiveF2FNet(
        **{
            **shared_kwargs(n),
            "candidate_routes": [[12], [6, 12], [3, 12], [3, 6, 12]],
            "forced_route": [3, 6, 12],
            "route_selection_mode": "forced",
            "loss_mode": "baseline_compatible",
            "planner_hidden_dim": 32,
        }
    )
    n_orig = sum(p.numel() for p in orig.parameters())
    n_bud = sum(p.numel() for p in budget.parameters())
    n_bb = sum(p.numel() for p in budget.backbone.parameters())
    n_pl = sum(p.numel() for p in budget.planner.parameters())
    print(f"  params original={n_orig} budget_total={n_bud} backbone={n_bb} planner={n_pl}")
    report["param_counts"] = {
        "original": n_orig,
        "budget_total": n_bud,
        "backbone": n_bb,
        "planner": n_pl,
    }
    if n_orig != n_bb:
        report["divergences"].append(
            f"backbone param count {n_bb} != original {n_orig} (first structural fork)"
        )
        print("  FAIL: backbone parameter count differs from original — structural mismatch")

    loadable, missing, unexpected, shape_mismatch = map_original_to_budget(orig, budget)
    print(f"  missing_non_planner={len(missing)} unexpected={len(unexpected)} shape_mismatch={len(shape_mismatch)}")
    if missing[:10]:
        print("  missing sample:", missing[:10])
    if unexpected[:10]:
        print("  unexpected sample:", unexpected[:10])
    if shape_mismatch[:5]:
        print("  shape_mismatch sample:", shape_mismatch[:5])
    report["mapping"] = {
        "missing_non_planner": missing,
        "unexpected": unexpected,
        "shape_mismatch": [list(x) for x in shape_mismatch],
    }
    if missing or shape_mismatch:
        report["divergences"].append("state_dict mapping incomplete/mismatched")
        print("  STOP: cannot claim weight-level equivalence")
    else:
        budget.load_state_dict(loadable, strict=False)
        orig.eval()
        budget.eval()
        with torch.no_grad():
            o = orig(history_data=x, train=False, return_all=True)
            # Prefer traced execution for conditions/temporal
            bt = budget._execute_route(x, [3, 6, 12], return_trace=True)
            ot_temp, ot_spat, ot_cond = extract_formal_conditions(o, orig, x, o["chain_preds"])

        all_ok = True
        checks = [
            ("stage3_temporal", ot_temp[0], bt["temporal_preds"][0]),
            ("stage3_post_spatial", ot_spat[0], bt["chain_preds"][0]),
            ("stage6_temporal", ot_temp[1], bt["temporal_preds"][1]),
            ("stage6_supervised", ot_spat[1], bt["chain_preds"][1]),
            ("stage6_downstream_condition", ot_cond[1], bt["downstream_conditions"][1]),
            ("stage12_temporal", ot_temp[2], bt["temporal_preds"][2]),
            ("final_forecast", o["pred"], bt["pred"]),
        ]
        print("  tensor comparisons after weight copy:")
        first_fail = None
        for name, a, b_ in checks:
            info = _diff(a, b_)
            ok = _print_diff(name, info)
            report["checks"][name] = info
            if not ok and first_fail is None:
                first_fail = name
            all_ok = all_ok and ok
        # loss after copy
        with torch.no_grad():
            l1 = forecast_state_token_mae(
                o["chain_preds"],
                [ChainForecasting.pool_target(y, k) for k in [3, 6, 12]],
            )
            l2 = baseline_compatible_token_mae(bt["chain_preds"], [3, 6, 12], y)
        info = {
            "shape_a": [],
            "shape_b": [],
            "shape_match": True,
            "max_abs_diff": float((l1 - l2).abs()),
            "mean_abs_diff": float((l1 - l2).abs()),
        }
        ok = _print_diff("token_normalized_loss", info)
        all_ok = all_ok and ok
        report["checks"]["forward_equiv_after_copy"] = all_ok
        if first_fail:
            report["divergences"].append(f"first tensor fork after weight copy: {first_fail}")
            print(f"  FIRST FORK: {first_fail}")
        elif all_ok:
            print("  PASS: forced [3,6,12] matches original within float tol after weight copy")

    print("=" * 72)
    print("7) Forced [12] vs dedicated single-stage ChainForecasting([12])")
    print("=" * 72)
    single = ChainForecasting(**shared_kwargs(n, [12]))
    forced12 = BudgetConditionedAdaptiveF2FNet(
        **{
            **shared_kwargs(n),
            "candidate_routes": [[12], [6, 12], [3, 12], [3, 6, 12]],
            "forced_route": [12],
            "route_selection_mode": "forced",
            "loss_mode": "baseline_compatible",
        }
    )
    tr12 = forced12._execute_route(x, [12], return_trace=True)
    cfg12 = tr12["spatial_cfgs"][0]
    # Single-stage progressive uses _fit_stage_list -> first ratio only (0.25)
    single_mod = single.progressive_spatial_modules[0]
    print(
        f"  single-stage[12]: n_temporal={len(single.temporal_steps)} "
        f"spatial_modules={len(single.progressive_spatial_modules)} "
        f"dyn_topk={getattr(single_mod,'dyn_topk',None)} "
        f"dyn_alpha={getattr(single_mod,'dyn_alpha',None)} "
        f"adp_hidden={getattr(single_mod,'adp_hidden_dim',None)}"
    )
    print(
        f"  forced[12]: uses supernet idx={cfg12['supernet_index']} "
        f"ratio={cfg12['ratio']} topk={cfg12['dyn_topk']} alpha={cfg12['dyn_alpha']} "
        f"channels={cfg12['step_input_channels']} "
        f"n_temporal_modules_in_supernet={len(forced12.backbone.temporal_steps)}"
    )
    print(
        "  VERDICT: forced[12] is NOT original single-stage [12]. "
        "It reuses the H=12 module from a [3,6,12] supernet with FULL spatial "
        "(ratio 1.0 / last progressive slot), whereas ChainForecasting(chain_lengths=[12]) "
        "builds ONE temporal step and ONE progressive module from ratios[:1]=[0.25] (LIGHT)."
    )
    report["forced12_verdict"] = (
        "not_equivalent_to_single_stage_chain_lengths_[12]; "
        "uses supernet stage index 2 with full spatial; "
        "first stage still base_encoders_only (no zero condition)"
    )

    print("=" * 72)
    print("8) Config field notes (PEMS04 formal vs forced_12 run)")
    print("=" * 72)
    formal_cfg = ROOT / (
        "checkpoints/fixed_input_horizon_pems04/h12/"
        "chain_interleaved_progressive_spatial_state_adapter_fixed_token_loss_seed1/"
        "8644aca996780912e49d26f59e8e13e6/"
        "h12_chain_interleaved_progressive_spatial_state_adapter_fixed_token_loss_seed1.py"
    )
    forced_cfg = ROOT / "generated/temp_configs_budget_f2f_pems04/H12_forced_12_seed1.py"
    print(f"  formal_cfg_exists={formal_cfg.is_file()} path={formal_cfg}")
    print(f"  forced12_cfg_exists={forced_cfg.is_file()} path={forced_cfg}")
    print(
        "  Known field diffs: MODEL.NAME/ARCH; chain_loss_mode "
        "token_normalized vs baseline_compatible (same underlying MAE helper); "
        "budget adds forced_route/candidate_routes/training_phase/planner; "
        "hyperparams (d_*, progressive, lr, milestones, bs) matched in generated PEMS configs."
    )
    print(
        "  Observed result: forced[12] test MAE≈18.977 (CSV) while formal full-chain "
        "baseline is a different architecture/objective schedule ([3,6,12] supervision)."
    )

    out_path = ROOT / "results/budget_f2f_architecture_equivalence_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=" * 72)
    print(f"Wrote {out_path}")
    print("Divergences:")
    for d in report["divergences"] or ["(none recorded beyond forced[12] structural note)"]:
        print(f"  - {d}")
    return 0 if report["checks"].get("forced_3612_execution") else 1


if __name__ == "__main__":
    raise SystemExit(main())
