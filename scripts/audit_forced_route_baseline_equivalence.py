#!/usr/bin/env python3
"""Strict architecture + raw-scale loss equivalence audit:

original ChainForecasting (formal [3,6,12] token-loss settings)
vs
BudgetConditionedAdaptiveF2FNet(forced=[3,6,12], baseline_compatible)

CPU-only, no training.

Raw-scale checks use a synthetic affine inverse transform so that null_val=0.0
masks physical zeros, not normalized-scale zeros.
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

SYNTHETIC_MEAN = 100.0
SYNTHETIC_STD = 20.0


def synthetic_rescale_pair(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        pred * SYNTHETIC_STD + SYNTHETIC_MEAN,
        target * SYNTHETIC_STD + SYNTHETIC_MEAN,
    )


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
    for k, v in dst.items():
        if k.startswith("planner.") and k not in loadable:
            loadable[k] = v
    return loadable, missing, unexpected, shape_mismatch


def extract_formal_conditions(orig_out: dict, orig: ChainForecasting, history, chain_preds):
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


def audit_null_mask_semantics() -> tuple[bool, float, float]:
    """Distinguish raw-scale null mask from normalized-scale null mask."""
    # normalized target [-5, 0, 1] -> raw [0, 100, 120]
    # normalized pred   [-4, 2, 3] -> raw [20, 140, 160]
    # null_val=0 masks raw 0 only; valid MAE = (40+40)/2 = 40
    pred = torch.tensor([[[[-4.0, 2.0, 3.0]]]], dtype=torch.float32)
    target = torch.tensor([[[[-5.0, 0.0, 1.0]]]], dtype=torch.float32)
    loss_raw = forecast_state_token_mae(
        [pred],
        [target],
        null_val=0.0,
        rescale_pair=synthetic_rescale_pair,
    )
    loss_norm = forecast_state_token_mae(
        [pred],
        [target],
        null_val=0.0,
        rescale_pair=None,
    )
    raw_val = float(loss_raw)
    norm_val = float(loss_norm)
    ok = abs(raw_val - 40.0) < 1e-6 and abs(norm_val - 40.0) > 1e-3
    return ok, raw_val, norm_val


def audit_raw_scale_scalar_and_grad(
    preds: list[torch.Tensor],
    resolutions: list[int],
    full_target: torch.Tensor,
) -> dict:
    targets = [ChainForecasting.pool_target(full_target, k) for k in resolutions]
    preds_original = [p.detach().clone().requires_grad_(True) for p in preds]
    preds_budget = [p.detach().clone().requires_grad_(True) for p in preds]

    l_original = forecast_state_token_mae(
        preds_original,
        targets,
        null_val=0.0,
        rescale_pair=synthetic_rescale_pair,
    )
    l_budget = baseline_compatible_token_mae(
        preds_budget,
        resolutions,
        full_target,
        null_val=0.0,
        rescale_pair=synthetic_rescale_pair,
    )
    loss_diff = float((l_original.detach() - l_budget.detach()).abs())
    l_original.backward()
    l_budget.backward()

    stage_grads = []
    all_grad_ok = True
    for res, po, pb in zip(resolutions, preds_original, preds_budget):
        assert po.grad is not None and pb.grad is not None
        gdiff = (po.grad - pb.grad).abs()
        max_d = float(gdiff.max())
        mean_d = float(gdiff.mean())
        ok = max_d < 1e-7
        all_grad_ok = all_grad_ok and ok
        stage_grads.append(
            {
                "resolution": int(res),
                "max_abs_diff": max_d,
                "mean_abs_diff": mean_d,
                "ok": ok,
            }
        )
    return {
        "l_original": float(l_original.detach()),
        "l_budget": float(l_budget.detach()),
        "loss_diff": loss_diff,
        "loss_ok": loss_diff < 1e-7,
        "grad_ok": all_grad_ok,
        "stage_grads": stage_grads,
    }


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
        (
            "loss",
            "forecast_state_token_mae(+rescale_pair)",
            "runner _token_mae_for_resolutions / baseline_compatible(+rescale_pair)",
        ),
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
    print("5) Raw-scale scalar + gradient loss equivalence")
    print("=" * 72)
    preds = [p.detach() for p in traced["chain_preds"]]
    resolutions = list(traced["chain_resolutions"])
    loss_audit = audit_raw_scale_scalar_and_grad(preds, resolutions, y)
    print(f"  original_token_mae(raw)={loss_audit['l_original']:.8f}")
    print(f"  baseline_compatible(raw)={loss_audit['l_budget']:.8f}")
    print(f"  loss abs_diff={loss_audit['loss_diff']:.3e}")
    report["checks"]["raw_scale_loss_equiv"] = loss_audit["loss_ok"]
    report["raw_scale_loss_diff"] = loss_audit["loss_diff"]
    if not loss_audit["loss_ok"]:
        report["divergences"].append(f"raw-scale loss abs_diff={loss_audit['loss_diff']}")
    for sg in loss_audit["stage_grads"]:
        status = "PASS" if sg["ok"] else "FAIL"
        print(
            f"  [{status}] stage res={sg['resolution']} "
            f"grad max_abs_diff={sg['max_abs_diff']:.3e} "
            f"mean_abs_diff={sg['mean_abs_diff']:.3e}"
        )
    report["checks"]["raw_scale_grad_equiv"] = loss_audit["grad_ok"]
    report["stage_grads"] = loss_audit["stage_grads"]
    if not loss_audit["grad_ok"]:
        report["divergences"].append("raw-scale gradient mismatch")

    print("=" * 72)
    print("5b) Null-mask semantics (raw vs normalized)")
    print("=" * 72)
    mask_ok, raw_loss, norm_loss = audit_null_mask_semantics()
    print(f"  raw_scale_loss={raw_loss:.6f} (expect 40.0)")
    print(f"  normalized_scale_loss={norm_loss:.6f} (must differ from 40)")
    report["checks"]["null_mask_semantics"] = mask_ok
    report["null_mask_raw_loss"] = raw_loss
    report["null_mask_normalized_loss"] = norm_loss
    if not mask_ok:
        report["divergences"].append(
            f"null-mask semantics failed: raw={raw_loss} norm={norm_loss}"
        )
    else:
        print("  PASS: raw-scale mask yields 40.0; normalized-scale path differs")

    print("=" * 72)
    print("6) Weight-mapping + forward equivalence (forced [3,6,12])")
    print("=" * 72)
    torch.manual_seed(123)
    orig = ChainForecasting(**shared_kwargs(n, [3, 6, 12]))
    torch.manual_seed(123)
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
    report["checks"]["state_dict_mapping"] = not missing and not shape_mismatch
    if missing or shape_mismatch:
        report["divergences"].append("state_dict mapping incomplete/mismatched")
        print("  STOP: cannot claim weight-level equivalence")
        report["checks"]["forward_equiv_after_copy"] = False
        report["checks"]["weight_copy_raw_loss"] = False
    else:
        budget.load_state_dict(loadable, strict=False)
        orig.eval()
        budget.eval()
        with torch.no_grad():
            o = orig(history_data=x, train=False, return_all=True)
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
        forward_max = 0.0
        for name, a, b_ in checks:
            info = _diff(a, b_)
            ok = _print_diff(name, info)
            report["checks"][name] = info
            forward_max = max(forward_max, info["max_abs_diff"])
            if not ok and first_fail is None:
                first_fail = name
            all_ok = all_ok and ok
        report["forward_max_abs_diff"] = forward_max

        with torch.no_grad():
            l1 = forecast_state_token_mae(
                o["chain_preds"],
                [ChainForecasting.pool_target(y, k) for k in [3, 6, 12]],
                null_val=0.0,
                rescale_pair=synthetic_rescale_pair,
            )
            l2 = baseline_compatible_token_mae(
                bt["chain_preds"],
                [3, 6, 12],
                y,
                null_val=0.0,
                rescale_pair=synthetic_rescale_pair,
            )
        loss_abs = float((l1 - l2).abs())
        info = {
            "shape_a": [],
            "shape_b": [],
            "shape_match": True,
            "max_abs_diff": loss_abs,
            "mean_abs_diff": loss_abs,
        }
        ok_loss = _print_diff("raw_scale_token_loss_after_copy", info, tol=1e-7)
        all_ok = all_ok and ok_loss
        report["checks"]["weight_copy_raw_loss"] = ok_loss
        report["weight_copy_raw_loss_diff"] = loss_abs
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
    print("8) Config field notes")
    print("=" * 72)
    print(
        "  Known field diffs: MODEL.NAME/ARCH; chain_loss_mode "
        "token_normalized vs baseline_compatible (both must pass rescale_pair); "
        "budget adds forced_route/candidate_routes/training_phase/planner."
    )
    print(
        "  Old loss audits that compared helpers WITHOUT rescale_pair were "
        "normalized-scale false positives and are no longer accepted."
    )

    required = [
        "forced_overrides_sandwich",
        "forced_3612_execution",
        "stage3_no_zero_condition",
        "raw_scale_loss_equiv",
        "raw_scale_grad_equiv",
        "null_mask_semantics",
        "state_dict_mapping",
        "forward_equiv_after_copy",
        "weight_copy_raw_loss",
    ]
    all_pass = all(bool(report["checks"].get(k)) for k in required)

    out_path = ROOT / "results/budget_f2f_architecture_equivalence_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=" * 72)
    print(f"Wrote {out_path}")
    print("Divergences:")
    for d in report["divergences"] or ["(none)"]:
        print(f"  - {d}")
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
