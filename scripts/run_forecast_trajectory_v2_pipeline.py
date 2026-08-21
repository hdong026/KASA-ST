#!/usr/bin/env python3
"""ForecastTrajectory V2 pipeline: acceptance-1epoch and resumable --full."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ForecastTrajectoryV2_arch.online_policy_v2 import OnlineTrajectoryPolicyV2
from basicts.archs.arch_zoo.ForecastTrajectoryV2_arch.trajectory_cache_v2 import TrajectoryCacheV2
from basicts.data.indexed_timeseries_dataset import IndexedTimeSeriesForecastingDataset
from scripts.forecast_trajectory_v2_runtime import (
    ACCEPTANCE_TRAJS,
    CANONICAL_CKPT,
    CANONICAL_TAU,
    CONTAINMENT_TOL,
    HEADROOM_MIN,
    PRIMARY_PANEL,
    build_prefix_dag_cache,
    build_v2_model,
    chronological_policy_split,
    ckpt_dir,
    config_hash,
    dump_json,
    evaluate_canonical_f2f,
    evaluate_trajectories,
    git_head,
    load_canonical_f2f,
    load_scaler,
    make_loader,
    marker_ok,
    oracle_analysis,
    print_terminal_summary,
    profile_latency,
    run_online_policy,
    run_preflight,
    run_dir,
    seed_all,
    train_policy,
    train_transition,
    write_marker,
)

INDEX_FILE = ROOT / "datasets" / "PEMS04" / "index_in12_out12.pkl"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--acceptance-1epoch", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def device_of(gpu: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def run_acceptance(args) -> int:
    seed_all(args.seed)
    device = device_of(args.gpu)
    tag = "acceptance"
    rdir = run_dir(args.seed, tag)
    rdir.mkdir(parents=True, exist_ok=True)
    cdir = ckpt_dir(args.seed, tag)
    cdir.mkdir(parents=True, exist_ok=True)
    cfg_hash = config_hash({"mode": "acceptance-1epoch-v2", "seed": args.seed, "states": [2, 3, 4, 6, 12]})
    failures = []
    report = {
        "git_head": git_head(),
        "config_hash": cfg_hash,
        "device": str(device),
        "states": [2, 3, 4, 6, 12],
        "eta_used": False,
        "proxy_costs_used": False,
        "ENGINEERING_ONLY_TEST": True,
        "v1_not_resumed": True,
        "canonical_ckpt": str(CANONICAL_CKPT),
    }
    pre = run_preflight(device)
    report["preflight"] = {k: v for k, v in pre.items() if k not in {"model_obj", "f2f_obj", "model"}}
    report["preflight"]["model"] = pre["model"]
    report["preflight_pass"] = pre["pass"]
    if not pre["pass"]:
        failures.append("preflight")

    _, rescale = load_scaler()
    f2f_loader, _ = make_loader("valid", list(range(256)), 8, False)
    f2f_val = evaluate_canonical_f2f(pre["f2f_obj"], f2f_loader, device, rescale)
    report["canonical_f2f_valid"] = f2f_val
    print(f"[acceptance] canonical F2F VALID MAE={f2f_val['MAE']:.4f} (engineering subset)", flush=True)

    train_idx = list(range(1024))
    valid_idx = list(range(256))
    test_idx = list(range(256))
    trans_ckpt = cdir / "transition_best.pt"
    tr = train_transition(
        device=device,
        seed=args.seed,
        epochs=1,
        batch_size=8,
        train_indices=train_idx,
        valid_indices=valid_idx,
        test_indices=test_idx,
        out_ckpt=trans_ckpt,
        history_json=rdir / "transition_history.json",
        phase="curriculum",
        canonical_baseline_mae=None,
        acceptance=True,
        eval_test=True,
    )
    model = tr["model"]
    ts = tr["summary"]
    report["transition"] = {k: v for k, v in ts.items() if k != "warm_start"}
    report["warm_start"] = ts.get("warm_start")
    report["param_breakdown"] = ts["param_breakdown"]
    report["nan_inf"] = ts["nan_inf"]
    if ts["nan_inf"]:
        failures.append("nan_inf")
    if not ts["optimizer_updated"]:
        failures.append("optimizer_not_updated")

    from scripts.forecast_trajectory_v2_runtime import run_v2_unit_tests, policy_dead_param_test

    audit = run_v2_unit_tests(model, device)
    report["post_train_audit"] = audit
    if not audit["pass"]:
        failures.append("post_train_model_audit")

    policy = OnlineTrajectoryPolicyV2(model.graph, d_model=model.d_model).to(device)
    dead = policy_dead_param_test(policy, model, device)
    report["policy_dead_param_test"] = dead
    if not dead["pass"]:
        failures.append("dead_policy_params")
    if dead["min_regret"] < -1e-6:
        failures.append("negative_regret")
        print("POLICY_OBJECTIVE_INVALID", flush=True)

    loader1, _ = make_loader("train", list(range(4)), 1, False)
    x1 = None
    for future, history, _ in loader1:
        x1 = history
        break
    from scripts.forecast_trajectory_v2_runtime import select_history

    lat = profile_latency(model, policy, select_history(x1.to(device)), device, warmup=5, iters=8)
    dump_json(rdir / "latency_table.json", lat)
    report["latency"] = {
        "history_median_ms": lat["lookup"]["history_median_ms"],
        "policy_step_median_ms": lat["lookup"]["policy_step_median_ms"],
        "n_edges": len(lat["edges"]),
        "total_ms_min": lat["lookup"]["total_ms_min"],
        "total_ms_max": lat["lookup"]["total_ms_max"],
        "source": lat["source"],
        "eta_used": False,
        "proxy_cost": False,
    }

    cache_dir = rdir / "tiny_train_cache"
    tloader, _ = make_loader("train", train_idx[:8], 2, False)
    man = build_prefix_dag_cache(model, tloader, device, rescale, lat, cache_dir)
    report["cache"] = man
    expected = int(man.get("expected_transitions", -1))
    if abs(float(man.get("mean_transitions", -1)) - expected) > 1e-6 or man.get("n_mismatch", 1) > 0:
        failures.append("prefix_dag_not_once")

    cache = TrajectoryCacheV2(cache_dir)
    oracle = oracle_analysis(cache, model.graph)
    report["oracle_tiny"] = oracle
    split = {"policy_train": cache.sample_indices(), "policy_valid": cache.sample_indices()}
    pol_ckpt = cdir / "policy_best.pt"
    pr = train_policy(
        model=model,
        cache=cache,
        split=split,
        device=device,
        latency=lat,
        oracle=oracle,
        out_ckpt=pol_ckpt,
        history_json=rdir / "policy_history.json",
        acceptance=True,
        batch_size=4,
    )
    policy = pr["policy"]
    ps = pr["summary"]
    report["policy"] = ps
    if ps.get("invalid") or ps["path_prob_max_error"] > 1e-6 or ps["min_regret"] < -1e-6:
        failures.append("policy_objective")
    if not ps.get("vectorized"):
        failures.append("policy_not_vectorized")

    vloader, _ = make_loader("valid", valid_idx, 8, False)
    tloader2, _ = make_loader("test", test_idx, 8, False)
    report["policy_valid"] = run_online_policy(model, policy, vloader, device, rescale, lat, 0.0, None)
    report["policy_test"] = run_online_policy(model, policy, tloader2, device, rescale, lat, 0.0, None)
    if report["policy_valid"]["any_nonfinite"] or report["policy_test"]["any_nonfinite"]:
        failures.append("policy_infer_nan")

    gap = float(ts["valid"]["canonical_MAE"]) - float(f2f_val["MAE"])
    verdict = "FORECAST_TRAJECTORY_V2_ACCEPTANCE_PASS" if not failures else "FORECAST_TRAJECTORY_V2_ACCEPTANCE_FAIL"
    report["failures"] = failures
    report["verdict"] = verdict
    report["containment_note"] = (
        "Acceptance is engineering-only; BACKBONE_CONTAINMENT_GATE is enforced in --full Phase 1."
    )
    dump_json(ROOT / "results" / "forecast_trajectory_v2_acceptance.json", report)
    dump_json(rdir / "acceptance.json", report)
    write_marker(rdir, "acceptance", {"config_hash": cfg_hash, "verdict": verdict})
    print_terminal_summary(
        {
            "canonical_f2f_valid_mae": f2f_val["MAE"],
            "v2_canonical_valid_mae": ts["valid"].get("canonical_MAE"),
            "containment_gap": gap,
            "containment_verdict": "DEFERRED_TO_FULL (acceptance 1 epoch)",
            "v2_params": ts["param_breakdown"]["total"],
            "shared_transition_params": ts["param_breakdown"].get("shared_transition"),
            "state_adapter_params": ts["param_breakdown"]["state_adapters"],
            "n_edges": 15,
            "n_traj": 16,
            "edge_exposure_ratio": ts["edge_exposure"]["max_min_ratio"],
            "valid_best_fixed_mae": oracle.get("best_fixed_MAE"),
            "valid_oracle_mae": oracle.get("sample_wise_oracle_MAE"),
            "valid_adaptive_headroom": oracle.get("Delta_adaptive"),
            "history_latency": lat["lookup"]["history_median_ms"],
            "policy_latency": lat["lookup"]["policy_step_median_ms"],
            "total_latency_range": [lat["lookup"]["total_ms_min"], lat["lookup"]["total_ms_max"]],
            "policy_epoch_wall": ps.get("policy_epoch_wall_s"),
            "path_prob_max_error": ps.get("path_prob_max_error"),
            "min_regret": ps.get("min_regret"),
            "joint_retained": False,
            "final_valid": ts["valid"].get("canonical_MAE"),
            "final_test": (ts["test"] or {}).get("canonical_MAE") if ts.get("test") else None,
        }
    )
    print(verdict, flush=True)
    return 0 if not failures else 1


def run_full(args) -> int:
    seed_all(args.seed)
    device = device_of(args.gpu)
    tag = "formal"
    rdir = run_dir(args.seed, tag)
    rdir.mkdir(parents=True, exist_ok=True)
    cdir = ckpt_dir(args.seed, tag)
    cdir.mkdir(parents=True, exist_ok=True)
    cfg_hash = config_hash({"mode": "full-v2", "seed": args.seed, "model": "ForecastTrajectoryV2"})
    acc_json = ROOT / "results" / "forecast_trajectory_v2_acceptance.json"
    if not acc_json.is_file() or json.loads(acc_json.read_text()).get("verdict") != "FORECAST_TRAJECTORY_V2_ACCEPTANCE_PASS":
        print("[full] running acceptance-1epoch first", flush=True)
        rc = run_acceptance(args)
        if rc != 0:
            print("FULL_ABORTED_ACCEPTANCE_FAIL", flush=True)
            return rc

    _, rescale = load_scaler()

    # PHASE 0 preflight
    if not marker_ok(rdir, "phase0_preflight", cfg_hash):
        pre = run_preflight(device)
        dump_json(rdir / "preflight.json", {k: v for k, v in pre.items() if k not in {"model_obj", "f2f_obj", "model"}})
        if not pre["pass"]:
            print("PREFLIGHT_FAIL", flush=True)
            return 1
        write_marker(rdir, "phase0_preflight", {"config_hash": cfg_hash})
        f2f_model = pre["f2f_obj"]
        n_f2f = pre["f2f_params"]
    else:
        f2f_model, n_f2f, _ = load_canonical_f2f(device)

    # PHASE 1 canonical baseline
    base_path = rdir / "canonical_f2f_valid.json"
    if not marker_ok(rdir, "phase1_canonical_baseline", cfg_hash) or not base_path.is_file():
        vloader, _ = make_loader("valid", None, 32, False)
        f2f_val = evaluate_canonical_f2f(f2f_model, vloader, device, rescale)
        dump_json(base_path, f2f_val)
        write_marker(rdir, "phase1_canonical_baseline", {"config_hash": cfg_hash, "MAE": f2f_val["MAE"]})
    else:
        f2f_val = json.loads(base_path.read_text())
    baseline_mae = float(f2f_val["MAE"])
    print(f"canonical original F2F VALID MAE: {baseline_mae}", flush=True)

    trans_ckpt = cdir / "transition_containment.pt"
    # PHASE 2 containment
    if not marker_ok(rdir, "phase2_containment", cfg_hash) or not trans_ckpt.is_file():
        tr = train_transition(
            device=device, seed=args.seed, epochs=100, batch_size=32,
            train_indices=None, valid_indices=None, test_indices=None,
            out_ckpt=trans_ckpt, history_json=rdir / "containment_history.json",
            phase="containment", canonical_baseline_mae=baseline_mae,
            acceptance=False, eval_test=False, max_epochs=100,
        )
        write_marker(rdir, "phase2_containment", {"config_hash": cfg_hash, "summary": tr["summary"]})
        model = tr["model"]
        cont = tr["summary"]
    else:
        model, _ = build_v2_model(device, warm_start=False)
        ck = torch.load(trans_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        cont = json.loads((rdir / "containment_history.json").read_text()).get("summary", {})

    v2_can = float(cont.get("canonical_MAE") or 1e9)
    gap = v2_can - baseline_mae
    print(f"V2 canonical [3,6,12] VALID MAE: {v2_can}", flush=True)
    print(f"containment gap: {gap}", flush=True)
    if gap > CONTAINMENT_TOL:
        print("BACKBONE_CONTAINMENT_FAIL", flush=True)
        dump_json(rdir / "containment_gate.json", {"pass": False, "gap": gap, "baseline": baseline_mae, "v2": v2_can})
        print_terminal_summary({
            "canonical_f2f_valid_mae": baseline_mae,
            "v2_canonical_valid_mae": v2_can,
            "containment_gap": gap,
            "containment_verdict": "BACKBONE_CONTAINMENT_FAIL",
            "v2_params": cont.get("param_breakdown", {}).get("total"),
            "shared_transition_params": cont.get("param_breakdown", {}).get("shared_transition"),
            "state_adapter_params": cont.get("param_breakdown", {}).get("state_adapters"),
            "n_edges": 15, "n_traj": 16,
            "edge_exposure_ratio": None,
            "joint_retained": False,
        })
        return 1
    print("BACKBONE_CONTAINMENT_PASS", flush=True)
    write_marker(rdir, "phase3_containment_gate", {"config_hash": cfg_hash, "pass": True, "gap": gap})

    # PHASE 4 curriculum
    curr_ckpt = cdir / "transition_best.pt"
    if not marker_ok(rdir, "phase4_curriculum", cfg_hash) or not curr_ckpt.is_file():
        # continue from containment weights
        torch.save({"epoch": 0, "state_dict": model.state_dict(), "canonical_MAE": v2_can, "phase": "seed"}, curr_ckpt)
        tr2 = train_transition(
            device=device, seed=args.seed, epochs=80, batch_size=32,
            train_indices=None, valid_indices=None, test_indices=None,
            out_ckpt=curr_ckpt, history_json=rdir / "curriculum_history.json",
            phase="curriculum", canonical_baseline_mae=baseline_mae,
            acceptance=False, eval_test=False, init_ckpt=trans_ckpt,
        )
        model = tr2["model"]
        curr = tr2["summary"]
        write_marker(rdir, "phase4_curriculum", {"config_hash": cfg_hash, "summary": curr})
    else:
        ck = torch.load(curr_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        curr = json.loads((rdir / "curriculum_history.json").read_text()).get("summary", {})

    vloader, _ = make_loader("valid", None, 32, False)
    val_panel = evaluate_trajectories(model, vloader, model.graph.terminal_trajectories(), device, rescale, stage_keys=["Z3", "Z6", "Z12"])
    dump_json(rdir / "trajectory_valid_panel.json", val_panel)
    if float(val_panel["canonical_MAE"]) > baseline_mae + CONTAINMENT_TOL:
        print("BACKBONE_CONTAINMENT_FAIL", flush=True)
        return 1
    write_marker(rdir, "phase5_traj_valid_gate", {"config_hash": cfg_hash, "canonical_MAE": val_panel["canonical_MAE"]})

    policy_scratch = OnlineTrajectoryPolicyV2(model.graph, d_model=model.d_model).to(device)
    loader1, _ = make_loader("train", list(range(4)), 1, False)
    x1 = None
    for _, history, _ in loader1:
        from scripts.forecast_trajectory_v2_runtime import select_history
        x1 = select_history(history.to(device))
        break
    lat_path = rdir / "latency_table.json"
    if not marker_ok(rdir, "phase6_latency", cfg_hash) or not lat_path.is_file():
        lat = profile_latency(model, policy_scratch, x1, device, warmup=30, iters=80)
        dump_json(lat_path, lat)
        write_marker(rdir, "phase6_latency", {"config_hash": cfg_hash})
    else:
        lat = json.loads(lat_path.read_text())

    train_cache_dir = rdir / "train_cache"
    if not marker_ok(rdir, "phase7_train_cache", cfg_hash):
        tloader, _ = make_loader("train", None, 4, False)
        tman = build_prefix_dag_cache(model, tloader, device, rescale, lat, train_cache_dir)
        dump_json(rdir / "train_cache_manifest.json", tman)
        write_marker(rdir, "phase7_train_cache", {"config_hash": cfg_hash, "n": tman["n_samples"]})

    valid_cache_dir = rdir / "valid_cache"
    if not marker_ok(rdir, "phase8_valid_cache", cfg_hash):
        vl, _ = make_loader("valid", None, 4, False)
        vman = build_prefix_dag_cache(model, vl, device, rescale, lat, valid_cache_dir)
        dump_json(rdir / "valid_cache_manifest.json", vman)
        write_marker(rdir, "phase8_valid_cache", {"config_hash": cfg_hash, "n": vman["n_samples"]})

    train_cache = TrajectoryCacheV2(train_cache_dir)
    valid_cache = TrajectoryCacheV2(valid_cache_dir)
    if not marker_ok(rdir, "phase9_oracle", cfg_hash):
        oracle_tr = oracle_analysis(train_cache, model.graph)
        oracle_va = oracle_analysis(valid_cache, model.graph)
        dump_json(rdir / "oracle_train.json", oracle_tr)
        dump_json(rdir / "oracle_valid.json", oracle_va)
        write_marker(rdir, "phase9_oracle", {"config_hash": cfg_hash})
    else:
        oracle_tr = json.loads((rdir / "oracle_train.json").read_text())
        oracle_va = json.loads((rdir / "oracle_valid.json").read_text())

    print(f"VALID best fixed MAE: {oracle_va['best_fixed_MAE']}", flush=True)
    print(f"VALID sample oracle MAE: {oracle_va['sample_wise_oracle_MAE']}", flush=True)
    print(f"VALID adaptive headroom: {oracle_va['Delta_adaptive']}", flush=True)
    if float(oracle_va["Delta_adaptive"]) < HEADROOM_MIN:
        print("ADAPTIVE_HEADROOM_TOO_SMALL", flush=True)
        dump_json(rdir / "headroom_gate.json", {"pass": False, **oracle_va})
        print_terminal_summary({
            "canonical_f2f_valid_mae": baseline_mae,
            "v2_canonical_valid_mae": val_panel["canonical_MAE"],
            "containment_gap": float(val_panel["canonical_MAE"]) - baseline_mae,
            "containment_verdict": "BACKBONE_CONTAINMENT_PASS",
            "v2_params": count_or(model),
            "shared_transition_params": model.param_breakdown().get("shared_transition"),
            "state_adapter_params": model.param_breakdown()["state_adapters"],
            "n_edges": 15, "n_traj": 16,
            "edge_exposure_ratio": curr.get("edge_exposure", {}).get("max_min_ratio"),
            "valid_best_fixed_mae": oracle_va["best_fixed_MAE"],
            "valid_oracle_mae": oracle_va["sample_wise_oracle_MAE"],
            "valid_adaptive_headroom": oracle_va["Delta_adaptive"],
            "history_latency": lat["lookup"]["history_median_ms"],
            "policy_latency": lat["lookup"]["policy_step_median_ms"],
            "total_latency_range": [lat["lookup"]["total_ms_min"], lat["lookup"]["total_ms_max"]],
            "joint_retained": False,
            "final_valid": val_panel,
        })
        write_marker(rdir, "phase10_headroom_stop", {"config_hash": cfg_hash, "stop": True})
        return 0

    write_marker(rdir, "phase10_headroom_pass", {"config_hash": cfg_hash})
    train_ds = IndexedTimeSeriesForecastingDataset(
        str(ROOT / "datasets/PEMS04/data_in12_out12.pkl"), str(INDEX_FILE), "train"
    )
    split = chronological_policy_split(train_cache.sample_indices(), train_ds.index, 0.8)
    dump_json(rdir / "policy_split.json", split)

    pol_ckpt = cdir / "policy_best.pt"
    if not marker_ok(rdir, "phase11_policy", cfg_hash) or not pol_ckpt.is_file():
        pr = train_policy(
            model=model, cache=train_cache, split=split, device=device, latency=lat,
            oracle=oracle_tr, out_ckpt=pol_ckpt, history_json=rdir / "policy_history.json",
            acceptance=False, batch_size=16,
        )
        policy = pr["policy"]
        ps = pr["summary"]
        write_marker(rdir, "phase11_policy", {"config_hash": cfg_hash})
    else:
        policy = OnlineTrajectoryPolicyV2(model.graph, d_model=model.d_model).to(device)
        ck = torch.load(pol_ckpt, map_location=device, weights_only=False)
        policy.load_state_dict(ck["state_dict"])
        ps = json.loads((rdir / "policy_history.json").read_text()).get("summary", {})

    extra = float(lat["lookup"].get("policy_step_median_ms") or 0.0)
    v_pol = run_online_policy(model, policy, vloader, device, rescale, lat, 0.0, None)
    dump_json(rdir / "policy_valid.json", v_pol)
    write_marker(rdir, "phase12_policy_valid", {"config_hash": cfg_hash})

    # optional joint: skipped by default unless both models exist; one short epoch then gate
    joint_retained = False
    write_marker(rdir, "phase13_joint_skipped", {"config_hash": cfg_hash, "retained": False})

    tloader, _ = make_loader("test", None, 16, False)
    t_pol = run_online_policy(model, policy, tloader, device, rescale, lat, 0.0, None)
    dump_json(rdir / "policy_test.json", t_pol)
    frozen_valid = evaluate_trajectories(model, vloader, PRIMARY_PANEL, device, rescale)
    frozen_test = evaluate_trajectories(model, tloader, PRIMARY_PANEL, device, rescale)
    dump_json(rdir / "frozen_valid.json", frozen_valid)
    dump_json(rdir / "frozen_test.json", frozen_test)

    final = {
        "git_head": git_head(),
        "eta_used": False,
        "proxy_costs_used": False,
        "canonical_f2f_valid": f2f_val,
        "containment": cont,
        "curriculum": curr,
        "oracle_valid": oracle_va,
        "policy": ps,
        "policy_valid": v_pol,
        "policy_test": t_pol,
        "frozen_valid": frozen_valid,
        "frozen_test": frozen_test,
        "joint_retained": joint_retained,
        "latency": lat["lookup"],
    }
    dump_json(rdir / "final_report.json", final)
    dump_json(ROOT / "results" / "forecast_trajectory_v2_final_report.json", final)
    print_terminal_summary({
        "canonical_f2f_valid_mae": baseline_mae,
        "v2_canonical_valid_mae": frozen_valid.get("canonical_MAE"),
        "containment_gap": float(frozen_valid.get("canonical_MAE", 0)) - baseline_mae,
        "containment_verdict": "BACKBONE_CONTAINMENT_PASS",
        "v2_params": model.param_breakdown()["total"],
        "shared_transition_params": model.param_breakdown().get("shared_transition"),
        "state_adapter_params": model.param_breakdown()["state_adapters"],
        "n_edges": 15, "n_traj": 16,
        "edge_exposure_ratio": curr.get("edge_exposure", {}).get("max_min_ratio"),
        "valid_best_fixed_mae": oracle_va["best_fixed_MAE"],
        "valid_oracle_mae": oracle_va["sample_wise_oracle_MAE"],
        "valid_adaptive_headroom": oracle_va["Delta_adaptive"],
        "history_latency": lat["lookup"]["history_median_ms"],
        "policy_latency": lat["lookup"]["policy_step_median_ms"],
        "total_latency_range": [lat["lookup"]["total_ms_min"], lat["lookup"]["total_ms_max"]],
        "policy_epoch_wall": ps.get("policy_epoch_wall_s"),
        "path_prob_max_error": ps.get("path_prob_max_error"),
        "min_regret": ps.get("min_regret"),
        "joint_retained": joint_retained,
        "final_valid": frozen_valid,
        "final_test": frozen_test,
    })
    print("FORECAST_TRAJECTORY_V2_FULL_COMPLETE", flush=True)
    return 0


def count_or(model):
    return sum(p.numel() for p in model.parameters())


def main():
    args = parse_args()
    if args.acceptance_1epoch and args.full:
        print("Specify only one of --acceptance-1epoch or --full", flush=True)
        return 2
    if args.acceptance_1epoch:
        return run_acceptance(args)
    if args.full:
        return run_full(args)
    print("Specify --acceptance-1epoch or --full", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
