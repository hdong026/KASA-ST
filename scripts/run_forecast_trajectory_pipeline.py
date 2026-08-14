#!/usr/bin/env python3
"""ForecastTrajectory pipeline orchestrator (acceptance + full)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.forecast_trajectory_runtime import (
    ACCEPTANCE_TRAJECTORIES,
    VALIDATION_PANEL,
    all_traj_costs,
    build_latency_table,
    build_model,
    build_trajectory_cache,
    chronological_policy_split,
    ckpt_dir,
    config_hash,
    count_params,
    default_model_args,
    dump_json,
    evaluate_trajectories,
    git_head,
    load_pkl,
    load_scaler,
    make_loader,
    marker_ok,
    oracle_analysis,
    run_dir,
    run_online_policy,
    run_preflight_unit_tests,
    seed_all,
    train_policy,
    train_transition,
    write_marker,
)
from basicts.archs.arch_zoo.ForecastTrajectory_arch.online_trajectory_policy import (
    OnlineTrajectoryPolicy,
)
from basicts.archs.arch_zoo.ForecastTrajectory_arch.trajectory_cache import TrajectoryCache
from basicts.utils import load_pkl as _load_pkl
from basicts.data.indexed_timeseries_dataset import IndexedTimeSeriesForecastingDataset


INDEX_FILE = ROOT / "datasets" / "PEMS04" / "index_in12_out12.pkl"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--acceptance-1epoch", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--phase", default=None, help="optional single phase override")
    return p.parse_args()


def device_of(gpu: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def scientific_answers(bundle: dict) -> dict:
    return {
        "1_separate_F2_F3_F4_F6_F12": False,
        "2_one_shared_transition_core": True,
        "3_states": [2, 3, 4, 6, 12],
        "4_n_legal_edges": 15,
        "5_n_terminal_trajectories": 16,
        "6_history_encoder_executions_per_rollout": 1,
        "7_edge_3_to_12_explicitly_trained": True,
        "8_edge_4_to_12_exists": True,
        "9_edge_2_to_6_exists": True,
        "10_eta_in_method": False,
        "11_old_proxy_cost": False,
        "12_latency_source": "actual CUDA measurement (cuda events; cpu_perf_counter fallback)",
        "13_forecast_transition_loss": "token-normalized forecast MAE",
        "14_policy_objective": "exact expected forecast loss + lambda * normalized time",
        "15_PPO_GRPO_DQN": False,
        "16_crossfit_teachers": False,
        "17_adaptive_oracle_gain_no_budget": bundle.get("oracle", {}).get("Delta_adaptive"),
        "18_adaptive_oracle_gain_every_B": bundle.get("oracle", {}).get("budget_curve"),
        "19_best_fixed_trajectory": bundle.get("oracle", {}).get("best_fixed_trajectory"),
        "20_sample_optimal_trajectory_distribution": bundle.get("oracle", {}).get(
            "sample_optimal_trajectory_histogram"
        ),
        "21_transition_best_epoch": bundle.get("transition", {}).get("best_epoch"),
        "22_transition_convergence_verdict": bundle.get("transition", {}).get("verdict"),
        "23_policy_best_epoch": bundle.get("policy", {}).get("best_epoch"),
        "24_policy_validation_oracle_regret": bundle.get("policy", {}).get("best_valid_regret"),
        "25_final_VALID_metrics": bundle.get("valid_eval"),
        "26_final_TEST_metrics": bundle.get("test_eval"),
    }


def run_acceptance(args) -> int:
    seed_all(args.seed)
    device = device_of(args.gpu)
    tag = "acceptance"
    rdir = run_dir(args.seed, tag)
    rdir.mkdir(parents=True, exist_ok=True)
    cdir = ckpt_dir(args.seed, tag)
    cdir.mkdir(parents=True, exist_ok=True)
    cfg_hash = config_hash(
        {
            "mode": "acceptance-1epoch",
            "seed": args.seed,
            "states": [2, 3, 4, 6, 12],
            "model": default_model_args(),
        }
    )
    failures = []
    report = {
        "git_head": git_head(),
        "config_hash": cfg_hash,
        "device": str(device),
        "states": [2, 3, 4, 6, 12],
        "eta_used": False,
        "proxy_costs_used": False,
        "crossfit_used": False,
        "ENGINEERING_ONLY_TEST": True,
    }

    pre = run_preflight_unit_tests(device)
    report["preflight"] = {k: v for k, v in pre.items() if k != "model"}
    report["preflight_pass"] = pre["pass"]
    if not pre["pass"]:
        failures.append("preflight")

    train_idx = list(range(1024))
    valid_idx = list(range(256))
    test_idx = list(range(256))
    report["acceptance_train_n"] = 1024
    report["acceptance_valid_n"] = 256
    report["acceptance_test_n"] = 256

    trans_ckpt = cdir / "transition_best.pt"
    trans_hist = ROOT / "results" / "forecast_trajectory_transition_history.json"
    tr = train_transition(
        device=device,
        seed=args.seed,
        epochs=1,
        batch_size=16,
        train_indices=train_idx,
        valid_indices=valid_idx,
        test_indices=test_idx,
        out_ckpt=trans_ckpt,
        history_json=trans_hist,
        acceptance=True,
        auto_extend=False,
        eval_test=True,
    )
    model = tr["model"]
    ts = tr["summary"]
    report["model_parameter_count"] = ts["param_count"]
    report["transition_parameter_count"] = ts["transition_param_count"]
    report["train_loss"] = ts["train_loss"]
    report["valid_metrics"] = ts["valid"]
    report["test_metrics"] = ts["test"]
    report["transition_grad_norm"] = ts["grad_norm"]
    report["optimizer_updated"] = ts["optimizer_updated"]
    report["nan_inf"] = ts["nan_inf"]
    report["edge_count"] = 15
    report["trajectory_count"] = 16
    if not ts["optimizer_updated"]:
        failures.append("optimizer_not_updated")
    if ts["nan_inf"] or ts["valid"]["any_nonfinite"] or (ts["test"] and ts["test"]["any_nonfinite"]):
        failures.append("nan_inf")

    # history encoder / shared params / edges already in preflight; re-audit on trained model
    from scripts.forecast_trajectory_runtime import run_model_unit_tests

    audit = run_model_unit_tests(model, device)
    report["history_encoder_call_audit"] = audit["checks"].get(
        "history_encoder_once_per_rollout"
    )
    report["edge_execution_audit"] = audit.get("edge_execution")
    report["shared_transition_ids"] = audit["checks"].get("shared_transition_param_ids")
    if not audit["pass"]:
        failures.append("post_train_model_audit")

    train_loader, _ = make_loader("train", train_idx[:8], 1, False)
    lat = build_latency_table(model, train_loader, device, warmup=5, iters=10)
    dump_json(ROOT / "results" / "forecast_trajectory_latency_table.json", lat)
    report["latency_table_summary"] = {
        "history_median_ms": lat["lookup"]["history_median_ms"],
        "n_edges_profiled": len(lat["edges"]),
        "source": lat["source"],
    }

    cache_dir = rdir / "tiny_train_cache"
    cache_loader, _ = make_loader("train", train_idx[:32], 8, False)
    _, rescale = load_scaler()
    man = build_trajectory_cache(
        model, cache_loader, device, rescale, lat, cache_dir, max_samples=32
    )
    dump_json(ROOT / "results" / "forecast_trajectory_train_cache_manifest.json", man)
    report["trajectory_cache_count"] = man["n_samples"]

    valid_cache_dir = rdir / "tiny_valid_cache"
    vloader, _ = make_loader("valid", valid_idx[:16], 8, False)
    vman = build_trajectory_cache(
        model, vloader, device, rescale, lat, valid_cache_dir, max_samples=16
    )
    dump_json(ROOT / "results" / "forecast_trajectory_valid_cache_manifest.json", vman)

    cache = TrajectoryCache(cache_dir)
    oracle = oracle_analysis(cache, model.graph)
    dump_json(ROOT / "results" / "forecast_trajectory_oracle_analysis.json", oracle)

    train_ds = IndexedTimeSeriesForecastingDataset(
        str(ROOT / "datasets/PEMS04/data_in12_out12.pkl"),
        str(INDEX_FILE),
        "train",
    )
    split = chronological_policy_split(cache.sample_indices(), train_ds.index, 0.8)
    pol_ckpt = cdir / "policy_best.pt"
    pol_hist = ROOT / "results" / "forecast_trajectory_policy_history.json"
    pr = train_policy(
        model=model,
        cache=cache,
        split=split,
        device=device,
        latency_table=lat,
        oracle=oracle,
        out_ckpt=pol_ckpt,
        history_json=pol_hist,
        acceptance=True,
        min_epochs=1,
        max_epochs=1,
        patience=1,
        batch_size=8,
    )
    policy = pr["policy"]
    ps = pr["summary"]
    report["policy_parameter_count"] = ps["param_count"]
    report["path_probability_sum_error"] = ps["path_sum_err_max"]
    report["policy_initial_loss"] = ps["init_loss"]
    report["policy_final_loss"] = ps["final_loss"]
    report["policy_valid_metrics"] = ps["valid"]
    if ps["path_probability_invalid"]:
        failures.append("PATH_PROBABILITY_INVALID")

    extra = float(lat["lookup"].get("policy_step_median_ms") or 0.0)
    v_pol = run_online_policy(
        model, policy, vloader, device, rescale, lat, lam=0.0, B_ms=None, extra_per_edge=extra
    )
    tloader, _ = make_loader("test", test_idx[:16], 8, False)
    t_pol = run_online_policy(
        model, policy, tloader, device, rescale, lat, lam=0.0, B_ms=None, extra_per_edge=extra
    )
    report["policy_valid_online"] = v_pol
    report["policy_test_metrics"] = t_pol
    if v_pol["any_nonfinite"] or t_pol["any_nonfinite"]:
        failures.append("policy_nonfinite")

    dump_json(ROOT / "results" / "forecast_trajectory_valid_eval.json", {"acceptance": v_pol, "forecast": ts["valid"]})
    dump_json(
        ROOT / "results" / "forecast_trajectory_test_eval.json",
        {"acceptance": t_pol, "forecast": ts["test"], "ENGINEERING_ONLY": True},
    )

    verdict = (
        "FORECAST_TRAJECTORY_ACCEPTANCE_PASS"
        if not failures
        else "FORECAST_TRAJECTORY_ACCEPTANCE_FAIL"
    )
    report["failures"] = failures
    report["verdict"] = verdict
    report["scientific_answers"] = scientific_answers(
        {"oracle": oracle, "transition": ts, "policy": ps, "valid_eval": v_pol, "test_eval": t_pol}
    )
    dump_json(ROOT / "results" / "forecast_trajectory_acceptance_1epoch.json", report)
    dump_json(ROOT / "results" / "forecast_trajectory_final_report.json", report)
    write_marker(rdir, "acceptance", {"config_hash": cfg_hash, "verdict": verdict})
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
    cfg_hash = config_hash(
        {
            "mode": "full",
            "seed": args.seed,
            "states": [2, 3, 4, 6, 12],
            "model": default_model_args(),
        }
    )
    acc_json = ROOT / "results" / "forecast_trajectory_acceptance_1epoch.json"
    # PHASE 0
    if not marker_ok(rdir, "phase0_preflight", cfg_hash):
        pre = run_preflight_unit_tests(device)
        dump_json(rdir / "preflight.json", pre)
        if not pre["pass"]:
            print("PREFLIGHT_FAIL", flush=True)
            return 1
        write_marker(rdir, "phase0_preflight", {"config_hash": cfg_hash, "pass": True})

    # PHASE 1 acceptance marker
    if not acc_json.is_file() or json.loads(acc_json.read_text()).get("verdict") != "FORECAST_TRAJECTORY_ACCEPTANCE_PASS":
        print("[full] acceptance marker missing; running acceptance-1epoch first", flush=True)
        rc = run_acceptance(args)
        if rc != 0:
            print("FULL_ABORTED_ACCEPTANCE_FAIL", flush=True)
            return rc

    _, rescale = load_scaler()
    trans_ckpt = cdir / "transition_best.pt"
    trans_hist = ROOT / "results" / "forecast_trajectory_transition_history.json"

    # PHASE 2-3 transition train + restore
    if not marker_ok(rdir, "phase2_transition", cfg_hash) or not trans_ckpt.is_file():
        tr = train_transition(
            device=device,
            seed=args.seed,
            epochs=100,
            batch_size=32,
            train_indices=None,
            valid_indices=None,
            test_indices=None,
            out_ckpt=trans_ckpt,
            history_json=trans_hist,
            acceptance=False,
            auto_extend=True,
            planned_epochs=100,
            max_epochs=250,
            eval_test=False,
        )
        write_marker(
            rdir,
            "phase2_transition",
            {"config_hash": cfg_hash, "ckpt": str(trans_ckpt), "summary": tr["summary"]},
        )
        model = tr["model"]
        trans_summary = tr["summary"]
    else:
        model = build_model(device)
        ckpt = torch.load(trans_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        trans_summary = json.loads(trans_hist.read_text()).get("summary", {})

    # PHASE 4 latency
    lat_path = ROOT / "results" / "forecast_trajectory_latency_table.json"
    if not marker_ok(rdir, "phase4_latency", cfg_hash) or not lat_path.is_file():
        loader, _ = make_loader("train", list(range(8)), 1, False)
        lat = build_latency_table(model, loader, device, warmup=50, iters=200)
        dump_json(lat_path, lat)
        write_marker(rdir, "phase4_latency", {"config_hash": cfg_hash})
    else:
        lat = json.loads(lat_path.read_text())

    # PHASE 5 train cache
    train_cache_dir = rdir / "train_cache"
    train_man_path = ROOT / "results" / "forecast_trajectory_train_cache_manifest.json"
    if not marker_ok(rdir, "phase5_train_cache", cfg_hash):
        tloader, _ = make_loader("train", None, 8, False)
        man = build_trajectory_cache(model, tloader, device, rescale, lat, train_cache_dir)
        dump_json(train_man_path, man)
        write_marker(rdir, "phase5_train_cache", {"config_hash": cfg_hash, "n": man["n_samples"]})
    else:
        man = json.loads(train_man_path.read_text()) if train_man_path.is_file() else {}

    # PHASE 6 valid cache
    valid_cache_dir = rdir / "valid_cache"
    valid_man_path = ROOT / "results" / "forecast_trajectory_valid_cache_manifest.json"
    if not marker_ok(rdir, "phase6_valid_cache", cfg_hash):
        vloader, _ = make_loader("valid", None, 8, False)
        vman = build_trajectory_cache(model, vloader, device, rescale, lat, valid_cache_dir)
        dump_json(valid_man_path, vman)
        write_marker(rdir, "phase6_valid_cache", {"config_hash": cfg_hash, "n": vman["n_samples"]})

    # PHASE 7 oracle (TRAIN cache only for tuning)
    oracle_path = ROOT / "results" / "forecast_trajectory_oracle_analysis.json"
    if not marker_ok(rdir, "phase7_oracle", cfg_hash) or not oracle_path.is_file():
        cache = TrajectoryCache(train_cache_dir)
        oracle = oracle_analysis(cache, model.graph)
        dump_json(oracle_path, oracle)
        write_marker(rdir, "phase7_oracle", {"config_hash": cfg_hash})
    else:
        oracle = json.loads(oracle_path.read_text())

    # PHASE 8 split
    cache = TrajectoryCache(train_cache_dir)
    train_ds = IndexedTimeSeriesForecastingDataset(
        str(ROOT / "datasets/PEMS04/data_in12_out12.pkl"), str(INDEX_FILE), "train"
    )
    split = chronological_policy_split(cache.sample_indices(), train_ds.index, 0.8)
    dump_json(rdir / "policy_split.json", split)
    write_marker(rdir, "phase8_split", {"config_hash": cfg_hash})

    # PHASE 9-10 policy
    pol_ckpt = cdir / "policy_best.pt"
    pol_hist = ROOT / "results" / "forecast_trajectory_policy_history.json"
    if not marker_ok(rdir, "phase9_policy", cfg_hash) or not pol_ckpt.is_file():
        pr = train_policy(
            model=model,
            cache=cache,
            split=split,
            device=device,
            latency_table=lat,
            oracle=oracle,
            out_ckpt=pol_ckpt,
            history_json=pol_hist,
            acceptance=False,
        )
        policy = pr["policy"]
        pol_summary = pr["summary"]
        write_marker(rdir, "phase9_policy", {"config_hash": cfg_hash})
    else:
        policy = OnlineTrajectoryPolicy(model.graph, d_history=model.d_model).to(device)
        ckpt = torch.load(pol_ckpt, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt["state_dict"])
        pol_summary = json.loads(pol_hist.read_text()).get("summary", {})

    extra = float(lat["lookup"].get("policy_step_median_ms") or 0.0)
    lambda_grid = pol_summary.get("lambda_grid") or [0.0]
    unique_B = pol_summary.get("unique_B") or []

    # PHASE 11 VALID eval
    vloader, _ = make_loader("valid", None, 8, False)
    valid_eval = {
        "quality_only_lambda0_noB": run_online_policy(
            model, policy, vloader, device, rescale, lat, 0.0, None, extra
        ),
        "lambda_curve": {},
        "hardB_curve": {},
        "fixed_baselines": {},
        "oracle_regret": pol_summary.get("valid"),
    }
    for lam in lambda_grid:
        valid_eval["lambda_curve"][str(lam)] = run_online_policy(
            model, policy, vloader, device, rescale, lat, float(lam), None, extra
        )
    for B in unique_B:
        valid_eval["hardB_curve"][str(B)] = run_online_policy(
            model, policy, vloader, device, rescale, lat, 0.0, float(B), extra
        )
    for tau in VALIDATION_PANEL:
        valid_eval["fixed_baselines"][model.graph.trajectory_key(tau)] = evaluate_trajectories(
            model, vloader, [tau], device, rescale
        )
    dump_json(ROOT / "results" / "forecast_trajectory_valid_eval.json", valid_eval)

    # PHASE 12 TEST eval (frozen; no TEST oracle)
    tloader, _ = make_loader("test", None, 8, False)
    test_eval = {
        "quality_only_lambda0_noB": run_online_policy(
            model, policy, tloader, device, rescale, lat, 0.0, None, extra
        ),
        "lambda_curve": {},
        "hardB_curve": {},
        "fixed_baselines": {},
        "TEST_oracle": None,
        "note": "TEST not used for checkpoint/lambda/B/policy selection",
    }
    for lam in lambda_grid:
        test_eval["lambda_curve"][str(lam)] = run_online_policy(
            model, policy, tloader, device, rescale, lat, float(lam), None, extra
        )
    for B in unique_B:
        test_eval["hardB_curve"][str(B)] = run_online_policy(
            model, policy, tloader, device, rescale, lat, 0.0, float(B), extra
        )
    for tau in VALIDATION_PANEL:
        test_eval["fixed_baselines"][model.graph.trajectory_key(tau)] = evaluate_trajectories(
            model, tloader, [tau], device, rescale
        )
    dump_json(ROOT / "results" / "forecast_trajectory_test_eval.json", test_eval)

    final = {
        "git_head": git_head(),
        "states": [2, 3, 4, 6, 12],
        "edge_count": 15,
        "trajectory_count": 16,
        "transition": trans_summary,
        "policy": pol_summary,
        "oracle": oracle,
        "valid_eval": valid_eval,
        "test_eval": test_eval,
        "eta_used": False,
        "proxy_costs_used": False,
        "crossfit_used": False,
        "PPO_GRPO_DQN": False,
        "scientific_answers": scientific_answers(
            {
                "oracle": oracle,
                "transition": trans_summary,
                "policy": pol_summary,
                "valid_eval": valid_eval,
                "test_eval": test_eval,
            }
        ),
    }
    dump_json(ROOT / "results" / "forecast_trajectory_final_report.json", final)
    write_marker(rdir, "phase13_report", {"config_hash": cfg_hash})
    print("FORECAST_TRAJECTORY_FULL_COMPLETE", flush=True)
    return 0


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
