#!/usr/bin/env python3
"""Verify template lookup vs channel-3 prior on PEMS04 train samples."""

import argparse
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare weekly template lookup with channel-3 prior."
    )
    parser.add_argument(
        "--data-npz",
        default=os.path.join(ROOT, "datasets", "PEMS04", "data_in12_out12.npz"),
        help="Processed npz with processed_data and weekly_spectral_template.",
    )
    parser.add_argument(
        "--index-pkl",
        default=os.path.join(ROOT, "datasets", "PEMS04", "index_in12_out12.pkl"),
        help="Index pkl with train/valid/test split tuples (start, mid, end).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=128,
        help="Number of train samples to evaluate (default: first 128).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size used to compute the 3-batch sample count hint.",
    )
    return parser.parse_args()


def load_data(data_npz, index_pkl):
    if not os.path.exists(data_npz):
        raise FileNotFoundError(f"Missing npz: {data_npz}")
    if not os.path.exists(index_pkl):
        raise FileNotFoundError(f"Missing index pkl: {index_pkl}")

    archive = np.load(data_npz)
    required = ("processed_data", "weekly_spectral_template", "slots_per_day", "slots_per_week")
    missing = [key for key in required if key not in archive]
    if missing:
        raise KeyError(f"npz missing keys: {missing}")

    processed_data = archive["processed_data"]
    template = archive["weekly_spectral_template"]
    slots_per_day = int(archive["slots_per_day"])
    slots_per_week = int(archive["slots_per_week"])

    with open(index_pkl, "rb") as f:
        index = pickle.load(f)
    train_index = index["train"]

    return processed_data, template, slots_per_day, slots_per_week, train_index


def lookup(tod, dow, template, slots_per_day):
    """Lookup prior from ToD/DoW using train-only weekly spectral template."""
    tod_idx = np.floor(tod * slots_per_day + 1e-6).astype(np.int64)
    dow_idx = dow.astype(np.int64)
    tod_idx = np.clip(tod_idx, 0, slots_per_day - 1)
    dow_idx = np.clip(dow_idx, 0, 6)
    week_idx = dow_idx * slots_per_day + tod_idx

    node_idx = np.arange(template.shape[1])
    prior = template[week_idx, node_idx, 0]
    return prior, week_idx


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def stats(name, values):
    print(
        f"{name} mean/std/min/max: "
        f"{values.mean():.8f} / {values.std():.8f} / "
        f"{values.min():.8f} / {values.max():.8f}"
    )


def main():
    args = parse_args()
    processed_data, template, slots_per_day, slots_per_week, train_index = load_data(
        args.data_npz, args.index_pkl
    )

    num_samples = min(args.max_samples, len(train_index))
    three_batches = min(3 * args.batch_size, len(train_index))
    if num_samples < three_batches:
        print(
            f"Using first {num_samples} train samples "
            f"(max_samples={args.max_samples}, train_size={len(train_index)})."
        )
    else:
        print(
            f"Using first {num_samples} train samples "
            f"(>= 3 batches of {args.batch_size})."
        )

    print(f"processed_data shape: {processed_data.shape}")
    print(f"weekly_spectral_template shape: {template.shape}")
    print(f"slots_per_day={slots_per_day}, slots_per_week={slots_per_week}")

    hist_lookup_err = []
    fut_lookup_err = []
    hist_fut_err = []
    hist_fut_lookup_err = []
    hist_prior_all = []
    fut_prior_all = []

    first_hist_week_idx = None
    first_fut_week_idx = None

    for i, (start, mid, end) in enumerate(train_index[:num_samples]):
        history = processed_data[start:mid]
        future = processed_data[mid:end]

        hist_tod = history[..., 1]
        hist_dow = history[..., 2]
        hist_prior = history[..., 3]

        fut_tod = future[..., 1]
        fut_dow = future[..., 2]
        fut_prior = future[..., 3]

        hist_lookup, hist_week_idx = lookup(hist_tod, hist_dow, template, slots_per_day)
        fut_lookup, fut_week_idx = lookup(fut_tod, fut_dow, template, slots_per_day)

        hist_lookup_err.append(mae(hist_lookup, hist_prior))
        fut_lookup_err.append(mae(fut_lookup, fut_prior))
        hist_fut_err.append(mae(hist_prior, fut_prior))
        hist_fut_lookup_err.append(mae(hist_lookup, fut_lookup))

        hist_prior_all.append(hist_prior)
        fut_prior_all.append(fut_prior)

        if i == 0:
            first_hist_week_idx = hist_week_idx[:, 0]
            first_fut_week_idx = fut_week_idx[:, 0]

    hist_prior_all = np.concatenate(hist_prior_all, axis=0)
    fut_prior_all = np.concatenate(fut_prior_all, axis=0)

    print("\n=== prior equivalence report ===")
    print(f"1. MAE(lookup(history ToD/DoW), history channel3): {np.mean(hist_lookup_err):.8f}")
    print(f"2. MAE(lookup(future ToD/DoW), future channel3):   {np.mean(fut_lookup_err):.8f}")
    print(f"3. MAE(history channel3, future channel3):        {np.mean(hist_fut_err):.8f}")
    print(f"4. MAE(lookup(history ToD/DoW), lookup(future ToD/DoW)): {np.mean(hist_fut_lookup_err):.8f}")
    stats("5. history channel3", hist_prior_all)
    stats("6. future channel3", fut_prior_all)

    print(f"7. first sample history week_idx first 12: {first_hist_week_idx[:12].tolist()}")
    print(f"8. first sample future week_idx first 12:  {first_fut_week_idx[:12].tolist()}")

    print("\n=== interpretation ===")
    near_zero = 1e-5
    ok_hist = np.mean(hist_lookup_err) < near_zero
    ok_fut = np.mean(fut_lookup_err) < near_zero
    if ok_hist and ok_fut:
        print("- Lookup matches channel-3 prior on both history and future windows.")
        if np.mean(hist_fut_err) > near_zero or np.mean(hist_fut_lookup_err) > near_zero:
            print(
                "- History/future priors differ by time position; a model switch from "
                "history prior to future template lookup can change behavior even if lookup is correct."
            )
    else:
        if not ok_hist:
            print("- History lookup mismatch: check ToD/DoW indexing against channel 3.")
        if not ok_fut:
            print("- Future lookup mismatch: check ToD/DoW indexing against channel 3.")


if __name__ == "__main__":
    main()
