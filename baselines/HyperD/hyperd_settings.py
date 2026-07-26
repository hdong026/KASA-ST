"""Official HyperD dataset metadata and model hyper-parameters (12->12)."""

from __future__ import annotations

HYPERD_DATASETS = ("PEMS03", "PEMS04", "PEMS07", "PEMS08")

# Official HyperD repo settings; split is 6:2:2 on time steps.
DATASET_META = {
    "PEMS03": {
        "num_nodes": 358,
        "num_time_steps": 26208,
        "raw_npz": "datasets/raw_data/PEMS03/PEMS03.npz",
        "adj_path": "datasets/PEMS03/adj_mx.pkl",
    },
    "PEMS04": {
        "num_nodes": 307,
        "num_time_steps": 16992,
        "raw_npz": "datasets/raw_data/PEMS04/PEMS04.npz",
        "adj_path": "datasets/PEMS04/adj_mx.pkl",
    },
    "PEMS07": {
        "num_nodes": 883,
        "num_time_steps": 28224,
        "raw_npz": "datasets/raw_data/PEMS07/PEMS07.npz",
        "adj_path": "datasets/PEMS07/adj_mx.pkl",
    },
    "PEMS08": {
        "num_nodes": 170,
        "num_time_steps": 17856,
        "raw_npz": "datasets/raw_data/PEMS08/PEMS08.npz",
        "adj_path": "datasets/PEMS08/adj_mx.pkl",
    },
}

# Copied from official baselines/HyperD/PEMS*.py (12->12).
MODEL_PARAMS = {
    "PEMS03": {
        "alpha": 2,
        "F_low": 1,
        "embed_size": 64,
        "hidden_size": 128,
        "fc_hidden_size": 128,
    },
    "PEMS04": {
        "alpha": 2,
        "F_low": 3,
        "embed_size": 64,
        "hidden_size": 128,
        "fc_hidden_size": 128,
    },
    "PEMS07": {
        "alpha": 2,
        "F_low": 1,
        "embed_size": 16,
        "hidden_size": 32,
        "fc_hidden_size": 64,
    },
    "PEMS08": {
        "alpha": 0.5,
        "F_low": 2,
        "embed_size": 64,
        "hidden_size": 128,
        "fc_hidden_size": 512,
    },
}

INPUT_LEN = 12
OUTPUT_LEN = 12
TRAIN_VAL_TEST_RATIO = [0.6, 0.2, 0.2]
NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATES = {
    "PEMS03": 0.007,
    "PEMS04": 0.005,
    "PEMS07": 0.01,
    "PEMS08": 0.002,
}
