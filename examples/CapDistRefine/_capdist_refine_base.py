"""Shared CapDistRefine config fields for PeMS04 horizon configs."""
import os

CAPDIST_REFINE_PARAM = {
    "variant_name": "CapDistRefine",
    "spatial_placement": "temporal_first_capdist_refine",
    "post_spatial_mode": "adaptive_cluster_mix",
    "capdist_enabled": True,
    "capdist_cluster_method": "capdist_spectral_pair",
    "capdist_capacities": [2, 1],
    "capdist_use_road_distance": True,
    "capdist_sigma_d": 0.5,
    "capdist_lambda_d": 0.1,
    "capdist_lambda_mix": [0.5, 0.3],
    "capdist_alphas": [0.08, 0.08],
    "capdist_topks": [8, 16],
    "clustering_seed": 0,
    "dataset_name": "PEMS04",
    "cluster_road_distance_path": os.path.join(
        "datasets", "raw_data", "PEMS04", "adj_PEMS04_distance.pkl"
    ),
    "spatial_graph_loss_weights": [0.0, 0.0],
    "use_prev_condition": True,
    "use_patch_branch": True,
    "use_downsample_branch": True,
    "use_linear_residual_branch": True,
    "patch_embedding_mode": "serial_concat",
    "patch_data_input_mode": "all",
    "use_pre_temporal_spatial_enhancement": False,
    "keep_output_prior_residual": False,
    "use_input_prior_enhancement": False,
    "use_graph_spectral_calibration": False,
    "use_extra_prior_input": False,
    "unified_aux_loss_mode": "none",
    "chain_loss_weights": [0.2, 0.3, 1.0],
}
