import os
import sys
import shutil
import pickle
import argparse

import numpy as np
import pandas as pd

# 适配 BasicTS 路径 (复用你的标准逻辑)
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
from basicts.data.transform import standard_transform


def generate_data(args: argparse.Namespace):
    """Preprocess and generate train/valid/test datasets."""
    target_channel = args.target_channel
    future_seq_len = args.future_seq_len
    history_seq_len = args.history_seq_len
    add_time_of_day = args.tod
    add_day_of_week = args.dow
    output_dir = args.output_dir
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    data_file_path = args.data_file_path
    graph_file_path = args.graph_file_path
    steps_per_day = args.steps_per_day  # Added for Prior

    # read data
    df = pd.read_hdf(data_file_path)
    data = np.expand_dims(df.values, axis=-1)
    data = data[..., target_channel]
    print("raw time series shape: {0}".format(data.shape))

    l, n, f = data.shape
    num_samples = l - (history_seq_len + future_seq_len) + 1
    train_num_short = round(num_samples * train_ratio)
    valid_num_short = round(num_samples * valid_ratio)
    test_num_short = num_samples - train_num_short - valid_num_short
    print("number of training samples:{0}".format(train_num_short))
    print("number of validation samples:{0}".format(valid_num_short))
    print("number of test samples:{0}".format(test_num_short))

    index_list = []
    for t in range(history_seq_len, num_samples + history_seq_len):
        index = (t-history_seq_len, t, t+future_seq_len)
        index_list.append(index)

    train_index = index_list[:train_num_short]
    valid_index = index_list[train_num_short: train_num_short + valid_num_short]
    test_index = index_list[train_num_short +
                            valid_num_short: train_num_short + valid_num_short + test_num_short]

    scaler = standard_transform
    data_norm = scaler(data, output_dir, train_index, history_seq_len, future_seq_len)

    # -------------------------------------------------------------------------
    # 🔥 HoloST Logic: Calculate Prior (Weekly Average + FFT)
    # -------------------------------------------------------------------------
    print("Generating Prior Channel with Frequency Domain Filtering...")
    
    # 1. 计算基础统计量
    train_end = train_index[-1][1]
    train_data = data[:train_end, :, 0]
    train_mean, train_std = np.mean(train_data), np.std(train_data)
    
    steps_per_week = steps_per_day * 7
    num_weeks = train_data.shape[0] // steps_per_week
    train_reshaped = train_data[:num_weeks * steps_per_week].reshape(num_weeks, steps_per_week, n)
    weekly_profile = np.mean(train_reshaped, axis=0) # [2016, N]
    
    # 2. FFT 低通滤波
    fft_coeffs = np.fft.rfft(weekly_profile, axis=0)
    top_k = int(len(fft_coeffs) * 0.1) 
    fft_coeffs[top_k:] = 0
    smooth_weekly_profile = np.fft.irfft(fft_coeffs, n=weekly_profile.shape[0], axis=0)
    weekly_spectral_template = ((smooth_weekly_profile - train_mean) / train_std)[..., np.newaxis].astype(np.float32)
    
    # 3. 平铺到全长并归一化
    prior_full = np.tile(smooth_weekly_profile, (l // steps_per_week + 1, 1))[:l]
    prior_norm = (prior_full - train_mean) / train_std
    prior_norm = prior_norm[..., np.newaxis]
    # -------------------------------------------------------------------------

    # add external feature
    feature_list = [data_norm]
    if add_time_of_day:
        tod = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        tod_tiled = np.tile(tod, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(tod_tiled)

    if add_day_of_week:
        dow = df.index.dayofweek
        dow_tiled = np.tile(dow, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)
        
    feature_list.append(prior_norm) # Add Prior

    processed_data = np.concatenate(feature_list, axis=-1)
    print("Final Data Shape: {0}".format(processed_data.shape))

    # dump data
    index = {}
    index["train"] = train_index
    index["valid"] = valid_index
    index["test"] = test_index
    with open(output_dir + "/index_in{0}_out{1}.pkl".format(history_seq_len, future_seq_len), "wb") as f:
        pickle.dump(index, f)

    data_dict = {}
    data_dict["processed_data"] = processed_data
    with open(output_dir + "/data_in{0}_out{1}.pkl".format(history_seq_len, future_seq_len), "wb") as f:
        pickle.dump(data_dict, f)

    npz_path = output_dir + "/data_in{0}_out{1}.npz".format(history_seq_len, future_seq_len)
    np.savez(
        npz_path,
        processed_data=processed_data.astype(np.float32),
        weekly_spectral_template=weekly_spectral_template,
        slots_per_day=steps_per_day,
        slots_per_week=steps_per_week,
    )
    print("Saved processed npz: {0}".format(npz_path))
        
    # copy adj
    if os.path.exists(args.graph_file_path):
        shutil.copyfile(args.graph_file_path, output_dir + "/adj_mx.pkl")
    else:
        print("Warning: graph file not found, skipping copy.")


if __name__ == "__main__":
    # Settings
    HISTORY_SEQ_LEN = 12
    FUTURE_SEQ_LEN = 12
    TRAIN_RATIO = 0.7
    VALID_RATIO = 0.1
    TARGET_CHANNEL = [0]
    STEPS_PER_DAY = 288
    DATASET_NAME = "PEMS-BAY"
    TOD = True
    DOW = True
    
    # ================= 核心修改：完全复用 PEMS04 的路径逻辑 =================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 这里的 dirname 层数取决于你的文件结构，假设此脚本在 scripts/data_preparation/PEMS-BAY/
    # script_dir = .../PEMS-BAY
    # dirname(script_dir) = .../data_preparation
    # dirname(...) = .../scripts
    # dirname(...) = .../LSTNN (Project Root)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    OUTPUT_DIR = os.path.join(project_root, "datasets", DATASET_NAME)
    DATA_FILE_PATH = os.path.join(project_root, "datasets", "raw_data", DATASET_NAME, f"{DATASET_NAME}.h5")
    GRAPH_FILE_PATH = os.path.join(project_root, "datasets", "raw_data", DATASET_NAME, f"adj_{DATASET_NAME}.pkl")
    # ======================================================================

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--data_file_path", type=str, default=DATA_FILE_PATH)
    parser.add_argument("--graph_file_path", type=str, default=GRAPH_FILE_PATH)
    parser.add_argument("--history_seq_len", type=int, default=HISTORY_SEQ_LEN)
    parser.add_argument("--future_seq_len", type=int, default=FUTURE_SEQ_LEN)
    parser.add_argument("--steps_per_day", type=int, default=STEPS_PER_DAY)
    parser.add_argument("--tod", type=bool, default=TOD)
    parser.add_argument("--dow", type=bool, default=DOW)
    parser.add_argument("--target_channel", type=list, default=TARGET_CHANNEL)
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--valid_ratio", type=float, default=VALID_RATIO)
    args_metr = parser.parse_args()

    # Paths absolute Check
    if not os.path.isabs(args_metr.output_dir):
        args_metr.output_dir = os.path.abspath(args_metr.output_dir)
    if not os.path.isabs(args_metr.data_file_path):
        args_metr.data_file_path = os.path.abspath(args_metr.data_file_path)
    if not os.path.isabs(args_metr.graph_file_path):
        args_metr.graph_file_path = os.path.abspath(args_metr.graph_file_path)
    
    # 打印参数以供检查
    print("-"*(20+45+5))
    for key, value in sorted(vars(args_metr).items()):
        print("|{0:>20} = {1:<45}|".format(key, str(value)))
    print("-"*(20+45+5))
    
    if os.path.exists(args_metr.output_dir):
        # 自动覆盖，保持脚本的流畅性，或者你可以改回 input 询问
        pass 
    else:
        os.makedirs(args_metr.output_dir, exist_ok=True)
    
    generate_data(args_metr)