import os
import sys
import shutil
import pickle
import argparse

import numpy as np

# 尝试导入，如果没有也不影响主逻辑
try:
    from generate_adj_mx import generate_adj_pems08
except ImportError:
    pass

# 适配 BasicTS 路径
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
from basicts.data.transform import standard_transform


def generate_data(args: argparse.Namespace):
    """
    Preprocess and generate train/valid/test datasets.
    Modified for HoloST to include Channel 4: Prior Knowledge.
    """
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
    steps_per_day = args.steps_per_day

    # 1. Read data
    data = np.load(data_file_path)["data"]
    data = data[..., target_channel] # [L, N, 1]
    print("raw time series shape: {0}".format(data.shape))

    l, n, f = data.shape
    num_samples = l - (history_seq_len + future_seq_len) + 1
    train_num_short = round(num_samples * train_ratio)
    valid_num_short = round(num_samples * valid_ratio)
    test_num_short = num_samples - train_num_short - valid_num_short
    print("number of training samples:{0}".format(train_num_short))
    print("number of validation samples:{0}".format(valid_num_short))
    print("number of test samples:{0}".format(test_num_short))

    # 2. Generate Index
    index_list = []
    for t in range(history_seq_len, num_samples + history_seq_len):
        index = (t-history_seq_len, t, t+future_seq_len)
        index_list.append(index)

    train_index = index_list[:train_num_short]
    valid_index = index_list[train_num_short: train_num_short + valid_num_short]
    test_index = index_list[train_num_short + valid_num_short: train_num_short + valid_num_short + test_num_short]

    # 3. Normalize Data (Standard Scaler)
    scaler = standard_transform
    data_norm = scaler(data, output_dir, train_index, history_seq_len, future_seq_len)

    # -------------------------------------------------------------------------
    # 🔥 New Logic: Calculate Prior (Weekly Average + FFT Low-pass Filter)
    # -------------------------------------------------------------------------
    print("Generating Prior Channel with Frequency Domain Filtering...")
    
    # 1. 计算基础周平均 (同原代码)
    train_end = train_index[-1][1]
    train_data = data[:train_end, :, 0]
    train_mean, train_std = np.mean(train_data), np.std(train_data)
    
    steps_per_week = steps_per_day * 7
    num_weeks = train_data.shape[0] // steps_per_week
    train_reshaped = train_data[:num_weeks * steps_per_week].reshape(num_weeks, steps_per_week, n)
    weekly_profile = np.mean(train_reshaped, axis=0) # [2016, N]
    
    # 2. 🔥 核心改进: FFT 低通滤波 (Frequency Domain Filtering)
    # 目的: 去除周平均曲线中的高频抖动，只保留“主趋势”
    
    # 转到频域
    fft_coeffs = np.fft.rfft(weekly_profile, axis=0)
    
    # 定义截断频率 (Cutoff): 只保留前 k 个低频分量
    # 例如保留前 10% 的频率 (通常能量集中在前几项)
    # 这个参数可以调，越小曲线越平滑，越大越接近原数据
    top_k = int(len(fft_coeffs) * 0.1) 
    
    # 高频部分置零 (Low-pass Filter)
    fft_coeffs[top_k:] = 0
    
    # 逆变换回时域 (取实部)
    smooth_weekly_profile = np.fft.irfft(fft_coeffs, n=weekly_profile.shape[0], axis=0)
    weekly_spectral_template = ((smooth_weekly_profile - train_mean) / train_std)[..., np.newaxis].astype(np.float32)
    
    # 3. 平铺到全长并归一化
    # 使用滤波后的 smooth_weekly_profile
    prior_full = np.tile(smooth_weekly_profile, (l // steps_per_week + 1, 1))[:l]
    prior_norm = (prior_full - train_mean) / train_std
    prior_norm = prior_norm[..., np.newaxis]
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    # 4. Concatenate Features
    feature_list = [data_norm] # Channel 0: Flow
    
    if add_time_of_day:
        tod = [i % steps_per_day / steps_per_day for i in range(data_norm.shape[0])]
        tod = np.array(tod)
        tod_tiled = np.tile(tod, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(tod_tiled) # Channel 1: TOD

    if add_day_of_week:
        dow = [(i // steps_per_day) % 7 for i in range(data_norm.shape[0])]
        dow = np.array(dow)
        dow_tiled = np.tile(dow, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled) # Channel 2: DOW
        
    # Add Prior Channel
    feature_list.append(prior_norm) # Channel 3: Prior

    processed_data = np.concatenate(feature_list, axis=-1)
    print("Final Data Shape: {0}".format(processed_data.shape))

    # 5. Dump Data
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
        
    # Copy Adj
    if os.path.exists(args.graph_file_path):
        shutil.copyfile(args.graph_file_path, output_dir + "/adj_mx.pkl")
    else:
        try:
            generate_adj_pems08()
            shutil.copyfile(graph_file_path, output_dir + "/adj_mx.pkl")
        except:
            print("Warning: adj generation skipped")


if __name__ == "__main__":
    # Settings (Keep same as original)
    HISTORY_SEQ_LEN = 12
    FUTURE_SEQ_LEN = 12
    TRAIN_RATIO = 0.7
    VALID_RATIO = 0.1
    TARGET_CHANNEL = [0]
    STEPS_PER_DAY = 288
    DATASET_NAME = "PEMS08"
    TOD = True
    DOW = True
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    OUTPUT_DIR = os.path.join(project_root, "datasets", DATASET_NAME)
    DATA_FILE_PATH = os.path.join(project_root, "datasets", "raw_data", DATASET_NAME, f"{DATASET_NAME}.npz")
    GRAPH_FILE_PATH = os.path.join(project_root, "datasets", "raw_data", DATASET_NAME, f"adj_{DATASET_NAME}.pkl")

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

    # Paths absolute
    if not os.path.isabs(args_metr.output_dir):
        args_metr.output_dir = os.path.abspath(args_metr.output_dir)
    if not os.path.isabs(args_metr.data_file_path):
        args_metr.data_file_path = os.path.abspath(args_metr.data_file_path)
    if not os.path.isabs(args_metr.graph_file_path):
        args_metr.graph_file_path = os.path.abspath(args_metr.graph_file_path)
    
    if os.path.exists(args_metr.output_dir):
         # Auto confirm for smoother experience
         pass
    else:
        os.makedirs(args_metr.output_dir, exist_ok=True)
    
    generate_data(args_metr)