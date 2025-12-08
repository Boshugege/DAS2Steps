#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
用途：
  从多通道分布式光纤 CSV（每列 ch_*，每行一个时间采样）中，
  计算时域上“脚步存在的通道概率分布”（channels × time），并绘图标出每个时刻最可能的通道。

依赖：
  pip install numpy pandas scipy matplotlib seaborn

使用方法（示例）：
  python CSV2Steps.py --input your_data.csv

主要输出（保存到 outdir）：
  - energy_matrix.npy  (channels x frames, 短时能量)
  - prob_matrix.npy    (channels x frames, 每列归一化为概率)
  - prob_matrix.csv
  - argmax_channel.csv (每帧最可能的 channel index)
  - figures: raw_heatmap.png, filtered_heatmap.png, prob_heatmap_with_argmax.png
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter, find_peaks
from scipy.ndimage import median_filter, gaussian_filter1d
import matplotlib.pyplot as plt
import seaborn as sns

# ====== 注释/结构重组说明（仅注释改动，代码未改） ======
# 目标：对原脚本的注释进行编号与分组，便于阅读与维护。
# 约定的编号体系（在文件中按出现顺序使用）：
#   [0] 文件头与依赖说明
#   [1] 参数解析与默认配置
#   [2] 滤波器相关函数（设计与应用）
#   [3] 短时能量与概率相关函数
#   [4] 主流程：输入/预处理/滤波/均衡/能量/归一化/热道抑制
#   [5] 导出与可视化
# 注：只修改注释，不改动任何非注释代码行。

# ----------------------------
# [1] 参数解析与入口函数
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Compute time-channel footstep probability map from CSV.")
    p.add_argument("--input", "-i", required=True, help="输入 CSV 文件路径（列名包含 ch_0 ... ch_N）")
    p.add_argument("--outdir", "-o", default="output", help="结果输出目录（默认 ./output）")
    p.add_argument("--fs", type=float, default=None, help="采样率 Hz（可选，如未提供则用行索引作为时间单位）")
    p.add_argument("--lowcut", type=float, default=10.0, help="带通滤波低频截止（Hz），默认 10 Hz")
    p.add_argument("--highcut", type=float, default=60.0, help="带通滤波高频截止（Hz），默认 60 Hz")
    p.add_argument("--filter_order", type=int, default=4, help="Butterworth 滤波器阶数")
    p.add_argument("--energy_win_ms", type=float, default=50.0, help="短时能量窗口（ms），默认 50 ms")
    p.add_argument("--energy_step_ms", type=float, default=10.0, help="短时能量窗口滑动步长（ms），默认 10 ms")
    p.add_argument("--savgol_window", type=int, default=7, help="Savitzky-Golay 平滑窗口（奇数）用于信号平滑，默认 7")
    p.add_argument("--median_smooth_frames", type=int, default=5, help="对 argmax 结果的中值滤波窗口（帧），默认 5")
    p.add_argument("--gauss_smooth_sigma", type=float, default=1.0, help="对概率矩阵时间轴做高斯平滑的 sigma（帧），默认 1.0")
    p.add_argument("--downsample_factor", type=int, default=1, help="如数据很长，可按时间轴下采样（整数因子），1 表示不下采样")
    p.add_argument("--ignore_channels_start", type=int, default=0, help="从通道 0 开始屏蔽到指定通道索引（不含该索引），用于去掉光源附近异常信号，默认 0 表示不屏蔽")
    p.add_argument("--hot_channel_k", type=float, default=3.0, help="检测热通道的阈值参数 k（median + k*MAD），默认 3.0")
    p.add_argument("--hot_mask_scale", type=float, default=1.0, help="热通道幅度抑制比例因子，默认 1.0（即抑制到 median 水平）")
    p.add_argument("--hot_channel_post_threshold", type=float, default=0.1, help="二次热通道检测阈值：若某通道 argmax 占比 > threshold 则视为噪声，默认 0.1")
    p.add_argument("--hot_channel_post_scale", type=float, default=0.5, help="二次热通道屏蔽/抑制比例，0=完全屏蔽, 0<x<1=按比例缩小概率，默认 0.5")
    p.add_argument("--alpha_kalman", type=float, default=0.01, help="一阶卡尔曼平滑参数 alpha，默认 0.01，越大越贴近原高斯平滑轨迹")
    p.add_argument("--min_energy_threshold", type=float, default=0.001, help="事件最小能量阈值（相对）")
    p.add_argument("--max_speed_threshold", type=float, default=10.0, help="事件最大速度阈值（通道/秒）")
    return p.parse_args()

# ----------------------------
# [2] 滤波器设计与应用
#    - butter_bandpass: 设计带通滤波器系数
#    - bandpass_filter_array: 对多通道按列做零相位滤波（filtfilt）
# ----------------------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    设计 Butterworth 带通滤波器（返回滤波系数）
    需要 fs（采样率）来设计滤波器。如果 fs 未提供，调用方应避免调用此函数。
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if low <= 0: low = 1e-6
    if high >= 1: high = 0.999999
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter_array(X, fs, lowcut=10.0, highcut=60.0, order=4):
    """
    对多通道信号（shape = [T, C]）按列进行带通滤波并返回滤波后的信号（相同 shape）。
    使用 filtfilt 做零相位滤波。
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # 对每一列（通道）做 filtfilt
    Xf = np.zeros_like(X)
    for c in range(X.shape[1]):
        # filtfilt 对常数列或 NaN 会报错，因此要保证数值稳定
        col = X[:, c]
        # 去除直流
        col = col - np.mean(col)
        try:
            colf = filtfilt(b, a, col, axis=0, method="pad")
        except Exception as e:
            # 若 filtfilt 出错（短信号等），退回不滤波或用简单移动平均
            print(f"[Warning] 通道 {c} filtfilt 出现异常：{e}，将使用 Savitzky-Golay 作为代替平滑")
            colf = savgol_filter(col, 7 if col.size>=7 else 3, polyorder=2, mode='nearest')
        Xf[:, c] = colf
    return Xf

# ----------------------------
# [3] 短时能量与概率处理函数
#    - short_time_energy: 计算短时能量（返回 E: (C, n_frames)）
#    - normalize_prob_per_time: 将每帧归一化为通道概率分布
# ----------------------------
def short_time_energy(X, fs, win_ms=50.0, step_ms=10.0):
    """
    计算短时能量（Short-Time Energy, STE）。
    输入：
      X: ndarray (T, C) 原始或滤波后信号（时间轴为行）
      fs: 采样率（Hz）
      win_ms: 窗口长度（ms）
      step_ms: 窗口步长（ms）
    返回：
      E: ndarray (C, n_frames) 每个通道每个时间帧的能量
      frame_times: ndarray (n_frames,) 每帧对应的时间（s）
    说明：
      能量按窗口内平方和计算（不做均值），你可乘上 1/fs 做能量归一但对相对概率没影响。
    """
    if fs is None:
        raise ValueError("short_time_energy 需要传入采样率 fs")
    win_samps = max(1, int(round(win_ms * 1e-3 * fs)))
    step_samps = max(1, int(round(step_ms * 1e-3 * fs)))
    T, C = X.shape
    frames = []
    frame_centers = []
    for start in range(0, T - win_samps + 1, step_samps):
        frame = X[start:start + win_samps, :]  # shape (win, C)
        # 能量 = 窗口内信号平方和（按通道）
        e = np.sum(frame.astype(np.float64) ** 2, axis=0)  # shape (C,)
        frames.append(e)
        frame_centers.append((start + start + win_samps - 1) / 2.0 / fs)  # 中心时间（s）
    if len(frames) == 0:
        # 信号太短，使用单帧全长
        e = np.sum(X.astype(np.float64) ** 2, axis=0)[None, :]
        E = e.T  # (C,1)
        frame_times = np.array([ (T-1)/2.0 / fs if fs else 0.0 ])
        return E, frame_times
    frames = np.stack(frames, axis=1)  # shape (C, n_frames) if we transpose below
    E = frames.T.T  # convert to (C, n_frames)
    frame_times = np.array(frame_centers)
    return E, frame_times

def normalize_prob_per_time(E, eps=1e-12):
    """
    对每个时间帧 (column) 在通道维度上做归一化，使得每列和为 1（概率分布）。
    输入 E shape (C, Tframes)
    返回 P 同样 shape
    """
    col_sums = np.sum(E, axis=0, keepdims=True)  # (1, Tframes)
    P = E / (col_sums + eps)
    return P

# ----------------------------
# [4] 主流程（读取 -> 预处理 -> 滤波 -> 均衡 -> 能量 -> 归一化 -> 抑制 -> 平滑）
#     注：此处的注释分段反映了主函数内的逻辑步骤，编号与输出文件顺序一致。
# ----------------------------
def main():
    args = parse_args()
    INPUT_CSV = args.input
    OUTDIR = args.outdir
    os.makedirs(OUTDIR, exist_ok=True)

    # [4.1] 读取 CSV 并自动识别通道列（以 ch_ 开头）
    df = pd.read_csv(INPUT_CSV)
    colnames = df.columns.tolist()
    ch_cols = [c for c in colnames if c.lower().startswith("ch_")]
    if len(ch_cols) == 0:
        raise ValueError("未检测到以 'ch_' 开头的列，请检查 CSV 格式。列名示例：ch_0,ch_1,...")
    print(f"检测到 {len(ch_cols)} 个通道，使用列：{ch_cols[:6]} ...")

    # [4.2] 转为 numpy 并处理 NaN（列内线性插值后前后填充）
    X_raw = df[ch_cols].values.astype(np.float64)  # shape (T, C)
    if np.isnan(X_raw).any():
        print("[Message] 数据包含 NaN，使用列内线性插值填充，然后剩余用 0 填充。")
        Xdf = pd.DataFrame(X_raw)
        Xdf = Xdf.interpolate(axis=0).fillna(method='bfill').fillna(method='ffill').fillna(0.0)
        X_raw = Xdf.values

    # [4.3] 可选时间下采样（按行）
    if args.downsample_factor > 1:
        print(f"[Message] 在时间上进行下采样，因子 = {args.downsample_factor}")
        X_raw = X_raw[::args.downsample_factor, :]

    T_samples, C_channels = X_raw.shape
    print(f"数据形状：{T_samples} 时间采样点 × {C_channels} 通道")

    # [4.4] 采样率处理提示：若未提供，则用 fs=1.0 作为索引单位
    fs = args.fs
    if fs is None:
        print("[Message] 未提供采样率 fs；时间轴将以样本索引为单位（1,2,3,...）。若你知道采样率，请使用 --fs 参数。")
        fs = 1.0

    # [4.5] 去趋势与标准化（每通道）：去均值并除以标准差，避免某些通道幅度过大主导
    X = X_raw.copy()
    channel_means = np.mean(X, axis=0)
    channel_stds = np.std(X, axis=0, ddof=1) + 1e-9
    X = (X - channel_means[None, :]) / channel_stds[None, :]

    # [4.6] 滤波阶段：若提供 fs 则使用带通滤波，否则用 Savitzky-Golay 做弱平滑
    if args.fs is not None:
        print(f"[Step] 对每通道做带通滤波 {args.lowcut}-{args.highcut} Hz")
        X_filt = bandpass_filter_array(X, fs=args.fs, lowcut=args.lowcut, highcut=args.highcut, order=args.filter_order)
    else:
        print("[Step] 未提供采样率，跳过带通滤波；对每通道做 Savitzky-Golay 平滑作为弱去噪")
        X_filt = np.zeros_like(X)
        win = args.savgol_window if args.savgol_window % 2 == 1 else args.savgol_window + 1
        for c in range(C_channels):
            try:
                X_filt[:, c] = savgol_filter(X[:, c], win, polyorder=2, mode='nearest')
            except Exception:
                X_filt[:, c] = X[:, c]

    # [4.7] 屏蔽起始通道（例如靠近光源的若干通道）
    ignore_end = args.ignore_channels_start
    if ignore_end > 0:
        print(f"[Message] 屏蔽通道 0 到 {ignore_end-1} 的信号")
        X_filt[:, :ignore_end] = 0.0

    # ----------------------------
    # [4.8] 幅度均衡与坏通道（热通道）检测/抑制（多步）
    #   步骤：
    #     a) 估计每通道的背景噪声水平（使用 median(|x|) 作为鲁棒估计）
    #     b) 识别显著高噪通道并强力抑制
    #     c) 按通道噪声水平做归一化（whitening-like）
    # ----------------------------
    # a) 估计通道噪声水平（MAD 近似）
    channel_noise_level = np.median(np.abs(X_filt), axis=0)
    # b) 系统中位噪声
    system_median_noise = np.median(channel_noise_level)
    # c) 识别并强力压制明显高噪通道（阈值为 median 的 2.5 倍，可调整）
    kill_threshold_ratio = 2.5  
    bad_channels_mask = channel_noise_level > (kill_threshold_ratio * system_median_noise)

    if np.sum(bad_channels_mask) > 0:
        bad_idx = np.where(bad_channels_mask)[0]
        print(f"[Message] 强力抑制 {len(bad_idx)} 个高噪通道 (Ch {bad_idx[:5]}...): 噪声 > {kill_threshold_ratio}x Median")
        baseline = np.median(X_filt[:, bad_channels_mask], axis=0)   # 取每个坏通道当前的中位值作为统一噪声基底
        X_filt[:, bad_channels_mask] = X_filt[:, bad_channels_mask] - baseline[np.newaxis, :]
        # 防止负值削弱动态范围：截断到 0+
        X_filt[:, bad_channels_mask] = np.clip(X_filt[:, bad_channels_mask], 0, None)
        channel_noise_level[bad_channels_mask] = np.maximum(channel_noise_level[bad_channels_mask], 1e-6)

    # d) 幅度均衡：每通道除以其噪声水平，突出瞬态信号
    safe_scale = np.where(channel_noise_level > 1e-9, channel_noise_level, 1.0)
    X_filt = X_filt / safe_scale[None, :]

    # [4.9] 计算短时能量矩阵 E(c, tframe)
    win_ms = args.energy_win_ms
    step_ms = args.energy_step_ms
    E, frame_times = short_time_energy(X_filt, fs=fs, win_ms=win_ms, step_ms=step_ms)
    C, n_frames = E.shape
    print(f"[Step] 计算短时能量：窗口 {win_ms} ms，步长 {step_ms} ms => {n_frames} 帧")

    # [4.10] 从能量中减去通道背景（按通道中位数），并确保非负
    noise_floor = np.median(E, axis=1, keepdims=True)
    E = E - noise_floor
    E = np.maximum(E, 0.0)
    # 加入极小常数以避免后续除零
    E = E + 1e-9

    # # [4.11] （可选）对能量做时间轴高斯平滑以降低随机跳变 # Comment by wyh 1204: useless
    # if args.gauss_smooth_sigma > 0:
    #     E_sm = np.zeros_like(E)
    #     for c in range(C):
    #         E_sm[c, :] = gaussian_filter1d(E[c, :], sigma=args.gauss_smooth_sigma, mode='reflect')
    #     E = E_sm

    # [4.12] 按时间帧归一化为概率分布（每列和为 1）
    P = normalize_prob_per_time(E, eps=1e-12)  # shape (C, n_frames)
    P_global = E / (np.sum(E) + 1e-12)  # 全局归一化（仅用于可视化对比）

    # [4.13] 基于概率分布的二次热通道检测与抑制（若某通道在过多帧中为 argmax）
    temp_argmax = np.argmax(P, axis=0)
    channel_counts = np.bincount(temp_argmax, minlength=C)
    channel_freq = channel_counts / n_frames
    post_threshold = args.hot_channel_post_threshold
    post_hot_mask = channel_freq > post_threshold

    if np.sum(post_hot_mask) > 0:
        bad_channels = np.where(post_hot_mask)[0]
        print(f"[Message] 二次检测发现持续活跃通道 (占比 > {post_threshold*100:.1f}%)：{bad_channels}")
        print(f"          执行抑制，比例系数 scale = {args.hot_channel_post_scale}")
        scale = args.hot_channel_post_scale
        P[post_hot_mask, :] *= scale
        # 重新归一化，保证每列和为 1
        P = normalize_prob_per_time(P, eps=1e-12)

    # [4.14] 计算每帧最可能通道（argmax）并平滑以减少跳变
    argmax_raw = np.argmax(P, axis=0)
    # 使用高斯先平滑数值序列（便于随后卡尔曼一阶滤波）
    sigma_gauss = args.gauss_smooth_sigma if args.gauss_smooth_sigma > 0 else 1.0
    argmax_gauss = gaussian_filter1d(argmax_raw.astype(float), sigma=sigma_gauss, mode='reflect')

    # 一阶卡尔曼平滑（指数平滑）增强时序连续性
    print(f"[Message] 对 argmax 结果做一阶卡尔曼平滑，alpha = {args.alpha_kalman}")
    argmax_smooth = np.zeros_like(argmax_gauss)
    argmax_smooth[0] = argmax_gauss[0]
    for t in range(1, len(argmax_gauss)):
        argmax_smooth[t] = args.alpha_kalman * argmax_gauss[t] + (1 - args.alpha_kalman) * argmax_smooth[t-1]

    # 置信度指标：top1 - top2 值，差值越大表示越确定
    top1_vals = np.max(P, axis=0)
    sorted_idx = np.argsort(P, axis=0)  # 从小到大
    top2_vals = P[sorted_idx[-2, :], np.arange(n_frames)]
    confidence = (top1_vals - top2_vals)

    # ----------------------------
    # [5] 导出结果：numpy / csv / 绘图
    # ----------------------------
    np.save(os.path.join(OUTDIR, "energy_matrix.npy"), E)
    np.save(os.path.join(OUTDIR, "prob_matrix.npy"), P)
    prob_df = pd.DataFrame(P, index=ch_cols)
    prob_df.columns = [f"t_{i}" for i in range(n_frames)]
    prob_df.to_csv(os.path.join(OUTDIR, "prob_matrix.csv"))

    arg_df = pd.DataFrame({
        "frame_index": np.arange(n_frames),
        "time_s": frame_times,
        "argmax_raw_channel": argmax_raw,
        "argmax_median_smoothed": argmax_smooth,
        "top1_prob": top1_vals,
        "top2_prob": top2_vals,
        "confidence": confidence
    })
    arg_df.to_csv(os.path.join(OUTDIR, "argmax_channel.csv"), index=False)

    # 新增：从 argmax 序列聚合检测事件（找“团”）
    def detect_events_from_argmax(argmax_smooth, frame_times, P,
                                  min_duration_frames=5, min_jump=2.0,
                                  min_energy_threshold=0.1, max_speed_threshold=10.0):
        """
        从平滑 argmax 序列聚合检测事件，加入能量和速度限制。
        - argmax_smooth: (n_frames,) 平滑后的 argmax 通道
        - frame_times: (n_frames,)
        - P: (C, n_frames) 概率
        - min_energy_threshold: 事件平均能量最小值
        - max_speed_threshold: 事件速度最大值（通道/秒）
        返回 events: list of dict {time_s, channel_centroid, duration_s, confidence}
        """
        events = []
        n = len(argmax_smooth)
        dt = frame_times[1] - frame_times[0] if n > 1 else 1.0
        i = 0
        while i < n:
            start = i
            current_ch = argmax_smooth[i]
            while i < n and abs(argmax_smooth[i] - current_ch) < min_jump:
                i += 1
            end = i - 1
            duration_frames = end - start + 1
            if duration_frames >= min_duration_frames:
                ch_avg = np.mean(argmax_smooth[start:end+1])
                time_s = np.mean(frame_times[start:end+1])
                prob_seg = P[:, start:end+1]
                top1_avg = np.mean(np.max(prob_seg, axis=0))
                # 能量检查：平均 top1 > min_energy_threshold
                if top1_avg < min_energy_threshold:
                    continue
                # 速度检查：通道变化率 < max_speed_threshold
                ch_diff = np.diff(argmax_smooth[start:end+1])
                speed = np.mean(np.abs(ch_diff) / dt) if len(ch_diff) > 0 else 0.0
                if speed > max_speed_threshold:
                    continue
                events.append({
                    "time_s": time_s,
                    "channel_centroid": ch_avg,
                    "duration_s": duration_frames * dt,
                    "confidence": top1_avg,
                    "speed": speed
                })
        return events

    # 在 argmax_smooth 计算后添加事件检测
    events = detect_events_from_argmax(argmax_smooth, frame_times, P,
                                       min_duration_frames=5, min_jump=2.0,
                                       min_energy_threshold=args.min_energy_threshold,
                                       max_speed_threshold=args.max_speed_threshold)
    print(f"[Step] 从 argmax 聚合检测到 {len(events)} 个事件")

    # 修改 arg_df 为 events_df
    events_df = pd.DataFrame(events) if events else pd.DataFrame(columns=["time_s", "channel_centroid", "duration_s", "confidence", "speed"])
    events_df.to_csv(os.path.join(OUTDIR, "aggregated_events.csv"), index=False)

    print("[Output] 已保存 energy_matrix.npy, prob_matrix.npy, prob_matrix.csv, argmax_channel.csv 到", OUTDIR)

    # ----------------------------
    # [5.1] 可视化：热图与时间序列
    # ----------------------------
    sns.set(style="whitegrid")
    fig_w = 14
    fig_h = 8

    # [5.1.1] 原始信号热图（channels x time samples）
    fig1, ax1 = plt.subplots(figsize=(fig_w, 3.5))
    vmax = np.percentile(np.abs(X_raw), 99)
    im1 = ax1.imshow(X_raw.T, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax1.set_title("Raw signal (channels x time samples)")
    ax1.set_ylabel("Channel index")
    ax1.set_xlabel("Sample index (time)")
    plt.colorbar(im1, ax=ax1, orientation='vertical', label='amplitude')
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUTDIR, "raw_signal_heatmap.png"), dpi=200)

    # [5.1.2] 滤波后信号热图
    fig2, ax2 = plt.subplots(figsize=(fig_w, 3.5))
    vmax2 = np.percentile(np.abs(X_filt), 99)
    im2 = ax2.imshow(X_filt.T, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-vmax2, vmax=vmax2)
    ax2.set_title("Filtered signal (channels x time samples)")
    ax2.set_ylabel("Channel index")
    ax2.set_xlabel("Sample index (time)")
    plt.colorbar(im2, ax=ax2, orientation='vertical', label='amplitude')
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTDIR, "filtered_signal_heatmap.png"), dpi=200)

    # [5.1.3] 概率矩阵热图，并叠加聚合事件
    fig3, ax3 = plt.subplots(figsize=(fig_w, 6))
    extent = (frame_times[0], frame_times[-1], -0.5, C - 0.5)
    im3 = ax3.imshow(P, aspect='auto', origin='lower', cmap='viridis', extent=extent)
    ax3.set_title("Probability map P(channel, time-frame) (each column sums to 1)")
    ax3.set_ylabel("Channel index")
    ax3.set_xlabel("Time (s)")
    plt.colorbar(im3, ax=ax3, orientation='vertical', label='probability')
    # 标出聚合事件
    for ev in events:
        ax3.scatter(ev["time_s"], ev["channel_centroid"], color='red', s=50, marker='x', label='aggregated event' if ev == events[0] else "")
    ax3.legend(loc='upper right')
    fig3.tight_layout()
    fig3.savefig(os.path.join(OUTDIR, "prob_heatmap_with_aggregated_events.png"), dpi=200)

    # [5.1.4] 置信度随时间曲线（top1 - top2）
    fig4, ax4 = plt.subplots(figsize=(fig_w, 2.5))
    ax4.plot(frame_times, confidence, label='top1 - top2 (confidence)')
    ax4.set_xlabel("Time (s)" if args.fs else "Frame index")
    ax4.set_ylabel("Confidence (prob difference)")
    ax4.set_title("Confidence of selected channel over time")
    ax4.grid(True)
    fig4.tight_layout()
    fig4.savefig(os.path.join(OUTDIR, "confidence_over_time.png"), dpi=200)

    plt.close('all')
    print("[Finish] 所有图像已保存到:", OUTDIR)

if __name__ == "__main__":
    main()