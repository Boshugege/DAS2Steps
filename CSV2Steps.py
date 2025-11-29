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
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.ndimage import median_filter, gaussian_filter1d
import matplotlib.pyplot as plt
import seaborn as sns

# ----- 参数与函数 -----

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
    p.add_argument("--hot_channel_post_scale", type=float, default=0.0, help="二次热通道屏蔽/抑制比例，0=完全屏蔽, 0<x<1=按比例缩小概率，默认 0.0")
    p.add_argument("--alpha_kalman", type=float, default=0.001, help="一阶卡尔曼平滑参数 alpha，默认 0.001，越大越贴近原高斯平滑轨迹")
    return p.parse_args()

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

# ----- 主流程部分 -----

def main():
    args = parse_args()
    INPUT_CSV = args.input
    OUTDIR = args.outdir
    os.makedirs(OUTDIR, exist_ok=True)

    # ------- 1) 读取 CSV -------
    # 我们尝试自动识别以 ch_ 开头的列
    df = pd.read_csv(INPUT_CSV)
    colnames = df.columns.tolist()
    ch_cols = [c for c in colnames if c.lower().startswith("ch_")]
    if len(ch_cols) == 0:
        raise ValueError("未检测到以 'ch_' 开头的列，请检查 CSV 格式。列名示例：ch_0,ch_1,...")
    print(f"检测到 {len(ch_cols)} 个通道，使用列：{ch_cols[:6]} ...")

    # 将数据转换为 numpy array：时间轴为第一维（rows = 时间采样点）
    X_raw = df[ch_cols].values.astype(np.float64)  # shape (T, C)
    # 可能包含 NaN：用线性插值或 0 填充
    if np.isnan(X_raw).any():
        print("[Message] 数据包含 NaN，使用列内线性插值填充，然后剩余用 0 填充。")
        Xdf = pd.DataFrame(X_raw)
        Xdf = Xdf.interpolate(axis=0).fillna(method='bfill').fillna(method='ffill').fillna(0.0)
        X_raw = Xdf.values

    # 可选下采样（在时间上）
    if args.downsample_factor > 1:
        print(f"[Message] 在时间上进行下采样，因子 = {args.downsample_factor}")
        X_raw = X_raw[::args.downsample_factor, :]

    T_samples, C_channels = X_raw.shape
    print(f"数据形状：{T_samples} 时间采样点 × {C_channels} 通道")

    # 若未提供采样率 fs，则以 1.0 作为单位采样（时间轴按索引）
    fs = args.fs
    if fs is None:
        print("[Message] 未提供采样率 fs；时间轴将以样本索引为单位（1,2,3,...）。若你知道采样率，请使用 --fs 参数。")
        # 为了计算 short-time energy 时仍需要 fs，设为 1.0（但窗口参数将被当成采样点数）
        # 我们把 win_ms 和 step_ms 转换为采样点时会基于 fs；若 fs == 1.0 则认为 win_ms/step_ms 已被解释为样本数（谨慎使用）
        fs = 1.0

    # ------- 2) 简单去趋势和归一化（每通道） -------
    # 先去直流和减去每通道的中位数，随后除以标准差（避免某些通道幅度极大主导）
    X = X_raw.copy()
    channel_means = np.mean(X, axis=0)
    channel_stds = np.std(X, axis=0, ddof=1) + 1e-9
    X = (X - channel_means[None, :]) / channel_stds[None, :]

    # ------- 3) 带通滤波（如果提供 fs） -------
    if args.fs is not None:
        print(f"[Step] 对每通道做带通滤波 {args.lowcut}-{args.highcut} Hz")
        X_filt = bandpass_filter_array(X, fs=args.fs, lowcut=args.lowcut, highcut=args.highcut, order=args.filter_order)
    else:
        # 无 fs 时不做频域带通，仅做 Savitzky-Golay 平滑作为替代（弱去噪）
        print("[Step] 未提供采样率，跳过带通滤波；对每通道做 Savitzky-Golay 平滑作为弱去噪")
        X_filt = np.zeros_like(X)
        win = args.savgol_window if args.savgol_window % 2 == 1 else args.savgol_window + 1
        for c in range(C_channels):
            try:
                X_filt[:, c] = savgol_filter(X[:, c], win, polyorder=2, mode='nearest')
            except Exception:
                # 如果信号太短则直接使用原信号
                X_filt[:, c] = X[:, c]

    # ------- 3.1) 屏蔽起始通道异常 -------
    ignore_end = args.ignore_channels_start
    if ignore_end > 0:
        print(f"[Message] 屏蔽通道 0 到 {ignore_end-1} 的信号")
        X_filt[:, :ignore_end] = 0.0
    

    # ------- 3.5) 热通道检测与抑制 -------

    # 这种检测方式效果不好，我应该去除的是持续较高的信号。
    # # 检测每通道幅度中位数，超过 median + k * MAD 的通道被认为是热通道
    # hot_channel_k = 5.0  # 可调参数，越大越宽松
    # channel_median = np.median(np.abs(X_filt), axis=0)  # 每列中位数
    # median_of_medians = np.median(channel_median)
    # mad = np.median(np.abs(channel_median - median_of_medians)) + 1e-12
    # hot_mask = channel_median > (median_of_medians + hot_channel_k * mad)
    # if np.sum(hot_mask) > 0:
    #     print(f"[提示] 检测到 {np.sum(hot_mask)} 个热通道，进行幅度抑制")
    #     print(f"热通道屏蔽强度因子：{args.hot_mask_scale}（即抑制到 {args.hot_mask_scale} * median 水平）")
    #     X_filt[:, hot_mask] *= args.hot_mask_scale*(median_of_medians / (channel_median[hot_mask] + 1e-12))

    # TODO 更合理的热通道检测
    # 1) 计算每个通道的长期平均幅度（贯穿整个时域的信号强度）
    long_term_mean = np.mean(np.abs(X_filt), axis=0)  # shape (C,)

    # 2) 用 median + k*MAD 判断热通道（持续偏高的通道）
    median_lt = np.median(long_term_mean)
    mad_lt = np.median(np.abs(long_term_mean - median_lt)) + 1e-12
    hot_mask = long_term_mean > (median_lt + args.hot_channel_k * mad_lt)

    # 3) 输出检测信息
    if np.sum(hot_mask) > 0:
        print(f"[Message] 检测到 {np.sum(hot_mask)} 个持续性热通道")
        print(f"[Message] 热通道抑制强度因子：{args.hot_mask_scale}")

        # 4) 根据模式处理热通道
        # 按比例抑制：乘以 scale_factor * median / long_term_mean
        X_filt[:, hot_mask] *= args.hot_mask_scale * (median_lt / (long_term_mean[hot_mask] + 1e-12))



    # ------- 4) 计算短时能量矩阵 E(c, tframe) -------
    # 注意：short_time_energy 期望 fs 有意义。
    win_ms = args.energy_win_ms
    step_ms = args.energy_step_ms
    # 若 fs==1.0 且用户未真实提供采样率，则 interpret win_ms/step_ms as sample count (不常用)
    E, frame_times = short_time_energy(X_filt, fs=fs, win_ms=win_ms, step_ms=step_ms)
    # E shape = (C, n_frames)
    C, n_frames = E.shape
    print(f"[Step] 计算短时能量：窗口 {win_ms} ms，步长 {step_ms} ms => {n_frames} 帧")

    # ------- 5) 对能量矩阵做时间轴高斯平滑（可降低随机跳变） -------
    if args.gauss_smooth_sigma > 0:
        # 对每个通道沿时间做高斯滤波（仅平滑时间维的抖动）
        E_sm = np.zeros_like(E)
        for c in range(C):
            E_sm[c, :] = gaussian_filter1d(E[c, :], sigma=args.gauss_smooth_sigma, mode='reflect')
        E = E_sm

    # ------- 6) 归一化为概率（按时间帧归一化每一列） -------
    P = normalize_prob_per_time(E, eps=1e-12)  # shape (C, n_frames)
    # 也可得到全局归一化（用于可视化对比）
    P_global = E / (np.sum(E) + 1e-12)

    # ------- 6.5) 二次热通道屏蔽：检查最概然点占比 -------
    # args.hot_channel_post_threshold: 阈值，若某通道的 argmax 占比超过该比例，则认定为残留噪音
    # args.hot_channel_post_scale: 屏蔽或抑制强度（0=完全屏蔽, 0<x<1=按比例缩小概率）

    if hasattr(args, 'hot_channel_post_threshold') and args.hot_channel_post_threshold > 0:
        n_frames = P.shape[1]
        # 统计每个通道作为 argmax 的帧占比
        argmax_count = np.sum(np.argmax(P, axis=0)[None, :] == np.arange(P.shape[0])[:, None], axis=1)
        argmax_ratio = argmax_count / n_frames
        post_hot_mask = argmax_ratio > args.hot_channel_post_threshold
        if np.sum(post_hot_mask) > 0:
            print(f"[Message] 二次检测到 {np.sum(post_hot_mask)} 个占比过高通道，进行额外抑制")
            # scale 抑制概率矩阵
            for c in np.where(post_hot_mask)[0]:
                P[c, :] *= args.hot_channel_post_scale  # scale <1 会抑制
            # 重新归一化每列
            P = normalize_prob_per_time(P, eps=1e-12)


    # ------- 7) 每帧最可能通道（argmax）并做短时中值平滑以简化跳动 -------
    argmax_raw = np.argmax(P, axis=0)  # shape (n_frames,)
    # 中值滤波效果不良
    # # 中值滤波以体现“短时间不大幅度跳变”的先验（但不会改变概率矩阵）
    # med_k = args.median_smooth_frames if args.median_smooth_frames >= 1 else 1
    # if med_k > 1:
    #     argmax_smooth = median_filter(argmax_raw, size=med_k)
    # else:
    #     argmax_smooth = argmax_raw.copy()
    # 高斯平滑（仅平滑时间轴）
    sigma_gauss = args.gauss_smooth_sigma if args.gauss_smooth_sigma > 0 else 1.0
    argmax_gauss = gaussian_filter1d(argmax_raw.astype(float), sigma=sigma_gauss, mode='reflect')

    # 一阶卡尔曼平滑强化连续性
    alpha_kalman = 0.001  # 可调参数，越大越贴近原高斯平滑轨迹
    print(f"[Message] 对 argmax 结果做一阶卡尔曼平滑，alpha = {alpha_kalman}")
    argmax_smooth = np.zeros_like(argmax_gauss)
    argmax_smooth[0] = argmax_gauss[0]
    for t in range(1, len(argmax_gauss)):
        argmax_smooth[t] = alpha_kalman * argmax_gauss[t] + (1 - alpha_kalman) * argmax_smooth[t-1]

    # 计算置信度：top1 - top2 比值或差值
    top1_vals = np.max(P, axis=0)
    # 找到每列第二大的值
    sorted_idx = np.argsort(P, axis=0)  # 从小到大
    top2_vals = P[sorted_idx[-2, :], np.arange(n_frames)]
    confidence = (top1_vals - top2_vals)  # 差值，越大越确定

    # ------- 8) 导出结果 -------
    np.save(os.path.join(OUTDIR, "energy_matrix.npy"), E)
    np.save(os.path.join(OUTDIR, "prob_matrix.npy"), P)
    # CSV: 每列为一个时间帧，每行是通道概率（便于加载到 Unity 或其他工具）
    prob_df = pd.DataFrame(P, index=ch_cols)
    prob_df.columns = [f"t_{i}" for i in range(n_frames)]
    prob_df.to_csv(os.path.join(OUTDIR, "prob_matrix.csv"))

    # 输出 argmax 原始与平滑版本
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

    print("[Output] 已保存 energy_matrix.npy, prob_matrix.npy, prob_matrix.csv, argmax_channel.csv 到", OUTDIR)

    # ----- 9) 可视化绘图部分 -----
    # 配置绘图风格
    sns.set(style="whitegrid")
    # 设置 figure canvas 大小
    fig_w = 14
    fig_h = 8

    # 9.1 原始信号（部分通道或全部）热图（为了清晰只显示前若干通道或可全部）
    fig1, ax1 = plt.subplots(figsize=(fig_w, 3.5))
    # 为避免过长时间与过多通道把画面压扁，这里将信号按通道拼接成热图： rows=channels, cols=time_samples
    # 但原信号维度为 (T_samples, C). 为热图转置到 (C, T)
    vmax = np.percentile(np.abs(X_raw), 99)
    im1 = ax1.imshow(X_raw.T, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax1.set_title("Raw signal (channels x time samples)")
    ax1.set_ylabel("Channel index")
    ax1.set_xlabel("Sample index (time)")
    plt.colorbar(im1, ax=ax1, orientation='vertical', label='amplitude')
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUTDIR, "raw_signal_heatmap.png"), dpi=200)

    # 9.2 滤波后信号热图
    fig2, ax2 = plt.subplots(figsize=(fig_w, 3.5))
    vmax2 = np.percentile(np.abs(X_filt), 99)
    im2 = ax2.imshow(X_filt.T, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-vmax2, vmax=vmax2)
    ax2.set_title("Filtered signal (channels x time samples)")
    ax2.set_ylabel("Channel index")
    ax2.set_xlabel("Sample index (time)")
    plt.colorbar(im2, ax=ax2, orientation='vertical', label='amplitude')
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTDIR, "filtered_signal_heatmap.png"), dpi=200)

    # 9.3 概率矩阵（channels x frames）热图，并叠加 argmax 平滑曲线
    fig3, ax3 = plt.subplots(figsize=(fig_w, 6))
    # P is (C, n_frames). 显示时以 origin='lower' 使通道 0 在底部
    # 使用分量范围 [0, p99] 以增强可视化，但我们直接显示完整比例
    im3 = ax3.imshow(P, aspect='auto', origin='lower', cmap='viridis')
    ax3.set_title("Probability map P(channel, time-frame) (each column sums to 1)")
    ax3.set_ylabel("Channel index")
    ax3.set_xlabel("Frame index (time)")
    plt.colorbar(im3, ax=ax3, orientation='vertical', label='probability')

    # 叠加最可能通道（平滑后）
    ax3.plot(np.arange(n_frames), argmax_smooth, color='red', linewidth=1.5, label='most likely channel (smoothed)')
    ax3.scatter(np.arange(n_frames), argmax_raw, color='white', s=6, alpha=0.6, label='argmax raw', marker='.')
    ax3.legend(loc='upper right')
    fig3.tight_layout()
    fig3.savefig(os.path.join(OUTDIR, "prob_heatmap_with_argmax.png"), dpi=200)

    # 9.4 置信度随时间的曲线（可用于判断检测稳健性）
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
