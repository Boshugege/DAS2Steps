#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途（简洁）：
  - 读取 CSV（列名含 ch_ 开头）。将每列视为一通道时间序列。
  - 对每通道做 200 Hz 低通（可配采样率 fs），随后做自适应小波去噪：
      阈值计算：sigma = median(|detail|)/0.6745
                 uthresh = sigma * sqrt(2*ln(N)) / 1.5
  - 基于短时能量，用四状态 (Quiet, Trans, Voice, End) 做时间段拾取（简单状态机）。
  - 定位相邻通道对：要求“形状相似但波形相反”（全时段皮尔逊相关系数 < -0.7）
    并且每通道幅值分布呈双峰（直方图峰值检测）。
  - 输出：
      * filtered_data.npy (滤波+去噪后的时域矩阵)
      * states.csv (每帧状态：frame_idx, time_s, state)
      * detections.csv (满足条件的相邻通道对及度量)
依赖：
  numpy pandas scipy matplotlib pywt(optional)
用法示例：
  python d:\DAS2Steps\repeat.py --input D:\data\your.csv --outdir D:\data\out --fs 2000
"""
from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
import math
import matplotlib.pyplot as plt

# optional
try:
    import pywt
    _HAS_PYWT = True
except Exception:
    _HAS_PYWT = False

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--outdir", "-o", default="output")
    p.add_argument("--fs", type=float, default=2000.0, help="采样率 Hz（默认2000）")
    p.add_argument("--lowpass_cut", type=float, default=200.0, help="低通截止 Hz（默认200）")
    p.add_argument("--wavelet", default="db4")
    p.add_argument("--wavelet_level", type=int, default=3)
    p.add_argument("--energy_win_ms", type=float, default=50.0)
    p.add_argument("--energy_step_ms", type=float, default=10.0)
    p.add_argument("--corr_thresh", type=float, default=0.7, help="相关阈值绝对值（默认0.7）")
    p.add_argument("--peak_prominence_ratio", type=float, default=0.1, help="直方图峰 prominence 比例")
    return p.parse_args()

def lowpass_filter(X, fs, cutoff=200.0, order=4):
    nyq = 0.5 * fs
    if cutoff >= nyq:
        return X
    b, a = butter(order, cutoff/nyq, btype='low')
    Xf = np.zeros_like(X)
    for c in range(X.shape[1]):
        col = X[:, c].astype(np.float64)
        col = col - np.mean(col)
        try:
            Xf[:, c] = filtfilt(b, a, col, axis=0, method="pad")
        except Exception:
            Xf[:, c] = col
    return Xf

def wavelet_denoise_adaptive(X, wavelet="db4", level=3):
    if not _HAS_PYWT:
        raise RuntimeError("PyWavelets 未安装，无法做小波去噪（pip install pywt）")
    Xd = np.zeros_like(X)
    N = X.shape[0]
    for c in range(X.shape[1]):
        sig = X[:, c]
        coeffs = pywt.wavedec(sig, wavelet, level=level)
        # 使用最细尺度细节作为噪声估计
        detail = coeffs[-1] if len(coeffs) > 1 else np.array([])
        sigma = np.median(np.abs(detail)) / 0.6745 if detail.size > 0 else 0.0
        uthresh = 0.0
        if sigma > 0:
            uthresh = sigma * math.sqrt(2.0 * math.log(max(2, N))) / 1.5  # 除以 1.5 按用户要求
        coeffs_thr = [coeffs[0]] + [pywt.threshold(d, uthresh, mode='soft') for d in coeffs[1:]]
        rec = pywt.waverec(coeffs_thr, wavelet)
        Xd[:, c] = rec[:N]
    return Xd

def short_time_energy_matrix(X, fs, win_ms=50.0, step_ms=10.0):
    win = max(1, int(round(win_ms*1e-3*fs)))
    step = max(1, int(round(step_ms*1e-3*fs)))
    T, C = X.shape
    frames = []
    times = []
    for s in range(0, T - win + 1, step):
        seg = X[s:s+win, :]
        e = np.sum(seg**2, axis=0)
        frames.append(e)
        times.append((s + (win-1)/2.0)/fs)
    if len(frames) == 0:
        frames = [np.sum(X**2, axis=0)]
        times = [ (T-1)/2.0 / fs ]
    E = np.stack(frames, axis=1)    # (C, n_frames)
    return E, np.array(times)

def pick_states_from_energy(E, frame_times):
    # 简单四态检测：Quiet, Trans, Voice, End
    # 以每通道全局能量中位数+std 作为阈值参考，综合多个通道决定帧级状态
    # 我们用总能量（所有通道求和）作为判定基础
    total = np.sum(E, axis=0)  # (n_frames,)
    med = np.median(total)
    sd = np.std(total)
    low_thr = med + 0.2 * sd
    high_thr = med + 1.5 * sd
    states = []
    mode = "Quiet"
    for i in range(len(total)):
        val = total[i]
        if mode == "Quiet":
            if val > high_thr:
                mode = "Trans"
                states.append("Trans")
            elif val > low_thr:
                states.append("Trans")
            else:
                states.append("Quiet")
        elif mode == "Trans":
            if val > high_thr:
                mode = "Voice"
                states.append("Voice")
            elif val < low_thr:
                mode = "Quiet"
                states.append("Quiet")
            else:
                states.append("Trans")
        elif mode == "Voice":
            if val < low_thr:
                mode = "End"
                states.append("End")
            else:
                states.append("Voice")
        elif mode == "End":
            if val < low_thr:
                mode = "Quiet"
                states.append("Quiet")
            else:
                states.append("End")
        else:
            states.append("Quiet")
    return states

def is_bimodal(sig, bins=100, prominence_ratio=0.1):
    # 通过直方图找峰判断双峰
    h, edges = np.histogram(sig, bins=bins)
    if h.max() == 0:
        return False, []
    promin = h.max() * prominence_ratio
    peaks, props = find_peaks(h, prominence=promin)
    return (len(peaks) >= 2), peaks

def detect_adjacent_inverted_bimodal(X, corr_thresh=0.7, peak_prominence_ratio=0.1):
    T, C = X.shape
    results = []
    # 归一化每通道（零均值单位方差）用于相关计算
    Xn = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-12)
    for c in range(C-1):
        a = Xn[:, c]
        b = Xn[:, c+1]
        # pearson correlation
        corr = np.corrcoef(a, b)[0,1]
        # 形状相似且相反：corr < -corr_thresh
        if np.isfinite(corr) and corr < -abs(corr_thresh):
            # 检查双峰
            bim_a, peaks_a = is_bimodal(X[:, c], prominence_ratio=peak_prominence_ratio)
            bim_b, peaks_b = is_bimodal(X[:, c+1], prominence_ratio=peak_prominence_ratio)
            if bim_a or bim_b:
                results.append({
                    "ch": c,
                    "neighbor": c+1,
                    "corr": float(corr),
                    "bimodal_ch": bool(bim_a),
                    "bimodal_neighbor": bool(bim_b),
                    "peaks_ch": ",".join(map(str, peaks_a.tolist())) if len(peaks_a)>0 else "",
                    "peaks_neighbor": ",".join(map(str, peaks_b.tolist())) if len(peaks_b)>0 else ""
                })
    return results

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    cols = [c for c in df.columns if c.lower().startswith("ch_")]
    if len(cols) == 0:
        raise RuntimeError("未检测到 ch_ 列")
    X_raw = df[cols].values.astype(np.float64)

    # NaN 线性插值+前后填充
    if np.isnan(X_raw).any():
        Xdf = pd.DataFrame(X_raw)
        Xdf = Xdf.interpolate(axis=0).fillna(method='bfill').fillna(method='ffill').fillna(0.0)
        X_raw = Xdf.values

    fs = float(args.fs) if args.fs else 2000.0

    # 1) 200Hz 低通
    X_lp = lowpass_filter(X_raw, fs=fs, cutoff=args.lowpass_cut)

    # 2) 自适应小波去噪（若可用）
    if _HAS_PYWT:
        X_deno = wavelet_denoise_adaptive(X_lp, wavelet=args.wavelet, level=args.wavelet_level)
    else:
        X_deno = X_lp  # 未安装 pywt 则跳过

    # 保存滤波后数据
    np.save(os.path.join(args.outdir, "filtered_data.npy"), X_deno)
    pd.DataFrame(X_deno, columns=cols).to_csv(os.path.join(args.outdir, "filtered_data.csv"), index=False)

    # 3) 计算短时能量并用四态拾取
    E, frame_times = short_time_energy_matrix(X_deno, fs=fs, win_ms=args.energy_win_ms, step_ms=args.energy_step_ms)
    states = pick_states_from_energy(E, frame_times)
    df_states = pd.DataFrame({
        "frame_idx": np.arange(len(frame_times)),
        "time_s": frame_times,
        "state": states
    })
    df_states.to_csv(os.path.join(args.outdir, "states.csv"), index=False)

    # 4) 检测相邻通道对（负相关 + 双峰）
    detections = detect_adjacent_inverted_bimodal(X_deno, corr_thresh=args.corr_thresh, peak_prominence_ratio=args.peak_prominence_ratio)
    df_det = pd.DataFrame(detections)
    df_det.to_csv(os.path.join(args.outdir, "detections.csv"), index=False)

    # --- 可视化：全通道全时间波形图，并在上面标出检测到的相邻反向双峰通道对 ---
    try:
        T, C = X_deno.shape
        t = np.arange(T) / fs
        # 叠加绘图：按通道垂直偏移
        peak = np.nanmax(np.abs(X_deno)) if np.isfinite(np.nanmax(np.abs(X_deno))) else 1.0
        offset = peak * 2.0 if peak > 0 else 1.0
        fig_h = max(4.0, 0.25 * C)
        fig, ax = plt.subplots(figsize=(12, fig_h))
        for i in range(C):
            ax.plot(t, X_deno[:, i] + i * offset, color='k', linewidth=0.4)
        # 标出检测到的通道对：用半透明条带跨时域标注对应通道区间
        for det in detections:
            ch = int(det.get("ch", 0))
            nb = int(det.get("neighbor", ch+1))
            y0 = ch * offset - 0.5 * offset
            y1 = nb * offset + 0.5 * offset
            ax.axhspan(y0, y1, color='red', alpha=0.12)
            ax.text(t[0], (y0 + y1) / 2.0, f"det: {ch}-{nb}", color='red', fontsize=8, va='center')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Channels (stacked)")
        ax.set_yticks([i * offset for i in range(0, C, max(1, C//20))])
        ax.set_yticklabels([f"ch_{i}" for i in range(0, C, max(1, C//20))])
        ax.set_title("All channels (stacked) with detected adjacent inverted-bimodal pairs highlighted")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        png_out = os.path.join(args.outdir, "all_channels_with_detections.png")
        fig.savefig(png_out, dpi=150)
        print(f"[info] Saved overview plot: {png_out}")
        # 在支持交互的环境中弹出窗口供放缩查看
        try:
            plt.show()
        except Exception:
            # 若环境不支持交互则忽略
            pass
        plt.close(fig)
    except Exception as e:
        print(f"[warn] 无法生成全通道可视化: {e}")

    # 简要打印结果
    print(f"[info] saved filtered_data.npy, filtered_data.csv, states.csv, detections.csv to {args.outdir}")
    print(f"[info] detections_count = {len(detections)}")
    if len(detections) > 0:
        print(df_det.head().to_string(index=False))

if __name__ == "__main__":
    main()