# -*- coding: utf-8 -*-
"""
Stream TDMS data and send JSON packets over a network port.

Each JSON packet includes total channel count, timestamp, and per-channel signal.
"""

import argparse
import json
import socket
import sys
import time
from datetime import datetime, timedelta, timezone
import numpy as np
from scipy.signal import butter, sosfilt, lfilter, firwin
from nptdms import TdmsFile
import pywt


def _iter_all_channels(tdms_file):
    channels = []
    for group in tdms_file.groups():
        for channel in group.channels():
            channels.append(channel)
    return channels


def _pick_channels(all_channels, start_idx, end_idx):
    if end_idx is None:
        return all_channels[start_idx:]
    return all_channels[start_idx : end_idx + 1]


def _send_packet(sock, protocol, addr, payload):
    data = (json.dumps(payload, ensure_ascii=True) + "\n").encode("utf-8")
    if protocol == "udp":
        sock.sendto(data, addr)
    else:
        sock.sendall(data)


def _parse_base_time(start_time_iso):
    if not start_time_iso:
        return None
    iso_value = start_time_iso
    if iso_value.endswith("Z"):
        iso_value = iso_value[:-1] + "+00:00"
    base_time = datetime.fromisoformat(iso_value)
    if base_time.tzinfo is None:
        base_time = base_time.replace(tzinfo=timezone.utc)
    return base_time


def _open_socket(protocol, host, port):
    addr = (host, port)
    if protocol == "udp":
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    else:
        sock = socket.create_connection(addr)
    return sock, addr


def _prepare_bandpass(sample_rate, bp_low, bp_high, bp_order, channels):
    if bp_low is None or bp_high is None:
        return None, None
    nyq = 0.5 * sample_rate
    low = bp_low / nyq
    high = bp_high / nyq
    if not (0.0 < low < high < 1.0):
        return None, None
    sos = butter(bp_order, [low, high], btype="bandpass", output="sos")
    zi = [np.zeros((sos.shape[0], 2), dtype=float) for _ in channels]
    return sos, zi


def _prepare_fir_bandpass(sample_rate, bp_low, bp_high, delay_ms, taps, channels):
    if bp_low is None or bp_high is None:
        return None, None, None
    nyq = 0.5 * sample_rate
    if not (0.0 < bp_low < bp_high < nyq):
        return None, None, None

    if taps is None:
        delay_s = (delay_ms or 0.0) / 1000.0
        max_taps = int(round(2.0 * delay_s * sample_rate)) + 1
        taps = max(31, max_taps)
    if taps % 2 == 0:
        taps += 1

    coeffs = firwin(
        taps,
        [bp_low, bp_high],
        pass_zero=False,
        fs=sample_rate,
    )
    zi = [np.zeros(taps - 1, dtype=float) for _ in channels]
    return coeffs, zi, taps


def _wavelet_denoise(data, wavelet='db4', level=3, mode='soft'):
    """
    对单通道信号进行小波降噪
    
    参数:
        data: 1D 信号数组
        wavelet: 小波类型 (db2/db4/sym4)
        level: 分解层数
        mode: 阈值模式 ('soft' 软阈值 / 'hard' 硬阈值)
    
    返回:
        降噪后的信号
    """
    # 小波分解
    coeffs = pywt.wavedec(data, wavelet, level=level)
    
    # 估计噪声标准差 (使用最高频细节系数的 MAD)
    # sigma = MAD / 0.6745 (假设高斯噪声)
    detail_1 = coeffs[-1]  # 最高频细节
    sigma = np.median(np.abs(detail_1)) / 0.6745
    
    # 计算通用阈值 (VisuShrink): threshold = sigma * sqrt(2 * log(n))
    n = len(data)
    threshold = sigma * np.sqrt(2 * np.log(n)) if n > 1 else 0
    
    # 对细节系数应用阈值
    denoised_coeffs = [coeffs[0]]  # 保留近似系数不变
    for i in range(1, len(coeffs)):
        if mode == 'soft':
            # 软阈值: sign(x) * max(|x| - threshold, 0)
            denoised = pywt.threshold(coeffs[i], threshold, mode='soft')
        else:
            # 硬阈值: x if |x| > threshold else 0
            denoised = pywt.threshold(coeffs[i], threshold, mode='hard')
        denoised_coeffs.append(denoised)
    
    # 小波重构
    return pywt.waverec(denoised_coeffs, wavelet)[:len(data)]


def _iter_chunks(
    channels,
    start_sample,
    end_sample,
    chunk_samples,
    sos=None,
    zi=None,
    fir_coeffs=None,
    fir_zi=None,
    denoise_wavelet=None,
    denoise_level=3,
    denoise_mode='soft',
):
    """
    迭代读取数据块，支持带通滤波和小波降噪
    
    参数:
        denoise_wavelet: 降噪小波类型，None 表示不降噪
        denoise_level: 降噪分解层数
        denoise_mode: 'soft' 或 'hard' 阈值
    """
    for chunk_start in range(start_sample, end_sample, chunk_samples):
        chunk_end = min(chunk_start + chunk_samples, end_sample)
        chunk_data = [np.asarray(ch[chunk_start:chunk_end], dtype=float) for ch in channels]
        
        # 第一层：带通滤波
        if sos is not None:
            filtered = []
            for idx, data in enumerate(chunk_data):
                data, zi[idx] = sosfilt(sos, data, zi=zi[idx])
                filtered.append(data)
            chunk_data = filtered
        elif fir_coeffs is not None:
            filtered = []
            for idx, data in enumerate(chunk_data):
                data, fir_zi[idx] = lfilter(fir_coeffs, [1.0], data, zi=fir_zi[idx])
                filtered.append(data)
            chunk_data = filtered
        
        # 第二层：小波降噪
        if denoise_wavelet is not None:
            denoised = []
            for data in chunk_data:
                data = _wavelet_denoise(data, wavelet=denoise_wavelet, 
                                        level=denoise_level, mode=denoise_mode)
                denoised.append(data)
            chunk_data = denoised
        
        yield chunk_start, chunk_data


def _iter_decimated_samples(chunk_start, chunk_data, decimate_factor):
    chunk_len = len(chunk_data[0])
    for i in range(0, chunk_len, decimate_factor):
        sample_index = chunk_start + i
        sample_vector = [float(ch[i]) for ch in chunk_data]
        yield sample_index, sample_vector


class RealtimeWaveletDetector:
    """实时小波脚步检测器 - 滑动窗口小波变换"""
    
    def __init__(self, n_channels, wavelet='db4', level=3, 
                 threshold_factor=3.0, fs=2000):
        self.wavelet_obj = pywt.Wavelet(wavelet)
        self.wavelet_name = wavelet
        self.level = level
        self.threshold_factor = threshold_factor
        self.fs = fs
        self.n_channels = n_channels
        
        # 滤波器长度决定所需缓冲区大小
        self.filter_len = self.wavelet_obj.dec_len
        self.buffer_size = self.filter_len * (2 ** level)
        
        # 每通道的滑动缓冲区
        self.buffers = np.zeros((n_channels, self.buffer_size), dtype=float)
        self.buf_idx = 0
        self.samples_collected = 0
        
        # 自适应阈值的递归估计 (每通道)
        self.energy_baseline = np.ones(n_channels, dtype=float) * 1e-6
        self.energy_mad = np.ones(n_channels, dtype=float) * 1e-6
        self.alpha = 0.02  # 阈值更新速率
        
    def update(self, sample_vector):
        """
        处理新采样点，返回检测结果
        
        sample_vector: (n_channels,) 当前采样值
        返回: (energies, ratio, ready)
            - energies: 各通道小波能量 (n_channels,)
            - ratio: 能量/阈值比值 (n_channels,)
            - ready: 本次是否有有效输出
        """
        sample_arr = np.asarray(sample_vector, dtype=float)
        
        # 更新缓冲区
        self.buffers[:, self.buf_idx] = sample_arr
        self.buf_idx = (self.buf_idx + 1) % self.buffer_size
        self.samples_collected += 1
        
        # 缓冲区未满时不计算
        if self.samples_collected < self.buffer_size:
            return np.zeros(self.n_channels), np.zeros(self.n_channels), False
        
        # 只在缓冲区循环一周时计算 (每 buffer_size 个样本计算一次)
        if self.buf_idx != 0:
            return np.zeros(self.n_channels), np.zeros(self.n_channels), False
        
        # 对每个通道计算小波细节系数能量
        energies = np.zeros(self.n_channels, dtype=float)
        for ch in range(self.n_channels):
            # 重排缓冲区为时序正确的顺序
            buf = np.roll(self.buffers[ch], -self.buf_idx)
            
            # 小波分解
            coeffs = pywt.wavedec(buf, self.wavelet_obj, level=self.level)
            
            # 累加细节系数能量 (level 1-2 对应脚步的高频成分)
            for d in range(1, min(3, self.level + 1)):
                energies[ch] += np.sum(coeffs[d] ** 2)
        
        # 更新自适应阈值 (指数移动平均)
        self.energy_baseline = (1 - self.alpha) * self.energy_baseline + self.alpha * energies
        deviation = np.abs(energies - self.energy_baseline)
        self.energy_mad = (1 - self.alpha) * self.energy_mad + self.alpha * deviation
        
        # 计算相对强度比值
        threshold = self.energy_baseline + self.threshold_factor * self.energy_mad
        ratio = energies / np.maximum(threshold, 1e-9)
        
        return energies, ratio, True


def _smooth_ratio(ratio, neighbor_radius):
    if neighbor_radius <= 0:
        return ratio
    kernel = np.ones(2 * neighbor_radius + 1, dtype=float)
    return np.convolve(ratio, kernel, mode="same") / kernel.size


def _build_event_payload(
    smoothed,
    neighbor_radius,
    sta_lta_threshold,
    last_event_time,
    timestamp,
    event_refractory_s,
    total_channels,
    base_time,
):
    max_idx = int(np.argmax(smoothed))
    max_score = float(smoothed[max_idx])
    should_fire = max_score >= sta_lta_threshold
    if not should_fire:
        return None, last_event_time
    if last_event_time is not None and (timestamp - last_event_time) < event_refractory_s:
        return None, last_event_time
    left = max(0, max_idx - neighbor_radius)
    right = min(total_channels - 1, max_idx + neighbor_radius)
    weights = smoothed[left : right + 1]
    if np.sum(weights) > 0:
        centroid = float(np.sum(weights * np.arange(left, right + 1)) / np.sum(weights))
    else:
        centroid = float(max_idx)
    event_payload = {
        "packet_type": "event",
        "timestamp": timestamp,
        "channel_index": centroid,
        "channel_span": [left, right],
        "score": max_score,
    }
    if base_time is not None:
        event_payload["timestamp_iso"] = (base_time + timedelta(seconds=timestamp)).isoformat()
    return event_payload, timestamp


def _build_signal_payload(total_channels, timestamp, sample_rate, sample_vector, base_time):
    payload = {
        "packet_type": "signal",
        "total_channels": total_channels,
        "timestamp": timestamp,
        "sample_rate": sample_rate,
        "sample_count": 1,
        "signals": sample_vector,
    }
    if base_time is not None:
        payload["timestamp_iso"] = (base_time + timedelta(seconds=timestamp)).isoformat()
    return payload


def stream_tdms(
    tdms_path,
    host,
    port,
    protocol,
    sample_rate,
    channel_start,
    channel_end,
    start_sample,
    max_samples,
    start_time_iso,
    bp_low,
    bp_high,
    bp_order,
    filter_kind,
    fir_delay_ms,
    fir_taps,
    denoise_wavelet,
    denoise_level,
    denoise_mode,
    wavelet_type,
    wavelet_level,
    wavelet_threshold,
    event_refractory_s,
    neighbor_radius,
):
    # Read: load TDMS and slice channels/samples.
    tdms = TdmsFile.read(tdms_path)
    all_channels = _iter_all_channels(tdms)
    channels = _pick_channels(all_channels, channel_start, channel_end)
    # Fixed large read size to reduce TDMS I/O overhead.
    chunk_samples = 5000

    total_channels = len(channels)
    total_samples = len(channels[0])

    end_sample = total_samples
    if max_samples is not None:
        end_sample = min(total_samples, start_sample + max_samples)

    base_time = _parse_base_time(start_time_iso)

    # Output: network setup and pacing bookkeeping.
    sock, addr = _open_socket(protocol, host, port)
    seconds_per_sample = 1.0 / sample_rate
    decimate_factor = 10
    start_perf = time.perf_counter()
    next_send = start_perf
    last_report = start_perf
    last_sent_timestamp = None
    packets_since_report = 0
    target_packets_per_sec = sample_rate / decimate_factor
    last_lag_ms = 0.0
    max_lag_ms = None

    # Process: filters and wavelet detector state.
    sos, zi = None, None
    fir_coeffs, fir_zi, fir_numtaps = None, None, None
    if filter_kind == "fir":
        fir_coeffs, fir_zi, fir_numtaps = _prepare_fir_bandpass(
            sample_rate, bp_low, bp_high, fir_delay_ms, fir_taps, channels
        )
        if fir_coeffs is not None:
            delay_samples = (fir_numtaps - 1) / 2.0
            delay_ms = 1000.0 * delay_samples / sample_rate
            print(f"[Filter] FIR bandpass taps={fir_numtaps}, delay≈{delay_ms:.1f} ms")
    else:
        sos, zi = _prepare_bandpass(sample_rate, bp_low, bp_high, bp_order, channels)
        if sos is not None:
            print(f"[Filter] IIR bandpass order={bp_order}")
    
    # 打印降噪信息
    if denoise_wavelet:
        print(f"[Denoise] Using {denoise_wavelet} wavelet denoising, "
              f"level={denoise_level}, mode={denoise_mode}")
    
    # 初始化小波检测器
    wavelet_detector = RealtimeWaveletDetector(
        n_channels=total_channels,
        wavelet=wavelet_type,
        level=wavelet_level,
        threshold_factor=wavelet_threshold,
        fs=sample_rate
    )
    print(f"[Wavelet] Using {wavelet_type} wavelet, level={wavelet_level}, "
          f"buffer_size={wavelet_detector.buffer_size} samples")
    
    last_event_time = None
    last_valid_ratio = np.zeros(total_channels, dtype=float)

    try:
        for chunk_start, chunk_data in _iter_chunks(
            channels,
            start_sample,
            end_sample,
            chunk_samples,
            sos=sos,
            zi=zi,
            fir_coeffs=fir_coeffs,
            fir_zi=fir_zi,
            denoise_wavelet=denoise_wavelet, denoise_level=denoise_level,
            denoise_mode=denoise_mode
        ):
            for sample_index, sample_vector in _iter_decimated_samples(
                chunk_start, chunk_data, decimate_factor
            ):
                timestamp = sample_index * seconds_per_sample
                
                # 小波检测
                energies, ratio, ready = wavelet_detector.update(sample_vector)
                if ready:
                    last_valid_ratio = ratio
                    smoothed = _smooth_ratio(ratio, neighbor_radius)
                else:
                    # 缓冲区未满时使用上次的比值
                    smoothed = _smooth_ratio(last_valid_ratio, neighbor_radius)
                event_payload, last_event_time = _build_event_payload(
                    smoothed,
                    neighbor_radius,
                    1.0,  # 小波检测阈值已在检测器内部处理，这里用 1.0
                    last_event_time,
                    timestamp,
                    event_refractory_s,
                    total_channels,
                    base_time,
                )
                if event_payload is not None:
                    _send_packet(sock, protocol, addr, event_payload)

                payload = _build_signal_payload(
                    total_channels, timestamp, sample_rate, sample_vector, base_time
                )
                _send_packet(sock, protocol, addr, payload)
                packets_since_report += 1

                # Pace according to the timestamp of the last sample in this packet.
                last_sample_index = sample_index
                last_sent_timestamp = last_sample_index * seconds_per_sample
                next_send += decimate_factor * seconds_per_sample
                sleep_for = next_send - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                last_lag_ms = (time.perf_counter() - next_send) * 1000.0
                if max_lag_ms is None:
                    max_lag_ms = last_lag_ms
                elif last_lag_ms > max_lag_ms:
                    next_send = time.perf_counter()
                    last_lag_ms = 0.0

                now = time.perf_counter()
                if last_sent_timestamp is not None and (now - last_report) >= 1.0:
                    elapsed = now - last_report
                    rate = packets_since_report / elapsed if elapsed > 0 else 0.0
                    print(
                        f"sent up to timestamp {last_sent_timestamp:.3f}s "
                        f"(sample_index {last_sample_index}), "
                        f"packets {rate:.1f}/s "
                        f"(target {target_packets_per_sec:.1f}/s), "
                        f"lag {last_lag_ms:.1f} ms"
                    )
                    last_report = now
                    packets_since_report = 0
    finally:
        sock.close()


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Stream TDMS data and send JSON packets over UDP/TCP."
    )
    parser.add_argument("tdms_path", help="Path to .tdms file")
    parser.add_argument("--host", default="127.0.0.1", help="Target host")
    parser.add_argument("--port", type=int, default=9000, help="Target port")
    parser.add_argument(
        "--protocol",
        choices=["udp", "tcp"],
        default="udp",
        help="Network protocol",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        required=True,
        help="Sampling rate in Hz (used for pacing and timestamps)",
    )
    parser.add_argument(
        "--channel-start",
        type=int,
        default=0,
        help="Start channel index (inclusive)",
    )
    parser.add_argument(
        "--channel-end",
        type=int,
        default=None,
        help="End channel index (inclusive)",
    )
    parser.add_argument(
        "--start-sample",
        type=int,
        default=0,
        help="Start sample index",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to stream",
    )
    parser.add_argument(
        "--start-time-iso",
        default=None,
        help="Optional ISO-8601 base time for timestamp_iso field",
    )
    parser.add_argument(
        "--bp-low",
        type=float,
        default=10.0,
        help="Bandpass low cutoff (Hz)",
    )
    parser.add_argument(
        "--bp-high",
        type=float,
        default=60.0,
        help="Bandpass high cutoff (Hz)",
    )
    parser.add_argument(
        "--bp-order",
        type=int,
        default=2,
        help="Butterworth bandpass order (IIR only)",
    )
    parser.add_argument(
        "--filter-kind",
        choices=["fir", "iir"],
        default="fir",
        help="Bandpass filter type: fir (linear-phase) or iir (Butterworth)",
    )
    parser.add_argument(
        "--fir-delay-ms",
        type=float,
        default=200.0,
        help="Target FIR group delay in ms (used to size taps)",
    )
    parser.add_argument(
        "--fir-taps",
        type=int,
        default=None,
        help="Override FIR tap count (odd number). If set, --fir-delay-ms is ignored.",
    )
    parser.add_argument(
        "--denoise-wavelet",
        type=str,
        choices=["db2", "db4", "db8", "sym4", "sym8", "coif3"],
        default=None,
        help="Wavelet type for denoising (None=disabled, db4/sym4/etc)",
    )
    parser.add_argument(
        "--denoise-level",
        type=int,
        default=3,
        help="Wavelet decomposition level for denoising (default: 3)",
    )
    parser.add_argument(
        "--denoise-mode",
        type=str,
        choices=["soft", "hard"],
        default="soft",
        help="Threshold mode for denoising: soft (default) or hard",
    )
    parser.add_argument(
        "--wavelet-type",
        type=str,
        choices=["db2", "db4", "sym4"],
        default="db4",
        help="Wavelet type for detection (db2/db4/sym4, default: db4)",
    )
    parser.add_argument(
        "--wavelet-level",
        type=int,
        default=3,
        help="Wavelet decomposition level (default: 3)",
    )
    parser.add_argument(
        "--wavelet-threshold",
        type=float,
        default=3.0,
        help="Wavelet detection threshold factor (default: 3.0)",
    )
    parser.add_argument(
        "--event-refractory-s",
        type=float,
        default=0.2,
        help="Minimum time between events (seconds)",
    )
    parser.add_argument(
        "--neighbor-radius",
        type=int,
        default=2,
        help="Neighbor channel radius for event centroid",
    )
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    stream_tdms(
        tdms_path=args.tdms_path,
        host=args.host,
        port=args.port,
        protocol=args.protocol,
        sample_rate=args.sample_rate,
        channel_start=args.channel_start,
        channel_end=args.channel_end,
        start_sample=args.start_sample,
        max_samples=args.max_samples,
        start_time_iso=args.start_time_iso,
        bp_low=args.bp_low,
        bp_high=args.bp_high,
        bp_order=args.bp_order,
        filter_kind=args.filter_kind,
        fir_delay_ms=args.fir_delay_ms,
        fir_taps=args.fir_taps,
        denoise_wavelet=args.denoise_wavelet,
        denoise_level=args.denoise_level,
        denoise_mode=args.denoise_mode,
        wavelet_type=args.wavelet_type,
        wavelet_level=args.wavelet_level,
        wavelet_threshold=args.wavelet_threshold,
        event_refractory_s=args.event_refractory_s,
        neighbor_radius=args.neighbor_radius,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
