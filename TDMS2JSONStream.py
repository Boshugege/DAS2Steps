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
from scipy.signal import butter, sosfilt
from nptdms import TdmsFile


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


def _iter_chunks(channels, start_sample, end_sample, chunk_samples, sos=None, zi=None):
    for chunk_start in range(start_sample, end_sample, chunk_samples):
        chunk_end = min(chunk_start + chunk_samples, end_sample)
        chunk_data = [np.asarray(ch[chunk_start:chunk_end], dtype=float) for ch in channels]
        if sos is not None:
            filtered = []
            for idx, data in enumerate(chunk_data):
                data, zi[idx] = sosfilt(sos, data, zi=zi[idx])
                filtered.append(data)
            chunk_data = filtered
        yield chunk_start, chunk_data


def _iter_decimated_samples(chunk_start, chunk_data, decimate_factor):
    chunk_len = len(chunk_data[0])
    for i in range(0, chunk_len, decimate_factor):
        sample_index = chunk_start + i
        sample_vector = [float(ch[i]) for ch in chunk_data]
        yield sample_index, sample_vector


def _update_sta_lta(sta, lta, sta_alpha, lta_alpha, sample_vector):
    energy = np.abs(sample_vector)
    sta = (1.0 - sta_alpha) * sta + sta_alpha * energy
    lta = (1.0 - lta_alpha) * lta + lta_alpha * energy
    ratio = sta / np.maximum(lta, 1e-9)
    return sta, lta, ratio


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
    sta_window_s,
    lta_window_s,
    sta_lta_threshold,
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

    # Process: filters and STA/LTA state.
    sos, zi = _prepare_bandpass(sample_rate, bp_low, bp_high, bp_order, channels)
    dt = decimate_factor * seconds_per_sample
    sta_alpha = min(1.0, dt / sta_window_s) if sta_window_s > 0 else 1.0
    lta_alpha = min(1.0, dt / lta_window_s) if lta_window_s > 0 else 1.0
    sta = np.zeros(total_channels, dtype=float)
    lta = np.full(total_channels, 1e-6, dtype=float)
    last_event_time = None

    try:
        for chunk_start, chunk_data in _iter_chunks(
            channels, start_sample, end_sample, chunk_samples, sos=sos, zi=zi
        ):
            for sample_index, sample_vector in _iter_decimated_samples(
                chunk_start, chunk_data, decimate_factor
            ):
                timestamp = sample_index * seconds_per_sample
                sta, lta, ratio = _update_sta_lta(sta, lta, sta_alpha, lta_alpha, sample_vector)
                smoothed = _smooth_ratio(ratio, neighbor_radius)
                event_payload, last_event_time = _build_event_payload(
                    smoothed,
                    neighbor_radius,
                    sta_lta_threshold,
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
    parser.add_argument("--port", type=int, required=True, help="Target port")
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
        default=None,
        help="Butterworth bandpass low cutoff (Hz)",
    )
    parser.add_argument(
        "--bp-high",
        type=float,
        default=None,
        help="Butterworth bandpass high cutoff (Hz)",
    )
    parser.add_argument(
        "--bp-order",
        type=int,
        default=2,
        help="Butterworth bandpass order",
    )
    parser.add_argument(
        "--sta-window-s",
        type=float,
        default=0.03,
        help="STA window in seconds",
    )
    parser.add_argument(
        "--lta-window-s",
        type=float,
        default=0.5,
        help="LTA window in seconds",
    )
    parser.add_argument(
        "--sta-lta-threshold",
        type=float,
        default=2.0,
        help="STA/LTA trigger threshold",
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
        sta_window_s=args.sta_window_s,
        lta_window_s=args.lta_window_s,
        sta_lta_threshold=args.sta_lta_threshold,
        event_refractory_s=args.event_refractory_s,
        neighbor_radius=args.neighbor_radius,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
