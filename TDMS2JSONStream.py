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
):
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

    if start_time_iso:
        iso_value = start_time_iso
        if iso_value.endswith("Z"):
            iso_value = iso_value[:-1] + "+00:00"
        base_time = datetime.fromisoformat(iso_value)
        if base_time.tzinfo is None:
            base_time = base_time.replace(tzinfo=timezone.utc)
    else:
        base_time = None

    addr = (host, port)
    if protocol == "udp":
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    else:
        sock = socket.create_connection(addr)
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

    try:
        for chunk_start in range(start_sample, end_sample, chunk_samples):
            chunk_end = min(chunk_start + chunk_samples, end_sample)
            # Read chunk data per channel.
            chunk_data = [ch[chunk_start:chunk_end] for ch in channels]
            chunk_len = len(chunk_data[0])
            for i in range(0, chunk_len, decimate_factor):
                sample_index = chunk_start + i
                timestamp = sample_index * seconds_per_sample
                signals_pack = [[float(ch[i]) for ch in chunk_data]]

                payload = {
                    "total_channels": total_channels,
                    "timestamp": timestamp,
                    "sample_rate": sample_rate,
                    "sample_count": 1,
                    "signals": signals_pack[0],
                }
                if base_time is not None:
                    payload["timestamp_iso"] = (
                        base_time + timedelta(seconds=timestamp)
                    ).isoformat()

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
    )


if __name__ == "__main__":
    main(sys.argv[1:])
