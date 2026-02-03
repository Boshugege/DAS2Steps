# -*- coding: utf-8 -*-
"""
Receive newline-delimited JSON packets from TDMS2JSONStream and plot in real time.
"""

import argparse
import json
import queue
import socket
import threading
import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Receive JSON stream and plot signals in real time."
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=9000, help="Bind port")
    parser.add_argument(
        "--protocol",
        choices=["udp", "tcp"],
        default="udp",
        help="Network protocol",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=5.0,
        help="Plot window in seconds",
    )
    parser.add_argument(
        "--max-channels",
        type=int,
        default=0,
        help="Max channels to plot (0 for all)",
    )
    parser.add_argument(
        "--refresh-hz",
        type=float,
        default=20.0,
        help="Plot refresh rate",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=None,
        help="Fallback sample rate if packets omit it",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=2000,
        help="Max events to keep in memory",
    )
    parser.add_argument(
        "--show-events",
        action="store_true",
        default=False,
        help="Display event markers",
    )
    return parser.parse_args(argv)


class StreamBuffer:
    def __init__(self, window_seconds, max_channels):
        self.window_seconds = window_seconds
        self.max_channels = max_channels
        self.sample_rate = None
        self.total_channels = None
        self.plot_channels = None
        self.window_samples = None
        self.time_deque = None
        self.channel_deques = None
        self.last_timestamp = None

    def _init_buffers(self, total_channels, sample_rate):
        self.sample_rate = sample_rate
        self.total_channels = total_channels
        if self.max_channels is None or self.max_channels <= 0:
            self.plot_channels = total_channels
        else:
            self.plot_channels = min(total_channels, self.max_channels)
        self.window_samples = None
        self.time_deque = deque()
        self.channel_deques = [
            deque() for _ in range(self.plot_channels)
        ]
        self.last_timestamp = None

    def _ensure_buffers(self, total_channels, sample_rate):
        if self.time_deque is None:
            self._init_buffers(total_channels, sample_rate)
            return
        if total_channels != self.total_channels:
            self._init_buffers(total_channels, sample_rate)
            return
        if sample_rate != self.sample_rate:
            self._init_buffers(total_channels, sample_rate)
            return

    def append_sample(self, timestamp, values):
        if self.time_deque is None:
            return
        self.time_deque.append(timestamp)
        limit = min(self.plot_channels, len(values))
        for idx in range(limit):
            self.channel_deques[idx].append(values[idx])
        self.last_timestamp = timestamp

    def update_from_packet(self, payload, fallback_sample_rate):
        total_channels = int(payload.get("total_channels", 0))
        signals = payload.get("signals")
        if not total_channels or signals is None:
            return
        sample_rate = payload.get("sample_rate") or fallback_sample_rate
        if sample_rate is None:
            return
        self._ensure_buffers(total_channels, float(sample_rate))

        sample_count = int(payload.get("sample_count", 1))
        timestamp = payload.get("timestamp")
        if timestamp is None:
            timestamp = time.time()

        # Handle sample_count == 1 (common case).
        if sample_count <= 1 or not signals:
            values = signals
            if isinstance(values, list) and values:
                self.append_sample(float(timestamp), values)
            return

        # Handle batched samples if signals is list of lists.
        if isinstance(signals, list) and signals and isinstance(signals[0], list):
            step = 1.0 / self.sample_rate if self.sample_rate else 0.0
            base_ts = float(timestamp)
            for offset, values in enumerate(signals):
                self.append_sample(base_ts + offset * step, values)


class Receiver(threading.Thread):
    def __init__(self, host, port, protocol, out_queue):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.protocol = protocol
        self.out_queue = out_queue
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def _run_udp(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.host, self.port))
        sock.settimeout(0.5)
        try:
            while not self._stop_event.is_set():
                try:
                    data, _ = sock.recvfrom(65535)
                except socket.timeout:
                    continue
                try:
                    payload = json.loads(data.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                self.out_queue.put(payload)
        finally:
            sock.close()

    def _run_tcp(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(1)
        server.settimeout(0.5)
        conn = None
        buffer = b""
        try:
            while not self._stop_event.is_set():
                if conn is None:
                    try:
                        conn, _ = server.accept()
                        conn.settimeout(0.5)
                    except socket.timeout:
                        continue
                try:
                    chunk = conn.recv(65535)
                except socket.timeout:
                    continue
                if not chunk:
                    conn.close()
                    conn = None
                    buffer = b""
                    continue
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line:
                        continue
                    try:
                        payload = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError:
                        continue
                    self.out_queue.put(payload)
        finally:
            if conn is not None:
                conn.close()
            server.close()

    def run(self):
        if self.protocol == "udp":
            self._run_udp()
        else:
            self._run_tcp()


def main(argv=None):
    args = _parse_args(argv)
    packet_queue = queue.Queue()
    buffer = StreamBuffer(args.window_seconds, args.max_channels)
    event_times = deque(maxlen=args.max_events)
    event_channels = deque(maxlen=args.max_events)

    receiver = Receiver(args.host, args.port, args.protocol, packet_queue)
    receiver.start()

    fig, ax = plt.subplots()
    ax.set_title("TDMS JSON Stream")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")

    image = ax.imshow(
        np.zeros((1, 1), dtype=float),
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        cmap="seismic",
    )
    event_scatter = ax.scatter([], [], s=30, c="black", marker="x")
    event_scatter.set_visible(args.show_events)
    last_channel_count = None

    def _refresh(_frame):
        updated = False
        while True:
            try:
                payload = packet_queue.get_nowait()
            except queue.Empty:
                break
            if payload.get("packet_type") == "event":
                timestamp = payload.get("timestamp")
                channel_index = payload.get("channel_index")
                if timestamp is not None and channel_index is not None:
                    event_times.append(float(timestamp))
                    event_channels.append(float(channel_index))
            else:
                buffer.update_from_packet(payload, args.sample_rate)
            updated = True

        if not updated or buffer.time_deque is None:
            return (image,)

        times = list(buffer.time_deque)
        if not times:
            return (image,)

        data = np.array([list(channel) for channel in buffer.channel_deques], dtype=float)

        x_max = times[-1]
        x_min = times[0]
        image.set_data(data)
        image.set_extent((x_min, x_max, buffer.plot_channels - 0.5, -0.5))

        v_min = np.nanmin(data)
        v_max = np.nanmax(data)
        if np.isfinite(v_min) and np.isfinite(v_max):
            if v_min == v_max:
                v_min -= 1.0
                v_max += 1.0
            v_abs = max(abs(v_min), abs(v_max))
            image.set_clim(-v_abs, v_abs)

        nonlocal last_channel_count
        if last_channel_count != buffer.plot_channels:
            tick_step = 10
            ticks = np.arange(0, buffer.plot_channels, tick_step)
            ax.set_yticks(ticks)
            ax.set_yticklabels([str(idx) for idx in ticks])
            last_channel_count = buffer.plot_channels

        if args.show_events:
            if event_times and event_channels:
                offsets = np.column_stack([list(event_times), list(event_channels)])
                event_scatter.set_offsets(offsets)
            else:
                event_scatter.set_offsets(np.empty((0, 2)))
        else:
            event_scatter.set_offsets(np.empty((0, 2)))

        return (image, event_scatter)

    interval_ms = max(10, int(1000.0 / args.refresh_hz))
    anim = FuncAnimation(fig, _refresh, interval=interval_ms, blit=False)

    try:
        plt.show()
    finally:
        receiver.stop()
        receiver.join(timeout=1.0)


if __name__ == "__main__":
    main()
