#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create an interactive full-channel DAS wiggle plot as a standalone HTML file.

The signal is filtered on the full record first. The browser plot uses a
downsampled and quantized copy only for interactive viewing.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


def parse_args():
    p = argparse.ArgumentParser(description="Create interactive full-channel DAS wiggle HTML.")
    p.add_argument("--input", "-i", required=True, help="Input CSV with ch_* columns.")
    p.add_argument("--output", "-o", required=True, help="Output HTML path.")
    p.add_argument("--fs", type=float, required=True, help="Sampling rate in Hz.")
    p.add_argument("--lowcut", type=float, default=0.1, help="Bandpass low cut in Hz.")
    p.add_argument("--highcut", type=float, default=20.0, help="Bandpass high cut in Hz.")
    p.add_argument("--filter-order", type=int, default=4, help="Butterworth filter order.")
    p.add_argument(
        "--max-points-per-channel",
        type=int,
        default=25000,
        help="Maximum plotted points per channel after filtering. Default: 25000.",
    )
    p.add_argument(
        "--scale-percentile",
        type=float,
        default=99.0,
        help="Global abs percentile used to scale wiggle amplitude. Default: 99.",
    )
    p.add_argument(
        "--quantize-scale",
        type=float,
        default=1000.0,
        help="Stored integer units per scaled amplitude. Default: 1000.",
    )
    p.add_argument(
        "--clip-scaled-amplitude",
        type=float,
        default=6.0,
        help="Clip scaled amplitudes before embedding in HTML. Default: +/-6.",
    )
    return p.parse_args()


def channel_sort_key(name):
    try:
        return int(str(name).split("_", 1)[1])
    except (IndexError, ValueError):
        return str(name)


def butter_bandpass_sos(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-9)
    high = min(highcut / nyq, 0.999999)
    if not 0 < low < high < 1:
        raise ValueError(f"Invalid bandpass range {lowcut}-{highcut} Hz for fs={fs} Hz.")
    return butter(order, [low, high], btype="band", output="sos")


def bandpass_filter_array(X, fs, lowcut, highcut, order):
    sos = butter_bandpass_sos(lowcut, highcut, fs, order)
    Xf = np.zeros_like(X, dtype=np.float64)
    for c in range(X.shape[1]):
        col = X[:, c].astype(np.float64)
        col = col - np.nanmean(col)
        Xf[:, c] = sosfiltfilt(sos, col)
    return Xf


def build_html(payload):
    payload_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{payload["title"]}</title>
<style>
  html, body {{
    margin: 0;
    height: 100%;
    background: #f7f7f4;
    color: #202124;
    font-family: Arial, sans-serif;
    overflow: hidden;
  }}
  #toolbar {{
    height: 42px;
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0 12px;
    border-bottom: 1px solid #d7d7d2;
    background: #ffffff;
    box-sizing: border-box;
    font-size: 13px;
  }}
  button {{
    height: 28px;
    border: 1px solid #b8b8b2;
    background: #fff;
    border-radius: 4px;
    cursor: pointer;
  }}
  input {{
    width: 72px;
    height: 24px;
    box-sizing: border-box;
  }}
  #status {{
    margin-left: auto;
    color: #5f6368;
    white-space: nowrap;
  }}
  #plot {{
    display: block;
    width: 100vw;
    height: calc(100vh - 42px);
    cursor: grab;
  }}
  #plot.dragging {{
    cursor: grabbing;
  }}
</style>
</head>
<body>
<div id="toolbar">
  <strong>{payload["title"]}</strong>
  <button id="reset">Reset</button>
  <button id="allChannels">All channels</button>
  <label>t0 <input id="tMin" type="number" step="0.1"></label>
  <label>t1 <input id="tMax" type="number" step="0.1"></label>
  <label>ch0 <input id="chMin" type="number" step="1"></label>
  <label>ch1 <input id="chMax" type="number" step="1"></label>
  <button id="apply">Apply</button>
  <span id="status"></span>
</div>
<canvas id="plot"></canvas>
<script>
const payload = {payload_json};
const canvas = document.getElementById("plot");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const tMinInput = document.getElementById("tMin");
const tMaxInput = document.getElementById("tMax");
const chMinInput = document.getElementById("chMin");
const chMaxInput = document.getElementById("chMax");

let view = {{
  tMin: payload.tStart,
  tMax: payload.tEnd,
  chMin: 0,
  chMax: payload.channels.length - 1,
}};
let dragging = false;
let lastX = 0;
let lastY = 0;

function resizeCanvas() {{
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}}

function clampView() {{
  const minSpanT = payload.dt * 10;
  if (view.tMax - view.tMin < minSpanT) {{
    const mid = (view.tMin + view.tMax) / 2;
    view.tMin = mid - minSpanT / 2;
    view.tMax = mid + minSpanT / 2;
  }}
  if (view.tMin < payload.tStart) {{
    const span = view.tMax - view.tMin;
    view.tMin = payload.tStart;
    view.tMax = payload.tStart + span;
  }}
  if (view.tMax > payload.tEnd) {{
    const span = view.tMax - view.tMin;
    view.tMax = payload.tEnd;
    view.tMin = payload.tEnd - span;
  }}
  view.tMin = Math.max(payload.tStart, view.tMin);
  view.tMax = Math.min(payload.tEnd, view.tMax);
  view.chMin = Math.max(0, Math.min(payload.channels.length - 1, view.chMin));
  view.chMax = Math.max(0, Math.min(payload.channels.length - 1, view.chMax));
  if (view.chMax < view.chMin) {{
    const tmp = view.chMin;
    view.chMin = view.chMax;
    view.chMax = tmp;
  }}
}}

function xOf(t, w, left, right) {{
  return left + (t - view.tMin) / (view.tMax - view.tMin) * (w - left - right);
}}

function yOf(chValue, h, top, bottom) {{
  return h - bottom - (chValue - view.chMin) / (view.chMax - view.chMin + 1) * (h - top - bottom);
}}

function niceStep(span, targetTicks) {{
  const raw = span / targetTicks;
  const pow = Math.pow(10, Math.floor(Math.log10(raw)));
  const frac = raw / pow;
  if (frac <= 1) return pow;
  if (frac <= 2) return 2 * pow;
  if (frac <= 5) return 5 * pow;
  return 10 * pow;
}}

function drawAxes(w, h, left, right, top, bottom) {{
  ctx.strokeStyle = "#b9b9b3";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(left, top);
  ctx.lineTo(left, h - bottom);
  ctx.lineTo(w - right, h - bottom);
  ctx.stroke();

  ctx.fillStyle = "#333";
  ctx.font = "12px Arial";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const tStep = niceStep(view.tMax - view.tMin, 9);
  const firstT = Math.ceil(view.tMin / tStep) * tStep;
  for (let t = firstT; t <= view.tMax + 1e-9; t += tStep) {{
    const x = xOf(t, w, left, right);
    ctx.strokeStyle = "#e2e2dc";
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, h - bottom);
    ctx.stroke();
    ctx.fillText(t.toFixed(tStep < 1 ? 2 : 1), x, h - bottom + 6);
  }}

  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  const chSpan = Math.max(1, view.chMax - view.chMin + 1);
  const chStep = Math.max(1, Math.ceil(chSpan / 14));
  for (let ch = Math.ceil(view.chMin); ch <= Math.floor(view.chMax); ch += chStep) {{
    const y = yOf(ch, h, top, bottom);
    ctx.strokeStyle = "#ededE7";
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(w - right, y);
    ctx.stroke();
    ctx.fillStyle = "#333";
    ctx.fillText(payload.channels[ch], left - 8, y);
  }}

  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.fillText("Time (s)", (left + w - right) / 2, h - 4);
  ctx.save();
  ctx.translate(15, (top + h - bottom) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Channel", 0, 0);
  ctx.restore();
}}

function draw() {{
  clampView();
  const rect = canvas.getBoundingClientRect();
  const w = rect.width;
  const h = rect.height;
  const left = 64, right = 18, top = 16, bottom = 42;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fbfbf8";
  ctx.fillRect(0, 0, w, h);
  drawAxes(w, h, left, right, top, bottom);

  const i0 = Math.max(0, Math.floor((view.tMin - payload.tStart) / payload.dt));
  const i1 = Math.min(payload.nPoints - 1, Math.ceil((view.tMax - payload.tStart) / payload.dt));
  const visibleSamples = Math.max(1, i1 - i0 + 1);
  const stride = Math.max(1, Math.floor(visibleSamples / Math.max(1, (w - left - right) * 1.5)));
  const ch0 = Math.max(0, Math.floor(view.chMin));
  const ch1 = Math.min(payload.channels.length - 1, Math.ceil(view.chMax));

  ctx.strokeStyle = "#1565a9";
  ctx.lineWidth = 0.75;
  for (let ch = ch0; ch <= ch1; ch++) {{
    const arr = payload.data[ch];
    ctx.beginPath();
    let started = false;
    for (let i = i0; i <= i1; i += stride) {{
      const t = payload.tStart + i * payload.dt;
      const amp = arr[i] / payload.quantizeScale;
      const x = xOf(t, w, left, right);
      const y = yOf(ch + amp, h, top, bottom);
      if (!started) {{
        ctx.moveTo(x, y);
        started = true;
      }} else {{
        ctx.lineTo(x, y);
      }}
    }}
    ctx.stroke();
  }}

  tMinInput.value = view.tMin.toFixed(3);
  tMaxInput.value = view.tMax.toFixed(3);
  chMinInput.value = Math.round(view.chMin);
  chMaxInput.value = Math.round(view.chMax);
  statusEl.textContent = `${{view.tMin.toFixed(2)}}-${{view.tMax.toFixed(2)}} s | ch ${{Math.round(view.chMin)}}-${{Math.round(view.chMax)}} | plotted stride ${{stride}}`;
}}

canvas.addEventListener("wheel", (ev) => {{
  ev.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const xFrac = Math.min(1, Math.max(0, (ev.clientX - rect.left - 64) / Math.max(1, rect.width - 82)));
  const yFrac = Math.min(1, Math.max(0, 1 - (ev.clientY - rect.top - 16) / Math.max(1, rect.height - 58)));
  const zoom = ev.deltaY < 0 ? 0.82 : 1.22;
  if (ev.shiftKey) {{
    const chAt = view.chMin + yFrac * (view.chMax - view.chMin + 1);
    const span = (view.chMax - view.chMin + 1) * zoom;
    view.chMin = chAt - yFrac * span;
    view.chMax = view.chMin + span - 1;
  }} else {{
    const tAt = view.tMin + xFrac * (view.tMax - view.tMin);
    const span = (view.tMax - view.tMin) * zoom;
    view.tMin = tAt - xFrac * span;
    view.tMax = view.tMin + span;
  }}
  draw();
}}, {{ passive: false }});

canvas.addEventListener("mousedown", (ev) => {{
  dragging = true;
  lastX = ev.clientX;
  lastY = ev.clientY;
  canvas.classList.add("dragging");
}});

window.addEventListener("mouseup", () => {{
  dragging = false;
  canvas.classList.remove("dragging");
}});

window.addEventListener("mousemove", (ev) => {{
  if (!dragging) return;
  const rect = canvas.getBoundingClientRect();
  const dx = ev.clientX - lastX;
  const dy = ev.clientY - lastY;
  lastX = ev.clientX;
  lastY = ev.clientY;
  const tSpan = view.tMax - view.tMin;
  const chSpan = view.chMax - view.chMin + 1;
  const dt = -dx / Math.max(1, rect.width - 82) * tSpan;
  const dch = dy / Math.max(1, rect.height - 58) * chSpan;
  view.tMin += dt;
  view.tMax += dt;
  view.chMin += dch;
  view.chMax += dch;
  draw();
}});

document.getElementById("reset").addEventListener("click", () => {{
  view = {{tMin: payload.tStart, tMax: payload.tEnd, chMin: 0, chMax: payload.channels.length - 1}};
  draw();
}});

document.getElementById("allChannels").addEventListener("click", () => {{
  view.chMin = 0;
  view.chMax = payload.channels.length - 1;
  draw();
}});

document.getElementById("apply").addEventListener("click", () => {{
  view.tMin = Number(tMinInput.value);
  view.tMax = Number(tMaxInput.value);
  view.chMin = Number(chMinInput.value);
  view.chMax = Number(chMaxInput.value);
  draw();
}});

window.addEventListener("resize", resizeCanvas);
resizeCanvas();
</script>
</body>
</html>
"""


def main():
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Info] Loading CSV: {args.input}")
    df = pd.read_csv(args.input)
    ch_cols = [c for c in df.columns if str(c).lower().startswith("ch_")]
    ch_cols = sorted(ch_cols, key=channel_sort_key)
    if not ch_cols:
        raise ValueError("No ch_* columns found.")

    X = df[ch_cols].to_numpy(dtype=np.float64)
    n_samples, n_channels = X.shape
    duration = n_samples / args.fs
    print(f"[Info] Loaded {n_samples} samples, {n_channels} channels, duration {duration:.2f}s.")

    print(f"[Info] Filtering full signal with {args.lowcut:g}-{args.highcut:g} Hz bandpass.")
    Xf = bandpass_filter_array(
        X,
        fs=args.fs,
        lowcut=args.lowcut,
        highcut=args.highcut,
        order=args.filter_order,
    )

    step = max(1, int(np.ceil(n_samples / max(1, args.max_points_per_channel))))
    if step > 1:
        print(f"[Info] Downsampling for browser plot only: every {step} samples.")

    Xp = Xf[::step, :]
    scale = np.nanpercentile(np.abs(Xp), args.scale_percentile)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nanmax(np.abs(Xp))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    scaled = np.clip(Xp / scale, -args.clip_scaled_amplitude, args.clip_scaled_amplitude)
    quantized = np.round(scaled * args.quantize_scale).astype(np.int16)
    data = [quantized[:, ch].tolist() for ch in range(n_channels)]

    payload = {
        "title": f"{Path(args.input).stem} full wiggle | {args.lowcut:g}-{args.highcut:g} Hz",
        "channels": ch_cols,
        "data": data,
        "tStart": 0.0,
        "tEnd": float((Xp.shape[0] - 1) * step / args.fs),
        "dt": float(step / args.fs),
        "nPoints": int(Xp.shape[0]),
        "quantizeScale": float(args.quantize_scale),
        "scalePercentile": float(args.scale_percentile),
        "filter": f"{args.lowcut:g}-{args.highcut:g} Hz",
    }

    html = build_html(payload)
    output.write_text(html, encoding="utf-8")
    print(f"[Finish] Saved interactive HTML: {output}")


if __name__ == "__main__":
    main()
