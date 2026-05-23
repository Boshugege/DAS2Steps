#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collect raw and filtered waveform images from per-event folders.
"""

import argparse
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Copy per-event raw/filtered waveform PNGs into separate folders for browsing."
    )
    p.add_argument(
        "--input-dir",
        "-i",
        default="output/step_waveform_windows/events_020_050",
        help="Parent folder containing event_* subfolders.",
    )
    p.add_argument(
        "--outdir",
        "-o",
        default=None,
        help="Output folder. Default: <input-dir>/collected_images",
    )
    p.add_argument("--raw-name", default="raw_waveforms.png", help="Raw waveform image filename.")
    p.add_argument(
        "--filtered-name",
        default="bandpass_0p1_30Hz_waveforms.png",
        help="Filtered waveform image filename.",
    )
    return p.parse_args()


def copy_images(event_dirs, image_name, target_dir):
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing = []

    for event_dir in event_dirs:
        src = event_dir / image_name
        if not src.exists():
            missing.append(str(event_dir))
            continue

        dst = target_dir / f"{event_dir.name}__{image_name}"
        shutil.copy2(src, dst)
        copied += 1

    return copied, missing


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    outdir = Path(args.outdir) if args.outdir else input_dir / "collected_images"
    raw_dir = outdir / "raw"
    filtered_dir = outdir / "filtered"

    event_dirs = sorted([p for p in input_dir.iterdir() if p.is_dir() and p.name.startswith("event_")])
    if not event_dirs:
        raise ValueError(f"No event_* folders found in: {input_dir}")

    raw_count, raw_missing = copy_images(event_dirs, args.raw_name, raw_dir)
    filtered_count, filtered_missing = copy_images(event_dirs, args.filtered_name, filtered_dir)

    print(f"[Finish] Event folders scanned: {len(event_dirs)}")
    print(f"[Finish] Raw images copied: {raw_count} -> {raw_dir}")
    print(f"[Finish] Filtered images copied: {filtered_count} -> {filtered_dir}")

    if raw_missing:
        print(f"[Warn] Missing raw images in {len(raw_missing)} folders.")
    if filtered_missing:
        print(f"[Warn] Missing filtered images in {len(filtered_missing)} folders.")


if __name__ == "__main__":
    main()
