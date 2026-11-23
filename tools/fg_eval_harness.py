#!/usr/bin/env python3
"""Offline Quality Evaluation Harness for Gamescope FrameGen (PSNR / SSIM / VMAF).

Takes a 60 FPS gameplay clip, drops odd frames (60 -> 30 FPS), synthesizes the missing
midpoint frames (duplicate, blend, or OpenCV/DIS optical-flow reference), and scores
every generated frame against the true 60 FPS ground truth with PSNR, SSIM, and optional VMAF.

Usage:
  python3 tools/fg_eval_harness.py gameplay_60fps.mp4 --max-frames 300 --out-csv scores.csv
"""

import argparse
import csv
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


def compute_psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    mse = float(np.mean((gt.astype(np.float32) - pred.astype(np.float32)) ** 2))
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10((255.0 * 255.0) / mse)


def compute_ssim(gt: np.ndarray, pred: np.ndarray) -> float:
    g1 = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY).astype(np.float32)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1 = cv2.GaussianBlur(g1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(g2, (11, 11), 1.5)
    s1_sq = cv2.GaussianBlur(g1 * g1, (11, 11), 1.5) - mu1 * mu1
    s2_sq = cv2.GaussianBlur(g2 * g2, (11, 11), 1.5) - mu2 * mu2
    s12 = cv2.GaussianBlur(g1 * g2, (11, 11), 1.5) - mu1 * mu2
    num = (2.0 * mu1 * mu2 + c1) * (2.0 * s12 + c2)
    den = (mu1 * mu1 + mu2 * mu2 + c1) * (s1_sq + s2_sq + c2)
    return float(np.mean(num / np.maximum(den, 1e-8)))


def interp_dis_flow(prev_bgr: np.ndarray, curr_bgr: np.ndarray, dis) -> np.ndarray:
    h, w = prev_bgr.shape[:2]
    g0 = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    # Global scene-cut guard
    if float(np.mean(np.abs(g1.astype(np.float32) - g0.astype(np.float32)))) / 255.0 > 0.18:
        return curr_bgr.copy()
    flow = dis.calc(g0, g1, None)
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    w0 = cv2.remap(prev_bgr, grid_x - 0.5 * flow[..., 0], grid_y - 0.5 * flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    w1 = cv2.remap(curr_bgr, grid_x + 0.5 * flow[..., 0], grid_y + 0.5 * flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.addWeighted(w0, 0.5, w1, 0.5, 0.0)


def run_vmaf(gt_path: Path, pred_path: Path) -> float | None:
    if not shutil.which("ffmpeg"):
        return None
    cmd = [
        "ffmpeg", "-hide_banner", "-i", str(pred_path), "-i", str(gt_path),
        "-lavfi", "libvmaf=log_fmt=csv:log_path=/dev/stdout", "-f", "null", "-"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return None
    for line in res.stderr.splitlines():
        if "VMAF score:" in line:
            try:
                return float(line.split("VMAF score:")[-1].strip())
            except ValueError:
                pass
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline 60->30->60 FrameGen quality harness (PSNR/SSIM/VMAF)")
    ap.add_argument("clip", help="Input 60 FPS video clip (mp4/mkv)")
    ap.add_argument("--max-frames", type=int, default=240, help="Max input frames to evaluate (default: 240)")
    ap.add_argument("--out-csv", default="fg_eval_scores.csv", help="Output per-frame CSV path")
    ap.add_argument("--vmaf", action="store_true", help="Compute VMAF via ffmpeg libvmaf (slower)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.clip)
    if not cap.isOpened():
        print(f"Error: cannot open {args.clip}", file=sys.stderr)
        return 1

    frames = []
    while len(frames) < args.max_frames:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()

    if len(frames) < 5:
        print("Error: need at least 5 frames in input clip", file=sys.stderr)
        return 1

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    modes = ("duplicate", "blend", "dis_flow")
    totals = {m: {"psnr": [], "ssim": []} for m in modes}

    rows = []
    h, w = frames[0].shape[:2]
    tmp_dir = Path(tempfile.mkdtemp(prefix="fg_eval_")) if args.vmaf else None
    writers = {}
    if tmp_dir:
        fourcc = cv2.VideoWriter_fourcc(*"FFV1")
        writers["gt"] = cv2.VideoWriter(str(tmp_dir / "gt.mkv"), fourcc, 30.0, (w, h))
        for m in modes:
            writers[m] = cv2.VideoWriter(str(tmp_dir / f"{m}.mkv"), fourcc, 30.0, (w, h))

    # Evaluate odd frames (1, 3, 5, ...) from even anchors (0, 2, 4, ...)
    for idx in range(1, len(frames) - 1, 2):
        prev_f, gt_f, curr_f = frames[idx - 1], frames[idx], frames[idx + 1]
        preds = {
            "duplicate": curr_f,
            "blend": cv2.addWeighted(prev_f, 0.5, curr_f, 0.5, 0.0),
            "dis_flow": interp_dis_flow(prev_f, curr_f, dis),
        }
        if tmp_dir:
            writers["gt"].write(gt_f)
        row = {"frame_idx": idx}
        for m in modes:
            p = compute_psnr(gt_f, preds[m])
            s = compute_ssim(gt_f, preds[m])
            totals[m]["psnr"].append(p)
            totals[m]["ssim"].append(s)
            row[f"{m}_psnr"] = f"{p:.3f}"
            row[f"{m}_ssim"] = f"{s:.4f}"
            if tmp_dir:
                writers[m].write(preds[m])
        rows.append(row)

    for wr in writers.values():
        wr.release()

    with open(args.out_csv, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)

    print(f"Evaluated {len(rows)} midpoint frames from {args.clip} -> {args.out_csv}")
    for m in modes:
        avg_p = float(np.mean(totals[m]["psnr"]))
        avg_s = float(np.mean(totals[m]["ssim"]))
        vmaf_str = ""
        if tmp_dir:
            v = run_vmaf(tmp_dir / "gt.mkv", tmp_dir / f"{m}.mkv")
            vmaf_str = f" | VMAF={v:.2f}" if v is not None else " | VMAF=unavailable"
        print(f"  {m:10s}: PSNR={avg_p:.2f} dB | SSIM={avg_s:.4f}{vmaf_str}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
