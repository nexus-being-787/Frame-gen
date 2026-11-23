#!/usr/bin/env python3
"""v3 Trainer for 15.2K-param IntelMaskNet — fixes all v2 training blockers:
1. QAT-only checkpoint selection: ONLY saves best_val checkpoints for epoch >= qat_start (guarantees INT8-trained weights!).
2. Pre-extracted sequential frame cache (eliminates cv2 codec seeking drift & keeps RAM < 100 MB).
3. Larger default crop (512x512 -> 64x64 at 1/8 res) so 11px receptive field isn't border-dominated.
4. Supports REAL Vulkan pipeline dumps (--dumps_dir with prev/mid/curr PNGs + flow_conf_occl .npy) to eliminate domain gap!
5. Perceptual loss (Charbonnier + Sobel edge sharpness + anti-collapse) instead of PSNR-only.
"""

from __future__ import annotations

import argparse
import glob
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import IntelMaskNet, count_parameters, sobel_edge_loss, warp_rgb, ycocg_variance_clamp_torch


def preextract_triplets_from_videos(video_paths: list[str], cache_dir: Path, stride: int = 8, max_triplets: int = 600) -> list[Path]:
    """Sequentially decodes contiguous (t-1, t, t+1) triplets to disk cache (zero codec-seek drift, low RAM!)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    triplet_dirs: list[Path] = []
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)

    for vidx, vp in enumerate(video_paths):
        cap = cv2.VideoCapture(vp)
        prev2, prev1 = None, None
        fidx = 0
        while len(triplet_dirs) < max_triplets:
            ok, curr = cap.read()
            if not ok:
                break
            if prev2 is not None and prev1 is not None and (fidx % stride == 0):
                tdir = cache_dir / f"v{vidx:02d}_f{fidx:06d}"
                if not (tdir / "flow8.npy").exists():
                    tdir.mkdir(parents=True, exist_ok=True)
                    h, w = prev2.shape[:2]
                    eh, ew = max(8, h // 8), max(8, w // 8)
                    g0 = cv2.resize(cv2.cvtColor(prev2, cv2.COLOR_BGR2GRAY), (ew, eh), interpolation=cv2.INTER_AREA)
                    g1 = cv2.resize(cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY), (ew, eh), interpolation=cv2.INTER_AREA)
                    flow8 = dis.calc(g0, g1, None).astype(np.float16)
                    cv2.imwrite(str(tdir / "prev.png"), prev2)
                    cv2.imwrite(str(tdir / "mid.png"), prev1)
                    cv2.imwrite(str(tdir / "curr.png"), curr)
                    np.save(tdir / "flow8.npy", flow8)
                triplet_dirs.append(tdir)
            prev2, prev1 = prev1, curr
            fidx += 1
        cap.release()
    return triplet_dirs


class DiskTripletDataset(Dataset):
    """Loads pre-extracted or Vulkan-dumped triplets (prev.png, mid.png, curr.png, flow8.npy) with 512px crops."""

    def __init__(self, triplet_dirs: list[Path], crop: int = 512):
        self.dirs = triplet_dirs
        self.crop = crop - (crop % 8)

    def __len__(self) -> int:
        return len(self.dirs)

    def __getitem__(self, idx: int):
        d = self.dirs[idx]
        p0 = cv2.imread(str(d / "prev.png"))
        pm = cv2.imread(str(d / "mid.png"))
        p1 = cv2.imread(str(d / "curr.png"))
        flow8 = np.load(d / "flow8.npy").astype(np.float32)  # (H/8, W/8, 2) or (H/8, W/8, 4)

        h, w = p0.shape[:2]
        ch = min(self.crop, h - (h % 8))
        cw = min(self.crop, w - (w % 8))
        y8 = np.random.randint(0, max(1, (h - ch) // 8 + 1))
        x8 = np.random.randint(0, max(1, (w - cw) // 8 + 1))
        y0, x0 = y8 * 8, x8 * 8

        p0 = p0[y0 : y0 + ch, x0 : x0 + cw]
        pm = pm[y0 : y0 + ch, x0 : x0 + cw]
        p1 = p1[y0 : y0 + ch, x0 : x0 + cw]
        f8 = flow8[y8 : y8 + ch // 8, x8 : x8 + cw // 8, :2]

        def to_t(bgr):
            return torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).permute(2, 0, 1)

        return to_t(p0), to_t(pm), to_t(p1), torch.from_numpy(f8.copy()).permute(2, 0, 1)


def synthesize_with_masks(t0: torch.Tensor, t1: torch.Tensor, flow_8: torch.Tensor, conf_8: torch.Tensor, occl_8: torch.Tensor) -> torch.Tensor:
    h, w = t0.shape[-2:]
    flow_full = F.interpolate(flow_8, size=(h, w), mode="bilinear", align_corners=False) * 8.0
    conf_full = F.interpolate(conf_8, size=(h, w), mode="bilinear", align_corners=False)
    occl_full = F.interpolate(occl_8, size=(h, w), mode="bilinear", align_corners=False)

    w0 = warp_rgb(t0, -0.5 * flow_full)
    w1 = warp_rgb(t1, +0.5 * flow_full)
    w0_clamped = ycocg_variance_clamp_torch(w0, w1, gamma=1.2)

    err_p = (w0_clamped - t0).abs().mean(dim=1, keepdim=True)
    err_c = (w1 - t1).abs().mean(dim=1, keepdim=True)
    occl_safe = torch.where(err_c <= err_p, w1, w0_clamped)
    motion_comp = (1.0 - 0.75 * occl_full) * (0.5 * (w0_clamped + w1)) + (0.75 * occl_full) * occl_safe
    return conf_full * motion_comp + (1.0 - conf_full) * (0.5 * (t0 + t1))


def build_mask_input(t0: torch.Tensor, t1: torch.Tensor, flow_8: torch.Tensor):
    eh, ew = flow_8.shape[-2:]
    l0 = 0.2126 * t0[:, 0:1] + 0.7152 * t0[:, 1:2] + 0.0722 * t0[:, 2:3]
    l1 = 0.2126 * t1[:, 0:1] + 0.7152 * t1[:, 1:2] + 0.0722 * t1[:, 2:3]
    l0_8 = F.interpolate(l0, size=(eh, ew), mode="area")
    l1_8 = F.interpolate(l1, size=(eh, ew), mode="area")
    w0_8 = warp_rgb(l0_8, -0.5 * flow_8)
    w1_8 = warp_rgb(l1_8, +0.5 * flow_8)
    err = (w0_8 - w1_8).abs()
    base_conf = (1.0 - (err / 0.25).clamp(0.0, 1.0))
    base_occl = ((err - 0.08) / 0.30).clamp(0.0, 1.0)
    x = torch.cat([l0_8, l1_8, w0_8, w1_8, (flow_8.abs() / 16.0).clamp(0.0, 1.0), base_conf, base_occl], dim=1)
    return x, base_conf, base_occl


def main() -> int:
    ap = argparse.ArgumentParser(description="v3 QAT trainer for 15.2K-param IntelMaskNet")
    ap.add_argument("--videos_dir", default=None, help="60 FPS clips directory (sequentially pre-extracted to disk)")
    ap.add_argument("--dumps_dir", default=None, help="Pre-extracted or Vulkan-dumped triplets directory")
    ap.add_argument("--cache_dir", default=None, help="Disk cache for extracted triplets")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--crop", type=int, default=512, help="Crop size (default 512 -> 64x64 at 1/8 res)")
    ap.add_argument("--out", default="intel_mask_net_qat.pt")
    args = ap.parse_args()

    if args.dumps_dir:
        triplet_dirs = sorted(p for p in Path(args.dumps_dir).iterdir() if p.is_dir() and (p / "flow8.npy").exists())
    elif args.videos_dir:
        videos = sorted(glob.glob(str(Path(args.videos_dir) / "*.mp4")) + glob.glob(str(Path(args.videos_dir) / "*.mkv")))
        cdir = Path(args.cache_dir) if args.cache_dir else Path(tempfile.mkdtemp(prefix="fg_triplets_"))
        triplet_dirs = preextract_triplets_from_videos(videos, cdir)
    else:
        print("Provide either --dumps_dir or --videos_dir")
        return 1

    split = max(1, int(len(triplet_dirs) * 0.8))
    train_dl = DataLoader(DiskTripletDataset(triplet_dirs[:split], crop=args.crop), batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(DiskTripletDataset(triplet_dirs[split:], crop=args.crop), batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IntelMaskNet().to(device)
    print(f"IntelMaskNet ({count_parameters(model):,} params) | train={split} val={len(triplet_dirs)-split} | crop={args.crop} | device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    qat_start = int(args.epochs * 0.6)
    best_qat_score = -1e9

    for epoch in range(1, args.epochs + 1):
        if epoch == qat_start:
            model.set_qat(True)
            print(f"[Epoch {epoch}] Fixed-scale INT8 QAT enabled (checkpoint selection starts now)")

        model.train()
        for t0, tm, t1, flow_8 in train_dl:
            t0, tm, t1, flow_8 = t0.to(device), tm.to(device), t1.to(device), flow_8.to(device)
            x, _, _ = build_mask_input(t0, t1, flow_8)
            out = model(x)
            pred = synthesize_with_masks(t0, t1, flow_8, out[:, 0:1], out[:, 1:2])
            charb = torch.mean(torch.sqrt((pred - tm) ** 2 + 1e-6))
            edge = sobel_edge_loss(pred, tm)
            anti_collapse = 0.015 * F.relu(0.55 - out[:, 0:1].mean())
            loss = charb + 0.15 * edge + anti_collapse
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        base_edge_tot, net_edge_tot, base_psnr_tot, net_psnr_tot, n_val = 0.0, 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for t0, tm, t1, flow_8 in val_dl:
                t0, tm, t1, flow_8 = t0.to(device), tm.to(device), t1.to(device), flow_8.to(device)
                x, bc, bo = build_mask_input(t0, t1, flow_8)
                bp = synthesize_with_masks(t0, t1, flow_8, bc, bo)
                out = model(x)
                np_ = synthesize_with_masks(t0, t1, flow_8, out[:, 0:1], out[:, 1:2])
                base_psnr_tot += 10.0 * math.log10(1.0 / max(F.mse_loss(bp, tm).item(), 1e-8))
                net_psnr_tot += 10.0 * math.log10(1.0 / max(F.mse_loss(np_, tm).item(), 1e-8))
                base_edge_tot += sobel_edge_loss(bp, tm).item()
                net_edge_tot += sobel_edge_loss(np_, tm).item()
                n_val += 1

        if n_val > 0:
            bp, np_ = base_psnr_tot / n_val, net_psnr_tot / n_val
            be, ne = base_edge_tot / n_val, net_edge_tot / n_val
            # Composite perceptual score: PSNR gain minus edge-blur regression
            score = (np_ - bp) - 10.0 * (ne - be)
            tag = "[QAT]" if epoch >= qat_start else "[FP32]"
            print(f"{tag} Ep {epoch:02d}/{args.epochs} | Val PSNR: {bp:.2f}->{np_:.2f} dB | EdgeErr: {be:.4f}->{ne:.4f}")
            # CRITICAL FIX: Only select checkpoints AFTER QAT starts so exported model is always INT8-trained!
            if epoch >= qat_start and score > best_qat_score:
                best_qat_score = score
                torch.save(model.state_dict(), args.out)

    print(f"[OK] Saved best QAT-trained checkpoint to {args.out} (composite gain={best_qat_score:+.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
