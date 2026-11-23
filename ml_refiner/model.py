"""IntelMaskNet (~15.2K params): 1/8-res Mask-Only (Confidence + Occlusion) Refiner for Pre-FSR Gamescope.

Note on resolution & compute budget:
  With Level 1 (Pre-FSR game-res pipeline at 720p), 1/8 game-res is 160x90 = 14,400 pixels -> 0.218 GMAC/frame.
  Every channel width is a multiple of 4 (8 -> 24 -> 24 -> 24 -> 12 -> 4) for Intel Xe-LP DP4a.
  Quantization matches cs_framegen_mask_dp4a.comp bit-for-bit:
    - Input [0, 1] -> int8 [0, 127], INPUT_SCALE = 1/127
    - ReLU6 [0, 6] -> int8 [0, 127], RELU6_SCALE = 6/127
    - Zero padding (padding=1), symmetric per-output-channel int8 weights [-127, 127]
    - IEEE round-to-even (torch.round <-> GLSL roundEven)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_SCALE = 1.0 / 127.0
RELU6_SCALE = 6.0 / 127.0


def fake_quant_fixed(x: torch.Tensor, scale: float, qmin: int = 0, qmax: int = 127) -> torch.Tensor:
    q = torch.clamp(torch.round(x / scale), float(qmin), float(qmax))
    return x + (q * scale - x).detach()


def fake_quant_weight_per_channel(w: torch.Tensor) -> torch.Tensor:
    max_abs = w.detach().abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
    scale = max_abs / 127.0
    q = torch.clamp(torch.round(w / scale), -127.0, 127.0)
    return w + (q * scale - w).detach()


class FixedQConv3x3(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, act: bool = True):
        super().__init__()
        assert in_ch % 4 == 0 and out_ch % 4 == 0
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.act = act
        self.qat = False

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        w = fake_quant_weight_per_channel(self.conv.weight) if self.qat else self.conv.weight
        y = F.conv2d(x, w, self.conv.bias, stride=1, padding=1)
        if skip is not None:
            y = y + skip
        if self.act:
            y = F.relu6(y)
            if self.qat:
                y = fake_quant_fixed(y, RELU6_SCALE, 0, 127)
        return y


class IntelMaskNet(nn.Module):
    """15,208-parameter mask-only refiner at 1/8 game resolution (160x90 @ 720p, 0.218 GMAC/frame).

    Input (B, 8, H/8, W/8) in [0, 1]:
      [prev_luma, curr_luma, warp_prev_luma, warp_curr_luma,
       |flow_x|/16, |flow_y|/16, classical_conf, classical_occl]
    Output (B, 4, H/8, W/8) in [0, 1]:
      [refined_conf, refined_occl, pad0, pad1]
    """

    def __init__(self):
        super().__init__()
        self.conv1 = FixedQConv3x3(8, 24, act=True)   # 1,752 params
        self.res1a = FixedQConv3x3(24, 24, act=True)  # 5,208 params
        self.res1b = FixedQConv3x3(24, 24, act=True)  # 5,208 params (+ residual skip)
        self.conv2 = FixedQConv3x3(24, 12, act=True)  # 2,604 params
        self.head  = FixedQConv3x3(12, 4, act=False)  #   436 params

    def set_qat(self, enabled: bool) -> None:
        for m in self.modules():
            if isinstance(m, FixedQConv3x3):
                m.qat = enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0.0, 1.0)
        if self.conv1.qat:
            x = fake_quant_fixed(x, INPUT_SCALE, 0, 127)
        f1 = self.conv1(x)
        r1 = self.res1b(self.res1a(f1), skip=f1)
        f2 = self.conv2(r1)
        return torch.sigmoid(self.head(f2))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def warp_rgb(img: torch.Tensor, flow_px: torch.Tensor) -> torch.Tensor:
    b, _, h, w = img.shape
    ys = torch.linspace(-1.0, 1.0, h, device=img.device, dtype=img.dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=img.device, dtype=img.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
    dx = flow_px[:, 0] * (2.0 / max(w - 1, 1))
    dy = flow_px[:, 1] * (2.0 / max(h - 1, 1))
    return F.grid_sample(img, grid + torch.stack([dx, dy], dim=-1), mode="bilinear", padding_mode="border", align_corners=True)


def ycocg_variance_clamp_torch(warped_prev: torch.Tensor, warped_curr: torch.Tensor, gamma: float = 1.2) -> torch.Tensor:
    def to_ycocg(c: torch.Tensor) -> torch.Tensor:
        co = c[:, 0:1] - c[:, 2:3]
        t = c[:, 2:3] + 0.5 * co
        cg = c[:, 1:2] - t
        y = t + 0.5 * cg
        return torch.cat([y, co, cg], dim=1)

    def to_rgb(ycc: torch.Tensor) -> torch.Tensor:
        t = ycc[:, 0:1] - 0.5 * ycc[:, 2:3]
        g = ycc[:, 2:3] + t
        b = t - 0.5 * ycc[:, 1:2]
        r = b + ycc[:, 1:2]
        return torch.cat([r, g, b], dim=1).clamp(0.0, 1.0)

    yc = to_ycocg(warped_curr)
    yp = to_ycocg(warped_prev)
    p = F.pad(yc, (1, 1, 1, 1), mode="replicate")
    m1 = (yc + p[:, :, 1:-1, :-2] + p[:, :, 1:-1, 2:] + p[:, :, :-2, 1:-1] + p[:, :, 2:, 1:-1]) * 0.2
    m2 = (yc * yc + p[:, :, 1:-1, :-2] ** 2 + p[:, :, 1:-1, 2:] ** 2 + p[:, :, :-2, 1:-1] ** 2 + p[:, :, 2:, 1:-1] ** 2) * 0.2
    sigma = torch.sqrt((m2 - m1 * m1).clamp(min=0.0))
    return to_rgb(torch.max(torch.min(yp, m1 + gamma * sigma), m1 - gamma * sigma))


def sobel_edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Perceptual sharpness loss: penalizes blurry plain blends that fool plain PSNR."""
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=pred.device, dtype=pred.dtype).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    pg = 0.2126 * pred[:, 0:1] + 0.7152 * pred[:, 1:2] + 0.0722 * pred[:, 2:3]
    tg = 0.2126 * target[:, 0:1] + 0.7152 * target[:, 1:2] + 0.0722 * target[:, 2:3]
    gx_p, gy_p = F.conv2d(pg, kx, padding=1), F.conv2d(pg, ky, padding=1)
    gx_t, gy_t = F.conv2d(tg, kx, padding=1), F.conv2d(tg, ky, padding=1)
    return (gx_p - gx_t).abs().mean() + (gy_p - gy_t).abs().mean()
