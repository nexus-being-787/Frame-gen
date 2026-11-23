# Gamescope FrameGen Final (Intel i5-12450H / UHD Edition)

Low-overhead Vulkan compute frame generator (2x / 3x / 4x MFG) built directly into Valve's `gamescope`, engineered to stay well under **~4 ms** on **Intel Core i5-12450H (Alder Lake-H, Intel UHD 48 EUs, Mesa ANV)** on Arch Linux.

## Pipeline Architecture (< ~1.9 ms total synthetic GPU pass at 1080p)

1. **Pass 1A — 1/4-Res `RG8_UNORM` Luma + 8-Bit Census Prepass (`src/shaders/cs_framegen_luma.comp`, ~0.10–0.18 ms on real frames)**
   - Downsamples the overlay-free base game layer into `DRM_FORMAT_GR88` (`480×270` = 259 KB, fits in Intel L2/L3 cache): `R` = 3×3 Gaussian pre-blurred luma, `G` = illumination/flicker-invariant 8-bit Census transform.
2. **Pass 1B — Hierarchical Census Optical Flow + Median Filter (`src/shaders/cs_framegen_flow.comp`, ~0.50–0.90 ms once per real frame)**
   - 1 single 2-byte `RG8` fetch per candidate (nearest-filtered so Census bits stay intact), temporal flow warm-start from `frameGenFlow[prev]`, 5×5 coarse → 3×3 fine + parabolic sub-pixel fit, in-place shared-memory 5-tap median filter, and forward-backward consistency check for `confidence` and `occlusion`.
   - Epoch-cached: runs once per real frame and reuses the flow field across all sub-frames in 3x/4x MFG.
3. **Pass 2 — 9-Tap Occlusion-Aware Bidirectional Warp (`src/shaders/cs_framegen.comp`, ~0.60–0.95 ms)**
   - Hardware bilinear flow read + shared 4-neighbor cross for YCoCg variance clipping (anti-ghosting), static in-game HUD/text auto-protection, occlusion-safe blending, scene-cut fallback, and RCAS-lite synthetic sharpening (`--frame-gen-sharpness`).
4. **Overlay Exclusion & RCAS History Match (`src/rendervulkan.cpp`)**
   - Captures Layer 0 through 1-layer RCAS when FSR/SGSR + `--mangoapp` are active (matching real frame sharpness pixel-for-pixel) and composites live overlays cleanly on top.
5. **VBlank-Safe EMA Midpoint Pacing & CSV Telemetry (`src/steamcompmgr.cpp`)**
   - Forces vblank-synced flips when FrameGen is active, tracks real frame interval via EMA, caps midpoint alignment (`<= 2.5 ms` via `clock_nanosleep` + 300 µs pause spin) so flips never miss vblank, auto-throttles on >55 ms stutters, and logs line-buffered CSV rows via `--frame-gen-csv <path>`.
6. **Live Vulkan GPU Profiling & On-Screen 0–4 ms Budget Bar (`rendervulkan.cpp`, `cs_framegen.comp`)**
   - `--frame-gen-stats` logs per-pass `CmdWriteTimestamp` microsecond averages (`framegen flow`, `framegen warp`, `framegen total`).
   - Any debug view (`Super + K`) renders a top-right 0–4 ms GPU meter bar (cyan = flow, green = warp, red = >4 ms overflow, white ticks at 1/2/3 ms).

## Live In-Game Hotkeys
- **`Super + M`**: Cycle FrameGen mode (`off` → `duplicate` → `blend` → `opticalflow`)
- **`Super + J`**: Instant A/B toggle (`opticalflow` ↔ `duplicate`)
- **`Super + K`**: Cycle debug views (`off` → `split` → `flow` → `confidence` → `occlusion`) + live GPU budget bar
- **`Super + L`**: Cycle MFG multiplier (`2x` → `3x` → `4x`)

## Fast Rebuild Over Your Existing Build (~5 seconds)
```bash
cd gamescope-master
unzip -o ../gamescope-framegen-final-update.zip
ninja -C build-alderlake -j"$(nproc)"
sudo ninja -C build-alderlake install
```

## Recommended Launch Commands (Intel i5-12450H, 60 Hz Panel)
- **Outlast 1 (30 → 60 FPS, 1/4-res Census flow, GPU stats + pacing CSV):**
  ```bash
  gamemoderun /usr/local/bin/gamescope -w 1280 -h 720 -W 1920 -H 1080 -r 60 --framerate-limit 30 -F fsr --frame-gen opticalflow --frame-gen-multiplier 2 --frame-gen-flow-div 4 --frame-gen-stats --frame-gen-csv /tmp/outlast_fg.csv --mangoapp -f -- %command%
  ```
- **NFS Most Wanted (30 → 60 FPS, wider motion scale, VSync ON in-game):**
  ```bash
  gamemoderun /usr/local/bin/gamescope -w 1280 -h 720 -W 1920 -H 1080 -r 60 --framerate-limit 30 -F fsr --frame-gen opticalflow --frame-gen-multiplier 2 --frame-gen-flow-div 4 --frame-gen-flow-scale 1.6 --frame-gen-low-latency --mangoapp -f -- %command%
  ```
