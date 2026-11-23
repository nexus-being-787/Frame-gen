# Level 4 (v3): 15.2K-Parameter 1/8-Res INT8 DP4a Mask Refiner + Auto-Tuner

Complete PyTorch QAT + Vulkan DP4a (`dotPacked4x8EXT`) mask-only refiner for **Pre-FSR Gamescope (v11)** on **Intel Core i5-12450H (Alder Lake GT2 / Xe-LP, 48 EUs)**.

## Why 15.2K Params @ 1/8 Game-Res?
- With Level 1 (Pre-FSR 720p pipeline), 1/8 resolution is **`160×90`** (14,400 px).
- `IntelMaskNet` (`8 → 24 → 24 → 24 (+skip) → 12 → 4`, **15,208 params**, ~14.9 KB INT8) costs only **0.218 GMAC/frame** (16× lighter than a 100K 1/4-res full-flow CNN), leaving classical 3-level Census+DIS flow untouched while refining only `(confidence, occlusion)` masks.

## Files Included
- `ml_refiner/model.py`: Fixed-scale INT8 `IntelMaskNet` (`INPUT_SCALE = 1/127`, `RELU6_SCALE = 6/127`, `/16` flow normalization, Sobel edge loss).
- `ml_refiner/train.py`: Sequential disk-cached trainer (zero codec-seek drift, <100 MB RAM), 512px crops, QAT-only checkpoint selection, `--dumps_dir` real-pipeline dump loader, and baseline PSNR + EdgeErr comparison.
- `ml_refiner/export_vulkan_int8.py`: Exports separate element-aligned `weights_i32.bin` + `params_f32.bin` + `fg_mask_layer_meta.h`, plus PyTorch-vs-INT8-DP4a parity verifier (`<= 1 LSB`).
- `ml_refiner/cs_framegen_mask_pack.comp` & `ml_refiner/cs_framegen_mask_dp4a.comp`: Dedicated `set = 1` input packer + shared-memory zero-padded DP4a kernel using `roundEven()` and `precise` arithmetic.
- `tools/fg_autotune.py`, `tools/fg_eval_harness.py`, `tools/fg_order_test.cpp`: Perceptual auto-tuner, lossless-VMAF evaluator, and enhanced order test (translation + rotation + occluder + toggleable night-vision grain).
