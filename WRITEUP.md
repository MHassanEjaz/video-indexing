# Write-up

Write these answers yourself, in your own words, without AI.

## 1. Your timings (100 words)
The all-frames FP32 baseline took about 1,904 s on 5 videos. The fast version took about 152 s on the same videos. A full file took about 12 minutes.

## 2. Your pipeline (150 words)

The pipeline processes videos stored in Parquet files across three decoupled multiprocessing stages to maximize GPU utilization and CPU throughput:

1. **Producer Process:** Streams raw MP4 bytes row-by-row from `.parquet` files into a bounded task queue, isolating disk I/O from compute.
2. **Decoder Workers:** Spawned CPU workers decode videos via PyAV, handle rotation metadata, and subsample frames (defaulting to 4 frames per second). Each sampled frame is letterboxed to a uniform 640×640 tensor on the CPU using Ultralytics-identical padding math, then sent to the main process in chunks.
3. **Main GPU Process:** Groups incoming frames by shape into dynamic batches (up to 64 frames) and runs FP16 inference using `yolo11l.pt` (`conf >= 0.4`), featuring automatic batch-halving fallback on out-of-memory errors. Finally, detections are aggregated per second—requiring >=50% frame presence for object labels and taking the median for person counts—before streaming directly to a JSONL file.

## 3. Where the time goes (150 words)
The benchmark results reveal a clear shift from CPU-bound decoding to GPU-bound inference as the sampling rate increases:

* **Baseline CPU Floor (57s):** The `dry_decode_only` run proves that reading Parquet files, decoding H.264/MP4 streams via PyAV, and letterboxing frames on CPU workers requires a fixed baseline of ~57 seconds.
* **GPU Inference Scaling:** At 3 samples/sec (`fast_3`), total runtime is 96s—meaning GPU inference adds only ~39s on top of decoding. However, runtime scales almost linearly with frame count: jumping to 5 samples/sec (`fast_5`, 161s) and 8 samples/sec (`fast_8`, 248s) pushes the workload firmly into a **GPU-compute bottleneck**, as PyAV still decodes every frame regardless of sample rate while the GPU processes 1.6× to 2.6× more batches.
* **Precision Overhead:** Comparing `fast_4_fp32` (238s) against `fast_5` in FP16 (161s) shows that disabling half-precision nearly doubles inference latency without improving accuracy, confirming FP16 tensor cores are essential for throughput.

## 4. Checking correctness (100 words)
Each optimized run was evaluated via `score.py` against a baseline reference generated with `--samples 0` (every frame) and `--no-half` (FP32) using exact Ultralytics letterbox padding (`rect=False`).

`score.py` reports three accuracy metrics: **objects F1** (target ≥ 0.95), **people exact** (≥ 0.85), and **people within 1** (≥ 0.95). Testing revealed that FP16 matched FP32 precision (`0.9685` vs `0.9742` F1) at roughly half the runtime, while `fast_3` cleared all thresholds (`0.9620` F1, `0.9104` exact, `0.9909` within-1) with comfortable margins, eliminating the need for higher sample rates. Finally, validation checks verified continuous per-second indexing without gaps, graceful error logging for corrupted videos/Parquet files, and clean execution inside the pinned environment.

## 5. Working with AI (100 words)

First AI give training in one platform which is much longer and I don't follow and prompt the AI to train half half on two platforms
## 6. Next steps (100 words)

optimizing and improving model.