# Video indexing at speed

Results note: results.jsonl.gz covers Parquet files 0-26 only (26+1 of the 50 files, 891 videos),
produced with the default settings of this run.py (--samples 4, FP16). The remaining files were
processed on a second machine (Kaggle) but that output was lost before it was saved, and there
was no time left to redo it. The script works on any folder of Parquet files.

For every second of every video in a folder of FineVideo Parquet files, record which of
`person, car, dog, laptop, cell phone` appear and how many people are visible
(YOLO11l, `imgsz=640`, `conf>=0.4`, IoU 0.7). Output format and checks follow the brief.

## Install (grader steps)

```bash
# system packages (Ubuntu 22.04)
sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0

python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Preparation command (needs internet, not timed)

```bash
python run.py --prepare
```

Downloads `yolo11l.pt` into the repo directory. The timed run needs no internet.

## Run

```bash
python run.py --input <folder of .parquet files> --output results.jsonl
```

Optional flags: `--samples N` (sampled frames per second, default 4; `0` = every frame),
`--batch N` (GPU batch, default 64), `--workers N` (decoder processes, default CPUs-2),
`--no-half` (FP32), `--rect` (minimal-rectangle letterbox), `--limit N` (first N files).

## How it works

1. A producer process reads the `mp4` column of each Parquet file one row at a time.
2. Decoder worker processes decode with PyAV straight from the in-memory bytes. Seconds come
   from frame timestamps relative to the first decoded frame (not from frame count x fps).
   Frames are rotated if the video has a rotation tag. Only the sampled frames are converted to
   BGR arrays and letterboxed on the CPU with Ultralytics' exact arithmetic (checked
   byte-for-byte against `ultralytics.data.augment.LetterBox`).
3. The main process batches frames and runs YOLO11l in FP16 on the GPU, restricted to the five
   classes. Decoding runs in other processes, so it overlaps with GPU work.
4. Per second: an object is present if found in at least half of the sampled frames; people =
   median of the per-frame person counts, rounded down at the half.
5. One JSON line is written for every second up to the last frame's second. Failures give an
   error line (lines already produced for that video are kept). A damaged Parquet file or a
   crashed worker never stops the run.

## Checking correctness

`--samples 0 --no-half` runs every frame in FP32, which is the brief's reference method.
```bash
python run.py --input finevideo/data --output ref.jsonl  --samples 0 --no-half
python run.py --input finevideo/data --output fast.jsonl
python score.py --ref ref.jsonl --cand fast.jsonl     # object F1, people exact / within-1, PASS/FAIL
```

## Running all 50 files on Colab / Kaggle (limited disk)

Download and process 10 files at a time (about 5 GB), keep each batch's output, then merge:
```bash
for i in 0 1 2 3 4; do
  hf download HuggingFaceFV/finevideo --repo-type dataset --include "data/train-000${i}*" --local-dir finevideo
  python run.py --input finevideo/data --output out_${i}.jsonl
  rm -rf finevideo/data
done
cat out_0.jsonl out_1.jsonl out_2.jsonl out_3.jsonl out_4.jsonl | gzip > results.jsonl.gz
```
Video ids contain the Parquet file name, so batches never collide. Copy each `out_i.jsonl` to Drive
(Colab) or keep it in `/kaggle/working` as soon as it finishes.

## Tests included

- `score.py` implements the brief's three thresholds.
- `python run.py --input <dir> --output x.jsonl --dry-run` exercises the whole pipeline
  (Parquet, decode, rotation, bucketing, error lines) without a model or GPU.
