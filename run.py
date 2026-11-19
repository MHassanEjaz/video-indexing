#!/usr/bin/env python
"""Video indexing: for each second of each video, record which of five objects
appear and how many people are visible (YOLO11l, imgsz 640, conf >= 0.4).

    python run.py --input <folder of .parquet files> --output results.jsonl

Architecture
    producer process   reads the `mp4` column of each Parquet file row by row
        |  (video bytes)
    N decoder workers  PyAV decode -> pick sample frames -> Ultralytics-identical
        |              letterbox on CPU -> small uint8 arrays
        |  (frame chunks)
    main process       batches same-shape frames -> YOLO11l on GPU (FP16)
                       -> per-second aggregation -> JSONL lines

Setting --samples 0 uses every frame (and --no-half gives FP32): that is the
"baseline"/reference mode used by score.py to check faster settings.
"""
import argparse
import glob
import io
import json
import multiprocessing as mp
import os
import queue
import sys
import time

# Only light imports at module level: spawned workers re-import this file and
# must not pull in torch/CUDA.
import numpy as np

NAMES = ["person", "car", "dog", "laptop", "cell phone"]
COCO_IDS = [0, 2, 16, 63, 67]  # COCO class ids of NAMES, same order
ID2BIT = {c: i for i, c in enumerate(COCO_IDS)}
IMGSZ = 640
CONF = 0.4
IOU = 0.7
STRIDE = 32
HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# Letterbox: exact copy of Ultralytics' LetterBox arithmetic, so the model sees
# the same pixels as if Ultralytics had resized the frame itself.
#   rect=False (default): pad to a square 640x640. This is what predict() does by
#       default in ultralytics 8.3.2xx / 8.4.x (args.rect defaults to False).
#   rect=True: pad only to a multiple of 32 (what ultralytics <= ~8.3.0 did for
#       same-shape batches). Use --rect only if the reference was made that way.
# Frames leave the worker already letterboxed to a fixed shape, so Ultralytics'
# own letterbox is then a no-op.
# --------------------------------------------------------------------------
def letterbox(img, rect=False, new=IMGSZ, stride=STRIDE):
    import cv2

    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = new - nw, new - nh
    if rect:
        dw, dh = dw % stride, dh % stride
    dw /= 2
    dh /= 2
    if (w, h) != (nw, nh):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )


# --------------------------------------------------------------------------
# Per-second aggregation (spec section 5)
# --------------------------------------------------------------------------
def aggregate(frames):
    """frames: list of (mask, people) for one second -> (objects, people).

    Object present if found in >= half of the frames. People = median of the
    per-frame counts, rounded down when halfway between two integers.
    """
    n = len(frames)
    if n == 0:
        return [], 0
    objs = []
    for bit, name in enumerate(NAMES):
        c = sum(1 for m, _ in frames if (m >> bit) & 1)
        if 2 * c >= n:
            objs.append(name)
    p = sorted(pc for _, pc in frames)
    people = p[n // 2] if n % 2 else (p[n // 2 - 1] + p[n // 2]) // 2
    return objs, people


# --------------------------------------------------------------------------
# Producer: Parquet -> video bytes
# --------------------------------------------------------------------------
def producer(files, task_q, res_q, n_workers):
    import pyarrow.parquet as pq

    for f in files:
        name = os.path.basename(f)
        row = 0
        try:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=1, columns=["mp4"]):
                data = batch.column(0)[0].as_py()
                task_q.put((f"{name}:{row}", data))
                row += 1
        except Exception as e:  # damaged file: keep going with the next one
            res_q.put(("lines", [{"video": f"{name}:{row}", "error": f"parquet read failed: {str(e)[:80]}"}]))
    for _ in range(n_workers):
        task_q.put(None)


# --------------------------------------------------------------------------
# Decoder worker: video bytes -> sampled, letterboxed frames
# --------------------------------------------------------------------------
def _rotation_k(frame, stream):
    """Number of 90-degree counter-clockwise turns (np.rot90 k) to upright a frame."""
    rot = getattr(frame, "rotation", 0) or 0  # display-matrix, degrees CCW
    if rot:
        return int(round(rot / 90.0)) % 4
    tag = None
    try:
        tag = stream.metadata.get("rotate")
    except Exception:
        pass
    if tag:
        try:  # legacy 'rotate' tag: degrees clockwise to apply
            return (-int(round(float(tag) / 90.0))) % 4
        except ValueError:
            pass
    return 0


def worker(task_q, res_q, samples, thread_count, chunk, rect):
    import av
    import cv2

    cv2.setNumThreads(1)
    while True:
        item = task_q.get()
        if item is None:
            res_q.put(("wdone",))
            return
        vid, data = item
        res_q.put(("start", vid))
        secs_buf, frames_buf = [], []
        last_sec, err = -1, None

        def flush():
            if frames_buf:
                res_q.put(("frames", vid, list(secs_buf), np.stack(frames_buf)))
                secs_buf.clear()
                frames_buf.clear()

        try:
            if not data:
                raise ValueError("empty video")
            container = av.open(io.BytesIO(data))
            vs = next((s for s in container.streams if s.type == "video"), None)
            if vs is None:
                raise ValueError("no video stream")
            vs.thread_type = "AUTO"
            vs.codec_context.thread_count = thread_count
            fps = float(getattr(vs, "average_rate", None) or getattr(vs, "guessed_rate", None) or 25)
            t0, k_rot, last_slot, idx = None, None, None, 0
            try:
                for frame in container.decode(vs):
                    t = frame.time
                    if t is None:
                        t = idx / fps
                    idx += 1
                    if t0 is None:
                        t0 = t
                        k_rot = _rotation_k(frame, vs)
                    rel = max(0.0, t - t0)
                    sec = int(rel)
                    last_sec = max(last_sec, sec)
                    if samples > 0:
                        slot = (sec, int((rel - sec) * samples))
                        if slot == last_slot:
                            continue
                        last_slot = slot
                    img = frame.to_ndarray(format="bgr24")
                    if k_rot:
                        img = np.ascontiguousarray(np.rot90(img, k_rot))
                    lb = letterbox(img, rect)
                    if frames_buf and lb.shape != frames_buf[0].shape:
                        flush()  # resolution changed mid-video: never stack mixed shapes
                    secs_buf.append(sec)
                    frames_buf.append(lb)
                    if len(frames_buf) >= chunk:
                        flush()
            except Exception as e:
                err = f"decode failed: {str(e)[:80]}"
            finally:
                container.close()
            if t0 is None and err is None:
                err = "could not decode video"
        except Exception as e:
            err = f"could not decode video: {str(e)[:60]}"
        flush()
        res_q.put(("end", vid, last_sec, err))


# --------------------------------------------------------------------------
# Main process: GPU inference + output
# --------------------------------------------------------------------------
class Dry:
    """Stand-in model for pipeline tests (no torch needed): detects nothing."""

    def predict(self, frames, **kw):
        return [None] * len(frames)


def load_model(path, dry):
    if dry:
        return Dry(), False, "cpu"
    import torch
    from ultralytics import YOLO

    if not os.path.exists(path):
        local = os.path.join(HERE, path)
        path = local if os.path.exists(local) else path
    model = YOLO(path)
    dev = 0 if torch.cuda.is_available() else "cpu"
    return model, dev != "cpu", dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="folder of .parquet files")
    ap.add_argument("--output", default="results.jsonl")
    ap.add_argument("--samples", type=int, default=4,
                    help="sampled frames per second (0 = every frame = reference mode)")
    ap.add_argument("--batch", type=int, default=64, help="GPU batch size")
    ap.add_argument("--workers", type=int, default=0, help="decoder processes (0 = auto)")
    ap.add_argument("--weights", default="yolo11l.pt")
    ap.add_argument("--rect", action="store_true", help="minimal-rectangle letterbox (old Ultralytics behaviour)")
    ap.add_argument("--no-half", action="store_true", help="FP32 inference")
    ap.add_argument("--limit", type=int, default=0, help="only first N parquet files (dev)")
    ap.add_argument("--prepare", action="store_true", help="download weights and exit")
    ap.add_argument("--dry-run", action="store_true", help="skip the model (pipeline test)")
    args = ap.parse_args()

    if args.prepare:
        from ultralytics import YOLO

        YOLO(args.weights)  # downloads yolo11l.pt into the current directory
        print("weights ready:", os.path.abspath(args.weights))
        return
    if not args.input:
        ap.error("--input is required")

    t_start = time.time()
    files = sorted(glob.glob(os.path.join(args.input, "*.parquet")))
    if args.limit:
        files = files[: args.limit]
    ncpu = os.cpu_count() or 2
    n_workers = args.workers or max(1, ncpu - 2)
    thread_count = max(1, ncpu // n_workers)
    chunk = 16

    # Start CPU-side processes BEFORE initialising CUDA (spawn keeps them clean).
    ctx = mp.get_context("spawn")
    task_q = ctx.Queue(maxsize=n_workers * 2)
    res_q = ctx.Queue(maxsize=n_workers * 3)
    prod = ctx.Process(target=producer, args=(files, task_q, res_q, n_workers), daemon=True)
    workers = [
        ctx.Process(target=worker, args=(task_q, res_q, args.samples, thread_count, chunk, args.rect), daemon=True)
        for _ in range(n_workers)
    ]
    prod.start()
    for w in workers:
        w.start()

    model, use_half, device = load_model(args.weights, args.dry_run)
    use_half = use_half and not args.no_half
    if not args.dry_run:  # warm-up so CUDA init is not counted per batch
        model.predict(np.zeros((IMGSZ, IMGSZ, 3), np.uint8), imgsz=IMGSZ, device=device, half=use_half, verbose=False)

    out = open(args.output, "w", buffering=1 << 20)
    states = {}
    buckets = {}  # (h, w) -> list of (vid, sec, frame)
    buffered = 0
    n_done_workers = 0
    n_videos = 0

    def write(d):
        out.write(json.dumps(d) + "\n")

    def finish(vid):
        nonlocal n_videos
        st = states.pop(vid)
        n_videos += 1
        for s in range(st["last_sec"] + 1):
            objs, people = aggregate(st["secs"].get(s, []))
            write({"video": vid, "second": s, "objects": objs, "people": people})
        err = st["err"]
        if err is None and st["last_sec"] < 0:
            err = "could not decode video"
        if err:
            write({"video": vid, "error": err})

    def maybe_finish(vid):
        st = states.get(vid)
        if st and st["ended"] and st["done"] == st["sent"]:
            finish(vid)

    def infer(items):
        """Run the model on same-shape frames; returns list of (mask, people)."""
        frames = [x[2] for x in items]
        try:
            res = model.predict(frames, imgsz=IMGSZ, conf=CONF, iou=IOU, classes=COCO_IDS,
                                half=use_half, device=device, verbose=False, rect=args.rect)
        except Exception:
            if len(items) == 1:
                return [None]
            h = len(items) // 2  # e.g. out of memory: retry in halves
            return infer(items[:h]) + infer(items[h:])
        outp = []
        for r in res:
            if r is None:
                outp.append((0, 0))
                continue
            cls = r.boxes.cls.cpu().numpy().astype(int)
            mask, people = 0, 0
            for c in cls:
                mask |= 1 << ID2BIT[int(c)]
                people += int(c == 0)
            outp.append((mask, people))
        return outp

    def flush(key):
        nonlocal buffered
        items = buckets.pop(key, [])
        if not items:
            return
        buffered -= len(items)
        results = infer(items)
        touched = set()
        for (vid, sec, _), r in zip(items, results):
            st = states[vid]
            if r is None:
                st["err"] = st["err"] or "inference failed"
            else:
                st["secs"].setdefault(sec, []).append(r)
            st["done"] += 1
            touched.add(vid)
        for vid in touched:
            maybe_finish(vid)

    def flush_all():
        for key in list(buckets):
            flush(key)

    while n_done_workers < n_workers:
        try:
            msg = res_q.get(timeout=0.2)
        except queue.Empty:
            flush_all()
            if not any(w.is_alive() for w in workers):
                break  # every worker is gone (crash); finalise what we have
            continue
        kind = msg[0]
        if kind == "wdone":
            n_done_workers += 1
        elif kind == "lines":
            for d in msg[1]:
                write(d)
        elif kind == "start":
            states[msg[1]] = {"secs": {}, "last_sec": -1, "sent": 0, "done": 0, "ended": False, "err": None}
        elif kind == "frames":
            _, vid, secs, arr = msg
            st = states[vid]
            st["sent"] += len(secs)
            key = arr.shape[1:3]
            lst = buckets.setdefault(key, [])
            for s, a in zip(secs, arr):
                lst.append((vid, s, a))
            buffered += len(secs)
            if len(lst) >= args.batch:
                flush(key)
            elif buffered > 4 * args.batch:  # many odd shapes: bound memory
                flush(max(buckets, key=lambda k: len(buckets[k])))
        elif kind == "end":
            _, vid, last_sec, err = msg
            st = states[vid]
            st["ended"], st["last_sec"], st["err"] = True, last_sec, err
            maybe_finish(vid)

    flush_all()
    for vid in list(states):  # videos cut off by a crashed worker
        st = states[vid]
        st["ended"] = True
        st["err"] = st["err"] or "worker crashed"
        finish(vid)
    out.close()
    for p in [prod] + workers:
        if p.is_alive():
            p.terminate()
    print(f"{n_videos} videos in {time.time() - t_start:.1f}s -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
