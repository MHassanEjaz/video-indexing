#!/usr/bin/env python
"""Compare a results file with a reference using the brief's section 5 checks.

    python score.py --ref reference.jsonl --cand results.jsonl

Accepts .jsonl or .jsonl.gz. Seconds that exist in the reference but are missing
from the candidate count as "nothing found" (empty objects, 0 people). Videos
that have an error line in the reference are skipped.
"""
import argparse
import gzip
import json
import sys


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    secs, errs = {}, set()
    with op(path, "rt") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if "error" in d:
                errs.add(d["video"])
            else:
                secs[(d["video"], d["second"])] = (frozenset(d["objects"]), d["people"])
    return secs, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", required=True)
    a = ap.parse_args()
    ref, ref_err = load(a.ref)
    cand, cand_err = load(a.cand)

    tp = fp = fn = 0
    exact = within1 = total = 0
    for key, (robj, rppl) in ref.items():
        if key[0] in ref_err:
            continue
        cobj, cppl = cand.get(key, (frozenset(), 0))
        tp += len(robj & cobj)
        fp += len(cobj - robj)
        fn += len(robj - cobj)
        total += 1
        d = abs(cppl - rppl)
        exact += d == 0
        within1 += d <= 1
    extra = sum(1 for k in cand if k not in ref and k[0] not in ref_err)

    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    ex, w1 = exact / total, within1 / total
    print(f"seconds compared : {total}   (extra seconds in candidate: {extra})")
    print(f"objects F1       : {f1:.4f}  (precision {prec:.4f}, recall {rec:.4f})   need >= 0.95")
    print(f"people exact     : {ex:.4f}   need >= 0.85")
    print(f"people within 1  : {w1:.4f}   need >= 0.95")
    print(f"error videos     : ref={len(ref_err)} cand={len(cand_err)}")
    ok = f1 >= 0.95 and ex >= 0.85 and w1 >= 0.95
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
