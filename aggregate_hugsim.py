"""Aggregate HUGSIM closed-loop results into a paper-style RC / HD-Score table.

Scans <root>/<dataset>_FPD_Drive/<scene>_<mode>/eval.json produced by closed_loop.py,
groups by dataset and difficulty (easy/medium/hard/extreme parsed from <mode>),
and prints per-difficulty and overall averages of RC and HD-Score.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

DIFFS = ["easy", "medium", "hard", "extreme"]
DIFF_ABBR = {"easy": "E", "medium": "M", "hard": "H", "extreme": "X"}


def load_scores(eval_path):
    with open(eval_path) as f:
        data = json.load(f)
    # hugsim_evaluate/calculate may return a dict or a 1-element list
    if isinstance(data, list):
        data = data[0]
    return float(data["rc"]) * 100.0, float(data["hdscore"]) * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()

    # scores[dataset][difficulty] = list of (rc, hd)
    scores = defaultdict(lambda: defaultdict(list))
    n_found = 0
    for eval_path in glob.glob(os.path.join(args.root, "*", "*", "eval.json")):
        parent = os.path.basename(os.path.dirname(eval_path))          # scene-0383_hard_00
        ds_dir = os.path.basename(os.path.dirname(os.path.dirname(eval_path)))  # nuscenes_drivor / nuscenes_FPD_Drive
        # Dataset names contain no underscores; agent names may (FPD_Drive).
        dataset = ds_dir.split("_", 1)[0]
        parts = parent.split("_")
        difficulty = next((p for p in parts if p in DIFFS), None)
        if difficulty is None:
            continue
        try:
            rc, hd = load_scores(eval_path)
        except Exception as e:
            print(f"  [skip] {eval_path}: {e}")
            continue
        scores[dataset][difficulty].append((rc, hd))
        n_found += 1

    print(f"\nFound {n_found} completed scenarios under {args.root}\n")

    def avg(vals, idx):
        return sum(v[idx] for v in vals) / len(vals) if vals else float("nan")

    header = f"{'dataset':<12} {'metric':<9} " + " ".join(f"{DIFF_ABBR[d]:>7}" for d in DIFFS) + f" {'Avg':>7} {'n':>5}"
    print(header)
    print("-" * len(header))

    all_rc, all_hd = [], []
    for dataset in sorted(scores):
        per_diff = scores[dataset]
        flat = [v for d in DIFFS for v in per_diff.get(d, [])]
        all_rc += [v[0] for v in flat]
        all_hd += [v[1] for v in flat]
        rc_row = " ".join(f"{avg(per_diff.get(d, []), 0):7.1f}" for d in DIFFS)
        hd_row = " ".join(f"{avg(per_diff.get(d, []), 1):7.1f}" for d in DIFFS)
        rc_avg = sum(v[0] for v in flat) / len(flat) if flat else float("nan")
        hd_avg = sum(v[1] for v in flat) / len(flat) if flat else float("nan")
        print(f"{dataset:<12} {'RC':<9} {rc_row} {rc_avg:7.1f} {len(flat):5d}")
        print(f"{dataset:<12} {'HD-Score':<9} {hd_row} {hd_avg:7.1f} {len(flat):5d}")

    print("-" * len(header))
    if all_rc:
        print(f"{'OVERALL':<12} {'RC':<9} {'':>39} {sum(all_rc)/len(all_rc):7.1f} {len(all_rc):5d}")
        print(f"{'OVERALL':<12} {'HD-Score':<9} {'':>39} {sum(all_hd)/len(all_hd):7.1f} {len(all_hd):5d}")


if __name__ == "__main__":
    main()
