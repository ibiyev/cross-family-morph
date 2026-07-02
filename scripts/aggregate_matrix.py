"""
Aggregate per-cell cross-evaluation JSON files into a single transfer matrix.

Reads files like results/matrix/{train}_to_{test}_seed{N}.json
Writes a single JSON list ready for distance_correlation.py.
"""
import argparse
import glob
import json
import os
import re


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    pattern = os.path.join(args.results_dir,
                            f"*_to_*_seed{args.seed}.json")
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} cell files matching seed={args.seed}")

    rx = re.compile(r"(?P<train>\w+)_to_(?P<test>\w+)_seed(?P<seed>\d+)\.json$")
    cells = []
    for fp in files:
        m = rx.search(os.path.basename(fp))
        if not m:
            continue
        with open(fp, "r") as f:
            d = json.load(f)
        cells.append({
            "train_lang": m.group("train"),
            "test_lang": m.group("test"),
            "seed": int(m.group("seed")),
            "exact_match": d["metrics"]["exact_match"],
            "stem_accuracy": d["metrics"]["stem_accuracy"],
            "suffix_accuracy": d["metrics"]["suffix_accuracy"],
            "n_total": d["metrics"]["n_total"],
        })

    # Print matrix
    langs = sorted(set(c["train_lang"] for c in cells)
                   | set(c["test_lang"] for c in cells))
    print(f"\n{'='*70}")
    print(f"  Transfer matrix (Exact Match) — seed={args.seed}")
    print(f"{'='*70}")
    print("           " + "".join(f"  {l:>8}" for l in langs))
    em_lookup = {(c["train_lang"], c["test_lang"]): c["exact_match"]
                 for c in cells}
    for tr in langs:
        row_str = f"  {tr:<6} →  "
        for te in langs:
            v = em_lookup.get((tr, te))
            row_str += f"  {v*100:>7.2f}%" if v is not None else "  ──────  "
        print(row_str)
    print(f"{'='*70}\n")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(cells, f, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
