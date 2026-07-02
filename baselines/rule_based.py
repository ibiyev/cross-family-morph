"""
Rule-based morpheme segmentation baseline for AZ, TR, HU, FI.

For each test word, greedily strips the longest matching suffix from the end,
repeating until no suffix matches or the remaining stem is too short.

This is the simplest possible baseline. It establishes a floor for how much
the neural model improves.
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.morphology import SUFFIX_INVENTORIES


def rule_segment(word: str, lang: str, min_stem: int = 2) -> str:
    """Greedy longest-match suffix stripping from the end of the word."""
    suffixes = SUFFIX_INVENTORIES[lang]
    stripped = []
    remaining = word
    progress = True
    while progress:
        progress = False
        for sf in suffixes:
            if remaining.endswith(sf) and len(remaining) > len(sf) + min_stem - 1:
                stripped.insert(0, sf)
                remaining = remaining[: -len(sf)]
                progress = True
                break
    if stripped:
        return remaining + " + " + " + ".join(stripped)
    return word


def evaluate_baseline(test_path: str, lang: str) -> dict:
    """Evaluate the rule-based segmenter on a TSV test set."""
    n_total = 0
    n_em = 0
    n_stem = 0
    n_suffix = 0
    with open(test_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            inp = row["input"]
            gold = row["target"].strip()
            pred = rule_segment(inp, lang).strip()
            n_total += 1
            if pred == gold:
                n_em += 1
            p_parts = [s.strip() for s in pred.split("+")]
            g_parts = [s.strip() for s in gold.split("+")]
            if p_parts and g_parts and p_parts[0] == g_parts[0]:
                n_stem += 1
            if len(p_parts) > 1 and len(g_parts) > 1 and p_parts[1:] == g_parts[1:]:
                n_suffix += 1
            elif len(p_parts) == len(g_parts) == 1:
                n_suffix += 1
    return {
        "n_total": n_total,
        "exact_match": n_em / max(n_total, 1),
        "stem_accuracy": n_stem / max(n_total, 1),
        "suffix_accuracy": n_suffix / max(n_total, 1),
    }


def main():
    p = argparse.ArgumentParser(
        description="Rule-based morpheme segmentation baseline"
    )
    p.add_argument("--test_tsv", required=True)
    p.add_argument("--lang", required=True, choices=["aze", "tur", "hun", "fin"])
    p.add_argument("--output_json", default=None)
    p.add_argument("--show_examples", type=int, default=10,
                   help="Show N example predictions")
    args = p.parse_args()

    metrics = evaluate_baseline(args.test_tsv, args.lang)

    print(f"\n{'='*55}")
    print(f"  Rule-based baseline ({args.lang})")
    print(f"{'='*55}")
    print(f"  Test file:        {args.test_tsv}")
    print(f"  Total examples:   {metrics['n_total']:>10,}")
    print(f"  Exact match:      {metrics['exact_match']:>10.4f}")
    print(f"  Stem accuracy:    {metrics['stem_accuracy']:>10.4f}")
    print(f"  Suffix accuracy:  {metrics['suffix_accuracy']:>10.4f}")
    print(f"{'='*55}\n")

    if args.show_examples > 0:
        print(f"Example predictions (first {args.show_examples}):")
        with open(args.test_tsv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for i, row in enumerate(reader):
                if i >= args.show_examples:
                    break
                pred = rule_segment(row["input"], args.lang)
                marker = "✓" if pred.strip() == row["target"].strip() else "✗"
                print(f"  {marker} {row['input']:<25} pred: {pred:<35} gold: {row['target']}")
        print()

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump({"language": args.lang,
                       "test_tsv": args.test_tsv,
                       "metrics": metrics}, f, indent=2)


if __name__ == "__main__":
    main()
