"""
Morfessor 2.0 baseline for unsupervised morpheme segmentation.

Trains Morfessor on word frequencies extracted from the training TSV,
then evaluates on the test TSV using exact match against the supervised gold.

Reference: Virpioja et al. (2013). Morfessor 2.0: Python Implementation and
Extensions for Morfessor Baseline. Aalto University Publication Series.
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter


def train_morfessor(train_tsv: str):
    """Train a Morfessor Baseline model on word frequencies from train TSV."""
    try:
        import morfessor
    except ImportError:
        print("ERROR: morfessor not installed. Run: pip install morfessor")
        sys.exit(1)

    # Build word frequency dictionary from inputs
    counts = Counter()
    with open(train_tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            counts[row["input"]] += 1

    io = morfessor.MorfessorIO()
    model = morfessor.BaselineModel()
    # Format expected: list of (count, word) tuples
    model.load_data([(c, w) for w, c in counts.items()])
    model.train_batch()
    return model


def evaluate_morfessor(model, test_tsv: str) -> dict:
    """Evaluate Morfessor segmentations against gold TSV."""
    n_total = 0
    n_em = 0
    n_stem = 0
    with open(test_tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            inp = row["input"]
            gold = row["target"].strip()
            try:
                segments, _ = model.viterbi_segment(inp)
            except Exception:
                segments = [inp]
            pred = " + ".join(segments)
            n_total += 1
            if pred == gold:
                n_em += 1
            p_parts = [s.strip() for s in pred.split("+")]
            g_parts = [s.strip() for s in gold.split("+")]
            if p_parts and g_parts and p_parts[0] == g_parts[0]:
                n_stem += 1
    return {
        "n_total": n_total,
        "exact_match": n_em / max(n_total, 1),
        "stem_accuracy": n_stem / max(n_total, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_tsv", required=True)
    p.add_argument("--test_tsv", required=True)
    p.add_argument("--output_json", default=None)
    p.add_argument("--label", default=None)
    args = p.parse_args()

    print(f"Training Morfessor on: {args.train_tsv}")
    model = train_morfessor(args.train_tsv)
    print(f"Evaluating on: {args.test_tsv}")
    metrics = evaluate_morfessor(model, args.test_tsv)

    print(f"\n{'='*55}")
    print(f"  Morfessor 2.0 baseline" + (f" — {args.label}" if args.label else ""))
    print(f"{'='*55}")
    print(f"  Total examples:    {metrics['n_total']:>10,}")
    print(f"  Exact match:       {metrics['exact_match']:>10.4f}")
    print(f"  Stem accuracy:     {metrics['stem_accuracy']:>10.4f}")
    print(f"{'='*55}\n")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump({"label": args.label, "metrics": metrics}, f, indent=2)


if __name__ == "__main__":
    main()
