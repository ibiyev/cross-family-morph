"""
Linguistic distance correlation analysis.

For each (train_lang, test_lang) cell of the transfer matrix, compute
distance metrics and correlate them with transfer accuracy.

Distance metrics:
  - Suffix inventory overlap (Jaccard, from utils.morphology)
  - Vowel inventory overlap (Jaccard)
  - lang2vec syntactic distance (URIEL; Littell et al., 2017)
  - lang2vec phonological distance (URIEL)
  - lang2vec genetic distance (URIEL)
  - Same-family binary indicator
"""
import argparse
import csv
import json
import logging
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.morphology import (
    suffix_overlap,
    vowel_overlap,
    is_intra_family,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── lang2vec wrapper ──────────────────────────────────────────────

# ISO 639-3 codes used by lang2vec
LANG_TO_ISO = {
    "aze": "azj",   # North Azerbaijani
    "tur": "tur",
    "hun": "hun",
    "fin": "fin",
}


def lang2vec_distance(lang_a: str, lang_b: str, distance_type: str = "syntactic") -> float:
    """
    Compute lang2vec distance between two languages.
    Returns 1.0 (max distance) if lang2vec is unavailable.
    """
    try:
        import lang2vec.lang2vec as l2v
        iso_a = LANG_TO_ISO[lang_a]
        iso_b = LANG_TO_ISO[lang_b]
        d = l2v.distance(distance_type, iso_a, iso_b)
        if d is None:
            return 1.0
        return float(d)
    except Exception as e:
        logger.warning(f"lang2vec failed for {lang_a}-{lang_b}: {e}")
        return 1.0


def compute_all_distances(lang_a: str, lang_b: str) -> Dict[str, float]:
    """Compute all distance metrics for a language pair."""
    return {
        "suffix_overlap": suffix_overlap(lang_a, lang_b),
        "vowel_overlap": vowel_overlap(lang_a, lang_b),
        "intra_family": 1.0 if is_intra_family(lang_a, lang_b) else 0.0,
        "syntactic_distance": lang2vec_distance(lang_a, lang_b, "syntactic"),
        "phonological_distance": lang2vec_distance(lang_a, lang_b, "phonological"),
        "genetic_distance": lang2vec_distance(lang_a, lang_b, "genetic"),
    }


# ─── Correlation analysis ──────────────────────────────────────────

def pearson_correlation(xs: List[float], ys: List[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(xs, ys))
    sx = sum((xi - mx) ** 2 for xi in xs) ** 0.5
    sy = sum((yi - my) ** 2 for yi in ys) ** 0.5
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


def main():
    p = argparse.ArgumentParser(
        description="Compute distance-vs-transfer-accuracy correlations"
    )
    p.add_argument("--matrix_json", required=True,
                   help="JSON: list of {train_lang, test_lang, exact_match}")
    p.add_argument("--output_json", required=True)
    p.add_argument("--output_csv", default=None)
    args = p.parse_args()

    # Load transfer matrix
    with open(args.matrix_json, "r") as f:
        cells = json.load(f)
    logger.info(f"Loaded {len(cells)} transfer matrix cells")

    # Compute all distances + collect EM
    rows = []
    for cell in cells:
        train_lang = cell["train_lang"]
        test_lang = cell["test_lang"]
        em = cell["exact_match"]
        if train_lang == test_lang:
            continue  # skip diagonal (in-distribution)
        dists = compute_all_distances(train_lang, test_lang)
        rows.append({
            "train_lang": train_lang,
            "test_lang": test_lang,
            "exact_match": em,
            **dists,
        })

    # Compute correlations
    em_values = [r["exact_match"] for r in rows]
    correlations = {}
    metric_names = ["suffix_overlap", "vowel_overlap", "intra_family",
                    "syntactic_distance", "phonological_distance", "genetic_distance"]
    for m in metric_names:
        m_values = [r[m] for r in rows]
        correlations[m] = pearson_correlation(m_values, em_values)

    print(f"\n{'='*60}")
    print(" Pearson correlation: distance metric vs transfer EM")
    print(f"{'='*60}")
    print(f"  N (off-diagonal cells): {len(rows)}")
    print()
    for m, r in sorted(correlations.items(), key=lambda x: -abs(x[1])):
        sign = "+" if r >= 0 else "-"
        print(f"  {m:<25} r = {sign}{abs(r):.3f}")
    print(f"{'='*60}\n")

    # Save outputs
    out = {
        "n_pairs": len(rows),
        "correlations": correlations,
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Wrote {args.output_json}")

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            fieldnames = ["train_lang", "test_lang", "exact_match"] + metric_names
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r[k] for k in fieldnames})
        logger.info(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
