"""
Build morpheme segmentation training data for AZ, TR, HU, FI from real sources.

Sources (all open-license):
  - UniMorph (Batsuren et al., 2022, LREC): aze, tur, hun, fin
    https://github.com/unimorph/{lang}
  - Universal Dependencies (Nivre et al., 2020, LREC):
    UD_Azerbaijani-TueCL, UD_Turkish-IMST, UD_Turkish-BOUN,
    UD_Hungarian-Szeged, UD_Finnish-TDT
  - SIGMORPHON 2022 morpheme segmentation (Batsuren et al., 2022):
    word.train, word.dev for hun (Hungarian, only language we share)

Output: TSV with columns `input` and `target` where target uses ' + ' separator.
Example: "Bakıda" -> "Bakı + da"
"""
import argparse
import csv
import logging
import os
import sys
import urllib.request
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# Ensure utils on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.morphology import (
    VERB_INFINITIVE_SUFFIXES,
    CONSONANT_MUTATIONS,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── Conversion engine ──────────────────────────────────────────────

def inflection_to_segmentation(
    lemma: str, inflected: str, lang: str
) -> Optional[Tuple[str, str]]:
    """
    Convert (lemma, inflected_form) → (inflected_form, "stem + suffix").

    Returns None if the pair cannot be cleanly segmented.
    """
    if not lemma or not inflected:
        return None

    # Identity: no morphology
    if lemma == inflected:
        return (inflected, inflected)

    # Case 1: simple suffixation (most common)
    if inflected.startswith(lemma):
        suffix = inflected[len(lemma):]
        return (inflected, f"{lemma} + {suffix}") if suffix else (inflected, inflected)

    # Case 2: verb infinitive stripping
    for inf_suffix in VERB_INFINITIVE_SUFFIXES.get(lang, []):
        if lemma.endswith(inf_suffix):
            stem = lemma[:-len(inf_suffix)]
            if stem and inflected.startswith(stem):
                suffix = inflected[len(stem):]
                if suffix:
                    return (inflected, f"{stem} + {suffix}")

    # Case 3: consonant mutation (single-char mutations only)
    mutations = CONSONANT_MUTATIONS.get(lang, {})
    if len(lemma) >= 2 and lemma[-1] in mutations:
        target = mutations[lemma[-1]]
        if target:
            mutated = lemma[:-1] + target
            if inflected.startswith(mutated):
                suffix = inflected[len(mutated):]
                if suffix:
                    return (inflected, f"{mutated} + {suffix}")

    # Case 4: longest common prefix (fallback)
    common = 0
    for i in range(min(len(lemma), len(inflected))):
        if lemma[i] == inflected[i]:
            common = i + 1
        else:
            break
    if common >= 2:
        stem = inflected[:common]
        suffix = inflected[common:]
        if suffix:
            return (inflected, f"{stem} + {suffix}")

    return None


def convert_triples(
    triples: List[Tuple[str, str, str]],
    source: str,
    lang: str,
    min_suffix_len: int = 1,
    max_word_len: int = 50,
) -> List[Tuple[str, str]]:
    """Convert (lemma, inflected, features) triples into segmentation pairs."""
    pairs, stats = [], defaultdict(int)
    for lemma, inflected, _ in triples:
        if " " in lemma or " " in inflected:
            stats["skip_multiword"] += 1
            continue
        if not (2 <= len(inflected) <= max_word_len):
            stats["skip_length"] += 1
            continue
        result = inflection_to_segmentation(lemma, inflected, lang)
        if result is None:
            stats["skip_unsegmentable"] += 1
            continue
        inp, tgt = result
        parts = tgt.split(" + ")
        if len(parts) > 1:
            suffix_total = sum(len(p) for p in parts[1:])
            if suffix_total >= min_suffix_len:
                pairs.append((inp, tgt))
                stats["suffixed"] += 1
            else:
                stats["skip_short_suffix"] += 1
        else:
            pairs.append((inp, tgt))
            stats["bare"] += 1
    unique = list(set(pairs))
    logger.info(
        f"  [{source}] {len(triples):,} triples -> {len(unique):,} unique "
        f"({stats['suffixed']} suffixed, {stats['bare']} bare, "
        f"{stats['skip_unsegmentable']} unsegmentable)"
    )
    return unique


# ─── UniMorph loader (raw GitHub TSV) ───────────────────────────────

UNIMORPH_URLS = {
    "aze": "https://raw.githubusercontent.com/unimorph/aze/master/aze",
    "tur": "https://raw.githubusercontent.com/unimorph/tur/master/tur",
    "hun": "https://raw.githubusercontent.com/unimorph/hun/master/hun",
    "fin": "https://raw.githubusercontent.com/unimorph/fin/master/fin",
}


def load_unimorph(lang: str, cache_dir: str) -> List[Tuple[str, str, str]]:
    """Download UniMorph TSV and parse to (lemma, inflected, features) triples."""
    if lang not in UNIMORPH_URLS:
        return []
    cache_file = os.path.join(cache_dir, f"unimorph_{lang}.tsv")
    if not os.path.exists(cache_file):
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"  Downloading UniMorph {lang}...")
        try:
            urllib.request.urlretrieve(UNIMORPH_URLS[lang], cache_file)
        except Exception as e:
            logger.warning(f"  Failed to download UniMorph {lang}: {e}")
            return []
    triples = []
    with open(cache_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                triples.append((parts[0], parts[1], parts[2]))
            elif len(parts) == 2:
                triples.append((parts[0], parts[1], ""))
    logger.info(f"  UniMorph {lang}: {len(triples):,} entries")
    return triples


# ─── Universal Dependencies loader (.conllu files) ──────────────────

UD_TREEBANKS: Dict[str, List[str]] = {
    "aze": ["UD_Azerbaijani-TueCL"],
    "tur": ["UD_Turkish-IMST", "UD_Turkish-BOUN"],
    "hun": ["UD_Hungarian-Szeged"],
    "fin": ["UD_Finnish-TDT"],
}


def parse_conllu_dir(dir_path: str) -> List[Tuple[str, str, str]]:
    """Parse all .conllu files in dir_path, extract (lemma, form, feats)."""
    triples = []
    if not os.path.isdir(dir_path):
        return triples
    for root, _, files in os.walk(dir_path):
        for fname in files:
            if not fname.endswith(".conllu"):
                continue
            with open(os.path.join(root, fname), "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    cols = line.split("\t")
                    # Skip multi-word tokens (e.g., "1-2" or "1.1")
                    if len(cols) < 6 or "-" in cols[0] or "." in cols[0]:
                        continue
                    form, lemma, feats = cols[1], cols[2], cols[5]
                    if form != "_" and lemma != "_":
                        triples.append((lemma, form, feats if feats != "_" else ""))
    return triples


def load_ud(lang: str, ud_root: str) -> List[Tuple[str, str, str]]:
    """Load all UD treebanks for a given language."""
    triples = []
    for treebank in UD_TREEBANKS.get(lang, []):
        path = os.path.join(ud_root, treebank)
        tb_triples = parse_conllu_dir(path)
        if tb_triples:
            logger.info(f"  UD {treebank}: {len(tb_triples):,} entries")
            triples.extend(tb_triples)
    return triples


# ─── SIGMORPHON 2022 morpheme segmentation (Hungarian only) ─────────

def load_sigmorphon_hun(sigmorphon_root: str) -> List[Tuple[str, str]]:
    """
    Load SIGMORPHON 2022 word-level morpheme segmentation for Hungarian.
    Format: word<TAB>segmentation<TAB>category
    Files: hun.word.train.tsv, hun.word.dev.tsv
    """
    pairs = []
    if not os.path.isdir(sigmorphon_root):
        return pairs
    for fname in ["hun.word.train.tsv", "hun.word.dev.tsv"]:
        path = os.path.join(sigmorphon_root, fname)
        # Check nested locations
        for candidate in [path,
                           os.path.join(sigmorphon_root, "data", fname),
                           os.path.join(sigmorphon_root, "2022SegmentationST",
                                        "data", fname)]:
            if os.path.exists(candidate):
                path = candidate
                break
        else:
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    word, seg = parts[0], parts[1]
                    # SIGMORPHON uses '@@' separator; convert to ' + '
                    seg_normalized = seg.replace(" @@", " + ").replace("@@", " + ")
                    pairs.append((word, seg_normalized))
    if pairs:
        logger.info(f"  SIGMORPHON 2022 hun: {len(pairs):,} pairs")
    return pairs


# ─── Top-level dataset builder ──────────────────────────────────────

def build_language_dataset(
    lang: str,
    cache_dir: str,
    ud_root: Optional[str] = None,
    sigmorphon_root: Optional[str] = None,
) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    """Build full dataset for one language. Returns (pairs, source_stats)."""
    all_pairs = []
    stats = {}

    # Source 1: UniMorph
    triples = load_unimorph(lang, cache_dir)
    if triples:
        p = convert_triples(triples, f"UniMorph-{lang}", lang)
        stats[f"unimorph_{lang}"] = len(p)
        all_pairs.extend(p)

    # Source 2: UD treebanks
    if ud_root:
        triples = load_ud(lang, ud_root)
        if triples:
            p = convert_triples(triples, f"UD-{lang}", lang)
            stats[f"ud_{lang}"] = len(p)
            all_pairs.extend(p)

    # Source 3: SIGMORPHON 2022 (Hungarian only — others not in shared task)
    if lang == "hun" and sigmorphon_root:
        sig_pairs = load_sigmorphon_hun(sigmorphon_root)
        if sig_pairs:
            stats["sigmorphon_hun"] = len(sig_pairs)
            all_pairs.extend(sig_pairs)

    # Deduplicate
    before = len(all_pairs)
    all_pairs = list(set(all_pairs))
    logger.info(f"  Dedup: {before:,} -> {len(all_pairs):,}")
    return all_pairs, stats


# ─── CLI ────────────────────────────────────────────────────────────

def write_tsv(pairs: List[Tuple[str, str]], path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".",
                exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["input", "target"])
        for inp, tgt in pairs:
            w.writerow([inp, tgt])
    logger.info(f"Wrote {len(pairs):,} pairs to {path}")


def main():
    p = argparse.ArgumentParser(
        description="Build morpheme segmentation dataset for AZ, TR, HU, or FI"
    )
    p.add_argument("--lang", required=True, choices=["aze", "tur", "hun", "fin"])
    p.add_argument("--cache_dir", default="data/cache",
                   help="Directory for cached UniMorph downloads")
    p.add_argument("--ud_root", default=None,
                   help="Path to directory containing UD_* treebank folders")
    p.add_argument("--sigmorphon_root", default=None,
                   help="Path to SIGMORPHON 2022 segmentation data")
    p.add_argument("--output", required=True, help="Output TSV file")
    p.add_argument("--preview", action="store_true",
                   help="Print 10 random sample pairs")
    args = p.parse_args()

    logger.info(f"Building dataset for language: {args.lang}")
    pairs, stats = build_language_dataset(
        args.lang, args.cache_dir, args.ud_root, args.sigmorphon_root
    )

    if not pairs:
        logger.error("No data collected from any source!")
        sys.exit(1)

    # Stats
    print(f"\n{'='*55}")
    print(f" Dataset composition for '{args.lang}'")
    print(f"{'='*55}")
    for src, count in sorted(stats.items(), key=lambda x: -x[1]):
        pct = count / len(pairs) * 100
        print(f"  {src:<25} {count:>10,} ({pct:5.1f}%)")
    print(f"  {'TOTAL':<25} {len(pairs):>10,}")
    print(f"{'='*55}\n")

    if args.preview:
        import random
        random.seed(42)
        sample = random.sample(pairs, min(10, len(pairs)))
        print("Sample pairs:")
        for inp, tgt in sample:
            print(f"  {inp:<25} -> {tgt}")
        print()

    write_tsv(pairs, args.output)


if __name__ == "__main__":
    main()
