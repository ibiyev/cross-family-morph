"""
Cross-language evaluation: load a trained model M, evaluate it on a test TSV.

Used to build the 4x4 train-on-X-test-on-Y transfer matrix that is the core
finding of the paper.
"""
import argparse
import csv
import json
import logging
import os
import sys
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train import MorphDataset, evaluate

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(
        description="Evaluate a trained model on a test TSV"
    )
    p.add_argument("--model_dir", required=True,
                   help="Directory of trained model (the 'best/' subdirectory)")
    p.add_argument("--test_tsv", required=True,
                   help="Test TSV with input,target columns")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_input_len", type=int, default=64)
    p.add_argument("--max_target_len", type=int, default=96)
    p.add_argument("--output_json", default=None,
                   help="Optional path to save results as JSON")
    p.add_argument("--label", default=None,
                   help="Optional label for this evaluation (e.g., 'TR_model_on_AZ_test')")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Model: {args.model_dir}")
    logger.info(f"Test:  {args.test_tsv}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir).to(device)

    ds = MorphDataset(args.test_tsv, tokenizer,
                      args.max_input_len, args.max_target_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=2, pin_memory=True)

    metrics = evaluate(model, tokenizer, loader, device, args.max_target_len)

    print(f"\n{'='*60}")
    if args.label:
        print(f" {args.label}")
        print(f"{'='*60}")
    print(f"  Total examples:    {metrics['n_total']:>10,}")
    print(f"  Exact match:       {metrics['exact_match']:>10.4f}")
    print(f"  Stem accuracy:     {metrics['stem_accuracy']:>10.4f}")
    print(f"  Suffix accuracy:   {metrics['suffix_accuracy']:>10.4f}")
    print(f"{'='*60}\n")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump({
                "label": args.label,
                "model_dir": args.model_dir,
                "test_tsv": args.test_tsv,
                "metrics": metrics,
            }, f, indent=2)
        logger.info(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
