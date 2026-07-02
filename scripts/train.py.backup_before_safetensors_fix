"""
Train ByT5-small for cross-family morpheme segmentation.

Designed for NVIDIA Tesla V100 (Volta, sm_70):
  - V100 supports fp16 but NOT bfloat16 → use fp32 for ByT5 (fp16 causes NaN
    due to byte-level activation overflow; see Xue et al., 2022 TACL).
  - 16 GB VRAM → batch_size=16 with gradient_accumulation=4 (effective 64)

Resume-safe: full optimizer/scheduler/scaler state saved every epoch.
Includes NaN detection to bail out early if numerical issues appear.
"""
import argparse
import csv
import logging
import math
import os
import sys
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── Dataset ─────────────────────────────────────────────────────────

class MorphDataset(Dataset):
    def __init__(self, tsv_path: str, tokenizer, max_input_len: int = 64,
                 max_target_len: int = 96, task_prefix: str = "segment: "):
        self.examples = []
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                self.examples.append((row["input"], row["target"]))
        logger.info(f"Loaded {len(self.examples):,} examples from {tsv_path}")
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.task_prefix = task_prefix

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        inp, tgt = self.examples[idx]
        src = self.task_prefix + inp
        src_enc = self.tokenizer(src, max_length=self.max_input_len,
                                  padding="max_length", truncation=True,
                                  return_tensors="pt")
        tgt_enc = self.tokenizer(tgt, max_length=self.max_target_len,
                                  padding="max_length", truncation=True,
                                  return_tensors="pt")
        labels = tgt_enc.input_ids.squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100  # mask pad
        return {
            "input_ids": src_enc.input_ids.squeeze(0),
            "attention_mask": src_enc.attention_mask.squeeze(0),
            "labels": labels,
        }


# ─── Evaluation ──────────────────────────────────────────────────────

def evaluate(model, tokenizer, loader, device, max_target_len: int = 96):
    model.eval()
    n_total = 0
    n_em = 0
    n_stem_correct = 0
    n_suffix_correct = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model.generate(input_ids=input_ids,
                                      attention_mask=attn_mask,
                                      max_length=max_target_len,
                                      num_beams=4, early_stopping=True)
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            label_ids = labels.clone()
            label_ids[label_ids == -100] = tokenizer.pad_token_id
            golds = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
            for p, g in zip(preds, golds):
                p = p.strip()
                g = g.strip()
                n_total += 1
                if p == g:
                    n_em += 1
                p_parts = [s.strip() for s in p.split("+")]
                g_parts = [s.strip() for s in g.split("+")]
                if p_parts and g_parts and p_parts[0] == g_parts[0]:
                    n_stem_correct += 1
                if len(p_parts) > 1 and len(g_parts) > 1:
                    if p_parts[1:] == g_parts[1:]:
                        n_suffix_correct += 1
                elif len(p_parts) == len(g_parts) == 1:
                    n_suffix_correct += 1
    return {
        "exact_match": n_em / max(n_total, 1),
        "stem_accuracy": n_stem_correct / max(n_total, 1),
        "suffix_accuracy": n_suffix_correct / max(n_total, 1),
        "n_total": n_total,
    }


# ─── Training ────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU: {gpu_name}")
        if "V100" in gpu_name and (args.fp16 or args.bf16):
            logger.warning(
                "V100 detected with fp16 or bf16 enabled! "
                "ByT5 produces NaN with fp16 on V100, and V100 does not support bf16. "
                "Forcing fp32 (set --no_amp to silence this)."
            )
            args.fp16 = False
            args.bf16 = False

    # Reproducibility
    torch.manual_seed(args.seed)

    # Tokenizer + model
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name).to(device)

    # Data
    full = MorphDataset(args.data_path, tokenizer,
                        args.max_input_len, args.max_target_len)
    n_train = int(0.85 * len(full))
    n_val = int(0.10 * len(full))
    n_test = len(full) - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        full, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )
    logger.info(f"Split: train={n_train:,} val={n_val:,} test={n_test:,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                   weight_decay=args.weight_decay)
    total_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # AMP setup
    use_amp = False
    amp_dtype = None
    scaler = None
    if args.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported():
        use_amp, amp_dtype = True, torch.bfloat16
        logger.info("Using bfloat16 mixed precision")
    elif args.fp16 and device.type == "cuda":
        use_amp, amp_dtype = True, torch.float16
        scaler = torch.amp.GradScaler("cuda")
        logger.warning("Using fp16 — may cause NaN with ByT5; prefer bf16 or fp32")
    else:
        logger.info("Using fp32 (recommended for ByT5 on V100)")

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)
    state_path = os.path.join(args.output_dir, "training_state.pt")

    # Resume from checkpoint
    best_em = 0.0
    patience = 0
    start_epoch = 0
    if args.resume and os.path.exists(state_path):
        state = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if scaler and "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_em = state["best_em"]
        patience = state["patience"]
        logger.info(f"Resumed from epoch {start_epoch}, best_em={best_em:.4f}, "
                    f"patience={patience}/{args.early_stopping_patience}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(bar):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    out = model(input_ids=input_ids,
                                attention_mask=attn_mask, labels=labels)
                    loss = out.loss / args.gradient_accumulation_steps
                if scaler:
                    scaler.scale(loss).backward()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        scheduler.step()
                else:
                    loss.backward()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
            else:
                out = model(input_ids=input_ids,
                            attention_mask=attn_mask, labels=labels)
                loss = out.loss / args.gradient_accumulation_steps
                loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

            full_loss = loss.item() * args.gradient_accumulation_steps
            if math.isnan(full_loss) or math.isinf(full_loss):
                logger.error(f"NaN/Inf loss detected at epoch {epoch+1}, step {step}!")
                logger.error("This usually indicates fp16 overflow with ByT5.")
                logger.error("Try --no_amp (fp32) or --bf16 (A100/H100 only).")
                return
            epoch_loss += full_loss
            n_batches += 1
            bar.set_postfix({"loss": f"{epoch_loss/n_batches:.4f}",
                             "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        avg_loss = epoch_loss / max(n_batches, 1)
        logger.info(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        # Validation
        metrics = evaluate(model, tokenizer, val_loader, device,
                           args.max_target_len)
        logger.info(
            f"Epoch {epoch+1} val: EM={metrics['exact_match']:.4f}, "
            f"stem={metrics['stem_accuracy']:.4f}, "
            f"sfx={metrics['suffix_accuracy']:.4f}"
        )

        # Save best
        improved = metrics["exact_match"] > best_em
        if improved:
            best_em = metrics["exact_match"]
            patience = 0
            best_dir = os.path.join(args.output_dir, "best")
            model.save_pretrained(best_dir, safe_serialization=False)
            tokenizer.save_pretrained(best_dir)
            logger.info(f"  ↑ New best EM: {best_em:.4f}")
        else:
            patience += 1
            logger.info(f"  No improvement ({patience}/{args.early_stopping_patience})")

        # Save full state
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_em": best_em,
            "patience": patience,
            "train_loss": avg_loss,
            "val_metrics": metrics,
        }
        if scaler:
            state["scaler"] = scaler.state_dict()
        torch.save(state, state_path)

        # Early stopping
        if patience >= args.early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # Final test evaluation
    best_dir = os.path.join(args.output_dir, "best")
    if os.path.exists(best_dir):
        logger.info("Loading best model for final test evaluation...")
        best_model = T5ForConditionalGeneration.from_pretrained(best_dir).to(device)
    else:
        best_model = model

    test_metrics = evaluate(best_model, tokenizer, test_loader, device,
                            args.max_target_len)
    logger.info(
        f"TEST: EM={test_metrics['exact_match']:.4f}, "
        f"stem={test_metrics['stem_accuracy']:.4f}, "
        f"sfx={test_metrics['suffix_accuracy']:.4f}"
    )

    # Write final results JSON
    import json
    final_results_path = os.path.join(args.output_dir, "final_results.json")
    with open(final_results_path, "w") as f:
        json.dump({
            "best_val_em": best_em,
            "test_metrics": test_metrics,
            "config": vars(args),
        }, f, indent=2)
    logger.info(f"Final results saved to {final_results_path}")


def main():
    p = argparse.ArgumentParser(description="Train ByT5 for morpheme segmentation")
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_name", default="google/byt5-small")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16,
                   help="Per-device batch size (V100=16, A100=64)")
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4,
                   help="V100=4 (eff. batch 64), A100=1")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_input_len", type=int, default=64)
    p.add_argument("--max_target_len", type=int, default=96)
    p.add_argument("--early_stopping_patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true",
                   help="Use fp16 (NOT recommended for ByT5 on V100 — causes NaN)")
    p.add_argument("--bf16", action="store_true",
                   help="Use bf16 (only A100/H100; V100 not supported)")
    p.add_argument("--no_amp", action="store_true",
                   help="Force fp32 (default, recommended for V100)")
    p.add_argument("--resume", action="store_true",
                   help="Resume from training_state.pt if present")
    args = p.parse_args()

    if args.no_amp:
        args.fp16 = False
        args.bf16 = False

    train(args)


if __name__ == "__main__":
    main()
