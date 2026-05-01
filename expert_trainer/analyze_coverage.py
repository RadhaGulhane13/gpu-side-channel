#!/usr/bin/env python3
"""
Comprehensive dataset coverage analysis for router datasets.

Checks:
  - Total tokens, shard sizes, shard consistency
  - Token vocab coverage (input + pred) vs full vocab
  - Token frequency distribution (long-tail / rare token pitfalls)
  - Expert usage per layer (load imbalance, dead experts)
  - Router score statistics (raw logits vs softmax, collapse, confidence)
  - Data quality (NaN/Inf, out-of-range token IDs, negative scores)
  - Label alignment sanity check

Usage:
  python analyze_coverage.py --data-dir ./router_dataset_v4_new
  python analyze_coverage.py --data-dir ./router_dataset_v4_new --expert-sample-every 5
  python analyze_coverage.py --data-dir ./some_other_dataset --vocab-size 200000
"""

import argparse
import glob
import math
import os
import sys
from collections import Counter

import numpy as np
import torch
from transformers import AutoConfig


# ── model config ─────────────────────────────────────────────────────────────

def resolve_from_model(model_id, revision=None):
    """
    Load only the model config (no weights) and return vocab_size, num_experts.
    Returns (None, None) if the model cannot be fetched.
    """
    print(f"Fetching config from {model_id!r} (no weights downloaded)...")
    try:
        cfg = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    except Exception as e:
        print(f"  WARNING: could not load model config: {e}")
        return None, None

    vocab_size  = getattr(cfg, "vocab_size", None)
    num_experts = getattr(cfg, "num_local_experts", None) or getattr(cfg, "num_experts", None)
    print(f"  vocab_size={vocab_size},  num_experts={num_experts}")
    return vocab_size, num_experts


# ── helpers ──────────────────────────────────────────────────────────────────

def load_shards(data_dir, keys=None):
    """Yield (shard_path, dict) for every shard_*.pt in data_dir."""
    paths = sorted(glob.glob(os.path.join(data_dir, "shard_*.pt")))
    if not paths:
        raise FileNotFoundError(f"No shard_*.pt files found in {data_dir!r}")
    for p in paths:
        d = torch.load(p, map_location="cpu", weights_only=True)
        if keys:
            d = {k: d[k] for k in keys if k in d}
        yield p, d


def section(title):
    print(f"\n{'='*64}")
    print(f"  {title}")
    print(f"{'='*64}")


def subsection(title):
    print(f"\n--- {title} ---")


# ── analysis functions ────────────────────────────────────────────────────────

def analyze_tokens(shards_data, vocab_size, rare_threshold):
    """Token coverage, frequency distribution, top/bottom tokens."""
    section("TOKEN COVERAGE")

    input_counter: Counter = Counter()
    pred_counter:  Counter = Counter()
    total = 0
    shard_sizes = []

    for _path, d in shards_data:
        inp  = d["input_tokens"].numpy().astype(np.int32)
        pred = d["pred_tokens"].numpy().astype(np.int32)
        input_counter.update(inp.tolist())
        pred_counter.update(pred.tolist())
        total += len(inp)
        shard_sizes.append(len(inp))

    unique_input = len(input_counter)
    unique_pred  = len(pred_counter)
    all_seen     = len(set(input_counter) | set(pred_counter))

    print(f"\nShards          : {len(shard_sizes)}")
    print(f"Total tokens    : {total:,}")
    if vocab_size:
        print(f"Vocab size      : {vocab_size:,}")
        print(f"Unique input    : {unique_input:,}  ({100*unique_input/vocab_size:.1f}% of vocab)")
        print(f"Unique pred     : {unique_pred:,}  ({100*unique_pred/vocab_size:.1f}% of vocab)")
        print(f"Union coverage  : {all_seen:,}  ({100*all_seen/vocab_size:.1f}% of vocab)")
        print(f"Never seen      : {vocab_size - all_seen:,}  ({100*(vocab_size-all_seen)/vocab_size:.1f}% of vocab)")
    else:
        print(f"Unique input tokens : {unique_input:,}")
        print(f"Unique pred tokens  : {unique_pred:,}")
        print(f"Total unique seen   : {all_seen:,}")
        print(f"(pass --vocab-size to compute coverage %)")

    # Shard size consistency
    subsection("SHARD SIZE CONSISTENCY")
    sizes = np.array(shard_sizes)
    print(f"Min  : {sizes.min():,}")
    print(f"Max  : {sizes.max():,}")
    print(f"Mean : {sizes.mean():,.0f}")
    print(f"Std  : {sizes.std():,.0f}")
    non_uniform = (sizes != sizes[0]).sum()
    if non_uniform > 1:
        print(f"WARNING: {non_uniform} shards have non-uniform size (last shard may be partial — check resume logic)")

    # Rarity breakdown
    subsection("TOKEN RARITY BREAKDOWN (input_tokens)")
    freq_vals = np.array(list(input_counter.values()))
    thresholds = sorted({1, 2, 5, 10, 50, 100, 500, 1000, rare_threshold})
    for t in thresholds:
        n = (freq_vals < t).sum()
        pct_tok = 100 * n / unique_input
        pct_occ = 100 * freq_vals[freq_vals < t].sum() / total if n else 0.0
        flag = "  ← rare_threshold" if t == rare_threshold else ""
        print(f"  < {t:>5} appearances : {n:>7,} tokens  ({pct_tok:.1f}% of seen vocab, {pct_occ:.3f}% of total occurrences){flag}")

    print(f"\n  Median frequency : {np.median(freq_vals):.0f}")
    print(f"  Mean  frequency  : {freq_vals.mean():.1f}")

    # Log10 frequency histogram
    subsection("FREQUENCY DISTRIBUTION (log10 buckets, input_tokens)")
    buckets: dict = {}
    for cnt in freq_vals:
        b = int(math.log10(max(cnt, 1)))
        buckets[b] = buckets.get(b, 0) + 1
    for b in sorted(buckets):
        lo, hi = 10**b, 10**(b+1) - 1
        bar = "#" * min(buckets[b] // 200, 50)
        print(f"  [{lo:>8,} – {hi:>9,}] : {buckets[b]:>7,}  {bar}")

    # Top tokens (potential label imbalance)
    subsection("TOP 20 MOST FREQUENT INPUT TOKENS (imbalance check)")
    top20 = input_counter.most_common(20)
    top20_pct = sum(cnt for _, cnt in top20) / total * 100
    print(f"  Top-20 tokens account for {top20_pct:.1f}% of all occurrences")
    for tid, cnt in top20:
        print(f"    id={tid:>7d}  count={cnt:>12,}  ({100*cnt/total:.2f}%)")

    # Rarest tokens
    subsection("20 RAREST SEEN INPUT TOKENS")
    rare20 = input_counter.most_common()[:-21:-1]
    for tid, cnt in rare20:
        print(f"    id={tid:>7d}  count={cnt:>12,}")

    return input_counter, pred_counter, total


def analyze_experts(data_dir, num_sample_shards, num_experts):
    """Expert routing coverage and load balance per layer."""
    section("EXPERT COVERAGE & LOAD BALANCE")

    paths = sorted(glob.glob(os.path.join(data_dir, "shard_*.pt")))
    step = max(1, len(paths) // num_sample_shards)
    sampled = paths[::step][:num_sample_shards]
    print(f"\nSampling {len(sampled)}/{len(paths)} shards for expert stats...")

    all_idx    = []
    all_scores = []

    for p in sampled:
        d = torch.load(p, map_location="cpu", weights_only=True)
        all_idx.append(d["expert_idx"].numpy())           # [N, L, k]
        all_scores.append(d["expert_scores"].float().numpy())

    expert_idx    = np.concatenate(all_idx)    # [N, L, k]
    expert_scores = np.concatenate(all_scores)

    N, num_layers, topk = expert_idx.shape
    if num_experts is None:
        num_experts = int(expert_idx.max()) + 1
        print(f"Inferred num_experts={num_experts} from max idx")

    print(f"Sampled tokens : {N:,}")
    print(f"Layers         : {num_layers}")
    print(f"Top-k          : {topk}")
    print(f"Num experts    : {num_experts}")

    subsection("PER-LAYER EXPERT USAGE (top-k pooled)")
    print(f"  {'Layer':>5} | {'Used':>4} | {'Cov%':>5} | {'Dom expert':>10} | {'Max%':>6} | {'Min%':>6} | {'Std':>6} | {'Imbalance?'}")
    print(f"  {'-'*5}-+-{'-'*4}-+-{'-'*5}-+-{'-'*10}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*10}")
    imbalanced_layers = []
    for layer in range(num_layers):
        layer_idx = expert_idx[:, layer, :].flatten()
        unique, counts = np.unique(layer_idx, return_counts=True)
        coverage = len(unique) / num_experts * 100
        freqs = counts / counts.sum() * 100
        dom = unique[np.argmax(counts)]
        flag = ""
        if freqs.max() > 25:
            flag = "HIGH-LOAD"
            imbalanced_layers.append((layer, dom, freqs.max()))
        if len(unique) < num_experts:
            flag = flag + " DEAD-EXPERT" if flag else "DEAD-EXPERT"
        print(f"  {layer:>5} | {len(unique):>4} | {coverage:>4.0f}% | {dom:>10} | {freqs.max():>5.1f}% | {freqs.min():>6.3f}% | {freqs.std():>6.3f} | {flag}")

    if imbalanced_layers:
        print(f"\n  WARNING: {len(imbalanced_layers)} layers with a single expert handling >25% of tokens:")
        for layer, dom, pct in imbalanced_layers:
            print(f"    Layer {layer}: expert {dom} gets {pct:.1f}%")

    # Score statistics
    subsection("ROUTER SCORE STATISTICS")
    top1 = expert_scores[:, :, 0].flatten()
    print(f"  Top-1 : mean={top1.mean():.4f}  std={top1.std():.4f}  min={top1.min():.4f}  max={top1.max():.4f}")
    if topk >= 2:
        top2 = expert_scores[:, :, 1].flatten()
        print(f"  Top-2 : mean={top2.mean():.4f}  std={top2.std():.4f}  min={top2.min():.4f}  max={top2.max():.4f}")
        gap = top1 - top2
        print(f"\n  Top-1 minus Top-2 gap:")
        print(f"    mean={gap.mean():.4f}  std={gap.std():.4f}")
        print(f"    Near-uniform (gap < 0.01) : {(gap < 0.01).mean()*100:.1f}%  (ambiguous routing)")
        print(f"    Confident    (gap > 0.50) : {(gap > 0.50).mean()*100:.1f}%")
        print(f"    Very conf.   (gap > 1.00) : {(gap > 1.00).mean()*100:.1f}%")

    # Detect raw logits vs softmax
    n_above_one = (expert_scores > 1.0).mean() * 100
    n_negative  = (expert_scores < 0.0).mean() * 100
    subsection("SCORE TYPE DETECTION")
    print(f"  Values > 1.0 : {n_above_one:.1f}%")
    print(f"  Values < 0.0 : {n_negative:.2f}%")
    if n_above_one > 5 or n_negative > 1:
        print(f"  → Scores appear to be RAW LOGITS (not softmax-normalized)")
        print(f"    If your model trains on these directly, consider softmax-normalizing first.")
    else:
        print(f"  → Scores appear to be softmax probabilities (values mostly in [0,1])")

    # NaN / Inf
    subsection("NaN / Inf CHECK")
    inf_count = np.isinf(expert_scores).sum()
    nan_count = np.isnan(expert_scores).sum()
    print(f"  Inf values : {inf_count}")
    print(f"  NaN values : {nan_count}")
    if inf_count or nan_count:
        print(f"  WARNING: corrupted values detected — check collection script fp16 overflow")


def analyze_quality(data_dir, vocab_size):
    """Token ID range, label alignment, sequence boundary sanity."""
    section("DATA QUALITY CHECKS")

    paths = sorted(glob.glob(os.path.join(data_dir, "shard_*.pt")))

    # Only need to inspect a few shards for quality checks
    check_paths = paths[:3] + paths[-2:]
    print(f"\nSpot-checking {len(check_paths)} shards (first 3 + last 2)...")

    total_neg = 0
    total_oob = 0
    total_consecutive_dup = 0
    total_n   = 0

    for p in check_paths:
        d = torch.load(p, map_location="cpu", weights_only=True)
        inp  = d["input_tokens"].numpy().astype(np.int64)
        pred = d["pred_tokens"].numpy().astype(np.int64)
        n = len(inp)
        total_n += n

        total_neg += (inp < 0).sum() + (pred < 0).sum()
        if vocab_size:
            total_oob += (inp >= vocab_size).sum() + (pred >= vocab_size).sum()
        total_consecutive_dup += (inp[1:] == inp[:-1]).sum()

    subsection("TOKEN ID RANGE")
    print(f"  Negative IDs        : {total_neg}")
    if vocab_size:
        print(f"  Out-of-vocab IDs    : {total_oob}  (vocab_size={vocab_size:,})")
    else:
        print(f"  (pass --vocab-size to check OOV tokens)")
    dup_rate = 100 * total_consecutive_dup / (total_n - len(check_paths))
    print(f"  Consecutive dup rate: {dup_rate:.2f}%  (expected ~2-5% for natural text)")

    subsection("LABEL ALIGNMENT (pred = true next token, not model argmax)")
    d0 = torch.load(paths[0], map_location="cpu", weights_only=True)
    inp0  = d0["input_tokens"].numpy()
    pred0 = d0["pred_tokens"].numpy()
    match_rate = (inp0 == pred0).mean() * 100
    print(f"  input_tokens == pred_tokens rate in shard0: {match_rate:.2f}%")
    print(f"  (This is the token repetition rate, not model accuracy — typically 5-15% for natural text)")

    subsection("SEQUENCE CHUNK BOUNDARY NOTE")
    print(f"  collect_data.py uses non-overlapping chunks of --seq-len tokens.")
    print(f"  No context is carried across chunk boundaries, so the model never")
    print(f"  sees inter-chunk dependencies during collection. This is fine for")
    print(f"  router prediction but note that very long-range dependencies are absent.")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze router dataset coverage and quality.")
    parser.add_argument("--data-dir", default="./router_dataset_v4_new",
                        help="Path to directory containing shard_*.pt files")
    parser.add_argument("--model", default="openai/gpt-oss-20b",
                        help="HuggingFace model ID to pull vocab_size / num_experts from config "
                             "(no weights are downloaded). Set to '' to skip. (default: openai/gpt-oss-20b)")
    parser.add_argument("--model-revision", default=None,
                        help="Optional model revision / branch")
    parser.add_argument("--vocab-size", type=int, default=None,
                        help="Override vocab size instead of reading from --model config")
    parser.add_argument("--num-experts", type=int, default=None,
                        help="Override num experts instead of reading from --model config")
    parser.add_argument("--rare-threshold", type=int, default=10,
                        help="Tokens seen fewer than this many times are flagged as rare (default: 10)")
    parser.add_argument("--expert-sample-shards", type=int, default=20,
                        help="Number of shards to sample for expert analysis (default: 20)")
    parser.add_argument("--skip-experts", action="store_true",
                        help="Skip expert coverage analysis (faster)")
    parser.add_argument("--skip-quality", action="store_true",
                        help="Skip data quality spot checks")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"ERROR: data directory not found: {args.data_dir!r}", file=sys.stderr)
        sys.exit(1)

    print(f"Dataset directory : {os.path.abspath(args.data_dir)}")

    # ── resolve vocab_size / num_experts from model config ────────────────────
    vocab_size  = args.vocab_size
    num_experts = args.num_experts
    if args.model and (vocab_size is None or num_experts is None):
        cfg_vocab, cfg_experts = resolve_from_model(args.model, args.model_revision)
        if vocab_size  is None: vocab_size  = cfg_vocab
        if num_experts is None: num_experts = cfg_experts

    # ── token coverage ────────────────────────────────────────────────────────
    shards_data = list(load_shards(args.data_dir, keys=["input_tokens", "pred_tokens"]))
    analyze_tokens(shards_data, vocab_size=vocab_size, rare_threshold=args.rare_threshold)

    # ── expert coverage ───────────────────────────────────────────────────────
    if not args.skip_experts:
        analyze_experts(
            args.data_dir,
            num_sample_shards=args.expert_sample_shards,
            num_experts=num_experts,
        )

    # ── quality checks ────────────────────────────────────────────────────────
    if not args.skip_quality:
        analyze_quality(args.data_dir, vocab_size=vocab_size)

    print(f"\n{'='*64}")
    print("  SUMMARY OF POTENTIAL TRAINING PITFALLS")
    print(f"{'='*64}")
    print("""
  1. LONG-TAIL TOKEN IMBALANCE
     Tokens seen only once or twice will have near-zero gradient signal.
     Consider frequency-weighted loss or up-sampling rare tokens.

  2. DOMINANT TOP TOKENS
     If a handful of tokens account for >20% of the dataset, the model
     may overfit their routing pattern and generalize poorly to rare tokens.

  3. EXPERT LOAD IMBALANCE
     Layers where one expert handles >25% of tokens indicate the model
     relies heavily on that expert. The router predictor will be biased
     toward predicting that expert; ensure loss weighting accounts for this.

  4. RAW LOGITS vs SOFTMAX SCORES
     If expert_scores are raw logits (not normalized), using them as
     regression targets directly can cause unstable training. Consider
     softmax-normalizing before training if loss is high or unstable.

  5. NEAR-UNIFORM ROUTING (~3% of tokens)
     These positions have nearly equal top-1/top-2 scores — the router
     is uncertain. These are noisy labels and may hurt loss. Consider
     filtering or down-weighting positions where score gap < 0.01.

  6. SEQUENCE CHUNK BOUNDARIES
     No cross-chunk context in the collected data. If your predictor
     uses a context window longer than --seq-len, it will see padding
     or stale context at chunk starts during inference.
""")


if __name__ == "__main__":
    main()
