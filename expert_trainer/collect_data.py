#!/usr/bin/env python3
"""
Collect expert routing data from openai/gpt-oss-20b on vietgpt/openwebtext_en.

For each token position t in a sequence:
  - input:  token at position t
  - label:  token at position t+1  (next token)
  - expert_idx:    top-k expert indices at each of the MoE layers  [num_layers, topk]
  - expert_scores: top-k router scores at each of the MoE layers   [num_layers, topk]

Output shards: {
    "input_tokens":   Int32   [N]
    "pred_tokens":    Int32   [N]
    "expert_idx":     Int16   [N, num_layers, topk]
    "expert_scores":  Float16 [N, num_layers, topk]
}
"""

import argparse
import glob as glob_module
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--dataset", default="vietgpt/openwebtext_en")
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--out-dir", default="./router_dataset_v2")
    parser.add_argument("--shard-size", type=int, default=1_000_000,
                        help="tokens per shard file")
    parser.add_argument("--max-tokens", type=int, default=100_000_000)
    parser.add_argument("--seq-len", type=int, default=512,
                        help="chunk length for inference")
    parser.add_argument("--topk", type=int, default=2,
                        help="number of top experts to store per layer")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.model_revision
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.eval()
    
    if hasattr(model.config, "output_router_logits"):
        model.config.output_router_logits = True

    num_layers = model.config.num_hidden_layers
    num_experts = model.config.num_local_experts
    print(f"Layers: {num_layers}, Experts: {num_experts}, Top-k: {args.topk}")

    print("Loading dataset (streaming)...")
    ds = load_dataset(
        args.dataset,
        split="train",
        streaming=True
    )

    # Resume: detect existing shards and count already-collected tokens
    existing_shards = sorted(glob_module.glob(os.path.join(args.out_dir, "shard_*.pt")))
    tokens_to_skip = 0
    shard_idx = len(existing_shards)
    if existing_shards:
        for p in existing_shards:
            d = torch.load(p, map_location="cpu", weights_only=True)
            tokens_to_skip += d["input_tokens"].shape[0]
        print(f"Resuming: found {len(existing_shards)} existing shards, "
              f"skipping first {tokens_to_skip:,} tokens → starting at shard {shard_idx:05d}")

    # Shard accumulators
    buf_input = []
    buf_pred  = []
    buf_idx   = []
    buf_scores = []
    total_tokens = tokens_to_skip
    tokens_consumed = 0  # dataset tokens seen so far (skipped + collected)

    def flush_shard():
        nonlocal shard_idx
        path = os.path.join(args.out_dir, f"shard_{shard_idx:05d}.pt")
        payload = {
            "input_tokens":  torch.cat(buf_input).to(torch.int32),
            "pred_tokens":   torch.cat(buf_pred).to(torch.int32),
            "expert_idx":    torch.cat(buf_idx).to(torch.int16),
            "expert_scores": torch.cat(buf_scores).to(torch.float16),
        }
        torch.save(payload, path)
        n = payload["input_tokens"].shape[0]
        print(f"[shard {shard_idx:05d}] saved {n:,} tokens → {path}")
        shard_idx += 1
        buf_input.clear()
        buf_pred.clear()
        buf_idx.clear()
        buf_scores.clear()

    with torch.inference_mode():
        for example in ds:
            if total_tokens >= args.max_tokens:
                break

            token_ids = tokenizer.encode(
                example["text"], add_special_tokens=False
            )
            if len(token_ids) < 2:
                continue

            # Process in non-overlapping chunks of seq_len
            for start in range(0, len(token_ids) - 1, args.seq_len):
                chunk = token_ids[start : start + args.seq_len + 1]
                if len(chunk) < 2:
                    break

                inp = chunk[:-1]   # positions 0..S-1
                lbl = chunk[1:]    # positions 1..S  (next tokens)
                S = len(inp)

                tokens_consumed += S

                # Skip chunks that were already saved in existing shards
                if tokens_consumed <= tokens_to_skip:
                    continue

                input_tensor = torch.tensor(
                    [inp], dtype=torch.long, device=device
                )  # [1, S]
                

                outputs = model(
                    input_tensor,
                    use_cache=False,
                    output_router_logits=True,
                    return_dict=True,
                )

                # Predicted tokens: argmax over vocab at each position → [1, S]
                preds = torch.argmax(outputs.logits, dim=-1)  # [1, S]

                # router_logits: tuple of num_layers tensors, each [S, num_experts]
                # (batch dim is folded into seq for MoE models)
                per_layer_idx, per_layer_scores = [], []
                for layer_logits in outputs.router_logits:
                    scores, idx = torch.topk(layer_logits, k=args.topk, dim=-1)
                    per_layer_idx.append(idx)       # [S, topk]
                    per_layer_scores.append(scores)  # [S, topk]

                # [S, num_layers, topk]
                idx_stack    = torch.stack(per_layer_idx, dim=1).cpu()
                scores_stack = torch.stack(per_layer_scores, dim=1).cpu()

                buf_input.append(torch.tensor(inp, dtype=torch.int32))
                buf_pred.append(torch.tensor(lbl, dtype=torch.int32))
                buf_idx.append(idx_stack)
                buf_scores.append(scores_stack)

                total_tokens += S

                if total_tokens % 100_000 < S:
                    print(f"  {total_tokens:,} tokens collected...")

                buf_len = sum(t.shape[0] for t in buf_input)
                if buf_len >= args.shard_size:
                    flush_shard()

                if total_tokens >= args.max_tokens:
                    break

    if buf_input:
        flush_shard()

    print(f"Done. Total tokens collected: {total_tokens:,}")


if __name__ == "__main__":
    main()
