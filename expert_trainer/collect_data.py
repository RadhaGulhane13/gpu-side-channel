#!/usr/bin/env python3
"""
Collect expert routing data from openai/gpt-oss-20b on vietgpt/openwebtext_en.

For each token position t in a sequence:
  - input:  token at position t
  - label:  token at position t+1  (next token)
  - expert_logits: raw router logits at each of the 24 MoE layers  [24, 32]

Output shards: {
    "input_tokens":   Int32  [N]
    "pred_tokens":    Int32  [N]
    "expert_logits":  Float16 [N, num_layers, num_experts]
}
"""

import argparse
import glob as glob_module
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


NUM_LAYERS = 24
NUM_EXPERTS = 32


class RouterCapture:
    """Captures full router logits via MLP pre-hooks.

    The MLP's forward is replaced by a MegaBlocks kernel that bypasses Python
    module dispatch, so router forward hooks never fire.  Instead we hook the
    MLP's *input* (pre-hook), manually run just the router linear projection on
    those hidden states, and store the full [T, num_experts] logits.
    """

    def __init__(self, model):
        self._logits = [None] * NUM_LAYERS
        self._hooks = []
        for i in range(NUM_LAYERS):
            mlp = model.model.layers[i].mlp
            hook = mlp.register_forward_pre_hook(self._make_hook(i, mlp.router))
            self._hooks.append(hook)

    def _make_hook(self, layer_idx, router):
        def hook(module, args):
            hidden_states = args[0]                          # [B, S, H]
            bsz, seq_len, hidden_dim = hidden_states.shape
            hs_flat = hidden_states.reshape(-1, hidden_dim)  # [B*S, H]
            # router_logits = F.linear(hs, weight, bias): [B*S, num_experts]
            logits = torch.nn.functional.linear(
                hs_flat.float(),
                router.weight.float(),
                router.bias.float(),
            )
            self._logits[layer_idx] = logits.detach()
        return hook

    def get(self, seq_len):
        """Return stacked logits [seq_len, num_layers, num_experts]."""
        stacked = torch.stack(self._logits, dim=1)  # [T, L, E]
        return stacked.view(seq_len, NUM_LAYERS, NUM_EXPERTS)

    def remove(self):
        for h in self._hooks:
            h.remove()


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
        revision=args.model_revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    capture = RouterCapture(model)
    print(f"Hooked {NUM_LAYERS} router layers.")

    print("Loading dataset (streaming)...")
    ds = load_dataset(
        args.dataset,
        split="train",
        streaming=True,
        revision=args.dataset_revision,
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
    buf_logits = []
    total_tokens = tokens_to_skip
    tokens_consumed = 0  # dataset tokens seen so far (skipped + collected)

    def flush_shard():
        nonlocal shard_idx
        path = os.path.join(args.out_dir, f"shard_{shard_idx:05d}.pt")
        payload = {
            "input_tokens":  torch.cat(buf_input).to(torch.int32),
            "pred_tokens":   torch.cat(buf_pred).to(torch.int32),
            "expert_logits": torch.cat(buf_logits).to(torch.float16),
        }
        torch.save(payload, path)
        n = payload["input_tokens"].shape[0]
        print(f"[shard {shard_idx:05d}] saved {n:,} tokens → {path}")
        shard_idx += 1
        buf_input.clear()
        buf_pred.clear()
        buf_logits.clear()

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

                model(input_tensor, use_cache=False)

                # [S, 24, 32]
                logits = capture.get(S).cpu()

                buf_input.append(torch.tensor(inp, dtype=torch.int32))
                buf_pred.append(torch.tensor(lbl, dtype=torch.int32))
                buf_logits.append(logits)

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

    capture.remove()
    print(f"Done. Total tokens collected: {total_tokens:,}")


if __name__ == "__main__":
    main()
