#!/usr/bin/env python
import argparse
import json
import os
import time
from dataclasses import dataclass
from unittest import loader

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer


from read_data import get_dataloader_seq


@dataclass
class TrainState:
    tokens_seen: int = 0
    example_index: int = 0
    example_token_offset: int = 0
    step: int = 0


def _write_json_atomic(path, payload):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


class ExpertStream:
    def __init__(
        self,
        idx_path,
        dataset_name,
        dataset_revision,
        tokenizer,
        seq_len,
        max_tokens,
        batch_size,
        state: TrainState,
    ):
        self.idx_path = idx_path
        self.dataset_name = dataset_name
        self.dataset_revision = dataset_revision
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.state = state

        self.idx_mmap = np.load(self.idx_path, mmap_mode="r")
        if self.idx_mmap.shape != (self.max_tokens, 24, 4):
            raise ValueError(f"Unexpected idx shape {self.idx_mmap.shape}.")

    def __iter__(self):
        ds = load_dataset(
            self.dataset_name,
            split="train",
            streaming=True,
            revision=self.dataset_revision,
        )
        tokens_seen = self.state.tokens_seen
        example_index = self.state.example_index
        example_token_offset = self.state.example_token_offset

        batch_tokens = []
        batch_idx = []
        batch_mask = []

        for idx, example in enumerate(ds):
            if idx < example_index:
                continue
            if tokens_seen >= self.max_tokens:
                break

            token_ids = self.tokenizer.encode(
                example["text"], add_special_tokens=False
            )
            if idx == example_index and example_token_offset > 0:
                token_ids = token_ids[example_token_offset:]

            if not token_ids:
                example_index = idx + 1
                example_token_offset = 0
                continue

            pos = 0
            while pos < len(token_ids) and tokens_seen < self.max_tokens:
                remaining = self.max_tokens - tokens_seen
                current_len = min(self.seq_len, len(token_ids) - pos, remaining)
                if current_len <= 0:
                    break

                chunk = token_ids[pos:pos + current_len]
                idx_chunk = self.idx_mmap[tokens_seen:tokens_seen + current_len]

                tokens_seen += current_len
                pos += current_len
                example_token_offset += current_len

                pad_len = self.seq_len - current_len
                if pad_len:
                    chunk = chunk + [self.tokenizer.pad_token_id] * pad_len
                    idx_chunk = np.pad(
                        idx_chunk,
                        ((0, pad_len), (0, 0), (0, 0)),
                        mode="edge",
                    )

                mask = [1] * current_len + [0] * pad_len

                batch_tokens.append(chunk)
                batch_idx.append(idx_chunk)
                batch_mask.append(mask)

                if len(batch_tokens) >= self.batch_size:
                    yield {
                        "input_ids": torch.tensor(
                            np.asarray(batch_tokens, dtype=np.int64)
                        ),
                        "expert_idx": torch.tensor(
                            np.asarray(batch_idx, dtype=np.int64)
                        ),
                        "attention_mask": torch.tensor(
                            np.asarray(batch_mask, dtype=np.bool_)
                        ),
                        "state": TrainState(
                            tokens_seen=tokens_seen,
                            example_index=idx,
                            example_token_offset=example_token_offset,
                        ),
                    }
                    batch_tokens, batch_idx, batch_mask = [], [], []

            example_index = idx + 1
            example_token_offset = 0

        if batch_tokens:
            yield {
                "input_ids": torch.tensor(np.asarray(batch_tokens, dtype=np.int64)),
                "expert_idx": torch.tensor(np.asarray(batch_idx, dtype=np.int64)),
                "attention_mask": torch.tensor(
                    np.asarray(batch_mask, dtype=np.bool_)
                ),
                "state": TrainState(
                    tokens_seen=tokens_seen,
                    example_index=example_index,
                    example_token_offset=example_token_offset,
                ),
            }


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) * (1.0 / (x.size(-1) ** 0.5))
        return self.weight * x / (norm + self.eps)


class ExpertEncoderMultiHot(nn.Module):
    def __init__(
        self,
        num_experts,
        num_layers,
        d_model,
        layer_hidden,
        layer_proj,
        dropout,
        layer_gating,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.layer_gating = layer_gating
        if layer_gating:
            self.layer_gate = nn.Parameter(torch.zeros(num_layers))
        self.layer_norm = nn.LayerNorm(num_experts)
        self.layer_mlp = nn.Sequential(
            nn.Linear(num_experts, layer_hidden),
            nn.ReLU(),
            nn.Linear(layer_hidden, layer_proj),
        )
        self.proj = nn.Linear(num_layers * layer_proj, d_model)
        self.dropout = nn.Dropout(dropout)
        
    

    # def forward(self, expert_idx):
    #     # expert_idx: [B, S, L, K]
    #     bsz, seq_len, num_layers, _topk = expert_idx.shape
    #     multihot = torch.zeros(
    #         (bsz, seq_len, num_layers, self.num_experts),
    #         device=expert_idx.device,
    #         dtype=torch.float32,
    #     )
    #     multihot.scatter_(-1, expert_idx, 1.0)
    #     if self.layer_gating:
    #         gate = torch.sigmoid(self.layer_gate).view(1, 1, num_layers, 1)
    #         multihot = multihot * gate
    #     multihot = self.layer_norm(multihot)
    #     layer_repr = self.layer_mlp(multihot)  # [B,S,L,P]
    #     flat = layer_repr.reshape(bsz, seq_len, num_layers * layer_repr.size(-1))
    #     return self.dropout(self.proj(flat))
    
    def forward(self, expert_logits):
        """
        expert_logits: [B, S, L, num_experts] (raw scores or probabilities)
        """

        # Convert logits to probabilities if they are raw scores
        expert_probs = torch.softmax(expert_logits, dim=-1)  # [B, S, L, num_experts]

        if self.layer_gating:
            gate = torch.sigmoid(self.layer_gate).view(1, 1, self.num_layers, 1)
            expert_probs = expert_probs * gate

        # LayerNorm across experts
        # expert_probs = self.layer_norm(expert_probs)
        expert_probs = nn.functional.layer_norm(expert_probs, expert_probs.shape[-1:])

        # Pass through MLP for each layer independently
        layer_repr = self.layer_mlp(expert_probs)  # [B, S, L, P]

        # Flatten layers and project to d_model
        bsz, seq_len, num_layers, proj_dim = layer_repr.shape
        flat = layer_repr.reshape(bsz, seq_len, num_layers * proj_dim)
        return self.dropout(self.proj(flat))


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.attn_norm = RMSNorm(d_model)
        self.mlp_norm = RMSNorm(d_model)
        self.attn = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, d_ff)
        self.fc_out = nn.Linear(d_ff, d_model)

    def forward(self, x, attention_mask):
        bsz, seq_len, d_model = x.shape
        qkv = self.attn(self.attn_norm(x))
        q, k, v = qkv.split(d_model, dim=-1)
        d_head = d_model // self.n_head
        q = q.view(bsz, seq_len, self.n_head, d_head).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_head, d_head).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, d_head).transpose(1, 2)
        attn = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
        x = x + self.dropout(self.proj(attn))
        mlp = self.fc(self.mlp_norm(x))
        mlp = torch.relu(mlp).pow(2)
        x = x + self.dropout(self.fc_out(mlp))
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
        return x


class EncoderOnlyModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_experts,
        num_layers,
        topk,
        d_model,
        n_head,
        d_ff,
        n_layer,
        dropout,
        max_len,
        layer_gating,
        logit_softcap,
        layer_hidden,
        layer_proj,
    ):
        super().__init__()
        self.encoder_in = ExpertEncoderMultiHot(
            num_experts,  # full logits over all experts
            num_layers,
            d_model,
            layer_hidden,
            layer_proj,
            dropout,
            layer_gating,
        )
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, n_head, d_ff, dropout) for _ in range(n_layer)]
        )
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.logit_softcap = logit_softcap

    # def forward(self, expert_idx, attention_mask):
    #     bsz, seq_len = expert_idx.shape[:2]
    #     x = self.encoder_in(expert_idx)
    #     pos_ids = torch.arange(seq_len, device=expert_idx.device)pos_ids = torch.arange(seq_len, device=expert_idx.device)
    #     pos_ids = pos_ids.unsqueeze(0).expand(bsz, -1)
    #     x = x + self.pos_emb(pos_ids)
    
    def forward(self, expert_scores, attention_mask):
        """
        expert_scores: (batch, seq_len, num_layers, topk) -- routing scores from gating network
        attention_mask: (batch, seq_len)
        """
        bsz, seq_len = expert_scores.shape[:2]

        x = self.encoder_in(expert_scores)

        pos_ids = torch.arange(seq_len, device=expert_scores.device)
        pos_ids = pos_ids.unsqueeze(0).expand(bsz, -1)
        x = x + self.pos_emb(pos_ids)
        
        
        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.norm(x)
        logits = self.head(x)
        if self.logit_softcap and self.logit_softcap > 0:
            logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return logits


def split_muon_params(model):
    muon_params = []
    adam_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_matrix = p.ndim == 2
        is_embedding_or_head = (
            name.endswith("head.weight") or name.endswith("pos_emb.weight")
        )
        if is_matrix and not is_embedding_or_head:
            muon_params.append(p)
        else:
            adam_params.append(p)
    return muon_params, adam_params


def make_trapezoidal_lr(step_idx, max_steps, warmup_ratio, warmdown_ratio):
    warmup_steps = max(1, int(warmup_ratio * max_steps)) if warmup_ratio > 0 else 0
    warmdown_steps = max(1, int(warmdown_ratio * max_steps)) if warmdown_ratio > 0 else 0
    if warmup_steps > 0 and step_idx < warmup_steps:
        return float(step_idx + 1) / float(warmup_steps)
    warmdown_start = max_steps - warmdown_steps
    if step_idx < warmdown_start:
        return 1.0
    if warmdown_steps > 0 and step_idx < max_steps:
        remaining = max_steps - step_idx
        return max(0.0, float(remaining) / float(warmdown_steps))
    return 0.0


def main():
    parser = argparse.ArgumentParser(
        description="V5 encoder-only inverter (multihot + per-layer MLP)."
    )
    # parser.add_argument("--idx", required=True)
    # parser.add_argument("--dataset", default="vietgpt/openwebtext_en")
    # parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=200000000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--out", default="inverter_v5.pt")
    parser.add_argument("--state-path", default="train_state_v5.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--attn-impl",
        choices=["auto", "flash", "mem_efficient", "math"],
        default="auto",
    )
    parser.add_argument("--logit-softcap", type=float, default=0.0)
    parser.add_argument("--layer-gating", action="store_true")
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--layer-hidden", type=int, default=64)
    parser.add_argument("--layer-proj", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--adam-lr", type=float, default=3e-4)
    parser.add_argument("--muon-lr-factor", type=float, default=4.0)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.01)
    parser.add_argument("--warmdown-ratio", type=float, default=0.20)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="expert-inversion")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        if args.attn_impl != "auto":
            try:
                torch.backends.cuda.enable_flash_sdp(args.attn_impl == "flash")
                torch.backends.cuda.enable_mem_efficient_sdp(
                    args.attn_impl == "mem_efficient"
                )
                torch.backends.cuda.enable_math_sdp(args.attn_impl == "math")
            except AttributeError:
                pass

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            # entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args),
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    state = TrainState()
    if args.resume and os.path.exists(args.state_path):
        with open(args.state_path, "r") as f:
            payload = json.load(f)
        state = TrainState(
            tokens_seen=payload.get("tokens_seen", 0),
            example_index=payload.get("example_index", 0),
            example_token_offset=payload.get("example_token_offset", 0),
            step=payload.get("step", 0),
        )

    # stream = ExpertStream(
    #     idx_path=args.idx,
    #     dataset_name=args.dataset,
    #     dataset_revision=args.dataset_revision,
    #     tokenizer=tokenizer,
    #     seq_len=args.seq_len,
    #     max_tokens=args.max_tokens,
    #     batch_size=args.batch_size,
    #     state=state,
    # )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EncoderOnlyModel(
        vocab_size=len(tokenizer),
        num_experts=32,
        num_layers=args.layers,
        topk=4,
        d_model=args.d_model,
        n_head=args.n_head,
        d_ff=args.d_ff,
        n_layer=args.n_layer,
        dropout=args.dropout,
        max_len=args.seq_len,
        layer_gating=args.layer_gating,
        logit_softcap=args.logit_softcap,
        layer_hidden=args.layer_hidden,
        layer_proj=args.layer_proj,
    ).to(device)

    if args.compile and device.type == "cuda":
        model = torch.compile(model, dynamic=False)

    muon_params, adam_params = split_muon_params(model)
    if not muon_params:
        raise RuntimeError("No Muon parameters found; check parameter names.")

    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError("torch.optim.Muon not available in this environment.")

    optimizer_adam = torch.optim.AdamW(
        adam_params,
        lr=args.adam_lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    optimizer_muon = torch.optim.Muon(
        muon_params,
        lr=args.adam_lr * args.muon_lr_factor,
        weight_decay=args.weight_decay,
        momentum=0.95,
        nesterov=True,
        adjust_lr_fn="match_rms_adamw",
    )
    optimizers = [optimizer_adam, optimizer_muon]

    def lr_lambda(step_idx):
        return make_trapezoidal_lr(
            step_idx, args.steps, args.warmup_ratio, args.warmdown_ratio
        )

    schedulers = [
        torch.optim.lr_scheduler.LambdaLR(optimizer_adam, lr_lambda=lr_lambda),
        torch.optim.lr_scheduler.LambdaLR(optimizer_muon, lr_lambda=lr_lambda),
    ]

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    model.train()
    step = state.step
    micro_step = 0
    start_time = time.time()
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)

    # for batch in stream:
    loader = get_dataloader_seq(data_dir="./router_dataset_v2", batch_size=args.batch_size)

    # for batch in loader:
    #     if step >= args.steps:
    #         break

    #     micro_step += 1
    #     # input_ids = batch["input_ids"].to(device, non_blocking=True)
    #     # expert_idx = batch["expert_idx"][:, :, :args.layers].to(device, non_blocking=True)
    #     # attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        
    #     input_ids = batch["input_ids"].to(device, non_blocking=True)         # [B, S]
    #     labels = batch["labels"].to(device, non_blocking=True)               # [B, S]
    #     expert_idx = batch["expert_idx"][:, :, :args.layers].to(device)      # [B, S, L, K]
    #     attention_mask = batch["attention_mask"].to(device, non_blocking=True) # [B, S]

    #     # labels = input_ids.clone()
    #     labels[~attention_mask] = -100
        

    #     with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
    #         logits = model(expert_idx, attention_mask)
    #         loss = F.cross_entropy(
    #             logits.view(-1, logits.size(-1)),
    #             labels.view(-1),
    #             ignore_index=-100,
    #         )
    #         loss = loss / args.grad_accum

    #     scaler.scale(loss).backward()

    #     if micro_step % args.grad_accum != 0:
    #         continue

    #     for opt in optimizers:
    #         scaler.step(opt)
    #     scaler.update()
    #     for opt in optimizers:
    #         opt.zero_grad(set_to_none=True)
    #     for sched in schedulers:
    #         sched.step()

    #     step += 1
    #     state = batch["state"]
    #     state.step = step
    #     micro_step = 0

    #     if step % 10 == 0:
    #         elapsed = time.time() - start_time
    #         lr_adam = schedulers[0].get_last_lr()[0]
    #         lr_muon = schedulers[1].get_last_lr()[0]
    #         print(
    #             f"step {step} loss {loss.item():.4f} lr_adam {lr_adam:.6e} lr_muon {lr_muon:.6e}"
    #         )
    #         if wandb_run:
    #             wandb_run.log(
    #                 {
    #                     "train/loss": loss.item() * args.grad_accum,
    #                     "train/lr_adam": lr_adam,
    #                     "train/lr_muon": lr_muon,
    #                     "train/step": step,
    #                     "train/time_elapsed_s": elapsed,
    #                 },
    #                 step=step,
    #             )

    #     if step % args.save_every == 0:
    #         payload = {
    #             "model": model.state_dict(),
    #             "config": vars(args),
    #             "step": step,
    #         }
    #         torch.save(payload, args.out)
    #         _write_json_atomic(args.state_path, state.__dict__)

    
    for batch in loader:
        if step >= args.steps:
            break

        micro_step += 1

        labels = batch["labels"].to(device=device, dtype=torch.long, non_blocking=True)  # [B, S]
        expert_scores = batch["expert_logits"][:, :, :args.layers].to(device)  # [B, S, L, E]
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)  # [B, S]

        labels[~attention_mask] = -100

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(expert_scores, attention_mask)  # [B, S, vocab_size]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            loss = loss / args.grad_accum

        scaler.scale(loss).backward()

        if micro_step % args.grad_accum != 0:
            continue

        for opt in optimizers:
            scaler.step(opt)
        scaler.update()
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        for sched in schedulers:
            sched.step()

        step += 1
        state = batch.get("state", None)
        if state is not None:
            state.step = step
        micro_step = 0

        if step % 10 == 0:
            elapsed = time.time() - start_time
            lr_adam = schedulers[0].get_last_lr()[0]
            lr_muon = schedulers[1].get_last_lr()[0]
            print(
                f"step {step} loss {loss.item():.4f} lr_adam {lr_adam:.6e} lr_muon {lr_muon:.6e}"
            )
            if wandb_run:
                wandb_run.log(
                    {
                        "train/loss": loss.item() * args.grad_accum,
                        "train/lr_adam": lr_adam,
                        "train/lr_muon": lr_muon,
                        "train/step": step,
                        "train/time_elapsed_s": elapsed,
                    },
                    step=step,
                )

        if step % args.save_every == 0:
            payload = {
                "model": model.state_dict(),
                "config": vars(args),
                "step": step,
            }
            torch.save(payload, args.out)
            _write_json_atomic(args.state_path, state.__dict__ if state else {})


    
    payload = {
        "model": model.state_dict(),
        "config": vars(args),
        "step": step,
    }
    torch.save(payload, args.out)
    _write_json_atomic(args.state_path, state.__dict__ if state is not None else {})

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()