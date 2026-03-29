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


from read_data import get_train_val_dataloaders


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
        # Per-layer learned bias: gives each transformer layer its own "view" of the expert space.
        # Initialized to zero so the model starts from the same place as before.
        self.layer_emb = nn.Embedding(num_layers, num_experts)
        nn.init.zeros_(self.layer_emb.weight)
        self.layer_mlp = nn.Sequential(
            nn.Linear(num_experts, layer_hidden),
            nn.SiLU(),
            nn.Linear(layer_hidden, layer_proj),
        )
        self.proj = nn.Linear(num_layers * layer_proj, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, expert_logits):
        # expert_logits: [B, S, L, num_experts]
        bsz, seq_len, num_layers, _ = expert_logits.shape

        # Add per-layer learned bias before softmax
        layer_ids = torch.arange(num_layers, device=expert_logits.device)
        layer_bias = self.layer_emb(layer_ids)  # [L, E]
        expert_logits = expert_logits + layer_bias.unsqueeze(0).unsqueeze(0)

        expert_probs = torch.softmax(expert_logits, dim=-1)

        if self.layer_gating:
            gate = torch.sigmoid(self.layer_gate).view(1, 1, self.num_layers, 1)
            expert_probs = expert_probs * gate

        expert_probs = nn.functional.layer_norm(expert_probs, expert_probs.shape[-1:])
        layer_repr = self.layer_mlp(expert_probs)  # [B, S, L, P]

        flat = layer_repr.reshape(bsz, seq_len, num_layers * layer_repr.size(-1))
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


def compute_accuracy(logits, labels, ignore_index=-100):
    """Top-1 and top-5 accuracy over non-ignored positions."""
    mask = (labels != ignore_index).view(-1)
    if mask.sum() == 0:
        return 0.0, 0.0
    flat_logits = logits.view(-1, logits.size(-1))[mask]
    flat_labels = labels.view(-1)[mask]
    top1 = (flat_logits.argmax(dim=-1) == flat_labels).float().mean().item()
    top5 = (
        (flat_logits.topk(5, dim=-1).indices == flat_labels.unsqueeze(-1))
        .any(-1)
        .float()
        .mean()
        .item()
    )
    return top1, top5


@torch.no_grad()
def run_validation(model, val_loader, device, layers, max_batches=50):
    model.eval()
    total_loss, total_top1, total_top5, n = 0.0, 0.0, 0.0, 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        labels = batch["labels"].to(device=device, dtype=torch.long, non_blocking=True)
        expert_scores = batch["expert_logits"][:, :, :layers].to(device)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels[~attention_mask] = -100
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(expert_scores, attention_mask)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
            )
        top1, top5 = compute_accuracy(logits.float(), labels)
        total_loss += loss.item()
        total_top1 += top1
        total_top5 += top5
        n += 1
    model.train()
    return total_loss / max(n, 1), total_top1 / max(n, 1), total_top5 / max(n, 1)


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
    parser.add_argument("--data-dir", default="./router_dataset_v2")
    parser.add_argument("--val-shards", type=int, default=2)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--val-batches", type=int, default=50)
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

    loader, val_loader = get_train_val_dataloaders(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        val_shards=args.val_shards,
    )

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
            train_top1, train_top5 = compute_accuracy(logits.float().detach(), labels.detach())
            print(
                f"step {step} loss {loss.item() * args.grad_accum:.4f} "
                f"acc@1 {train_top1:.3f} acc@5 {train_top5:.3f} "
                f"lr_adam {lr_adam:.6e} lr_muon {lr_muon:.6e}"
            )
            if wandb_run:
                wandb_run.log(
                    {
                        "train/loss": loss.item() * args.grad_accum,
                        "train/acc_top1": train_top1,
                        "train/acc_top5": train_top5,
                        "train/lr_adam": lr_adam,
                        "train/lr_muon": lr_muon,
                        "train/step": step,
                        "train/time_elapsed_s": elapsed,
                    },
                    step=step,
                )

        if step % args.val_every == 0:
            val_loss, val_top1, val_top5 = run_validation(
                model, val_loader, device, args.layers, max_batches=args.val_batches
            )
            print(
                f"  [val] step {step} loss {val_loss:.4f} acc@1 {val_top1:.3f} acc@5 {val_top5:.3f}"
            )
            if wandb_run:
                wandb_run.log(
                    {
                        "val/loss": val_loss,
                        "val/acc_top1": val_top1,
                        "val/acc_top5": val_top5,
                        "val/step": step,
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