import sys
import torch
from transformers import AutoModelForCausalLM

DEVICE = "cuda"
DTYPE = torch.float16

LAYER_TYPE = sys.argv[1]   # attention | ffn | moe_router | moe_expert

def load_dense():
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        torch_dtype=DTYPE,
        device_map="auto"
    )
    block = model.model.layers[0]
    return model, block

def load_moe():
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        torch_dtype=DTYPE,
        device_map="auto"
    )
    block = model.model.layers[0]
    return model, block

@torch.no_grad()
def main():
    if LAYER_TYPE in ["attention", "ffn"]:
        model, block = load_dense()
    else:
        model, block = load_moe()

    model.eval()

    seq_len = 128
    hidden = model.config.hidden_size
    x = torch.randn(1, seq_len, hidden, device=DEVICE, dtype=DTYPE)

    torch.cuda.synchronize()

    if LAYER_TYPE == "attention":
        batch_size, seq_len, _ = x.shape
        position_ids = torch.arange(
            seq_len, device=x.device
        ).unsqueeze(0).expand(batch_size, -1)

        block.self_attn(
            x,
            attention_mask=None,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False
        )

    elif LAYER_TYPE == "ffn":
        block.mlp(x)

    elif LAYER_TYPE == "moe_router":
        block.block_sparse_moe.gate(x)

    elif LAYER_TYPE == "moe_expert":
        block.block_sparse_moe.experts[0](x)

    else:
        raise ValueError("Unknown layer type")

    torch.cuda.synchronize()

if __name__ == "__main__":
    main()
