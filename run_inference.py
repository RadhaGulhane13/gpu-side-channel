import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

DEVICE = "cuda"

MODELS = {
    # Dense (GPT-style)
    "dense_7b": "meta-llama/Llama-2-7b-hf",
    # "dense_13b": "meta-llama/Llama-2-13b-hf",
    # Uncomment only if GPU supports it
    # "dense_70b": "meta-llama/Llama-2-70b-hf",

    # MoE
    "moe_8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    # "moe_8x22b": "mistralai/Mixtral-8x22B-Instruct-v0.1",
}

PROMPTS = {
    "neutral": (
        "Summarize the following paragraph in one sentence:\n"
        "Artificial intelligence systems are increasingly deployed at scale."
    ),
    "math": "Solve step by step: ∫ x^2 e^x dx",
    "code": "Write an efficient Python implementation of quicksort"
}

def run(model_key, prompt_key):
    model_name = MODELS[model_key]
    prompt = PROMPTS[prompt_key]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False
        )

if __name__ == "__main__":
    model_key = sys.argv[1]
    prompt_key = sys.argv[2]
    run(model_key, prompt_key)
