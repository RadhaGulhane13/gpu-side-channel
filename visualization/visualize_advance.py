import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUTDIR = "traces"
MODELS = ["dense_7b", "moe_8x7b"]
PROMPTS = ["neutral", "math", "code"]

# Store mean power per (model, prompt)
mean_power = {p: {m: [] for m in MODELS} for p in PROMPTS}

for prompt in PROMPTS:
    for model in MODELS:
        files = glob.glob(f"{OUTDIR}/trace_{model}_{prompt}_run*.csv")
        for f in files:
            try:
                df = pd.read_csv(f)
            except pd.errors.EmptyDataError:
                print(f"[Warning] Empty file skipped: {f}")
                continue
            
            mean_power[prompt][model].append(df["power.draw"].mean())

# ---------------- Plot ----------------
fig, axes = plt.subplots(
    1, len(PROMPTS), figsize=(15, 4), sharey=True
)

for ax, prompt in zip(axes, PROMPTS):
    for model in MODELS:
        ax.hist(
            mean_power[prompt][model],
            bins=10,
            alpha=0.6,
            density=True,
            label=model.replace("_", " ").upper(),
        )

    ax.set_title(f"Prompt: {prompt}", fontsize=14)
    ax.set_xlabel("Mean GPU Power (W)", fontsize=12)
    ax.grid(alpha=0.3)

axes[0].set_ylabel("Density", fontsize=12)
axes[-1].legend(frameon=False)

plt.tight_layout()
plt.show()
plt.savefig(f"{OUTDIR}/mean_power_distribution_by_prompt.png")
plt.close()
print(f"Saved figure: {OUTDIR}/mean_power_distribution_by_prompt.png")