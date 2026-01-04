import glob
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# Config
# -----------------------
TRACE_DIR = "traces"
OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

# Columns
POWER = "power.draw"
GPU_UTIL = "utilization.gpu"
SM_CLOCK = "clocks.sm"
MEM_CLOCK = "clocks.mem"
MEM_UTIL = "utilization.memory"

# Colors and markers for architectures
COLORS = {"Dense": "blue", "MoE": "red"}
MARKERS = {"Dense": "o", "MoE": "s"}

# -----------------------
# Helper functions
# -----------------------
def parse_filename(fname):
    """trace_dense_7b_neutral_run9.csv -> model, prompt, run"""
    m = re.match(r"trace_(.+?)_(.+?)_run(\d+)\.csv", fname)
    if not m:
        raise ValueError(f"Unexpected filename: {fname}")
    return m.group(1), m.group(2), int(m.group(3))

def architecture(model):
    return "MoE" if "moe" in model else "Dense"

# -----------------------
# Load traces
# -----------------------
rows = []

for path in glob.glob(f"{TRACE_DIR}/trace_*.csv"):
    fname = os.path.basename(path)
    try:
        df = pd.read_csv(path)
        if df.empty:
            print(f"[Warning] Empty file skipped: {fname}")
            continue
    except pd.errors.EmptyDataError:
        print(f"[Warning] Empty file skipped: {fname}")
        continue

    required_cols = [POWER, GPU_UTIL, SM_CLOCK, MEM_CLOCK, MEM_UTIL]
    if not all(col in df.columns for col in required_cols):
        print(f"[Warning] Missing columns in {fname}, skipping")
        continue

    model, prompt, run = parse_filename(fname)
    arch = architecture(model)

    rows.append({
        "model": model,
        "architecture": arch,
        "prompt": prompt,
        "run": run,
        "mean_power": df[POWER].mean(),
        "mean_gpu_util": df[GPU_UTIL].mean(),
        "mean_sm_clock": df[SM_CLOCK].mean(),
        "mean_mem_clock": df[MEM_CLOCK].mean(),
        "mean_mem_util": df[MEM_UTIL].mean(),
    })

df_feat = pd.DataFrame(rows)

# -----------------------
# Multi-subfigure figure: one subplot per prompt
# -----------------------
prompts = df_feat["prompt"].unique()
n_prompts = len(prompts)

fig, axes = plt.subplots(1, n_prompts, figsize=(6*n_prompts, 5), sharey=True)

if n_prompts == 1:
    axes = [axes]

for ax, prompt in zip(axes, prompts):
    subset = df_feat[df_feat["prompt"] == prompt]
    for arch, group in subset.groupby("architecture"):
        ax.scatter(
            group["mean_power"],
            group["mean_gpu_util"],
            color=COLORS[arch],
            marker=MARKERS[arch],
            s=80,
            alpha=0.8,
            label=arch
        )
    ax.set_xlabel("Mean Power (W)")
    ax.set_ylabel("Mean GPU Utilization (%)")
    ax.set_title(f"Prompt: '{prompt}'")
    ax.grid(True)

# Only one legend for whole figure
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", fontsize=10)
fig.suptitle("GPU Side-Channel Features Across Architectures and Prompts", fontsize=14)
plt.tight_layout(rect=[0, 0, 0.95, 0.93])

# Save figure
out_file = f"{OUTDIR}/all_prompts_architecture_comparison.png"
plt.savefig(out_file, dpi=300)
plt.close()
print(f"Saved figure: {out_file}")
