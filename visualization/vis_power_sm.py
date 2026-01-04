import glob
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# Configuration
# -----------------------

TRACE_DIR = "../traces"
OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

# Columns from your log
POWER = "power.draw"
GPU_UTIL = "utilization.gpu"
SM_CLOCK = "clocks.sm"
MEM_CLOCK = "clocks.mem"
MEM_UTIL = "utilization.memory"

# Figure style
plt.style.use("ggplot")  # simple nice grid

MARKERS = {"Dense": "o", "MoE": "s"}

# -----------------------
# Helper functions
# -----------------------

def parse_filename(fname):
    """
    trace_dense_7b_neutral_run9.csv
    trace_dense_7b_code_run1.csv
    -> model=dense_7b, prompt=neutral|math|code, run=int
    """
    m = re.match(
        r"trace_(.+?)_(neutral|math|code)_run(\d+)\.csv",
        fname
    )
    if not m:
        raise ValueError(f"Unexpected filename: {fname}")
    return m.group(1), m.group(2), int(m.group(3))

def architecture(model):
    return "MoE" if "moe" in model else "Dense"

# -----------------------
# Load all runs
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

    # Check required columns
    required_cols = [POWER, GPU_UTIL, SM_CLOCK, MEM_CLOCK, MEM_UTIL]
    if not all(col in df.columns for col in required_cols):
        print(f"[Warning] Missing columns in {fname}, skipping")
        continue

    model, prompt, run = parse_filename(fname)

    rows.append({
        "model": model,
        "architecture": architecture(model),
        "prompt": prompt,
        "run": run,
        "mean_power": df[POWER].mean(),
        "std_power": df[POWER].std(),
        "mean_gpu_util": df[GPU_UTIL].mean() * 100,
        "mean_sm_clock": df[SM_CLOCK].mean(),
        "mean_mem_clock": df[MEM_CLOCK].mean(),
        "mean_mem_util": df[MEM_UTIL].mean(),
    })

df_feat = pd.DataFrame(rows)

print("Runs per prompt / architecture:")
print(df_feat.groupby(["prompt", "architecture"]).size())

# -----------------------
# Plotting multi-subfigure figure
# -----------------------

prompts = df_feat["prompt"].unique()
n_prompts = len(prompts)

fig, axes = plt.subplots(1, n_prompts, figsize=(6*n_prompts, 5), sharey=True)

if n_prompts == 1:
    axes = [axes]  # ensure iterable

for ax, prompt in zip(axes, prompts):
    subset = df_feat[df_feat["prompt"] == prompt]

    for arch, group in subset.groupby("architecture"):
        ax.scatter(
            group["mean_power"],
            group["mean_gpu_util"],
            label=f"{arch}",
            alpha=0.8,
            marker=MARKERS[arch],
            s=80
        )
    
    ax.set_xlabel("Mean Power (W)")
    ax.set_ylabel("Mean GPU Utilization (%)")
    ax.set_title(f"Prompt: '{prompt}'")
    ax.legend()
    ax.grid(True)

plt.suptitle("GPU Side-Channel Features Across Architectures (Power vs GPU Util)")
plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig(f"{OUTDIR}/power_gpu_util_by_arch_prompt.png")
plt.close()

print(f"Saved figure: {OUTDIR}/power_gpu_util_by_arch_prompt.png")

# -----------------------
# Optional: other features
# You can repeat similar subfigures for:
# mean SM clock, memory clock, memory utilization
# -----------------------

# Example: SM clock vs Power
fig, axes = plt.subplots(1, n_prompts, figsize=(6*n_prompts, 5), sharey=True)
if n_prompts == 1:
    axes = [axes]

for ax, prompt in zip(axes, prompts):
    subset = df_feat[df_feat["prompt"] == prompt]
    for arch, group in subset.groupby("architecture"):
        ax.scatter(
            group["mean_sm_clock"],
            group["mean_power"],
            label=f"{arch}",
            alpha=0.8,
            marker=MARKERS[arch],
            s=80
        )
    ax.set_xlabel("Mean SM Clock (MHz)")
    ax.set_ylabel("Mean Power (W)")
    ax.set_title(f"Prompt: '{prompt}'")
    ax.legend()
    ax.grid(True)

plt.suptitle("GPU Side-Channel Features Across Architectures (SM Clock vs Power)")
plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig(f"{OUTDIR}/smclock_vs_power_by_arch_prompt.png")
plt.close()

print(f"Saved figure: {OUTDIR}/smclock_vs_power_by_arch_prompt.png")

