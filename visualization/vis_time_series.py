import glob
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# Configuration
# -----------------------
TRACE_DIR = "traces"
OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

POWER = "power.draw"
COLORS = {
    "Dense": "#1f77b4",  # muted blue
    "MoE": "#ff7f0e"     # muted orange
}

# -----------------------
# Helper Functions
# -----------------------
def parse_filename(fname):
    """
    Extract architecture (Dense/MoE), prompt, and run number from filename.
    Example filenames:
        trace_dense_7b_neutral_run4.csv
        trace_moe_8b_math_run2.csv
    """
    m = re.match(r"trace_(dense|moe)_.+?_(code|math|neutral)_run(\d+)\.csv", fname, re.IGNORECASE)
    if not m:
        raise ValueError(f"Unexpected filename: {fname}")
    arch_str, prompt, run = m.groups()
    arch = "MoE" if arch_str.lower() == "moe" else "Dense"
    return arch, prompt, int(run)

# -----------------------
# Load traces
# -----------------------
time_series_data = {}

for path in glob.glob(f"{TRACE_DIR}/trace_*.csv"):
    fname = os.path.basename(path)
    try:
        df = pd.read_csv(path)
        if df.empty:
            continue
    except pd.errors.EmptyDataError:
        continue

    try:
        arch, prompt, run = parse_filename(fname)
    except ValueError:
        print(f"Skipping file (unexpected format): {fname}")
        continue

    key = (prompt, arch)
    if key not in time_series_data:
        time_series_data[key] = []
    time_series_data[key].append(df[POWER].values)

# -----------------------
# Fixed prompt order
# -----------------------
prompts = ["code", "math", "neutral"]
arches = ["Dense", "MoE"]

# -----------------------
# Create figure
# -----------------------
fig, axes = plt.subplots(len(prompts), len(arches), 
                         figsize=(6*len(arches), 4*len(prompts)), 
                         sharey=True)

# Ensure axes is always 2D
if len(prompts) == 1 and len(arches) == 1:
    axes = [[axes]]
elif len(prompts) == 1:
    axes = [axes]
elif len(arches) == 1:
    axes = [[ax] for ax in axes]

# -----------------------
# Plotting
# -----------------------
for i, prompt in enumerate(prompts):
    for j, arch in enumerate(arches):
        ax = axes[i][j]
        key = (prompt, arch)
        ax.set_facecolor("whitesmoke")  # optional for empty axes

        if key not in time_series_data:
            ax.set_title(f"Prompt: '{prompt}' | Arch: {arch}\n(No data)")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        runs = time_series_data[key]

        for run_trace in runs:
            ax.plot(run_trace, color=COLORS[arch], alpha=0.3, linewidth=0.8)  # thin lines

        # Mean trace slightly thicker but still thin
        mean_trace = pd.DataFrame(runs).mean(axis=0)
        ax.plot(mean_trace, color=COLORS[arch], linewidth=1.5, label=f"{arch} mean")  # thinner than default 2-3

        ax.set_title(f"Prompt: '{prompt}' | Arch: {arch}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Power (W)")
        ax.grid(True)
        ax.legend()

plt.suptitle("GPU Power Time-Series Across Prompts and Architectures", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])

out_file = f"{OUTDIR}/all_prompts_archs_timeseries.png"
plt.savefig(out_file, dpi=300)
plt.close()
print(f"Saved big time-series figure: {out_file}")
