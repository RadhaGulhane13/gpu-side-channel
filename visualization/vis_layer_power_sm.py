import glob
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Configuration ----------------
TRACE_DIR = "../layer_traces"

LAYERS = ["attention", "ffn", "moe_router", "moe_expert"]
COLORS = {
    "attention": "tab:blue",
    "ffn": "tab:orange",
    "moe_router": "tab:green",
    "moe_expert": "tab:red",
}

# ---------------- Load & Aggregate ----------------
records = []
scatter_data = []

for layer in LAYERS:
    files = glob.glob(f"{TRACE_DIR}/trace_{layer}_run*.csv")

    for f in files:
        try:
            df = pd.read_csv(f)
            if df.empty:
                continue
        except Exception:
            continue

        # ---- Aggregate per run (POC signal) ----
        records.append({
            "layer": layer,
            "mean_power": df["power.draw"].mean(),
            "std_power": df["power.draw"].std(),
        })

        # ---- Subsample for scatter (avoid overplotting) ----
        df = df.sample(frac=0.3, random_state=0)
        df["layer"] = layer
        scatter_data.append(df)

df_runs = pd.DataFrame(records)
df_scatter = pd.concat(scatter_data, ignore_index=True)

# ---------------- Plot ----------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ===== Left: Mean power distribution (CORE POC) =====
for layer in LAYERS:

    vals = df_runs[df_runs.layer == layer]["mean_power"].dropna()

    if len(vals) < 2:
        print(f"[WARN] Not enough data to plot histogram for {layer}")
        continue

    axes[0].hist(
        vals,
        bins=10,
        alpha=0.6,
        label=layer.replace("_", " ").title(),
        color=COLORS[layer],
        density=True,
    )

axes[0].set_xlabel("Mean GPU Power per Run (W)", fontsize=12)
axes[0].set_ylabel("Density", fontsize=12)
axes[0].set_title("Distribution of Mean GPU Power", fontsize=14)
axes[0].legend(frameon=False)
axes[0].grid(alpha=0.3)

# ===== Right: Power vs Utilization (MECHANISM) =====
for layer in LAYERS:
    sub = df_scatter[df_scatter.layer == layer]
    axes[1].scatter(
        sub["utilization.gpu"],
        sub["power.draw"],
        s=6,
        alpha=0.4,
        label=layer.replace("_", " ").title(),
        color=COLORS[layer],
    )

axes[1].set_xlabel("GPU Utilization (%)", fontsize=12)
axes[1].set_ylabel("Power Draw (W)", fontsize=12)
axes[1].set_title("Power–Utilization Manifold", fontsize=14)
axes[1].legend(frameon=False)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
plt.savefig(f"{TRACE_DIR}/layer_power_analysis.png")
plt.close()
print(f"Saved figure: {TRACE_DIR}/layer_power_analysis.png")