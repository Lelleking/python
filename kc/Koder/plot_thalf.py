import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ===============================
# ARGUMENT
# ===============================
if len(sys.argv) < 3:
    print("Usage: python plot_thalf.py <labname> <antibody> [title]")
    sys.exit(1)

lab = sys.argv[1]
antibody = sys.argv[2]
title = sys.argv[3] if len(sys.argv) > 3 else f"{antibody} t₀.₅"

# ===============================
# PATH
# ===============================
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
map_path = os.path.join(base_dir, lab, f"{lab}_map.csv")

df = pd.read_csv(map_path)

# ===============================
# FILTRERA ANTIBODY
# ===============================
df = df[df["antibody"] == antibody]

if df.empty:
    print(f"No data found for '{antibody}'")
    sys.exit(1)

df = df.dropna(subset=["half_time"])

# ===============================
# MATCHA ANTAL 0 µM REPLIKAT
# ===============================
rep_counts = (
    df[df["conc_uM"] != 0]
    .groupby("conc_uM")
    .size()
)

if not rep_counts.empty:
    target_reps = rep_counts.min()

    zero_rows = df[df["conc_uM"] == 0]
    if not zero_rows.empty:
        df = pd.concat([
            df[df["conc_uM"] != 0],
            zero_rows.iloc[:target_reps]
        ])

# ===============================
# GROUP MEAN + SD
# ===============================
summary = (
    df.groupby("conc_uM")["half_time"]
    .agg(["mean", "std"])
    .reset_index()
    .sort_values("conc_uM")
)

# Ta bort 0 för log-axel
summary_nonzero = summary[summary["conc_uM"] > 0]

# ===============================
# COLORBLIND-SAFE PALETTE
# ===============================
OKABE_ITO = [
    "#000000",  # 0 µM
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#999999",
    "#8B4513",
]

sorted_concs = sorted(summary["conc_uM"].unique())

color_map = {}
color_index = 1

for conc in sorted_concs:
    if conc == 0:
        color_map[conc] = "#000000"
    else:
        color_map[conc] = OKABE_ITO[color_index % len(OKABE_ITO)]
        color_index += 1

# ===============================
# PLOT
# ===============================
fig, ax = plt.subplots(figsize=(7,5))

for _, row in summary_nonzero.iterrows():
    conc = row["conc_uM"]

    ax.errorbar(
        conc,
        row["mean"],
        yerr=row["std"],
        fmt='o',
        capsize=4,
        elinewidth=1.5,
        markersize=7,
        color=color_map[conc],
        label=f"{conc} µM"
    )

ax.set_xscale("log")

# Ticks exakt vid dina konc
unique_conc = sorted(summary_nonzero["conc_uM"].unique())
ax.set_xticks(unique_conc)
ax.set_xticklabels([f"{c:g}" for c in unique_conc])

ax.xaxis.set_minor_locator(
    ticker.LogLocator(base=10.0, subs=np.arange(1,10)*0.1)
)

ax.grid(True, which="both", linestyle="--", linewidth=0.5)

ax.set_xlabel("Antibody concentration (µM)")
ax.set_ylabel("Half-time (h)")
ax.set_title(title)

plt.tight_layout()
plt.show()