#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

MATRIX_CSV = "rmsc_summary.csv"

mlips = ["mace_omol", "mace_off_small", "mace_off_medium", "mace_off_large"]
xtick_labels = ["OMOL", "OFF-S", "OFF-M", "OFF-L"]

col_order = ["MACE_mu_alpha-Raman", "MACE_mu_alpha-IR", "MACE_mu-IR"]

palette = {
    "MACE_mu_alpha-Raman": "royalblue",
    "MACE_mu_alpha-IR": "indianred",
    "MACE_mu-IR": "#DAA520",
}

df = pd.read_csv(MATRIX_CSV)
df = df.rename(columns={df.columns[0]: "Model"})

for c in col_order:
    if c not in df.columns:
        df[c] = np.nan
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.set_index("Model").reindex(mlips)

# Broken axis limits
y_low_max = 0.09
y_high_min = 0.76
y_high_max = 1.03

x = np.arange(len(mlips))
width = 0.18
bar_gap = 0.04

fig = plt.figure(figsize=(9, 5.2))
gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.05)
ax_top = fig.add_subplot(gs[0])
ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

# Bars on both axes
for i, src in enumerate(col_order):
    offset = (i - (len(col_order) - 1) / 2) * (width + bar_gap)
    vals = df[src].values
    ax_top.bar(x + offset, vals, width=width, color=palette.get(src, None),
               edgecolor="black", linewidth=1.2, label=src, alpha=0.9)
    ax_bot.bar(x + offset, vals, width=width, color=palette.get(src, None),
               edgecolor="black", linewidth=1.2, alpha=0.9)

ax_top.set_ylim(y_high_min, y_high_max)
ax_bot.set_ylim(0, y_low_max)

# Hide spines between axes
ax_top.spines["bottom"].set_visible(False)
ax_bot.spines["top"].set_visible(False)

# --- FIX 1: remove ALL x ticks/labels from ax_top (ticks + marks)
ax_top.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

# Keep x ticks only on bottom axis
ax_bot.xaxis.tick_bottom()

d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=14,
              linestyle="none", color='k', mec='k', mew=1.7, clip_on=False)
ax_top.plot([0, 1], [0, 0],transform=ax_top.transAxes, **kwargs)
ax_bot.plot([0, 1], [1, 1],transform=ax_bot.transAxes, **kwargs)
# Axis formatting
ax_top.set_ylabel(r"Match Score ($r_{\mathrm{msc}}$)", fontsize=14)
ax_bot.set_xlabel("MLIP Model", fontsize=14)

ax_bot.set_xticks(x)
ax_bot.set_xticklabels(xtick_labels, fontsize=13)

fmt = FuncFormatter(lambda y, _: f"{y:.2f}")
ax_top.yaxis.set_major_formatter(fmt)
ax_bot.yaxis.set_major_formatter(fmt)

ax_top.grid(axis="y", linestyle="--", alpha=0.7)
ax_bot.grid(axis="y", linestyle="--", alpha=0.7)

ax_top.legend(frameon=False, fontsize=11, loc="upper left")

for ax in (ax_top, ax_bot):
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(axis="y", labelsize=12, width=1.5)

plt.tight_layout()
plt.savefig("combined_RMSC_barplot_brokenaxis.png", dpi=400)
plt.savefig("combined_RMSC_barplot_brokenaxis.pdf")
plt.close()
print("✅ Saved combined_RMSC_barplot_brokenaxis.png and .pdf")
