# -*- coding: utf-8 -*-
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
import os

VOIGT_LABELS, VOIGT_INDICES = [], []
for i in range(6):
    for j in range(i, 6):
        VOIGT_LABELS.append(f"C{i+1}{j+1}")
        VOIGT_INDICES.append((i, j))

OUTPUT_DIR = "preprocessing_plots_clean"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading elastic tensor data...")
with open("dataset_equivariant/elastic_tensors.json", "r") as f:
    raw_data = json.load(f)
print(f"Total raw: {len(raw_data)}")

component_data = defaultdict(list)
material_ids, rejected = [], 0
for mat_id, tensor in raw_data.items():
    if tensor is None:
        rejected += 1
        continue
    arr = np.array(tensor)
    if arr.shape != (6, 6):
        rejected += 1
        continue
    if np.max(np.abs(arr)) >= 1000.0:
        rejected += 1
        continue
    material_ids.append(mat_id)
    for label, (i, j) in zip(VOIGT_LABELS, VOIGT_INDICES):
        component_data[label].append(arr[i, j])

print(f"Clean: {len(material_ids)}  Rejected: {rejected}")
for label in VOIGT_LABELS:
    component_data[label] = np.array(component_data[label])

# Summary Statistics
print("\n" + "=" * 95)
hdr = f"{'Component':<10} {'Count':>7} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Median':>10} {'Zeros%':>8}"
print(hdr)
print("=" * 95)
for label in VOIGT_LABELS:
    vals = component_data[label]
    zero_pct = 100.0 * np.sum(np.abs(vals) < 0.5) / len(vals)
    print(f"{label:<10} {len(vals):>7} {vals.mean():>10.2f} {vals.std():>10.2f} "
          f"{vals.min():>10.2f} {vals.max():>10.2f} {np.median(vals):>10.2f} {zero_pct:>7.1f}%")

# Outlier Detection
print("\n" + "=" * 75)
print("OUTLIER DETECTION on CLEAN data (IQR: Q1 - 1.5*IQR to Q3 + 1.5*IQR)")
print("=" * 75)
outlier_counts = {}
for label in VOIGT_LABELS:
    vals = component_data[label]
    Q1, Q3 = np.percentile(vals, 25), np.percentile(vals, 75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_out = np.sum((vals < lower) | (vals > upper))
    outlier_counts[label] = n_out
    print(f"{label}: {n_out:>5} outliers  IQR Range: [{lower:>8.1f}, {upper:>8.1f}]  "
          f"Actual: [{vals.min():>8.1f}, {vals.max():>8.1f}]")

# PLOT 1: All 21 histograms
print("\nGenerating clean histograms...")
fig, axes = plt.subplots(7, 3, figsize=(18, 28))
fig.suptitle(f"Distribution of All 21 Elastic Tensor Components (Cleaned, N={len(material_ids):,})",
             fontsize=16, fontweight="bold", y=1.0)

for idx, (label, (i, j)) in enumerate(zip(VOIGT_LABELS, VOIGT_INDICES)):
    ax = axes[idx // 3, idx % 3]
    vals = component_data[label]
    nonzero_vals = vals[np.abs(vals) > 0.5]
    zero_pct = 100.0 * (1 - len(nonzero_vals) / len(vals))
    if zero_pct > 50:
        if len(nonzero_vals) > 5:
            ax.hist(nonzero_vals, bins=60, color="#e74c3c", edgecolor="white", linewidth=0.3, alpha=0.85)
            ax.set_title(f"{label}  ({zero_pct:.0f}% zero, non-zero shown)", fontsize=10, fontweight="bold", color="#c0392b")
        else:
            ax.text(0.5, 0.5, f"{label}\n{zero_pct:.0f}% zero", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="gray")
            ax.set_title(f"{label}  (Nearly all zero)", fontsize=10, fontweight="bold", color="gray")
    else:
        ax.hist(vals, bins=80, color="steelblue", edgecolor="white", linewidth=0.3, alpha=0.85)
        ax.axvline(x=np.mean(vals), color="orange", linestyle="-", linewidth=1.5, label=f"Mean={np.mean(vals):.1f}")
        ax.axvline(x=np.median(vals), color="green", linestyle="--", linewidth=1.5, label=f"Median={np.median(vals):.1f}")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_title(f"{label}  (std={np.std(vals):.1f})", fontsize=10, fontweight="bold")
    ax.axvline(x=0, color="red", linestyle=":", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("GPa", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(os.path.join(OUTPUT_DIR, "histograms_all_21_clean.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: histograms_all_21_clean.png")

# PLOT 2: Diagonal detail
print("Generating diagonal detail plots...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Diagonal Stiffness Components (C11-C66) -- Detailed", fontsize=15, fontweight="bold")
diag_labels = ["C11", "C22", "C33", "C44", "C55", "C66"]
diag_colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12", "#1abc9c"]

for idx, (label, color) in enumerate(zip(diag_labels, diag_colors)):
    ax = axes[idx // 3, idx % 3]
    vals = component_data[label]
    ax.hist(vals, bins=100, color=color, edgecolor="white", linewidth=0.3, alpha=0.8)
    ax.axvline(x=np.mean(vals), color="black", linestyle="-", linewidth=1.5, label=f"Mean={np.mean(vals):.1f}")
    ax.axvline(x=np.median(vals), color="gray", linestyle="--", linewidth=1.5, label=f"Median={np.median(vals):.1f}")
    Q1, Q3 = np.percentile(vals, 25), np.percentile(vals, 75)
    ax.axvspan(Q1, Q3, alpha=0.15, color="yellow", label=f"IQR=[{Q1:.0f}, {Q3:.0f}]")
    ax.set_title(f"{label}  (N={len(vals):,})", fontsize=12, fontweight="bold")
    ax.set_xlabel("GPa", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "histograms_diagonal_detail.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: histograms_diagonal_detail.png")

# PLOT 3: Box plots
print("Generating clean box plots...")
fig, axes = plt.subplots(3, 1, figsize=(18, 16))
fig.suptitle("Box Plots -- Outlier Detection (Cleaned Dataset)", fontsize=15, fontweight="bold")
groups = [
    ("Diagonal (Stiffness)", ["C11", "C22", "C33", "C44", "C55", "C66"]),
    ("Primary Off-Diagonal", ["C12", "C13", "C23", "C14", "C15", "C16"]),
    ("Secondary Off-Diagonal", ["C24", "C25", "C26", "C34", "C35", "C36", "C45", "C46", "C56"]),
]
for ax, (group_name, labels) in zip(axes, groups):
    box_data = [component_data[l] for l in labels]
    bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True,
                    showfliers=True, flierprops=dict(marker=".", markersize=3, alpha=0.4, color="red"))
    colors_bp = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    for patch, c in zip(bp["boxes"], colors_bp):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_title(f"{group_name}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Value (GPa)", fontsize=11)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "boxplots_clean.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: boxplots_clean.png")

# PLOT 4: Sparsity Heatmap
print("Generating sparsity heatmap...")
zero_pct_matrix = np.zeros((6, 6))
for label, (i, j) in zip(VOIGT_LABELS, VOIGT_INDICES):
    vals = component_data[label]
    pct = 100.0 * np.sum(np.abs(vals) < 0.5) / len(vals)
    zero_pct_matrix[i, j] = pct
    zero_pct_matrix[j, i] = pct

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(zero_pct_matrix, cmap="YlOrRd_r", vmin=0, vmax=100)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("% of Crystals with |Value| < 0.5 GPa", fontsize=11)
for i in range(6):
    for j in range(6):
        text_color = "white" if zero_pct_matrix[i, j] < 40 else "black"
        ax.text(j, i, f"{zero_pct_matrix[i,j]:.0f}%", ha="center", va="center",
                fontsize=12, fontweight="bold", color=text_color)
labels_6 = [f"C{i+1}" for i in range(6)]
ax.set_xticks(range(6))
ax.set_yticks(range(6))
ax.set_xticklabels(labels_6, fontsize=11)
ax.set_yticklabels(labels_6, fontsize=11)
ax.set_title("Tensor Component Sparsity Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sparsity_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: sparsity_heatmap.png")

# PLOT 5: Outlier bar chart
print("Generating outlier bar chart...")
fig, ax = plt.subplots(figsize=(14, 6))
bar_colors = ["#e74c3c" if v > 500 else "#f39c12" if v > 200 else "#2ecc71" for v in outlier_counts.values()]
bars = ax.bar(outlier_counts.keys(), outlier_counts.values(), color=bar_colors, edgecolor="white", linewidth=0.5)
for bar, count in zip(bars, outlier_counts.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(count),
            ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xlabel("Tensor Component", fontsize=12)
ax.set_ylabel("Number of Outliers (IQR)", fontsize=12)
ax.set_title("Outlier Count per Component (Clean Dataset)", fontsize=14, fontweight="bold")
ax.tick_params(axis="x", rotation=45, labelsize=10)
ax.grid(axis="y", alpha=0.3)
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#e74c3c", label=">500 (High)"),
                   Patch(facecolor="#f39c12", label="200-500 (Medium)"),
                   Patch(facecolor="#2ecc71", label="<200 (Low)")]
ax.legend(handles=legend_elements, fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "outlier_counts_bar.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: outlier_counts_bar.png")

# PLOT 6: Correlation heatmap
print("Generating correlation heatmap...")
diag_data = np.column_stack([component_data[l] for l in diag_labels])
corr = np.corrcoef(diag_data.T)

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Pearson Correlation", fontsize=11)
for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=11, fontweight="bold",
                color="white" if abs(corr[i,j]) > 0.5 else "black")
ax.set_xticks(range(6))
ax.set_yticks(range(6))
ax.set_xticklabels(diag_labels, fontsize=11)
ax.set_yticklabels(diag_labels, fontsize=11)
ax.set_title("Correlation Between Diagonal Stiffness Components", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_diagonal.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: correlation_diagonal.png")

print("\n" + "=" * 60)
print(f"All clean plots saved to: {OUTPUT_DIR}/")
print("=" * 60)
