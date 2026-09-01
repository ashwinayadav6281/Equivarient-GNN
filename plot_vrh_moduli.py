import json
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
PROJECT_DIR = r"c:\Users\arvin\Downloads\UGP-Phy"
sys.path.append(PROJECT_DIR)
from tensor_utils import calculate_moduli, check_stability
TENSORS_JSON = os.path.join(PROJECT_DIR, "dataset_equivariant", "elastic_tensors.json")
PLOT_DIR = os.path.join(PROJECT_DIR, "Plots")

def main():
    print(f"Loading data from {TENSORS_JSON}...")
    with open(TENSORS_JSON, "r") as f:
        tensors = json.load(f)
        
    print(f"Loaded {len(tensors)} tensors.")
    
    K_vals = []
    G_vals = []
    labels = []
    
    stable_count = 0
    valid_count = 0
    
    for mat_id, C in tensors.items():
        C_arr = np.array(C)
        if C_arr.shape != (6, 6):
            continue
            
        moduli = calculate_moduli(C_arr)
        
        K = moduli.get('K_H', np.nan)
        G = moduli.get('G_H', np.nan)
        
        # We only want physically meaningful, computable values
        if not np.isnan(K) and not np.isnan(G):
            K_vals.append(K)
            G_vals.append(G)
            labels.append(mat_id)
            valid_count += 1
            if check_stability(C_arr):
                stable_count += 1
                
    print(f"Computed VRH moduli for {valid_count} compounds ({stable_count} are strictly Born-stable).")
    
    K_vals = np.array(K_vals)
    G_vals = np.array(G_vals)
    
    # Check for extreme outliers
    print("\n--- Summary Statistics (GPa) ---")
    print(f"Bulk Modulus  | Mean: {np.mean(K_vals):.1f} | Median: {np.median(K_vals):.1f} | Min: {np.min(K_vals):.1f} | Max: {np.max(K_vals):.1f}")
    print(f"Shear Modulus | Mean: {np.mean(G_vals):.1f} | Median: {np.median(G_vals):.1f} | Min: {np.min(G_vals):.1f} | Max: {np.max(G_vals):.1f}")
    
    # Let's find some top outliers
    print("\n--- Extreme Outliers (K > 1000 GPa or G > 1000 GPa or values < -500) ---")
    for i in range(len(K_vals)):
        if K_vals[i] > 1000 or G_vals[i] > 1000 or K_vals[i] < -500 or G_vals[i] < -500:
            print(f"{labels[i]:15} | Bulk: {K_vals[i]:7.1f} | Shear: {G_vals[i]:7.1f}")
            
    # Set up the plot
    os.makedirs(PLOT_DIR, exist_ok=True)
    plot_path = os.path.join(PLOT_DIR, "bulk_vs_shear_moduli.png")
    
    plt.figure(figsize=(10, 8), dpi=150)
    
    # We will constrain the axes a bit to ignore crazy numerical artifacts in plot
    # and show the main density distribution well.
    # We'll use a hexbin plot as there are 47k points.
    
    plt.hexbin(G_vals, K_vals, gridsize=100, cmap='viridis', mincnt=1, bins='log')
    cb = plt.colorbar()
    cb.set_label('log10(Count)')
    
    # Calculate Pugh's ratio = K/G boundary lines (Ductile vs Brittle boundary K/G = 1.75)
    # Materials with K/G > 1.75 are ductile (upper left region)
    # Materials with K/G < 1.75 are brittle (lower right region)
    g_range = np.linspace(0, min(1000, np.percentile(G_vals, 99.5)), 100)
    plt.plot(g_range, 1.75 * g_range, 'r--', alpha=0.8, label="Pugh's ratio = 1.75 (Ductile/Brittle boundary)")
    
    plt.xlabel('Shear Modulus, G (GPa)', fontsize=12)
    plt.ylabel('Bulk Modulus, K (GPa)', fontsize=12)
    plt.title('Bulk vs Shear Moduli (VRH) across ~47k Crystals', fontsize=14)
    
    # Focus axes on main distribution
    # Find 0.5% and 99.5% percentiles for reasonable limits
    g_min, g_max = np.percentile(G_vals, [0.5, 99.5])
    k_min, k_max = np.percentile(K_vals, [0.5, 99.5])
    
    # Expand slightly
    plt.xlim(max(-50, g_min - 20), g_max + 50)
    plt.ylim(max(-50, k_min - 50), k_max + 50)
    
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"\nPlot saved to {plot_path}")
    
if __name__ == "__main__":
    main()
