import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

PROJECT_DIR = r"c:\Users\arvin\Downloads\UGP-Phy"
sys.path.append(PROJECT_DIR)
from tensor_utils import calculate_moduli, check_stability

TENSORS_JSON = os.path.join(PROJECT_DIR, "dataset_equivariant", "elastic_tensors.json")
CLEAN_TENSORS_JSON = os.path.join(PROJECT_DIR, "dataset_equivariant", "elastic_tensors_clean.json")
PLOT_DIR = os.path.join(PROJECT_DIR, "Plots")

def main():
    print(f"Loading raw data from {TENSORS_JSON}...")
    with open(TENSORS_JSON, "r") as f:
        tensors = json.load(f)
        
    print(f"Loaded {len(tensors)} total tensors.")
    
    clean_tensors = {}
    K_vals = []
    G_vals = []
    
    for mat_id, C in tensors.items():
        C_arr = np.array(C)
        if C_arr.shape != (6, 6):
            continue
            
        moduli = calculate_moduli(C_arr)
        
        K = moduli.get('K_H', np.nan)
        G = moduli.get('G_H', np.nan)
        
        # Apply the physical filter:
        # VRH Modulus > 1000 GPa or < -50 GPa are excluded
        if not np.isnan(K) and not np.isnan(G):
            if -50 <= K <= 1000 and -50 <= G <= 1000:
                clean_tensors[mat_id] = C
                K_vals.append(K)
                G_vals.append(G)
                
    print(f"Filtered out extreme outliers.")
    print(f"Cleaned dataset contains {len(clean_tensors)} compounds (removed {len(tensors) - len(clean_tensors)} outliers).")
    
    # Save the cleaned dataset
    with open(CLEAN_TENSORS_JSON, "w") as f:
        json.dump(clean_tensors, f)
    print(f"Saved clean tensors to {CLEAN_TENSORS_JSON}")
    
    K_vals = np.array(K_vals)
    G_vals = np.array(G_vals)
    
    # Re-plot the cleaned data
    os.makedirs(PLOT_DIR, exist_ok=True)
    plot_path = os.path.join(PLOT_DIR, "bulk_vs_shear_moduli_clean.png")
    
    plt.figure(figsize=(10, 8), dpi=150)
    
    # Hexbin plot for density
    plt.hexbin(G_vals, K_vals, gridsize=80, cmap='viridis', mincnt=1, bins='log')
    cb = plt.colorbar()
    cb.set_label('log10(Count)')
    
    # Pugh's ratio = 1.75 line
    g_range = np.linspace(max(0, np.min(G_vals)), np.max(G_vals), 100)
    plt.plot(g_range, 1.75 * g_range, 'r--', alpha=0.8, label="Pugh's ratio = 1.75 (Ductile/Brittle)")
    
    plt.xlabel('Shear Modulus, G (GPa)', fontsize=12)
    plt.ylabel('Bulk Modulus, K (GPa)', fontsize=12)
    plt.title('Bulk vs Shear Moduli (VRH) - Cleaned Dataset', fontsize=14)
    
    # Limits tightly fit to the filtered bounds
    plt.xlim(max(-60, np.min(G_vals) - 10), min(1010, np.max(G_vals) + 20))
    plt.ylim(max(-60, np.min(K_vals) - 10), min(1010, np.max(K_vals) + 20))
    
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
