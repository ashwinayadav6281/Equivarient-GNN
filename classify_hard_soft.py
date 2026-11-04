"""
Classify full Materials Project + JARVIS dataset into Ultrahard and Ultrasoft materials
based on E3NN-predicted and DFT-calculated Bulk Modulus (K_H) and Shear Modulus (G_H).

Pipeline:
1. Load E3NN model & robust normalization stats
2. Run inference across all 41,376 materials (Materials Project + JARVIS)
3. Calculate robust VRH moduli (K_H, G_H), Pugh's ratio (G/K), Poisson's ratio (nu), and Vickers hardness (H_v)
4. Classify into:
   - Ultrahard: Pred Bulk >= 200 GPa AND Pred Shear >= 100 GPa (or Vickers Hardness >= 25 GPa)
   - Ultrasoft: Pred Bulk <= 25 GPa AND Pred Shear <= 15 GPa
   - Intermediate: other materials
5. Save:
   - ultrahard_materials.csv
   - ultrasoft_materials.csv
   - all_materials_classified.csv
6. Generate comparison analysis and publication plots:
   - Moduli space map (K vs G) highlighting ultrahard and ultrasoft clusters
   - Parity plots for both categories
"""
import os
import sys
import csv
import time
import math
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pymatgen.core import Structure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Server path setup
sys.path.insert(0, '/home/ashwinay23/UGP-Phy')
from equivariant_model import SimpleEquivariantElasticNet
from tensor_utils import calculate_moduli, check_stability

def estimate_vickers_hardness(K, G):
    """Chen's empirical model for Vickers hardness: Hv = 2 * (k^2 * G)^0.585 - 3"""
    if K <= 0 or G <= 0:
        return 0.0
    k = G / K
    val = 2.0 * ((k**2) * G)**0.585 - 3.0
    return max(0.0, float(val))

def calculate_poisson_ratio(K, G):
    """nu = (3K - 2G) / (2 * (3K + G))"""
    denom = 2.0 * (3.0 * K + G)
    if denom <= 0:
        return 0.0
    val = (3.0 * K - 2.0 * G) / denom
    return float(np.clip(val, -1.0, 0.5))

class FullEquivariantDataset(Dataset):
    def __init__(self, data_dir, max_radius=6.0):
        self.data_dir = data_dir
        self.max_radius = max_radius
        
        tensor_json = os.path.join(data_dir, 'elastic_tensors.json')
        with open(tensor_json, 'r') as f:
            self.raw_tensors = json.load(f)
            
        print(f"Loaded {len(self.raw_tensors)} tensors from {tensor_json}")
        
        # Scan CIFs in directory
        all_cifs = [f for f in os.listdir(data_dir) if f.endswith('.cif')]
        self.cif_ids = []
        for f in all_cifs:
            mid = f[:-4]
            if mid in self.raw_tensors and self.raw_tensors[mid] is not None:
                self.cif_ids.append(mid)
                
        print(f"Matched {len(self.cif_ids)} structures with valid tensors.")

    def __len__(self):
        return len(self.cif_ids)

    def __getitem__(self, idx):
        mat_id = self.cif_ids[idx]
        cif_path = os.path.join(self.data_dir, f"{mat_id}.cif")
        structure = Structure.from_file(cif_path)
        
        formula = structure.composition.reduced_formula
        num_atoms = len(structure)
        
        # Extract features
        atomic_numbers = []
        rich_features = []
        for site in structure:
            sp = site.specie
            num = min(sp.number, 99)
            atomic_numbers.append(num)
            
            X = sp.X if sp.X is not None and not math.isnan(sp.X) else 0.0
            radius = sp.atomic_radius if sp.atomic_radius is not None and not (isinstance(sp.atomic_radius, float) and math.isnan(sp.atomic_radius)) else 0.0
            mass = sp.atomic_mass if sp.atomic_mass is not None and not (isinstance(sp.atomic_mass, float) and math.isnan(sp.atomic_mass)) else 0.0
            group = sp.group if sp.group is not None else 0.0
            row = sp.row if sp.row is not None else 0.0
            
            rich_features.append([
                X / 4.0, radius / 3.0, mass / 200.0, group / 18.0, row / 7.0
            ])
            
        atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.long)
        one_hot = torch.nn.functional.one_hot(atomic_numbers, num_classes=100).to(torch.float32)
        rich_feat = torch.tensor(rich_features, dtype=torch.float32)
        node_features = torch.cat([one_hot, rich_feat], dim=-1)
        
        src_indices, dst_indices, edge_vectors = [], [], []
        for i in range(num_atoms):
            neighbors = structure.get_neighbors(structure[i], self.max_radius)
            for neighbor in neighbors:
                src_indices.append(i)
                dst_indices.append(neighbor.index)
                edge_vectors.append(neighbor.coords - structure[i].coords)
                
        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        edge_vec = torch.tensor(np.array(edge_vectors), dtype=torch.float32)
        target_tensor = torch.tensor(np.array(self.raw_tensors[mat_id]), dtype=torch.float32)
        
        return {
            'mat_id': mat_id,
            'formula': formula,
            'num_atoms': num_atoms,
            'node_features': node_features,
            'edge_vec': edge_vec,
            'edge_index': edge_index,
            'target_tensor': target_tensor
        }


def main():
    base_dir = '/home/ashwinay23/UGP-Phy'
    device = torch.device('cpu')
    
    print("="*75)
    print("ULTRAHARD & ULTRASOFT MATERIALS CLASSIFICATION PIPELINE")
    print("="*75)
    
    # 1. Load Model
    ckpt_path = os.path.join(base_dir, 'checkpoints_equivariant', 'model_best_equivariant.pth.tar')
    stats_path = os.path.join(base_dir, 'checkpoints_equivariant', 'normalization_stats.pt')
    
    stats = torch.load(stats_path, map_location=device, weights_only=False)
    target_mean_np = stats['mean'].cpu().numpy()
    target_std_np = stats['std'].cpu().numpy()
    
    model = SimpleEquivariantElasticNet(num_node_features=105).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path} (epoch {checkpoint.get('epoch', '?')})")
    
    # 2. Dataset & DataLoader
    data_dir = os.path.join(base_dir, 'dataset_equivariant')
    dataset = FullEquivariantDataset(data_dir=data_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    results = []
    start_time = time.time()
    
    print(f"\nRunning inference on {len(dataset)} materials with 4 DataLoader workers...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if (i + 1) % 2500 == 0 or (i + 1) == len(dataset):
                elapsed = time.time() - start_time
                rate = (i + 1) / max(elapsed, 0.001)
                eta = (len(dataset) - i - 1) / rate
                print(f"  [{i+1}/{len(dataset)}] {rate:.1f} mat/s | ETA: {eta:.0f}s")
                
            mat_id = batch['mat_id'][0]
            formula = batch['formula'][0]
            num_atoms = int(batch['num_atoms'][0])
            node_features = batch['node_features'][0].to(device)
            edge_vec = batch['edge_vec'][0].to(device)
            edge_index = batch['edge_index'][0].to(device)
            true_C = batch['target_tensor'][0].cpu().numpy()
            
            try:
                pred_std = model(node_features, edge_vec, edge_index).cpu().numpy()
                pred_C = pred_std * target_std_np + target_mean_np
                
                # Robust moduli calculation
                true_mod = calculate_moduli(true_C)
                pred_mod = calculate_moduli(pred_C)
                
                true_K = true_mod['K_H']
                pred_K = pred_mod['K_H']
                true_G = true_mod['G_H']
                pred_G = pred_mod['G_H']
                
                pred_pugh = pred_G / max(pred_K, 1e-4)
                true_pugh = true_G / max(true_K, 1e-4)
                pred_poisson = calculate_poisson_ratio(pred_K, pred_G)
                pred_hv = estimate_vickers_hardness(pred_K, pred_G)
                true_hv = estimate_vickers_hardness(true_K, true_G)
                
                source = 'JARVIS' if mat_id.startswith('JVASP') else 'MaterialsProject'
                
                # Classification Criteria
                # Ultrahard: High bulk resistance (K >= 200 GPa) AND high shear resistance (G >= 100 GPa)
                #            OR estimated Vickers hardness >= 25 GPa
                # Ultrasoft: Low bulk stiffness (K <= 25 GPa) AND low shear stiffness (G <= 15 GPa)
                if (pred_K >= 200.0 and pred_G >= 100.0) or pred_hv >= 25.0:
                    category = 'ultrahard'
                elif pred_K <= 25.0 and pred_G <= 15.0:
                    category = 'ultrasoft'
                else:
                    category = 'intermediate'
                    
                results.append({
                    'mat_id': mat_id,
                    'formula': formula,
                    'source': source,
                    'num_atoms': num_atoms,
                    'pred_bulk_modulus': round(pred_K, 3),
                    'true_bulk_modulus': round(true_K, 3),
                    'bulk_error': round(abs(pred_K - true_K), 3),
                    'pred_shear_modulus': round(pred_G, 3),
                    'true_shear_modulus': round(true_G, 3),
                    'shear_error': round(abs(pred_G - true_G), 3),
                    'pred_pugh_ratio': round(pred_pugh, 3),
                    'pred_poisson_ratio': round(pred_poisson, 3),
                    'pred_vickers_hardness': round(pred_hv, 2),
                    'true_vickers_hardness': round(true_hv, 2),
                    'pred_born_stable': pred_mod['is_born_stable'],
                    'true_born_stable': true_mod['is_born_stable'],
                    'category': category
                })
            except Exception as e:
                pass
                
    elapsed = time.time() - start_time
    print(f"\nInference completed in {elapsed:.1f}s ({len(results)} valid structures).")
    
    # 3. Separate into Subsets
    ultrahard = [r for r in results if r['category'] == 'ultrahard']
    ultrasoft = [r for r in results if r['category'] == 'ultrasoft']
    intermediate = [r for r in results if r['category'] == 'intermediate']
    
    print(f"\nDataset Breakdown:")
    print(f"  * Ultrahard materials: {len(ultrahard):,} ({100*len(ultrahard)/len(results):.2f}%)")
    print(f"  * Ultrasoft materials: {len(ultrasoft):,} ({100*len(ultrasoft)/len(results):.2f}%)")
    print(f"  * Intermediate:        {len(intermediate):,} ({100*len(intermediate)/len(results):.2f}%)")
    
    # 4. Save CSV files
    fieldnames = [
        'mat_id', 'formula', 'source', 'num_atoms',
        'pred_bulk_modulus', 'true_bulk_modulus', 'bulk_error',
        'pred_shear_modulus', 'true_shear_modulus', 'shear_error',
        'pred_pugh_ratio', 'pred_poisson_ratio',
        'pred_vickers_hardness', 'true_vickers_hardness',
        'pred_born_stable', 'true_born_stable', 'category'
    ]
    
    # Sort ultrahard by hardness/bulk modulus descending
    ultrahard_sorted = sorted(ultrahard, key=lambda x: (x['pred_vickers_hardness'], x['pred_bulk_modulus']), reverse=True)
    # Sort ultrasoft by bulk modulus ascending
    ultrasoft_sorted = sorted(ultrasoft, key=lambda x: (x['pred_bulk_modulus'], x['pred_shear_modulus']))
    
    hard_path = os.path.join(base_dir, 'ultrahard_materials.csv')
    soft_path = os.path.join(base_dir, 'ultrasoft_materials.csv')
    all_path  = os.path.join(base_dir, 'all_materials_classified.csv')
    
    with open(hard_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ultrahard_sorted)
    print(f"Saved: {hard_path}")
    
    with open(soft_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ultrasoft_sorted)
    print(f"Saved: {soft_path}")
    
    with open(all_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved: {all_path}")
    
    # 5. Comparative Evaluation
    print("\n" + "="*70)
    print("COMPARATIVE EVALUATION: ULTRAHARD VS ULTRASOFT")
    print("="*70)
    
    def print_stats(name, data_subset):
        k_pred = [d['pred_bulk_modulus'] for d in data_subset]
        k_true = [d['true_bulk_modulus'] for d in data_subset]
        g_pred = [d['pred_shear_modulus'] for d in data_subset]
        g_true = [d['true_shear_modulus'] for d in data_subset]
        hv     = [d['pred_vickers_hardness'] for d in data_subset]
        
        k_mae = np.mean(np.abs(np.array(k_pred) - np.array(k_true)))
        g_mae = np.mean(np.abs(np.array(g_pred) - np.array(g_true)))
        
        print(f"\n[{name.upper()}] (N = {len(data_subset):,})")
        print(f"  Mean Pred Bulk Modulus:   {np.mean(k_pred):.2f} GPa (range: {np.min(k_pred):.1f} - {np.max(k_pred):.1f} GPa)")
        print(f"  Mean Pred Shear Modulus:  {np.mean(g_pred):.2f} GPa (range: {np.min(g_pred):.1f} - {np.max(g_pred):.1f} GPa)")
        print(f"  Mean Vickers Hardness:    {np.mean(hv):.2f} GPa (max: {np.max(hv):.1f} GPa)")
        print(f"  Bulk Modulus MAE:         {k_mae:.2f} GPa")
        print(f"  Shear Modulus MAE:        {g_mae:.2f} GPa")
        
        # Top 5 formulas
        top_examples = [f"{d['formula']} (K={d['pred_bulk_modulus']:.0f}, G={d['pred_shear_modulus']:.0f}, Hv={d['pred_vickers_hardness']:.1f})" for d in data_subset[:5]]
        print(f"  Notable Examples:")
        for ex in top_examples:
            print(f"    - {ex}")
            
    print_stats("Ultrahard Materials", ultrahard_sorted)
    print_stats("Ultrasoft Materials", ultrasoft_sorted)
    
    # 6. Generate Publication Visualization Plots
    plot_dir = os.path.join(base_dir, 'Plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot 1: Moduli Phase Space Map (K vs G)
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Background intermediate
    ax.scatter([d['pred_bulk_modulus'] for d in intermediate],
               [d['pred_shear_modulus'] for d in intermediate],
               c='#bdbdbd', alpha=0.25, s=8, label=f"Standard / Intermediate (N={len(intermediate):,})")
               
    # Ultrasoft
    ax.scatter([d['pred_bulk_modulus'] for d in ultrasoft],
               [d['pred_shear_modulus'] for d in ultrasoft],
               c='#2b83ba', alpha=0.6, s=16, label=f"Ultrasoft (N={len(ultrasoft):,}, K≤25, G≤15 GPa)")
               
    # Ultrahard
    ax.scatter([d['pred_bulk_modulus'] for d in ultrahard],
               [d['pred_shear_modulus'] for d in ultrahard],
               c='#d7191c', alpha=0.7, s=22, label=f"Ultrahard (N={len(ultrahard):,}, K≥200, G≥100 or Hv≥25 GPa)")
               
    # Pugh line k = 0.57 (brittle-ductile transition)
    k_line = np.linspace(0, 450, 100)
    ax.plot(k_line, 0.57 * k_line, 'k--', alpha=0.5, label="Pugh's Critical Ratio (G/K = 0.57)")
    
    # Annotate famous ultrahard
    for d in ultrahard_sorted[:4]:
        ax.annotate(d['formula'], (d['pred_bulk_modulus'], d['pred_shear_modulus']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold', color='#990000')
                    
    ax.set_xlim(0, 420)
    ax.set_ylim(0, 320)
    ax.set_xlabel('Predicted Bulk Modulus $K_H$ (GPa)', fontsize=12)
    ax.set_ylabel('Predicted Shear Modulus $G_H$ (GPa)', fontsize=12)
    ax.set_title('Classification Map: Ultrahard vs. Ultrasoft Crystals (MP + JARVIS)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.35)
    
    plt.tight_layout()
    map_path = os.path.join(plot_dir, 'classification_moduli_map.png')
    plt.savefig(map_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {map_path}")
    
    # Plot 2: Parity plots for Ultrahard and Ultrasoft
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    
    # Ultrahard parity
    ax = axes[0]
    u_true_K = [d['true_bulk_modulus'] for d in ultrahard if d['true_bulk_modulus'] > 0]
    u_pred_K = [d['pred_bulk_modulus'] for d in ultrahard if d['true_bulk_modulus'] > 0]
    ax.scatter(u_true_K, u_pred_K, c='#d7191c', alpha=0.5, s=18, edgecolors='none')
    lim_h = [150, 420]
    ax.plot(lim_h, lim_h, 'k--', lw=1.5, label='Ideal ($y=x$)')
    ax.set_xlim(lim_h)
    ax.set_ylim(lim_h)
    u_mae = np.mean(np.abs(np.array(u_pred_K) - np.array(u_true_K)))
    ax.set_title(f'Ultrahard Bulk Modulus Parity\nMAE = {u_mae:.2f} GPa (N={len(u_true_K):,})', fontsize=12, fontweight='bold')
    ax.set_xlabel('True Bulk Modulus (GPa)', fontsize=11)
    ax.set_ylabel('Predicted Bulk Modulus (GPa)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.35)
    
    # Ultrasoft parity
    ax = axes[1]
    s_true_K = [d['true_bulk_modulus'] for d in ultrasoft if d['true_bulk_modulus'] > 0]
    s_pred_K = [d['pred_bulk_modulus'] for d in ultrasoft if d['true_bulk_modulus'] > 0]
    ax.scatter(s_true_K, s_pred_K, c='#2b83ba', alpha=0.5, s=18, edgecolors='none')
    lim_s = [0, 40]
    ax.plot(lim_s, lim_s, 'k--', lw=1.5, label='Ideal ($y=x$)')
    ax.set_xlim(lim_s)
    ax.set_ylim(lim_s)
    s_mae = np.mean(np.abs(np.array(s_pred_K) - np.array(s_true_K)))
    ax.set_title(f'Ultrasoft Bulk Modulus Parity\nMAE = {s_mae:.2f} GPa (N={len(s_true_K):,})', fontsize=12, fontweight='bold')
    ax.set_xlabel('True Bulk Modulus (GPa)', fontsize=11)
    ax.set_ylabel('Predicted Bulk Modulus (GPa)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.35)
    
    plt.tight_layout()
    parity_path = os.path.join(plot_dir, 'classification_parity_subsets.png')
    plt.savefig(parity_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {parity_path}")
    
    print("\nAll classifications, files, and plots completed successfully!")

if __name__ == '__main__':
    main()
