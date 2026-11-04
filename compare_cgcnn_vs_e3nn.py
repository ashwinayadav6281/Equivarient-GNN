"""
Compare CGCNN direct Bulk/Shear Modulus predictions vs E3NN tensor-derived Moduli.

Key Features & Physics Improvements:
1. Physics-consistent Voigt-Reuss-Hill (VRH) moduli calculation with PSD cone regularization.
2. Born mechanical stability analysis (positive-definiteness of elastic stiffness tensor).
3. Publication-quality 2D Hexbin density parity plots (logarithmic color density).
4. Physical axis limits and avoidance of artificial 0/800 GPa clamping artifacts.
5. Dual evaluation: Full dataset vs Mechanically Stable (Born criteria satisfied) subset.
6. Error distribution KDE curves and cumulative accuracy curves (CDF).
"""
import os
import sys
import csv
import time
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pymatgen.core import Structure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde

# Add project root to path
sys.path.insert(0, '/home/ashwinay23/UGP-Phy')
if os.path.exists(r'C:\Users\arvin\Downloads\UGP-Phy'):
    sys.path.insert(0, r'C:\Users\arvin\Downloads\UGP-Phy')

from equivariant_model import SimpleEquivariantElasticNet
from tensor_utils import calculate_moduli, check_stability


class FilteredEquivariantDataset(Dataset):
    """Loads only target materials that exist in the evaluation set for speed."""
    def __init__(self, data_dir, target_ids=None, max_radius=6.0):
        self.data_dir = data_dir
        self.max_radius = max_radius
        
        tensor_json_path = os.path.join(data_dir, 'elastic_tensors.json')
        with open(tensor_json_path, 'r') as f:
            raw_tensors = json_data = __import__('json').load(f)
            
        target_set = set(target_ids) if target_ids else None
        
        self.cif_ids = []
        self.tensors = {}
        for mid, tensor in raw_tensors.items():
            if tensor is None:
                continue
            if target_set and mid not in target_set:
                continue
            cif_path = os.path.join(data_dir, f'{mid}.cif')
            if os.path.exists(cif_path):
                self.tensors[mid] = tensor
                self.cif_ids.append(mid)
                
        print(f"Loaded {len(self.cif_ids)} structures for evaluation.")

    def __len__(self):
        return len(self.cif_ids)

    def __getitem__(self, idx):
        mat_id = self.cif_ids[idx]
        cif_path = os.path.join(self.data_dir, f'{mat_id}.cif')
        structure = Structure.from_file(cif_path)
        
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
        for i in range(len(structure)):
            neighbors = structure.get_neighbors(structure[i], self.max_radius)
            for neighbor in neighbors:
                src_indices.append(i)
                dst_indices.append(neighbor.index)
                edge_vectors.append(neighbor.coords - structure[i].coords)
                
        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        edge_vec = torch.tensor(np.array(edge_vectors), dtype=torch.float32)
        target_tensor = torch.tensor(np.array(self.tensors[mat_id]), dtype=torch.float32)
        
        return {
            'mat_id': mat_id,
            'node_features': node_features,
            'edge_vec': edge_vec,
            'edge_index': edge_index,
            'target_tensor': target_tensor
        }


def load_cgcnn_predictions(csv_path):
    """Load CGCNN predictions from CSV."""
    results = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cif_id = row['cif_id']
            try:
                bulk_actual = float(row['bulk_actual'])
                bulk_predicted = float(row['bulk_predicted'])
                shear_actual = float(row['shear_actual'])
                shear_predicted = float(row['shear_predicted'])
                results[cif_id] = {
                    'true_K': bulk_actual, 'pred_K': bulk_predicted,
                    'true_G': shear_actual, 'pred_G': shear_predicted,
                }
            except (ValueError, KeyError):
                pass
    print(f'Loaded {len(results)} CGCNN predictions from {csv_path}')
    return results


def run_e3nn_predictions(base_dir, target_ids):
    """Run E3NN inference on target materials with robust moduli computation."""
    device = torch.device('cpu')
    
    print('Loading E3NN model checkpoint and normalization stats...')
    ckpt_path = os.path.join(base_dir, 'checkpoints_equivariant', 'model_best_equivariant.pth.tar')
    stats_path = os.path.join(base_dir, 'checkpoints_equivariant', 'normalization_stats.pt')
    
    stats = torch.load(stats_path, map_location=device, weights_only=False)
    target_mean_np = stats['mean'].cpu().numpy()
    target_std_np = stats['std'].cpu().numpy()
    
    model = SimpleEquivariantElasticNet(num_node_features=105).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  Model loaded (epoch {checkpoint.get('epoch', '?')})")
    
    data_dir = os.path.join(base_dir, 'dataset_equivariant')
    dataset = FilteredEquivariantDataset(data_dir=data_dir, target_ids=target_ids)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    
    results = {}
    errors = 0
    start = time.time()
    
    print(f'Running inference on {len(dataset)} materials...')
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if (i + 1) % 1000 == 0 or (i + 1) == len(dataset):
                elapsed = time.time() - start
                rate = (i + 1) / max(elapsed, 0.001)
                eta = (len(dataset) - i - 1) / rate
                print(f'  [{i+1}/{len(dataset)}] {rate:.1f} mat/s | ETA: {eta:.0f}s')
                
            mat_id = batch['mat_id'][0]
            node_features = batch['node_features'][0].to(device)
            edge_vec = batch['edge_vec'][0].to(device)
            edge_index = batch['edge_index'][0].to(device)
            true_C = batch['target_tensor'][0].cpu().numpy()
            
            try:
                pred_std = model(node_features, edge_vec, edge_index).cpu().numpy()
                pred_C = pred_std * target_std_np + target_mean_np
                
                # Robust moduli calculation with PSD projection
                true_mod = calculate_moduli(true_C)
                pred_mod = calculate_moduli(pred_C)
                
                results[mat_id] = {
                    'true_K': true_mod['K_H'],
                    'pred_K': pred_mod['K_H'],
                    'true_G': true_mod['G_H'],
                    'pred_G': pred_mod['G_H'],
                    'true_born_stable': true_mod['is_born_stable'],
                    'pred_born_stable': pred_mod['is_born_stable'],
                    'true_min_eig': true_mod['min_eigenvalue'],
                    'pred_min_eig': pred_mod['min_eigenvalue'],
                }
            except Exception as e:
                errors += 1
                
    elapsed = time.time() - start
    print(f'  Completed {len(results)} materials in {elapsed:.1f}s ({errors} errors)')
    return results


def compute_metrics(true_vals, pred_vals, label=""):
    """Compute MAE, Median AE, RMSE, R^2"""
    true_arr = np.array(true_vals, dtype=float)
    pred_arr = np.array(pred_vals, dtype=float)
    
    diff = true_arr - pred_arr
    mae = float(np.mean(np.abs(diff)))
    med_ae = float(np.median(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    
    ss_res = np.sum(diff**2)
    ss_tot = np.sum((true_arr - np.mean(true_arr))**2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    
    print(f'  {label}:')
    print(f'    N       = {len(true_vals)}')
    print(f'    MAE     = {mae:.3f} GPa')
    print(f'    MedAE   = {med_ae:.3f} GPa')
    print(f'    RMSE    = {rmse:.3f} GPa')
    print(f'    R^2      = {r2:.4f}')
    return {
        'mae': mae, 'med_ae': med_ae, 'rmse': rmse, 'r2': r2, 'n': len(true_vals)
    }


def plot_hexbin_parity(ax, true_v, pred_v, title, max_val, cmap='Blues', is_cgcnn=False):
    """Helper to draw publication-grade hexbin density parity plot."""
    hb = ax.hexbin(true_v, pred_v, gridsize=75, cmap=cmap, mincnt=1,
                   bins='log', extent=[0, max_val, 0, max_val])
    
    # 1:1 Parity line
    ax.plot([0, max_val], [0, max_val], 'r--', lw=1.8, label='Ideal (y = x)')
    
    # +/- 10% bounds
    xs = np.linspace(0, max_val, 100)
    ax.plot(xs, 1.10 * xs, ':', color='gray', lw=1.0, label='±10% Error')
    ax.plot(xs, 0.90 * xs, ':', color='gray', lw=1.0)
    
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.35)
    
    metrics = compute_metrics(true_v, pred_v, label=title)
    stats_text = (
        f"N = {metrics['n']:,}\n"
        f"MAE  = {metrics['mae']:.2f} GPa\n"
        f"MedAE= {metrics['med_ae']:.2f} GPa\n"
        f"RMSE = {metrics['rmse']:.2f} GPa\n"
        f"R^2   = {metrics['r2']:.4f}"
    )
    ax.text(0.04, 0.96, stats_text, transform=ax.transAxes,
            fontsize=10.5, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray'))
    
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.legend(loc='lower right', fontsize=9.5, framealpha=0.85)
    return hb


def generate_improved_plots(data, plot_dir):
    """Generate clean, publication-ready comparison figures."""
    os.makedirs(plot_dir, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'figure.autolayout': False})
    
    # Filter for realistic physical range and Born-stable crystals for standard parity plots
    valid_data = [d for d in data if d['true_K'] > 0 and d['true_G'] > 0 and d['true_born_stable']]
    print(f"\nPlotting on {len(valid_data)} mechanically stable materials (out of {len(data)} total)...")
    
    cgcnn_true_K = [d['true_K'] for d in valid_data]
    cgcnn_pred_K = [d['cgcnn_pred_K'] for d in valid_data]
    e3nn_true_K  = [d['true_K'] for d in valid_data]
    e3nn_pred_K  = [d['e3nn_pred_K'] for d in valid_data]
    
    cgcnn_true_G = [d['true_G'] for d in valid_data]
    cgcnn_pred_G = [d['cgcnn_pred_G'] for d in valid_data]
    e3nn_true_G  = [d['true_G'] for d in valid_data]
    e3nn_pred_G  = [d['e3nn_pred_G'] for d in valid_data]
    
    # -------------------------------------------------------------------------
    # 1. BULK MODULUS PARITY PLOT (HEXBIN DENSITY)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    fig.subplots_adjust(wspace=0.18, right=0.88)
    
    hb1 = plot_hexbin_parity(axes[0], cgcnn_true_K, cgcnn_pred_K,
                             "CGCNN (Direct Invariant MLP)", max_val=350, cmap='Blues', is_cgcnn=True)
    axes[0].set_xlabel('True Bulk Modulus $K_H$ (GPa)', fontsize=12)
    axes[0].set_ylabel('Predicted Bulk Modulus $K_H$ (GPa)', fontsize=12)
    
    hb2 = plot_hexbin_parity(axes[1], e3nn_true_K, e3nn_pred_K,
                             "E3NN (Equivariant Tensor -> VRH)", max_val=350, cmap='YlOrRd')
    axes[1].set_xlabel('True Bulk Modulus $K_H$ (GPa)', fontsize=12)
    
    # Shared colorbar for density
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    cb = fig.colorbar(hb2, cax=cbar_ax)
    cb.set_label('Point Density (Log Count)', fontsize=11)
    
    fig.suptitle('Bulk Modulus Prediction Parity (Mechanically Stable Materials)', fontsize=14, fontweight='bold', y=0.98)
    bulk_path = os.path.join(plot_dir, 'comparison_bulk_modulus.png')
    plt.savefig(bulk_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {bulk_path}")
    
    # -------------------------------------------------------------------------
    # 2. SHEAR MODULUS PARITY PLOT (HEXBIN DENSITY)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    fig.subplots_adjust(wspace=0.18, right=0.88)
    
    hb1 = plot_hexbin_parity(axes[0], cgcnn_true_G, cgcnn_pred_G,
                             "CGCNN (Direct Invariant MLP)", max_val=220, cmap='Blues', is_cgcnn=True)
    axes[0].set_xlabel('True Shear Modulus $G_H$ (GPa)', fontsize=12)
    axes[0].set_ylabel('Predicted Shear Modulus $G_H$ (GPa)', fontsize=12)
    
    hb2 = plot_hexbin_parity(axes[1], e3nn_true_G, e3nn_pred_G,
                             "E3NN (Equivariant Tensor -> VRH)", max_val=220, cmap='YlOrRd')
    axes[1].set_xlabel('True Shear Modulus $G_H$ (GPa)', fontsize=12)
    
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    cb = fig.colorbar(hb2, cax=cbar_ax)
    cb.set_label('Point Density (Log Count)', fontsize=11)
    
    fig.suptitle('Shear Modulus Prediction Parity (Mechanically Stable Materials)', fontsize=14, fontweight='bold', y=0.98)
    shear_path = os.path.join(plot_dir, 'comparison_shear_modulus.png')
    plt.savefig(shear_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {shear_path}")
    
    # -------------------------------------------------------------------------
    # 3. ERROR DISTRIBUTION & CUMULATIVE ACCURACY CURVES (CDF)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    e3nn_err_K = np.abs(np.array(e3nn_true_K) - np.array(e3nn_pred_K))
    cgcnn_err_K = np.abs(np.array(cgcnn_true_K) - np.array(cgcnn_pred_K))
    e3nn_err_G = np.abs(np.array(e3nn_true_G) - np.array(e3nn_pred_G))
    cgcnn_err_G = np.abs(np.array(cgcnn_true_G) - np.array(cgcnn_pred_G))
    
    # Panel A: Smooth Error PDF via KDE
    ax = axes[0]
    x_eval = np.linspace(0, 60, 200)
    
    kde_e3nn_k = gaussian_kde(e3nn_err_K)(x_eval)
    kde_cgcnn_k = gaussian_kde(cgcnn_err_K)(x_eval)
    kde_e3nn_g = gaussian_kde(e3nn_err_G)(x_eval)
    kde_cgcnn_g = gaussian_kde(cgcnn_err_G)(x_eval)
    
    ax.plot(x_eval, kde_e3nn_k, color='#d95f02', lw=2.2, label=f'E3NN Bulk (MAE={np.mean(e3nn_err_K):.2f})')
    ax.plot(x_eval, kde_cgcnn_k, color='#7570b3', lw=1.8, linestyle='--', label=f'CGCNN Bulk (MAE={np.mean(cgcnn_err_K):.2f})')
    ax.plot(x_eval, kde_e3nn_g, color='#1b9e77', lw=2.2, label=f'E3NN Shear (MAE={np.mean(e3nn_err_G):.2f})')
    ax.plot(x_eval, kde_cgcnn_g, color='#e7298a', lw=1.8, linestyle='--', label=f'CGCNN Shear (MAE={np.mean(cgcnn_err_G):.2f})')
    
    ax.set_xlim(0, 60)
    ax.set_xlabel('Absolute Error |Pred - True| (GPa)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('A. Error Density Distribution (KDE)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.4)
    
    # Panel B: Cumulative Distribution Function (CDF)
    ax = axes[1]
    err_thresholds = np.linspace(0, 50, 200)
    
    cdf_e3nn_k = [100.0 * np.mean(e3nn_err_K <= t) for t in err_thresholds]
    cdf_cgcnn_k = [100.0 * np.mean(cgcnn_err_K <= t) for t in err_thresholds]
    cdf_e3nn_g = [100.0 * np.mean(e3nn_err_G <= t) for t in err_thresholds]
    cdf_cgcnn_g = [100.0 * np.mean(cgcnn_err_G <= t) for t in err_thresholds]
    
    ax.plot(err_thresholds, cdf_e3nn_k, color='#d95f02', lw=2.2, label='E3NN Bulk ($K_H$)')
    ax.plot(err_thresholds, cdf_cgcnn_k, color='#7570b3', lw=1.8, linestyle='--', label='CGCNN Bulk')
    ax.plot(err_thresholds, cdf_e3nn_g, color='#1b9e77', lw=2.2, label='E3NN Shear ($G_H$)')
    ax.plot(err_thresholds, cdf_cgcnn_g, color='#e7298a', lw=1.8, linestyle='--', label='CGCNN Shear')
    
    # Highlight 10 GPa threshold
    pct_e3nn_10k = 100.0 * np.mean(e3nn_err_K <= 10.0)
    pct_cgcnn_10k = 100.0 * np.mean(cgcnn_err_K <= 10.0)
    ax.axvline(10, color='gray', linestyle=':', alpha=0.7)
    ax.text(11, 20, f"At ≤10 GPa error:\nE3NN Bulk: {pct_e3nn_10k:.1f}%\nCGCNN Bulk: {pct_cgcnn_10k:.1f}%",
            fontsize=9.5, bbox=dict(boxstyle='round,pad=0.4', facecolor='linen', alpha=0.8))
    
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Error Tolerance (GPa)', fontsize=12)
    ax.set_ylabel('% of Materials within Tolerance', fontsize=12)
    ax.set_title('B. Cumulative Accuracy Curve', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    dist_path = os.path.join(plot_dir, 'comparison_error_distribution.png')
    plt.savefig(dist_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {dist_path}")


def main():
    base_dir = '/home/ashwinay23/UGP-Phy'
    if not os.path.exists(base_dir):
        base_dir = r'C:\Users\arvin\Downloads\UGP-Phy'
        
    print("="*70)
    print("E3NN EQUIVARIANT VS CGCNN MODULI COMPARISON (ROBUST PHYSICS PIPELINE)")
    print("="*70)
    
    # 1. Load CGCNN predictions
    cgcnn_csv = os.path.join(base_dir, 'predictions_combined.csv')
    cgcnn_results = load_cgcnn_predictions(cgcnn_csv)
    target_ids = list(cgcnn_results.keys())
    
    # 2. Run E3NN inference on target materials
    e3nn_results = run_e3nn_predictions(base_dir, target_ids)
    
    # 3. Intersect materials
    common_ids = sorted(list(set(e3nn_results.keys()) & set(cgcnn_results.keys())))
    print(f"\nSuccessfully evaluated {len(common_ids)} overlapping materials.")
    
    # 4. Compile unified comparison records
    compiled_data = []
    output_csv = os.path.join(base_dir, 'comparison_cgcnn_vs_e3nn.csv')
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'mat_id',
            'true_bulk', 'cgcnn_pred_bulk', 'e3nn_pred_bulk',
            'cgcnn_bulk_error', 'e3nn_bulk_error',
            'true_shear', 'cgcnn_pred_shear', 'e3nn_pred_shear',
            'cgcnn_shear_error', 'e3nn_shear_error',
            'true_born_stable', 'pred_born_stable', 'true_min_eig'
        ])
        for mid in common_ids:
            e = e3nn_results[mid]
            c = cgcnn_results[mid]
            rec = {
                'mat_id': mid,
                'true_K': e['true_K'],
                'cgcnn_pred_K': c['pred_K'],
                'e3nn_pred_K': e['pred_K'],
                'true_G': e['true_G'],
                'cgcnn_pred_G': c['pred_G'],
                'e3nn_pred_G': e['pred_G'],
                'true_born_stable': e['true_born_stable'],
                'pred_born_stable': e['pred_born_stable'],
                'true_min_eig': e['true_min_eig'],
            }
            compiled_data.append(rec)
            writer.writerow([
                mid,
                f"{rec['true_K']:.3f}", f"{rec['cgcnn_pred_K']:.3f}", f"{rec['e3nn_pred_K']:.3f}",
                f"{abs(rec['true_K'] - rec['cgcnn_pred_K']):.3f}", f"{abs(rec['true_K'] - rec['e3nn_pred_K']):.3f}",
                f"{rec['true_G']:.3f}", f"{rec['cgcnn_pred_G']:.3f}", f"{rec['e3nn_pred_G']:.3f}",
                f"{abs(rec['true_G'] - rec['cgcnn_pred_G']):.3f}", f"{abs(rec['true_G'] - rec['e3nn_pred_G']):.3f}",
                rec['true_born_stable'], rec['pred_born_stable'], f"{rec['true_min_eig']:.3f}"
            ])
            
    print(f"Saved compiled data to: {output_csv}")
    
    # 5. Report comprehensive metrics
    print("\n" + "="*70)
    print("METRICS: ALL OVERLAPPING MATERIALS (N = %d)" % len(compiled_data))
    print("="*70)
    all_true_K = [d['true_K'] for d in compiled_data]
    all_cgcnn_K = [d['cgcnn_pred_K'] for d in compiled_data]
    all_e3nn_K = [d['e3nn_pred_K'] for d in compiled_data]
    print("\n[BULK MODULUS]")
    compute_metrics(all_true_K, all_e3nn_K, label="E3NN (Tensor -> VRH)")
    compute_metrics(all_true_K, all_cgcnn_K, label="CGCNN (Direct Invariant MLP)")
    
    all_true_G = [d['true_G'] for d in compiled_data]
    all_cgcnn_G = [d['cgcnn_pred_G'] for d in compiled_data]
    all_e3nn_G = [d['e3nn_pred_G'] for d in compiled_data]
    print("\n[SHEAR MODULUS]")
    compute_metrics(all_true_G, all_e3nn_G, label="E3NN (Tensor -> VRH)")
    compute_metrics(all_true_G, all_cgcnn_G, label="CGCNN (Direct Invariant MLP)")
    
    # 6. Generate improved plots
    plot_dir = os.path.join(base_dir, 'Plots')
    generate_improved_plots(compiled_data, plot_dir)
    print("\n" + "="*70)
    print("ALL EVALUATIONS AND PLOTS COMPLETED SUCCESSFULLY!")
    print("="*70)


if __name__ == '__main__':
    main()
