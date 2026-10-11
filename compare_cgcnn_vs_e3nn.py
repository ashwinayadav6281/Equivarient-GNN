"""
Compare CGCNN direct Bulk Modulus predictions vs E3NN tensor-derived Bulk Modulus.

Pipeline:
1. Load the E3NN model and run inference on the entire equivariant dataset
2. For each material, predict the full 6x6 tensor -> compute VRH bulk modulus (K_H)
3. Load CGCNN predictions from predictions_bulk.csv
4. Match materials by ID and compute MAE, R^2 for both approaches
5. Generate comparison plots
"""
import os
import sys
import csv
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, '/home/ashwinay23/UGP-Phy')

from equivariant_model import SimpleEquivariantElasticNet
from equivariant_dataset import EquivariantTensorDataset
from tensor_utils import calculate_moduli

def run_e3nn_predictions():
    """Run E3NN model on entire dataset, return dict of {mat_id: (true_K, pred_K, true_G, pred_G)}"""
    device = torch.device('cpu')
    
    # Load model
    print('Loading E3NN model (105-dim features)...')
    model = SimpleEquivariantElasticNet(num_node_features=105).to(device)
    
    ckpt_path = '/home/ashwinay23/UGP-Phy/checkpoints_equivariant/model_best_equivariant.pth.tar'
    stats_path = '/home/ashwinay23/UGP-Phy/checkpoints_equivariant/normalization_stats.pt'
    
    stats = torch.load(stats_path, map_location=device, weights_only=False)
    target_mean_np = stats['mean'].cpu().numpy()
    target_std_np = stats['std'].cpu().numpy()
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f'  Model loaded (epoch {checkpoint.get("epoch", "?")}, val_loss={checkpoint.get("val_loss", 0.0):.6f})')
    
    # Load dataset
    print('Loading dataset...')
    dataset = EquivariantTensorDataset(data_dir='/home/ashwinay23/UGP-Phy/dataset_equivariant')
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    results = {}
    errors = 0
    start = time.time()
    
    print(f'Running inference on {len(dataset)} materials...')
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if (i+1) % 2000 == 0:
                elapsed = time.time() - start
                rate = (i+1) / elapsed
                eta = (len(dataset) - i - 1) / rate
                print(f'  [{i+1}/{len(dataset)}] {rate:.1f} mat/s, ETA: {eta:.0f}s')
            
            mat_id = batch['mat_id'][0]
            node_features = batch['node_features'][0].to(device)
            edge_vec = batch['edge_vec'][0].to(device)
            edge_index = batch['edge_index'][0].to(device)
            true_C = batch['target_tensor'][0].cpu().numpy()
            
            try:
                pred_standard = model(node_features, edge_vec, edge_index).cpu().numpy()
                pred_C = pred_standard * target_std_np + target_mean_np
                
                # Enforce tensor symmetry
                pred_C = (pred_C + pred_C.T) / 2.0
                
                true_moduli = calculate_moduli(true_C)
                pred_moduli = calculate_moduli(pred_C)
                
                true_K = true_moduli.get('K_H', float('nan'))
                pred_K = pred_moduli.get('K_H', float('nan'))
                true_G = true_moduli.get('G_H', float('nan'))
                pred_G = pred_moduli.get('G_H', float('nan'))
                
                # Prevent non-physical negative values or extreme outliers from matrix inversions
                if pred_K < 0: pred_K = 0.0
                if pred_G < 0: pred_G = 0.0
                if pred_K > 800: pred_K = 800.0
                if pred_G > 800: pred_G = 800.0
                
                if not (np.isnan(true_K) or np.isnan(pred_K)):
                    results[mat_id] = {
                        'true_K': true_K, 'pred_K': pred_K,
                        'true_G': true_G, 'pred_G': pred_G,
                    }
            except Exception as e:
                errors += 1
    
    elapsed = time.time() - start
    print(f'  Done! {len(results)} valid predictions in {elapsed:.1f}s ({errors} errors)')
    return results


def load_cgcnn_predictions():
    """Load CGCNN bulk modulus predictions from predictions_combined.csv"""
    results = {}
    csv_path = '/home/ashwinay23/UGP-Phy/predictions_combined.csv'
    
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


def compute_metrics(true_vals, pred_vals, label=""):
    """Compute MAE, RMSE, R^2"""
    true_arr = np.array(true_vals)
    pred_arr = np.array(pred_vals)
    
    mae = np.mean(np.abs(true_arr - pred_arr))
    rmse = np.sqrt(np.mean((true_arr - pred_arr)**2))
    
    ss_res = np.sum((true_arr - pred_arr)**2)
    ss_tot = np.sum((true_arr - np.mean(true_arr))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    
    print(f'  {label}:')
    print(f'    MAE  = {mae:.3f} GPa')
    print(f'    RMSE = {rmse:.3f} GPa')
    print(f'    R²   = {r2:.4f}')
    print(f'    N    = {len(true_vals)}')
    return mae, rmse, r2


def main():
    # Step 1: Run E3NN predictions
    e3nn_results = run_e3nn_predictions()
    
    # Step 2: Load CGCNN predictions
    cgcnn_results = load_cgcnn_predictions()
    
    # Step 3: Find overlapping material IDs
    common_ids = set(e3nn_results.keys()) & set(cgcnn_results.keys())
    print(f'\nOverlapping materials: {len(common_ids)}')
    print(f'  E3NN-only: {len(e3nn_results) - len(common_ids)}')
    print(f'  CGCNN-only: {len(cgcnn_results) - len(common_ids)}')
    
    if len(common_ids) == 0:
        print('\nERROR: No overlapping IDs found! The datasets may use different ID schemes.')
        print('E3NN sample IDs:', list(e3nn_results.keys())[:10])
        print('CGCNN sample IDs:', list(cgcnn_results.keys())[:10])
        return
    
    # Step 4: Compute metrics on overlapping set
    # For E3NN: true bulk vs tensor-derived bulk
    e3nn_true_K = [e3nn_results[mid]['true_K'] for mid in common_ids]
    e3nn_pred_K = [e3nn_results[mid]['pred_K'] for mid in common_ids]
    
    # For CGCNN: true bulk vs directly predicted bulk
    cgcnn_true_K = [cgcnn_results[mid]['true_K'] for mid in common_ids]
    cgcnn_pred_K = [cgcnn_results[mid]['pred_K'] for mid in common_ids]
    
    print('\n' + '='*60)
    print('BULK MODULUS COMPARISON')
    print('='*60)
    compute_metrics(e3nn_true_K, e3nn_pred_K, label='E3NN (tensor → VRH K_H)')
    compute_metrics(cgcnn_true_K, cgcnn_pred_K, label='CGCNN (direct prediction)')
    
    # Same for shear
    e3nn_true_G = [e3nn_results[mid]['true_G'] for mid in common_ids]
    e3nn_pred_G = [e3nn_results[mid]['pred_G'] for mid in common_ids]
    cgcnn_true_G = [cgcnn_results[mid]['true_G'] for mid in common_ids]
    cgcnn_pred_G = [cgcnn_results[mid]['pred_G'] for mid in common_ids]
    
    print('\n' + '='*60)
    print('SHEAR MODULUS COMPARISON')
    print('='*60)
    compute_metrics(e3nn_true_G, e3nn_pred_G, label='E3NN (tensor → VRH G_H)')
    compute_metrics(cgcnn_true_G, cgcnn_pred_G, label='CGCNN (direct prediction)')
    
    # Step 5: Save comparison CSV
    output_csv = '/home/ashwinay23/UGP-Phy/comparison_cgcnn_vs_e3nn.csv'
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'mat_id',
            'true_bulk', 'cgcnn_pred_bulk', 'e3nn_pred_bulk',
            'cgcnn_bulk_error', 'e3nn_bulk_error',
            'true_shear', 'cgcnn_pred_shear', 'e3nn_pred_shear',
            'cgcnn_shear_error', 'e3nn_shear_error',
        ])
        for mid in sorted(common_ids):
            e = e3nn_results[mid]
            c = cgcnn_results[mid]
            writer.writerow([
                mid,
                f"{e['true_K']:.3f}", f"{c['pred_K']:.3f}", f"{e['pred_K']:.3f}",
                f"{abs(e['true_K'] - c['pred_K']):.3f}", f"{abs(e['true_K'] - e['pred_K']):.3f}",
                f"{e['true_G']:.3f}", f"{c['pred_G']:.3f}", f"{e['pred_G']:.3f}",
                f"{abs(e['true_G'] - c['pred_G']):.3f}", f"{abs(e['true_G'] - e['pred_G']):.3f}",
            ])
    print(f'\nComparison saved to: {output_csv}')
    
    # Step 6: Generate plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        plot_dir = '/home/ashwinay23/UGP-Phy/Plots'
        os.makedirs(plot_dir, exist_ok=True)
        
        # --- Plot 1: Parity plots side by side for Bulk Modulus ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # CGCNN
        ax = axes[0]
        ax.scatter(cgcnn_true_K, cgcnn_pred_K, alpha=0.15, s=8, c='royalblue')
        lims = [min(min(cgcnn_true_K), min(cgcnn_pred_K)), max(max(cgcnn_true_K), max(cgcnn_pred_K))]
        ax.plot(lims, lims, 'r--', lw=1.5, label='y = x')
        cgcnn_mae, _, cgcnn_r2 = compute_metrics(cgcnn_true_K, cgcnn_pred_K, 'CGCNN (plot)')
        ax.set_xlabel('True Bulk Modulus (GPa)', fontsize=12)
        ax.set_ylabel('Predicted Bulk Modulus (GPa)', fontsize=12)
        ax.set_title(f'CGCNN Direct Prediction\nMAE={cgcnn_mae:.2f} GPa, R²={cgcnn_r2:.4f}', fontsize=13)
        ax.legend(fontsize=10)
        ax.set_aspect('equal', adjustable='box')
        
        # E3NN
        ax = axes[1]
        ax.scatter(e3nn_true_K, e3nn_pred_K, alpha=0.15, s=8, c='darkorange')
        lims = [min(min(e3nn_true_K), min(e3nn_pred_K)), max(max(e3nn_true_K), max(e3nn_pred_K))]
        ax.plot(lims, lims, 'r--', lw=1.5, label='y = x')
        e3nn_mae, _, e3nn_r2 = compute_metrics(e3nn_true_K, e3nn_pred_K, 'E3NN (plot)')
        ax.set_xlabel('True Bulk Modulus (GPa)', fontsize=12)
        ax.set_ylabel('Predicted Bulk Modulus (GPa)', fontsize=12)
        ax.set_title(f'E3NN Tensor → VRH\nMAE={e3nn_mae:.2f} GPa, R²={e3nn_r2:.4f}', fontsize=13)
        ax.legend(fontsize=10)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        bulk_path = os.path.join(plot_dir, 'comparison_bulk_modulus.png')
        plt.savefig(bulk_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {bulk_path}')
        plt.close()
        
        # --- Plot 2: Parity plots side by side for Shear Modulus ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        ax = axes[0]
        ax.scatter(cgcnn_true_G, cgcnn_pred_G, alpha=0.15, s=8, c='royalblue')
        lims = [min(min(cgcnn_true_G), min(cgcnn_pred_G)), max(max(cgcnn_true_G), max(cgcnn_pred_G))]
        ax.plot(lims, lims, 'r--', lw=1.5, label='y = x')
        cgcnn_mae_g, _, cgcnn_r2_g = compute_metrics(cgcnn_true_G, cgcnn_pred_G, 'CGCNN Shear (plot)')
        ax.set_xlabel('True Shear Modulus (GPa)', fontsize=12)
        ax.set_ylabel('Predicted Shear Modulus (GPa)', fontsize=12)
        ax.set_title(f'CGCNN Direct Prediction\nMAE={cgcnn_mae_g:.2f} GPa, R²={cgcnn_r2_g:.4f}', fontsize=13)
        ax.legend(fontsize=10)
        ax.set_aspect('equal', adjustable='box')
        
        ax = axes[1]
        ax.scatter(e3nn_true_G, e3nn_pred_G, alpha=0.15, s=8, c='darkorange')
        lims = [min(min(e3nn_true_G), min(e3nn_pred_G)), max(max(e3nn_true_G), max(e3nn_pred_G))]
        ax.plot(lims, lims, 'r--', lw=1.5, label='y = x')
        e3nn_mae_g, _, e3nn_r2_g = compute_metrics(e3nn_true_G, e3nn_pred_G, 'E3NN Shear (plot)')
        ax.set_xlabel('True Shear Modulus (GPa)', fontsize=12)
        ax.set_ylabel('Predicted Shear Modulus (GPa)', fontsize=12)
        ax.set_title(f'E3NN Tensor → VRH\nMAE={e3nn_mae_g:.2f} GPa, R²={e3nn_r2_g:.4f}', fontsize=13)
        ax.legend(fontsize=10)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        shear_path = os.path.join(plot_dir, 'comparison_shear_modulus.png')
        plt.savefig(shear_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {shear_path}')
        plt.close()
        
        # --- Plot 3: Error distribution comparison ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        cgcnn_errors_K = [abs(cgcnn_results[mid]['true_K'] - cgcnn_results[mid]['pred_K']) for mid in common_ids]
        e3nn_errors_K = [abs(e3nn_results[mid]['true_K'] - e3nn_results[mid]['pred_K']) for mid in common_ids]
        
        ax = axes[0]
        ax.hist(cgcnn_errors_K, bins=80, alpha=0.6, label=f'CGCNN (MAE={np.mean(cgcnn_errors_K):.2f})', color='royalblue', range=(0, 100))
        ax.hist(e3nn_errors_K, bins=80, alpha=0.6, label=f'E3NN (MAE={np.mean(e3nn_errors_K):.2f})', color='darkorange', range=(0, 100))
        ax.set_xlabel('Absolute Error (GPa)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Bulk Modulus Error Distribution', fontsize=13)
        ax.legend(fontsize=10)
        
        cgcnn_errors_G = [abs(cgcnn_results[mid]['true_G'] - cgcnn_results[mid]['pred_G']) for mid in common_ids]
        e3nn_errors_G = [abs(e3nn_results[mid]['true_G'] - e3nn_results[mid]['pred_G']) for mid in common_ids]
        
        ax = axes[1]
        ax.hist(cgcnn_errors_G, bins=80, alpha=0.6, label=f'CGCNN (MAE={np.mean(cgcnn_errors_G):.2f})', color='royalblue', range=(0, 100))
        ax.hist(e3nn_errors_G, bins=80, alpha=0.6, label=f'E3NN (MAE={np.mean(e3nn_errors_G):.2f})', color='darkorange', range=(0, 100))
        ax.set_xlabel('Absolute Error (GPa)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Shear Modulus Error Distribution', fontsize=13)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        error_path = os.path.join(plot_dir, 'comparison_error_distribution.png')
        plt.savefig(error_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {error_path}')
        plt.close()
        
    except ImportError:
        print('matplotlib not available, skipping plots')
    
    print('\n' + '='*60)
    print('COMPARISON COMPLETE!')
    print('='*60)


if __name__ == '__main__':
    main()

