import os
import torch
import numpy as np
from equivariant_model import SimpleEquivariantElasticNet
from equivariant_dataset import EquivariantTensorDataset
from torch.utils.data import DataLoader

def main():
    device = torch.device('cpu')
    
    # 1. Load the dataset
    print('Loading dataset...')
    dataset = EquivariantTensorDataset(data_dir='dataset_equivariant')
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 2. Initialize the model
    print('Initializing model...')
    model = SimpleEquivariantElasticNet(num_node_features=100).to(device)
    
    # 3. Load the trained weights and normalization stats
    ckpt_path = 'checkpoints_equivariant/model_best_equivariant.pth.tar'
    stats_path = 'checkpoints_equivariant/normalization_stats.pt'
    
    if not os.path.exists(ckpt_path) or not os.path.exists(stats_path):
        print(f'Error: Could not find model weights at {ckpt_path}')
        print('Please ensure training is complete and files are downloaded.')
        return

    # Load stats
    stats = torch.load(stats_path, map_location=device)
    target_mean_np = stats['mean'].cpu().numpy()
    target_std_np = stats['std'].cpu().numpy()
    
    # Load weights
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print('Model loaded successfully!\n')
    
    print(f'{'Material ID':<15} | {'Component':<10} | {'Predicted (GPa)':<18} | {'True (GPa)':<18} | {'Error'}')
    print('-' * 80)
    
    # 4. Predict and compare individual components
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 10:
                break
                
            mat_id = batch['mat_id'][0]
            node_features = batch['node_features'][0].to(device)
            edge_vec = batch['edge_vec'][0].to(device)
            edge_index = batch['edge_index'][0].to(device)
            true_C = batch['target_tensor'][0].cpu().numpy()

            # Forward pass
            pred_standard = model(node_features, edge_vec, edge_index).cpu().numpy()
            pred_C = pred_standard * target_std_np + target_mean_np
            
            components = {
                'C11': (0,0), 'C22': (1,1), 'C33': (2,2), 'C44': (3,3), 'C55': (4,4), 'C66': (5,5),
                'C12': (0,1), 'C13': (0,2), 'C23': (1,2),
                'C14': (0,3), 'C15': (0,4), 'C16': (0,5),
                'C24': (1,3), 'C25': (1,4), 'C26': (1,5),
                'C34': (2,3), 'C35': (2,4), 'C36': (2,5),
                'C45': (3,4), 'C46': (3,5), 'C56': (4,5)
            }
            
            for comp_name, (row, col) in components.items():
                p_val = pred_C[row, col]
                t_val = true_C[row, col]
                error = abs(p_val - t_val)
                print(f'{mat_id:<15} | {comp_name:<10} | {p_val:>18.2f} | {t_val:>18.2f} | {error:>10.2f}')
            print('-' * 80)

if __name__ == '__main__':
    main()
