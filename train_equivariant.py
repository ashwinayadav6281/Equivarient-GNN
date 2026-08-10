import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from equivariant_dataset import EquivariantTensorDataset
from equivariant_model import SimpleEquivariantElasticNet
from tensor_utils import calculate_moduli, check_stability

def train(epochs=10):
    dataset = EquivariantTensorDataset()
    
    if len(dataset) == 0:
        print("Dataset is empty. Run fetch_elastic_tensors.py first.")
        return
        
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # We use batch_size=1 to avoid graph batching complexities
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    # The dataset now outputs 100 node features (one-hot encoding)
    model = SimpleEquivariantElasticNet(num_node_features=100).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Calculate dataset statistics for Z-score standardization
    print("Calculating training data statistics...")
    all_targets = []
    for batch in train_loader:
        all_targets.append(batch['target_tensor'][0].flatten())
    all_targets = torch.stack(all_targets)
    target_mean = all_targets.mean(dim=0).to(device)
    target_std = all_targets.std(dim=0).to(device)
    # Avoid zero division
    target_std[target_std < 1e-4] = 1.0
    
    # Reshape back to 6x6
    target_mean = target_mean.view(6, 6)
    target_std = target_std.view(6, 6)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            node_features = batch['node_features'][0].to(device)
            pos = batch['pos'][0].to(device)
            edge_index = batch['edge_index'][0].to(device)
            
            # Standardize target
            target_tensor = batch['target_tensor'][0].to(device)
            standardized_target = (target_tensor - target_mean) / target_std
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_C = model(node_features, pos, edge_index)
            
            # Loss against standardized target
            loss = F.mse_loss(pred_C, standardized_target)
            loss.backward()
            
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                node_features = batch['node_features'][0].to(device)
                pos = batch['pos'][0].to(device)
                edge_index = batch['edge_index'][0].to(device)
                
                target_tensor = batch['target_tensor'][0].to(device)
                standardized_target = (target_tensor - target_mean) / target_std
                
                pred_C = model(node_features, pos, edge_index)
                loss = F.mse_loss(pred_C, standardized_target)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        # Step the scheduler
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
    print("Training complete. Evaluating some samples...")
    evaluate(model, val_loader, device, target_mean, target_std)
    
def evaluate(model, val_loader, device, target_mean, target_std):
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 3: 
                break
                
            mat_id = batch['mat_id'][0]
            node_features = batch['node_features'][0].to(device)
            pos = batch['pos'][0].to(device)
            edge_index = batch['edge_index'][0].to(device)
            
            target_tensor = batch['target_tensor'][0].cpu().numpy()
            
            # Scale prediction back to original magnitude
            pred_standard = model(node_features, pos, edge_index).cpu().numpy()
            target_std_np = target_std.cpu().numpy()
            target_mean_np = target_mean.cpu().numpy()
            
            # Since pred_C and the targets are flattened in the MSE loss, 
            # wait, the target_tensor is 6x6, so pred_C is 6x6.
            # standardized_target = (target_tensor - target_mean) / target_std
            # So target_tensor = standardized_target * target_std + target_mean
            pred_C = pred_standard * target_std_np + target_mean_np
            
            true_moduli = calculate_moduli(target_tensor)
            pred_moduli = calculate_moduli(pred_C)
            
            true_k = true_moduli.get('K_H', float('nan'))
            true_g = true_moduli.get('G_H', float('nan'))
            pred_k = pred_moduli.get('K_H', float('nan'))
            pred_g = pred_moduli.get('G_H', float('nan'))
            
            true_stable = check_stability(target_tensor)
            pred_stable = check_stability(pred_C)
            
            print(f"\n--- Material: {mat_id} ---")
            print(f"TRUE: Bulk={true_k:.1f}, Shear={true_g:.1f}, Stable={true_stable}")
            print(f"PRED: Bulk={pred_k:.1f}, Shear={pred_g:.1f}, Stable={pred_stable}")
            print(f"Tensor MSE: {np.mean((target_tensor - pred_C)**2):.4f}")

if __name__ == "__main__":
    train(epochs=10)
