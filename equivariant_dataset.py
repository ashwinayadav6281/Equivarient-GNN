import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pymatgen.core import Structure

class EquivariantTensorDataset(Dataset):
    def __init__(self, data_dir="dataset_equivariant", max_radius=5.0):
        self.data_dir = data_dir
        self.max_radius = max_radius
        
        # Load tensors
        tensor_json_path = os.path.join(data_dir, "elastic_tensors.json")
        if not os.path.exists(tensor_json_path):
            self.tensors = {}
            self.cif_ids = []
            return
            
        with open(tensor_json_path, 'r') as f:
            raw_data = json.load(f)
            
        self.tensors = {}
        # Filter out extreme outliers (DFT errors in MP database)
        for mid, tensor in raw_data.items():
            if tensor is None:
                continue
            tensor_np = np.array(tensor)
            if np.max(np.abs(tensor_np)) < 2000.0:
                self.tensors[mid] = tensor
                
        self.cif_ids = []
        for filename in os.listdir(data_dir):
            if filename.endswith(".cif"):
                mat_id = filename.replace(".cif", "")
                if mat_id in self.tensors:
                    self.cif_ids.append(mat_id)
                    
        print(f"Loaded dataset with {len(self.cif_ids)} valid structures (filtered {len(raw_data) - len(self.cif_ids)} outliers).")

    def __len__(self):
        return len(self.cif_ids)

    def __getitem__(self, idx):
        mat_id = self.cif_ids[idx]
        cif_path = os.path.join(self.data_dir, f"{mat_id}.cif")
        
        # 1. Parse Structure
        structure = Structure.from_file(cif_path)
        
        # 2. Node features (One-hot encoded atomic numbers)
        # We one-hot encode atomic numbers up to 100 to give the model categorical context.
        atomic_numbers = torch.tensor([site.specie.number for site in structure], dtype=torch.long)
        # Ensure we don't exceed 99
        atomic_numbers = torch.clamp(atomic_numbers, 0, 99) 
        one_hot_features = torch.nn.functional.one_hot(atomic_numbers, num_classes=100).to(torch.float32)
        
        # 3. Positions
        pos = torch.tensor(structure.cart_coords, dtype=torch.float32)
        
        # 4. Edge Index (Find neighbors)
        # We loop over all pairs using get_distance to avoid pymatgen Cython bugs on Windows
        src_indices = []
        dst_indices = []
        
        N = len(structure)
        for i in range(N):
            for j in range(N):
                if i != j:
                    # get_distance returns the shortest distance to any image of j
                    if structure.get_distance(i, j) <= self.max_radius:
                        src_indices.append(i)
                        dst_indices.append(j)
                
        if not src_indices: # Handle isolated atoms edge case
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
            
        # 5. Target Tensor
        target_matrix = self.tensors[mat_id]
        target_tensor = torch.tensor(target_matrix, dtype=torch.float32)
        
        return {
            'mat_id': mat_id,
            'node_features': one_hot_features,
            'pos': pos,
            'edge_index': edge_index,
            'target_tensor': target_tensor
        }

def collate_graphs(batch):
    """
    Collate function for graph data since number of nodes/edges varies.
    """
    # For a simple batching, we just return a list of dicts.
    # A true PyTorch Geometric DataLoader would batch them into a single disconnected graph,
    # but since our current equivariant model doesn't use scatter/batch indices natively yet,
    # we'll process them one by one or we'd need to adapt the model for batching.
    
    # We will just return the list of items.
    return batch

if __name__ == "__main__":
    dataset = EquivariantTensorDataset()
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample mat_id: {sample['mat_id']}")
        print(f"Node features shape: {sample['node_features'].shape}")
        print(f"Pos shape: {sample['pos'].shape}")
        print(f"Edge index shape: {sample['edge_index'].shape}")
        print(f"Target tensor shape: {sample['target_tensor'].shape}")
