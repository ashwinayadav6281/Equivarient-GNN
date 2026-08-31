import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pymatgen.core import Structure
import warnings
warnings.filterwarnings('ignore')

class EquivariantTensorDataset(Dataset):
    def __init__(self, data_dir='dataset_equivariant', max_radius=5.0):
        self.data_dir = data_dir
        self.max_radius = max_radius
        
        tensor_json_path = os.path.join(data_dir, 'elastic_tensors.json')
        if not os.path.exists(tensor_json_path):
            self.tensors = {}
            self.cif_ids = []
            return
            
        with open(tensor_json_path, 'r') as f:
            raw_data = json.load(f)
            
        self.tensors = {}
        for mid, tensor in raw_data.items():
            if tensor is None:
                continue
            tensor_np = np.array(tensor)
            if np.max(np.abs(tensor_np)) < 1000.0:
                self.tensors[mid] = tensor
                
        self.cif_ids = []
        for filename in os.listdir(data_dir):
            if filename.endswith('.cif'):
                mat_id = filename.replace('.cif', '')
                if mat_id in self.tensors:
                    self.cif_ids.append(mat_id)
                    
        print(f'Loaded dataset with {len(self.cif_ids)} valid structures.')

    def __len__(self):
        return len(self.cif_ids)

    def __getitem__(self, idx):
        mat_id = self.cif_ids[idx]
        cif_path = os.path.join(self.data_dir, f'{mat_id}.cif')
        
        structure = Structure.from_file(cif_path)
        
        atomic_numbers = torch.tensor([site.specie.number for site in structure], dtype=torch.long)
        atomic_numbers = torch.clamp(atomic_numbers, 0, 99) 
        one_hot_features = torch.nn.functional.one_hot(atomic_numbers, num_classes=100).to(torch.float32)
        
        src_indices = []
        dst_indices = []
        edge_vectors = []
        
        N = len(structure)
        for i in range(N):
            neighbors = structure.get_neighbors(structure[i], self.max_radius)
            for neighbor in neighbors:
                j = neighbor.index
                vec = neighbor.coords - structure[i].coords
                src_indices.append(i)
                dst_indices.append(j)
                edge_vectors.append(vec)
                
        if not src_indices: 
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_vec = torch.empty((0, 3), dtype=torch.float32)
        else:
            edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
            edge_vec = torch.tensor(np.array(edge_vectors), dtype=torch.float32)
            
        target_tensor = torch.tensor(self.tensors[mat_id], dtype=torch.float32)
        
        return {
            'mat_id': mat_id,
            'node_features': one_hot_features,
            'edge_index': edge_index,
            'edge_vec': edge_vec,
            'target_tensor': target_tensor
        }
