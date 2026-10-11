import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pymatgen.core import Structure
import warnings
warnings.filterwarnings('ignore')

class EquivariantTensorDataset(Dataset):
    def __init__(self, data_dir='dataset_equivariant', max_radius=6.0):
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
        
        import math
        atomic_numbers = []
        rich_features = []
        for site in structure:
            sp = site.specie
            num = min(sp.number, 99)
            atomic_numbers.append(num)
            
            # Extract rich elemental properties, fallback to 0.0 if missing or NaN
            X = sp.X if sp.X is not None and not math.isnan(sp.X) else 0.0
            radius = sp.atomic_radius if sp.atomic_radius is not None and not (isinstance(sp.atomic_radius, float) and math.isnan(sp.atomic_radius)) else 0.0
            mass = sp.atomic_mass if sp.atomic_mass is not None and not (isinstance(sp.atomic_mass, float) and math.isnan(sp.atomic_mass)) else 0.0
            group = sp.group if sp.group is not None else 0.0
            row = sp.row if sp.row is not None else 0.0
            
            # Normalize them roughly to 0-1 range for stable training
            rich_features.append([
                X / 4.0,           # Electronegativity max ~ 4.0
                radius / 3.0,      # Radius max ~ 3.0 A
                mass / 200.0,      # Mass max ~ 200+
                group / 18.0,      # Group max = 18
                row / 7.0          # Row max = 7
            ])
            
        atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.long)
        one_hot_features = torch.nn.functional.one_hot(atomic_numbers, num_classes=100).to(torch.float32)
        
        rich_features = torch.tensor(rich_features, dtype=torch.float32)
        # Concatenate: 100 (one-hot) + 5 (rich) = 105 features
        node_features = torch.cat([one_hot_features, rich_features], dim=-1)
        
        src_indices = []
        dst_indices = []
        edge_vectors = []
        
        N = len(structure)
        for i in range(N):
            # Using get_all_neighbors to avoid Windows C-extension bug on local machine, and it's cleaner
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
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_vec': edge_vec,
            'target_tensor': target_tensor
        }
