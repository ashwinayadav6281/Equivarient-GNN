import torch
import torch.nn as nn
from e3nn import o3
from e3nn.nn import FullyConnectedNet, Gate
from e3nn.math import soft_one_hot_linspace

class FastImprovedEquivariantElasticNet(nn.Module):
    def __init__(self, num_node_features=100, max_radius=5.0):
        super().__init__()
        self.max_radius = max_radius
        
        self.atom_embedding = nn.Linear(num_node_features, 32)
        self.irreps_node_input = o3.Irreps("32x0e")
        self.irreps_node_hidden = o3.Irreps("32x0e + 16x1o + 8x2e")
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=2)
        
        self.num_layers = 3
        self.tps = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.self_connections = nn.ModuleList()
        
        irreps_scalars = o3.Irreps("32x0e")
        act_scalars = [torch.nn.functional.silu]
        irreps_gates = o3.Irreps("24x0e") 
        act_gates = [torch.sigmoid]
        irreps_non_scalars = o3.Irreps("16x1o + 8x2e")
        gate = Gate(irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_non_scalars)
        
        irreps_pre_gate = gate.irreps_in
        current_irreps = self.irreps_node_input
        
        for i in range(self.num_layers):
            tp = o3.FullyConnectedTensorProduct(current_irreps, self.irreps_sh, irreps_pre_gate, shared_weights=False)
            self.tps.append(tp)
            # Using soft one-hot linspace as a radial basis function (size 20)
            self.fcs.append(FullyConnectedNet([20, 32, tp.weight_numel], act=torch.nn.functional.silu))
            self.self_connections.append(o3.Linear(current_irreps, irreps_pre_gate))
            self.gates.append(gate)
            current_irreps = gate.irreps_out
            
        self.readout_mlp = nn.Sequential(
            nn.Linear(32 + 16*3 + 8*5, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 21)
        )
        
    def forward(self, node_features, edge_vec, edge_index):
        src, dst = edge_index
        current_node_features = self.atom_embedding(node_features)
        
        # Add epsilon to prevent NaN in backprop and sh normalization
        edge_dist = edge_vec.norm(dim=1, keepdim=True) + 1e-8
        
        # Radial basis encoding
        radial_basis = soft_one_hot_linspace(
            edge_dist.squeeze(-1), 
            start=0.0, 
            end=self.max_radius, 
            number=20, 
            basis='gaussian', 
            cutoff=True
        )
        
        sh = o3.spherical_harmonics(self.irreps_sh, edge_vec, normalize=True, normalization='component')
        
        for i in range(self.num_layers):
            weight = self.fcs[i](radial_basis)
            messages = self.tps[i](current_node_features[src], sh, weight)
            
            node_hidden = torch.zeros(current_node_features.shape[0], messages.shape[1], device=current_node_features.device)
            node_hidden.index_add_(0, dst, messages)
            
            self_out = self.self_connections[i](current_node_features)
            pre_gate = node_hidden + self_out
            current_node_features = self.gates[i](pre_gate)
        
        graph_feature = current_node_features.mean(dim=0)
        out = self.readout_mlp(graph_feature)
        
        C = torch.zeros(6, 6, device=out.device)
        idx = 0
        for i in range(6):
            for j in range(i, 6):
                C[i, j] = out[idx]
                C[j, i] = out[idx]
                idx += 1
        return C

SimpleEquivariantElasticNet = FastImprovedEquivariantElasticNet
