import torch
import torch.nn as nn
from e3nn import o3
from e3nn.nn import FullyConnectedNet, Gate, BatchNorm
from e3nn.math import soft_one_hot_linspace

class ImprovedEquivariantElasticNet(nn.Module):
    """Highly Optimized Equivariant GNN for Elastic Tensor Prediction.
    
    Architectural Improvements:
    - Initial Node MLP (Linear -> SiLU -> Linear) for richer feature embedding.
    - Deep Radial Network (3 layers: 64 -> 64 -> weight_numel) for better distance embedding.
    - 20 Gaussian radial bases for finer spatial resolution.
    - Equivariant BatchNorm after message passing to stabilize gradients.
    - Mean and Sum graph pooling concatenation for better structural representation.
    - Deeper readout MLP with Dropout for regularization.
    """
    def __init__(self, num_node_features=100, max_radius=5.0):
        super().__init__()
        self.max_radius = max_radius
        self.num_radial_basis = 20
        
        # 1. Richer Node Embedding MLP
        self.atom_embedding = nn.Sequential(
            nn.Linear(num_node_features, 64),
            nn.SiLU(),
            nn.Linear(64, 32)
        )
        self.irreps_node_input = o3.Irreps("32x0e")
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=2)
        
        self.num_layers = 3
        self.tps = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.self_connections = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        irreps_scalars = o3.Irreps("32x0e")
        act_scalars = [torch.nn.functional.silu]
        irreps_gates = o3.Irreps("24x0e")
        act_gates = [torch.sigmoid]
        irreps_non_scalars = o3.Irreps("16x1o + 8x2e")
        gate = Gate(irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_non_scalars)
        
        irreps_pre_gate = gate.irreps_in
        current_irreps = self.irreps_node_input
        
        for i in range(self.num_layers):
            tp = o3.FullyConnectedTensorProduct(
                current_irreps, self.irreps_sh, irreps_pre_gate, shared_weights=False
            )
            self.tps.append(tp)
            
            # Deeper Radial Network
            self.fcs.append(FullyConnectedNet(
                [self.num_radial_basis, 64, 64, tp.weight_numel], 
                act=torch.nn.functional.silu
            ))
            
            self.self_connections.append(o3.Linear(current_irreps, irreps_pre_gate))
            self.gates.append(gate)
            
            # Equivariant Batch Norm for stable training
            self.batch_norms.append(BatchNorm(gate.irreps_out))
            
            current_irreps = gate.irreps_out
        
        # Readout feature dim is doubled because we will concatenate Mean and Sum pooling
        base_feat_dim = 32 + 16*3 + 8*5  # = 120
        feat_dim = base_feat_dim * 2     # = 240
        
        # Deeper Readout MLP with Dropout
        self.readout_mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 21)
        )
        
    def forward(self, node_features, edge_vec, edge_index):
        src, dst = edge_index
        current_node_features = self.atom_embedding(node_features)
        
        # Epsilon to prevent NaN gradients in backwards pass for perfectly overlapping atoms
        edge_dist = edge_vec.norm(dim=1, keepdim=True) + 1e-8
        
        # Expanded Gaussian radial basis
        radial_basis = soft_one_hot_linspace(
            edge_dist.squeeze(-1), start=0.0, end=self.max_radius,
            number=self.num_radial_basis, basis='smooth_finite', cutoff=True
        )
        
        sh = o3.spherical_harmonics(self.irreps_sh, edge_vec, normalize=True, normalization='component')
        
        for i in range(self.num_layers):
            weight = self.fcs[i](radial_basis)
            messages = self.tps[i](current_node_features[src], sh, weight)
            
            node_hidden = torch.zeros(
                current_node_features.shape[0], messages.shape[1], 
                device=current_node_features.device
            )
            node_hidden.index_add_(0, dst, messages)
            
            self_out = self.self_connections[i](current_node_features)
            pre_gate = node_hidden + self_out
            
            # Apply Gate and BatchNorm
            node_features_out = self.gates[i](pre_gate)
            current_node_features = self.batch_norms[i](node_features_out)
        
        # Pooling: Concatenate Mean and Sum for context size invariance and local extreme representation
        graph_mean = current_node_features.mean(dim=0)
        graph_sum = current_node_features.sum(dim=0)
        graph_feature = torch.cat([graph_mean, graph_sum], dim=0)
        
        out = self.readout_mlp(graph_feature)
        
        C = torch.zeros(6, 6, device=out.device)
        idx = 0
        for i in range(6):
            for j in range(i, 6):
                C[i, j] = out[idx]
                C[j, i] = out[idx]
                idx += 1
        return C

SimpleEquivariantElasticNet = ImprovedEquivariantElasticNet

if __name__ == "__main__":
    model = ImprovedEquivariantElasticNet(num_node_features=100)
    params = sum(p.numel() for p in model.parameters())
    print(f"Improved model: {params:,} parameters")
    nodes = torch.randn(4, 100)
    edge_vec = torch.randn(6, 3)
    edge_index = torch.tensor([[0, 0, 1, 2, 2, 3], [1, 2, 2, 3, 0, 1]])
    C = model(nodes, edge_vec, edge_index)
    print("Output shape:", C.shape)
