import torch
import torch.nn as nn
from e3nn import o3
from e3nn.nn import FullyConnectedNet

class SimpleEquivariantElasticNet(nn.Module):
    """
    A simplified E(3)-Equivariant Neural Network for predicting elastic tensors.
    Takes in invariant node features (e.g. atomic numbers) and 3D coordinates.
    """
    def __init__(self, num_node_features, max_radius=5.0):
        super().__init__()
        self.max_radius = max_radius
        
        # Node features are invariant scalars: 0e
        self.irreps_node_input = o3.Irreps(f"{num_node_features}x0e")
        
        # Hidden node features (scalars and vectors): 0e + 1o
        # We increase the number of channels to give the network more capacity
        self.irreps_node_hidden = o3.Irreps("64x0e + 32x1o")
        
        # Spherical harmonics for edge vectors (L=0, 1, 2)
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=2)
        
        # We will use 3 message passing layers
        self.num_layers = 3
        self.tps = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.self_connections = nn.ModuleList()
        
        # Layer 1: Input to Hidden
        self.tps.append(o3.FullyConnectedTensorProduct(self.irreps_node_input, self.irreps_sh, self.irreps_node_hidden, shared_weights=False))
        self.fcs.append(FullyConnectedNet([1, 32, self.tps[0].weight_numel], act=torch.nn.functional.silu))
        self.self_connections.append(o3.Linear(self.irreps_node_input, self.irreps_node_hidden))
        
        # Layers 2 and 3: Hidden to Hidden
        for i in range(1, self.num_layers):
            tp = o3.FullyConnectedTensorProduct(self.irreps_node_hidden, self.irreps_sh, self.irreps_node_hidden, shared_weights=False)
            self.tps.append(tp)
            self.fcs.append(FullyConnectedNet([1, 32, tp.weight_numel], act=torch.nn.functional.silu))
            self.self_connections.append(o3.Linear(self.irreps_node_hidden, self.irreps_node_hidden))
            
        # Readout layer: pool node features and map to the elastic tensor irreps.
        self.readout_tp = o3.Linear(
            self.irreps_node_hidden,
            o3.Irreps("21x0e")  # outputting 21 invariant scalars
        )
        
    def forward(self, node_features, pos, edge_index):
        """
        node_features: [N, num_node_features]
        pos: [N, 3] 3D coordinates
        edge_index: [2, E] edge indices
        """
        src, dst = edge_index
        
        # Edge vectors
        edge_vec = pos[dst] - pos[src]
        edge_dist = edge_vec.norm(dim=1, keepdim=True)
        
        # Spherical harmonics of edge vectors
        sh = o3.spherical_harmonics(self.irreps_sh, edge_vec, normalize=True, normalization='component')
        
        # Radial inputs for FC networks
        # We need to process layers sequentially
        current_node_features = node_features
        
        for i in range(self.num_layers):
            # Radial weights for this layer
            weight = self.fcs[i](edge_dist)
            
            # Messages from neighbors
            messages = self.tps[i](current_node_features[src], sh, weight)
            
            # Aggregate messages to destination nodes
            node_hidden = torch.zeros(current_node_features.shape[0], self.irreps_node_hidden.dim, device=current_node_features.device)
            node_hidden.index_add_(0, dst, messages)
            
            # Add self-connection (residual)
            self_out = self.self_connections[i](current_node_features)
            
            # Update node features for next layer
            current_node_features = node_hidden + self_out
        
        # Global pooling (mean over all nodes to stabilize features)
        graph_feature = node_hidden.mean(dim=0)
        
        # Map to 21 elastic components
        out = self.readout_tp(graph_feature)
        
        # Reshape to a symmetric 6x6 matrix (Voigt notation)
        # This requires placing the 21 independent components into a 6x6 matrix
        # C11, C22, C33, C44, C55, C66, C12, C13, C23, C14, C15, C16, C24, C25, C26, C34, C35, C36, C45, C46, C56
        # To ensure the matrix is symmetric, we construct it explicitly.
        
        C = torch.zeros(6, 6, device=out.device)
        idx = 0
        for i in range(6):
            for j in range(i, 6):
                C[i, j] = out[idx]
                C[j, i] = out[idx]
                idx += 1
                
        return C

# Test the model
if __name__ == "__main__":
    model = SimpleEquivariantElasticNet(num_node_features=100)
    print("Model initialized.")
    
    # Dummy data
    nodes = torch.randn(4, 100)
    pos = torch.randn(4, 3)
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 3]])
    
    C = model(nodes, pos, edge_index)
    print("Output shape:", C.shape)
    print("Is symmetric:", torch.allclose(C, C.T))
