import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, heads=4):
        super().__init__()
        self.gat = GATConv(in_channels, hidden_channels, heads=heads, concat=True)

    def forward(self, X, edge_index):
        """
        Args:
            X: (V, C) node features
            edge_index: (2, E) edge list for all graphs

        Returns:
            X: (V, C) refined node features
        """
        X_gat = self.gat(X, edge_index)
        return X_gat

def run_gat(gat, Xs, As):
    """
    Reshape input tensors and run GAT on each graph in the batch.
    Xs: (N, T, V, C) node features
    As: (N, T, K, V, V) adjacency matrices
    """
    N, T, V, C = Xs.shape
    assert As.shape == (N, T, As.shape[2], V, V)
    
    sequence = []
    for n in range(N):
        batch = []
        for t in range(T):
            X = Xs[n, t]
            A = As[n, t]
            edge_index = get_edge_indices(A)
            ref_feat = gat(X, edge_index)
            batch.append(ref_feat)
        sequence.append(torch.stack(batch, dim=0))
    sequence = torch.stack(sequence, dim=0)
    return sequence


def get_edge_indices(A):
    """
    Convert dense multi-relational adjacency matrices to PyG-style edge_index.

    Args:
        A: (K, V, V) adjacency matrix with multiple relation types

    Returns:
        edge_index: (2, E) edge list across all graphs in batch and time
    """
    K, V, V_check = A.shape
    assert V == V_check, "A must have shape (K, V, V)"

    edge_indices = []

    # Combine all K relation types at this timestep
    A_combined = A.sum(dim=0)  # shape: (V, V)
    edge_src, edge_dst = (A_combined > 0).nonzero(as_tuple=True)
    edge_indices.append(torch.stack([edge_src, edge_dst], dim=0))

    edge_index = torch.cat(edge_indices, dim=1)  # shape: (2, E_total)
    return edge_index
