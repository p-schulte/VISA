# The basic building block of spatiotemporal graph convolutional networks.
# This module applies a temporal convolution followed by a spatial graph convolution.
# Optionally, it can use GAT-based attention for the spatial graph convolution.

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from config.config_loader import CONFIG




class ConvTemporalGraphical(nn.Module):

    r"""
    The basic module for applying spatiotemporal graph convolution.

    This module first applies a temporal convolution to extract temporal features from a sequence of graph-structured data.
    Then it applies a graph convolution over each frame's spatial structure using a learnable edge importance weighting.
    Optionally, it can use a GAT (Graph Attention Network) layer instead of standard convolution.

    Args:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output feature channels.
        kernel_size (int): Number of spatial kernels (i.e., edge types) to convolve over.
        t_kernel_size (int): Size of the temporal convolutional kernel.
        t_stride (int): Stride for the temporal convolution. Default: 1.
        t_padding (int): Padding for the temporal convolution. Default: 0.
        t_dilation (int): Dilation for the temporal convolution. Default: 1.
        bias (bool): Whether to include a learnable bias term. Default: True.

    Input:
        - X_seq: Tensor of shape (N, C_in, T, V), input features for each node over time.
        - A_seq: Tensor of shape (N, T, K, V, V), adjacency matrices per time step and edge type.

    Output:
        - X_out: Tensor of shape (N, C_out, T, V), output node features.
        - A_seq: Same as input (unchanged), passed through for consistency.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels


        # loading information about model from config file
        conf = CONFIG.ac
        self.pooling_method = conf['temp_spat_conv']

        # Temporal convolution
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)
        

        # Learnable weight matrix for graph convolution
        self.edge_importance = nn.Parameter(torch.ones(self.kernel_size, 1, 1))  # shape: (K, 1, 1)


        # GAT layer (note: input dim = output dim from temporal conv)
        self.use_gat = False
        self.gat = GATConv(out_channels, out_channels, heads=1, concat=True)



    def forward(self, X_seq, A_seq):
        """
        X_seq: Input graph sequence of shape (N, C, T, V)
        A_seq: Sequence of adjacency matrices of shape (N, T, K, V, V)
        """
        batch_size = X_seq.size(0)
        assert X_seq.shape == (batch_size, self.in_channels, X_seq.shape[2], X_seq.shape[3]), "Input shape is incorrect"
        assert A_seq.shape[3] == A_seq.shape[4], "Adjacency matrix must be square"
        assert A_seq.size(1) == X_seq.size(2), "Number of adjacency matrices must match number of time steps"
        assert A_seq.size(3) == X_seq.size(3), "Number of nodes in adjacency matrix must match number of nodes in X_seq"


        # Expand the feature channels into K×out_channels so that each spatial kernel gets a separate set of features
        X_seq = self.conv(X_seq)  
        n, k_times_c, t, v = X_seq.size()
        X_seq = X_seq.view(n, self.kernel_size, k_times_c // self.kernel_size, t, v)

        # Apply graph convolution at each time step with its corresponding adjacency matrix
        x_out = []
        for t_idx in range(t):
            x_t = X_seq[:, :, :, t_idx, :]  # Select features for time step t
            A_t = A_seq[:,t_idx]  # Select adjacency matrix for time step t


            # Apply graph convolution
            if self.use_gat:
                # GAT logic
                # x_t: (N, K, C, V) → (N, C, V) after temporal conv
                # -> flatten for GAT: (V, C), run once per graph
                gat_out = []
                for i in range(batch_size):
                    x_feat = x_t[i].mean(dim=0).T  # shape: (V, C)  -- collapsed K
                    A = A_t[i]                     # shape: (K, V, V)
                    edge_index = self.dense_adj_to_edge_index(A)
                    gat_out.append(self.gat(x_feat, edge_index))  # (V, out_channels)
                x_t = torch.stack(gat_out, dim=0).permute(0, 2, 1)  # (N, C, V)

            else:
                # Original einsum-based graph conv

                # Learned part: Edge weighting
                A_t = A_t * self.edge_importance.view(1, -1, 1, 1)  # shape stays (N, K, V, V)
                
                # x_skip = x_t.mean(dim=1)  # shape: [N, C, V] => skip connection
                x_t = torch.stack([
                    torch.einsum('kcv,kvw->cw', x_t[i], A_t[i])
                    for i in range(x_t.shape[0]) # for every element in the batch
                ]) 
                # x_t = x_t + x_skip  # Skip connection


            x_out.append(x_t)

        X_seq = torch.stack(x_out, dim=2)  # Stack back across time

        return X_seq.contiguous(), A_seq
    


    def dense_adj_to_edge_index(self, A):  # A: [K, V, V]
        A_combined = A.sum(dim=0)  # combine across edge types: (V, V)
        edge_index = (A_combined > 0).nonzero(as_tuple=False).T  # shape: (2, E)
        return edge_index
