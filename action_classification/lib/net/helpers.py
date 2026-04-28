import torch
import torch.nn as nn
import torch.nn.functional as F

class NodeAttention(nn.Module):
    def __init__(self, in_features, hidden_dim):
        """
        Attention mechanism to compute weighted node features.

        Args:
            in_features: Dimension of input node features
            hidden_dim: Dimension of hidden layer in attention MLP
        """
        super(NodeAttention, self).__init__()
        
        # Learnable MLP to compute attention scores
        self.attn_mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Outputs scalar score for each node
        )

    def forward(self, X):
        """
        X: Node features of shape (N, T, V, C)

        Returns:
            Weighted sum of node features: (N, T, C)
        """
        N, T, V, C = X.shape  # Batch, Time, Nodes, Channels

        # Compute attention scores (N, T, V, 1)
        attn_scores = self.attn_mlp(X)
        
        # Normalize attention scores using softmax across nodes (V)
        attn_weights = F.softmax(attn_scores, dim=2)  # Shape: (N, T, V, 1)

        # Compute weighted sum over nodes (V) to get scene-level embedding
        weighted_X = (attn_weights * X).sum(dim=2)  # Shape: (N, T, C)

        return weighted_X  # Returns aggregated feature per time step


class MultiHeadNodeAttention_old(nn.Module):
    def __init__(self, in_features, hidden_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.attn_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_heads)
        ])

    def forward(self, X):  # (N, T, V, C)
        attn_weights = [F.softmax(attn(X), dim=2) for attn in self.attn_mlp]
        attn_weights = torch.stack(attn_weights, dim=-1)  # (N, T, V, num_heads)

        # Aggregate over different heads
        weighted_X = (attn_weights.mean(dim=-1) * X).sum(dim=2)
        return weighted_X
    
class MultiHeadNodeAttention(nn.Module):
    def __init__(self, in_features, hidden_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        # Learnable attention projections
        self.query = nn.Linear(in_features, hidden_dim * num_heads, bias=False)
        self.key = nn.Linear(in_features, hidden_dim * num_heads, bias=False)
        self.value = nn.Linear(in_features, hidden_dim * num_heads, bias=False)

        # Final projection after attention
        self.fc_out = nn.Linear(hidden_dim * num_heads, in_features)  

    def forward(self, X):  # (N, T, V, C)
        N, T, V_dim, C = X.shape  # Batch, Time, Nodes, Features
        
        # Project queries, keys, and values
        Q = self.query(X)  # (N, T, V, num_heads * hidden_dim)
        K = self.key(X)    # (N, T, V, num_heads * hidden_dim)
        V = self.value(X)  # (N, T, V, num_heads * hidden_dim)


        # Reshape for multi-head attention: (batch, time, nodes, heads, dim_per_head)
        Q = Q.view(N, T, V_dim, self.num_heads, self.hidden_dim).permute(0, 1, 3, 2, 4)  # (N, T, H, V, D)
        K = K.view(N, T, V_dim, self.num_heads, self.hidden_dim).permute(0, 1, 3, 2, 4)  # (N, T, H, V, D)
        V = V.view(N, T, V_dim, self.num_heads, self.hidden_dim).permute(0, 1, 3, 2, 4)  # (N, T, H, V, D)

        # Compute scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_dim ** 0.5)  # (N, T, H, V, V)
        attn_weights = F.softmax(attn_scores, dim=-1)  # Normalize across nodes (V)

        # Apply attention weights to values
        weighted_V = torch.matmul(attn_weights, V)  # (N, T, H, V, D)

        # Reshape back: concatenate heads and project
        weighted_V = weighted_V.permute(0, 1, 3, 2, 4).contiguous().view(N, T, V_dim, self.num_heads * self.hidden_dim)
        output = self.fc_out(weighted_V)  # Project back to original feature dimension

        return output, attn_weights  # Return attended features & attention scores



class RoleEmbeddingGenerator(nn.Module):
    def __init__(self, feature_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(feature_dim, out_dim)  # num_classes → out_dim

    def forward(self, action_name_features):  # (N, T, C) → (N, T, out_dim)
        action_name_features = self.fc(action_name_features)
        return action_name_features  # Dynamic role embeddings
