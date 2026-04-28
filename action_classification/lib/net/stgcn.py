import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.net.tgcn import ConvTemporalGraphical

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from config.config_loader import CONFIG



class STGCNLayer(nn.Module):
    r"""Applies a Spatial-Temporal Graph Convolution (ST-GCN) to a sequence of graphs.

    This layer first performs spatial graph convolution over each frame using multiple 
    adjacency matrices (e.g., edge types), followed by temporal convolution to capture 
    patterns over time. It also supports optional residual connections.

    Args:
        in_channels (int): Number of input feature channels per node
        out_channels (int): Number of output feature channels per node
        kernel_size (tuple): (temporal_kernel_size, spatial_kernel_size)
        stride (int, optional): Temporal stride. Default is 1.
        residual (bool, optional): Whether to apply a residual connection. Default is True.

    Shape:
        - Input[0]: X_seq of shape (N, C_in, T_in, V)
        - Input[1]: A_seq of shape (N, T_in, K, V, V)
        - Output[0]: Updated X_seq of shape (N, C_out, T_out, V)
        - Output[1]: Same A_seq (unchanged adjacency matrix)
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 residual=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.residual = residual


        # loading information about model from config file
        conf = CONFIG.ac
        dropout_tcn = conf['stgcn_layer']['dropout_tcn']

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)


        # DEFINE LAYERS

        # Spatial convolution
        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])
        
        # Temporal convolution
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout_tcn, inplace=True),
        )

        # Residual connection
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        # ReLU activation
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, X_seq, A_seq):
        """
        Applies the ST-GCN layer to a batch of input graph sequences.

        Args:
            X_seq (torch.Tensor): Input tensor of shape (N, C_in, T_in, V)
            A_seq (torch.Tensor): Adjacency matrices of shape (N, T, K, V, V)

        Returns:
            torch.Tensor: Updated graph features of shape (N, C_out, T_out, V)
            torch.Tensor: Unchanged adjacency matrix (A_seq)
        """
        # Node feature dimensions
        N, C, T, V = X_seq.shape

        # Compute residual connection
        res = self.residual(X_seq)

        # Apply spatial graph convolution
        # Output shape: (N, C_out, T_out, V)
        x_out, _ = self.gcn(X_seq, A_seq)  
        
        # Ablation study: remove temporal convolution
        if CONFIG.ac['ablation_studies']['remove_tcn']:
            # Do not apply temporal convolution
            x_out = x_out + res
        else:
            # Apply temporal convolution
            x_out = self.tcn(x_out) + res

        # validate output shape
        assert X_seq.shape[1] == self.in_channels, f"Input shape {X_seq.shape} does not match in_channels {self.in_channels}"
        assert A_seq.shape[3] == A_seq.shape[4], f"Adjacency matrix shape {A_seq.shape} is not square"
        assert x_out.shape[1] == self.out_channels, f"Output shape {x_out.shape} does not match out_channels {self.out_channels}"
        assert x_out.shape[2] == T, f"Output sequence length {x_out.shape[2]} does not match input sequence length {T}"
        assert x_out.shape[3] == V, f"Output number of nodes {x_out.shape[3]} does not match input number of nodes {V}"

        # Return processed sequence and adjacency sequence
        return self.relu(x_out), A_seq







class STGCN(nn.Module):
    r"""Spatial-Temporal Graph Convolutional Network (ST-GCN)

    This model applies a series of spatial-temporal graph convolutional layers 
    to sequences of graphs, followed by pooling, optional LSTM temporal modeling, 
    and dual-head classification for structured action prediction.

    Args:
        in_channels (int): Number of input feature channels per node
        num_classes (int): Number of action classes to predict
        spatial_kernel_size (int): Number of edge types in adjacency matrix (K)
        temporal_kernel_size (int): Size of the temporal convolution kernel
        num_nodes (int): Number of nodes in the graph (V)

    Attributes:
        st_gcn_networks (nn.ModuleList): Stack of ST-GCN layers
        lstm (nn.LSTM): Optional bidirectional LSTM for sequence modeling
        data_bn (nn.BatchNorm2d): Batch normalization over input features
        action_name_fcn (nn.Sequential): Fully-connected head for action name classification
        args_mlp / args_mlp1 / args_mlp2: MLP heads for argument classification
        attn_fc (nn.Linear): Attention layer for node-level importance pooling
    """

    def __init__(self, in_channels, 
                 num_classes, 
                 spatial_kernel_size=3,
                 temporal_kernel_size=9,
                 num_nodes=12,
                 ):
        
        super(STGCN, self).__init__()
        self.num_classes = num_classes
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.spatial_kernel_size = spatial_kernel_size
        self.temporal_kernel_size = temporal_kernel_size

        # Ablation studies
        if CONFIG.ac['ablation_studies']['remove_tcn']:
            # Do not apply temporal convolution
            self.temporal_kernel_size = 1
        if CONFIG.ac['ablation_studies']['use_simple_edge_type']:
            self.spatial_kernel_size = 2

        # build networks
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        # Normalization layer
        self.data_bn = nn.BatchNorm2d(self.in_channels)

        # loading information about model from config file
        conf = CONFIG.ac
        self.pooling_method = conf['stgcn_model']['pooling_method']
        fcn_dropout = conf['stgcn_model']['fcn_dropout']
        self.action_name_prediction_method = conf['stgcn_model']['action_name_prediction']['method']
        self.argument_arguments_prediction_method = conf['stgcn_model']['argument_prediction']['method']

        # Load MLP configuration from the YAML config file
        mlp_config = conf['stgcn_model']['classifier_head_architecture']

        self.use_lstm = conf['stgcn_model']['use_lstm']
        if self.use_lstm:
            self.lstm_layers = conf['stgcn_model']['lstm_num_layers']

        stgcn_architecture = conf['stgcn_model']['stgcn_architecture']
        layers = stgcn_architecture['layers']


        # ST-GCN layers
        self.st_gcn_networks = nn.ModuleList()

        for index, layer_out_channels in enumerate(layers):
            self.st_gcn_networks.append(
                STGCNLayer(in_channels, layer_out_channels, kernel_size, 1, residual = (index != 0))
            )
            in_channels = layer_out_channels

        self.out_channels = layers[-1]


        # Attention FC layer to learn node importance -> for pooling
        self.attn_fc = nn.Linear(in_features=self.out_channels, out_features=1)  # Maps node features to attention scores



        if self.use_lstm:
            # **LSTM layer to process temporal dynamics**
            self.lstm = nn.LSTM(
                input_size=self.out_channels,  
                hidden_size=self.out_channels // 2,  
                num_layers=self.lstm_layers,  
                batch_first=True,  
                bidirectional=True  
            )

        # ------
        # FINAL CLASSIFICATION HEAD
        # ------


        # ------
        # 1. ACTION NAME PREDICTION

        if self.action_name_prediction_method == 'whole_sequence':

            # Fully connected layer for action_name prediction
            self.action_name_fcn = nn.Sequential( 
                nn.Linear(self.out_channels, self.out_channels),
                nn.ReLU(),
                nn.Dropout(fcn_dropout),
                nn.Linear(self.out_channels, self.num_classes),
            )
        elif self.action_name_prediction_method == 'pairwise':   

            # Fully connected layer for action_name prediction
            self.action_name_fcn = nn.Sequential(
                nn.Linear(2 * self.out_channels, 128),
                nn.ReLU(),
                nn.Dropout(fcn_dropout),
                nn.Linear(128, self.num_classes),
            )
        

        # ------
        # 2. ARGUMENT PREDICTION

        # Reduce features
        self.reduced_feature_dim = 32
        self.reduce_features = nn.Sequential(
            nn.LayerNorm(self.out_channels),     # normalize 256-d features
            nn.Linear(self.out_channels, self.reduced_feature_dim),    # reduce to 32
            nn.ReLU()
        )


        # Attention over nodes
        input_dim = 2 * self.reduced_feature_dim
        self.node_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4, batch_first=True)


        # Simple FCNs for argument prediction
        input_dim = (
                        2 * self.reduced_feature_dim +      # node feature pair
                        1 * self.reduced_feature_dim +      # delta
                        1 +                                 # sim
                        self.num_classes                    # action info
                    )                   


        # Create the MLP using the dynamically constructed layers
        self.args_mlp1 = self.create_mlp_from_config(input_dim, mlp_config['layers'])
        self.args_mlp2 = self.create_mlp_from_config(input_dim, mlp_config['layers'])

        if self.argument_arguments_prediction_method == 'set_prediction': # Only use one head
            self.args_mlp = self.args_mlp1
            self.args_mlp2 = None



    def forward(self, Xs, As):
        """
        Forward pass of ST-GCN model.

        Args:
            Xs (torch.Tensor): Input node features of shape (N, T, V, C)
            As (torch.Tensor): Adjacency matrices of shape (N, T, K, V, V)

        Returns:
            pred_action_name (torch.Tensor): Action class logits, shape (N, T-1, num_classes)
            [pred_arg1, pred_arg2] (list of torch.Tensor): Argument logits per node
            _debug_dict (dict): Intermediate tensors for debugging and analysis
        """
        batch_size = Xs.shape[0]
        assert Xs.shape[0] == batch_size and Xs.shape[2] == self.num_nodes and Xs.shape[3] == self.in_channels, f"Input shape {Xs.shape} does not match expected shape with batch_size {batch_size}, num_nodes {self.num_nodes}, and in_channels {self.in_channels}"
        assert As.shape[0] == batch_size, f"Batch size {As.shape[0]} does not match expected batch size {batch_size}"
        assert As.shape[2:] == (self.spatial_kernel_size, self.num_nodes, self.num_nodes), f"Adjacency matrix shape {As.shape[2:]} does not match expected shape ({self.spatial_kernel_size}, {self.num_nodes}, {self.num_nodes})"

        # Data normalization
        N, T, V, C = Xs.shape
        Xs = Xs.permute(0, 3, 1, 2)  # (N, C, T, V)
        Xs = self.data_bn(Xs)
        input_features = snapshot(Xs)

        assert Xs.shape[0] == batch_size and Xs.shape[1] == self.in_channels and Xs.shape[3] == self.num_nodes, f"Input shape {Xs.shape} does not match expected shape with batch_size {batch_size}, in_channels {self.in_channels}, and num_nodes {self.num_nodes}"


        # Forward through ST-GCN layers
        Xs_aggregated = Xs
        for gcn in self.st_gcn_networks:
            Xs_aggregated, _ = gcn(Xs_aggregated, As)

        assert Xs_aggregated.shape[0] == batch_size, f"Batch size {Xs_aggregated.shape[0]} does not match expected batch size {batch_size}"
        assert Xs_aggregated.shape[1] == self.out_channels, f"Output channels {Xs_aggregated.shape[1]} do not match expected out_channels {self.out_channels}"
        assert Xs_aggregated.shape[3] == self.num_nodes, f"Number of nodes {Xs_aggregated.shape[3]} does not match expected num_nodes {self.num_nodes}"
        aggregated_features = snapshot(Xs_aggregated)


        # Pooling over nodes
        Xs_pooled = None
        if self.pooling_method == 'avg':
            # Pool over nodes, keeping T
            Xs_pooled = F.avg_pool2d(Xs_aggregated, (1, Xs_aggregated.size(3)))  # (N, C, T, 1)
            Xs_pooled = Xs_pooled.squeeze(-1)  # (N, C, T)

            # Transpose for FCN
            Xs_pooled = Xs_pooled.permute(0, 2, 1)  # (N, T, C)
        elif self.pooling_method == 'attention':
            # Permute to shape (N, T, V, C) for attention calculation
            Xs_pooled = Xs_aggregated.permute(0, 2, 3, 1)  # (N, T, V, C)
            # Apply attention FC to get attention scores per node
            attn_scores = self.attn_fc(Xs_pooled)  # (N, T, V, 1)
            # Apply softmax over nodes (V) to get attention weights
            attn_weights = torch.softmax(attn_scores, dim=2)  # (N, T, V, 1)
            # Compute weighted sum over nodes to get scene-level embedding
            Xs_pooled = (attn_weights * Xs_pooled).sum(dim=2)  # (N, T, C)
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling_method}")
        
        assert Xs_pooled.shape[0] == batch_size, f"Batch size {Xs_pooled.shape[0]} does not match expected batch size {batch_size}"
        assert Xs_pooled.shape[2] == self.out_channels, f"Output channels {Xs_pooled.shape[2]} do not match expected out_channels {self.out_channels}"


        if self.use_lstm:
            # **LSTM layer to process temporal dynamics**
            Xs_pooled, _ = self.lstm(Xs_pooled)

        assert Xs_pooled.shape[0] == batch_size, f"Batch size {Xs_pooled.shape[0]} does not match expected batch size {batch_size}"
        assert Xs_pooled.shape[2] == self.out_channels, f"Output channels {Xs_pooled.shape[2]} do not match expected out_channels {self.out_channels}"
        


        # ------
        # FINAL ACTION_NAME CLASSIFICATION HEAD FOR EACH GRAPH IN THE SEQUENCE
        pooled_features = None
        if self.action_name_prediction_method == 'whole_sequence':
            pred_action_name = self.action_name_fcn(Xs_pooled)  # Predicting the action_name using aggregated node features
            pred_action_name = pred_action_name[:, 1:, :]  # Shape: (1, sequence_length-1, num_classes)

        elif self.action_name_prediction_method == 'pairwise': # Pairwise prediction between graph i and graph i+1

            # Create (i, i+1) pairs
            g_i = Xs_pooled[:,:-1]
            g_i1 = Xs_pooled[:,1:]
            pair_features = torch.cat([g_i, g_i1], dim=-1)
            pooled_features = pair_features

            # Classify relationship between each consecutive graph pair
            pred_action_name = self.action_name_fcn(pair_features)

             
        assert pred_action_name.shape[0] == batch_size, f"Batch size {pred_action_name.shape[0]} does not match expected batch size {batch_size}"
        assert pred_action_name.shape[2] == self.num_classes, f"Number of classes {pred_action_name.shape[2]} does not match expected num_classes {self.num_classes}"

        
        # ------
        # FINAL ARGUMENT CLASSIFICATION HEAD FOR EACH GRAPH IN THE SEQUENCE
        action_name_info = pred_action_name
        N, C, T, V = Xs_aggregated.shape
        # Shape: (N, T, V, C)
        Xs_aggregated = Xs_aggregated.permute(0, 2, 3, 1).contiguous()
        


        x_t = Xs_aggregated[:, :-1, :, :]     # [N, T-1, V, C]
        x_t1 = Xs_aggregated[:, 1:, :, :]     # [N, T-1, V, C]
        N, T, V, C = x_t.shape

        # 1. Reduce
        x_t_reduced   = self.reduce_features(x_t)   # [N, T, V, C_r]    (C_r = 32, new reduced feature dim)
        x_t1_reduced  = self.reduce_features(x_t1)  # [N, T, V, C_r]

        # 2. Stack
        x_stack = torch.cat([x_t_reduced, x_t1_reduced], dim=-1)  # [N, T, V, 2*C_r]

        # 3. Attention over nodes
        x_attn_in = x_stack.view(N*T, V, 64)
        x_attn_out, _ = self.node_attention(x_attn_in, x_attn_in, x_attn_in)  # [N*T, V, 2*C_r]
        NT, V, D = x_attn_out.shape
        x_attn_out = x_attn_out.view(N, T, V, D)  # [N, T, V, 2*C_r]

        # 4. Compute delta, sim
        delta = x_t1_reduced - x_t_reduced  # [N, T, V, C_r]
        sim = (x_t_reduced * x_t1_reduced).sum(dim=-1, keepdim=True)  # [N, T, V, 1]

        # 5. Expand action logits
        action_info = pred_action_name.unsqueeze(2).expand(-1, -1, V, -1)  # [N, T, V, num_classes]
        action_info = torch.softmax(action_info, dim=-1)

        # 6. Final input
        x_final = torch.cat([x_attn_out, delta, sim, action_info], dim=-1) # [N, T, V, 2*C_r + C_r + 1 + num_classes]

        # 7. Classify
        if self.argument_arguments_prediction_method == 'set_prediction':
            pred_arg1 = self.args_mlp(x_final).squeeze(-1)
            pred_arg2 = torch.zeros_like(pred_arg1)  # Dummy tensor for arg2
        elif self.argument_arguments_prediction_method == 'two_head_prediction':
            pred_arg1 = self.args_mlp1(x_final).squeeze(-1)  # [N, T, V]
            pred_arg2 = self.args_mlp2(x_final).squeeze(-1)  # [N, T, V]
            

        # add information for debugging
        _debug_dict = {
            'x_t': x_t,
            'x_t1': x_t1,
            'x_t_reduced': x_t_reduced,
            'x_t1_reduced': x_t1_reduced,
            'x_stack': x_stack,
            'x_attn_in': x_attn_in,
            'x_attn_out': x_attn_out,
            'delta': delta,
            'sim': sim,
            'action_info': action_info,
            'x_final': x_final,
            'x_pooled_paired': pooled_features,
            'x_aggregated_features': aggregated_features,
            'x_input_features': input_features,
            'pred_arg1': pred_arg1,
            'pred_arg2': pred_arg2
        }
        
        # Return predictions
        return pred_action_name, [pred_arg1, pred_arg2], _debug_dict  # Action predictions per time step
    















    def create_mlp_from_config(self, input_dim, config):
        """
        Dynamically creates an MLP (multi-layer perceptron) from a layer config list.

        Args:
            input_dim (int): Input feature dimension for the first layer
            config (list[dict]): List of layer specifications from YAML config, e.g.:
                                [{'type': 'Linear', 'out_features': 128}, {'type': 'ReLU'}, ...]

        Returns:
            nn.Sequential: Instantiated MLP as a PyTorch Sequential module
        """
        layers = []
        for layer_config in config:
            layer_type = layer_config['type']
            if layer_type == 'Linear':
                layers.append(nn.Linear(input_dim, layer_config['out_features']))
                input_dim = layer_config['out_features']
            elif layer_type == 'ReLU':
                layers.append(nn.ReLU())
            elif layer_type == 'Dropout':
                layers.append(nn.Dropout(layer_config['p']))
            elif layer_type == 'LayerNorm':
                layers.append(nn.LayerNorm(input_dim))
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")
        return nn.Sequential(*layers)









'''
self.args_mlp1 = nn.Sequential(
    nn.LayerNorm(input_dim),
    nn.Linear(input_dim, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)

self.st_gcn_networks = nn.ModuleList((
    STGCNLayer(in_channels, 64, kernel_size, 1, residual=False),
    STGCNLayer(64, 64, kernel_size, 1),
    STGCNLayer(64, 64, kernel_size, 1),
    STGCNLayer(64, 64, kernel_size, 1),
    STGCNLayer(64, 128, kernel_size, 1),
    STGCNLayer(128, 128, kernel_size, 1),
    STGCNLayer(128, 128, kernel_size, 1),
    STGCNLayer(128, 256, kernel_size, 1),
    STGCNLayer(256, 256, kernel_size, 1),
    STGCNLayer(256, 256, kernel_size, 1),
))

# Attention FC layer to learn node importance
self.attn_fc = nn.Linear(in_features=256, out_features=1)  # Maps node features to attention scores

# fcn for prediction
self.action_name_fcn = nn.Sequential(
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, self.num_classes),
)   
'''

def snapshot(t: torch.Tensor, to_cpu: bool = True):
    # Detach from graph and clone data so it won't be freed/modified.
    with torch.no_grad():
        out = t.detach().clone()
        if to_cpu:
            out = out.cpu()
    return out
