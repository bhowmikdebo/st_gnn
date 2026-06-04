
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Temporal Attention Module ────────────────────────────────────────────────

class TemporalAttention(nn.Module):

    def __init__(self, features: int, window: int):
        super().__init__()
        # Small network that scores each time step
        self.score_net = nn.Sequential(
            nn.Linear(features, 32),
            nn.Tanh(),
            nn.Linear(32, 1),          # one score per time step
        )

    def forward(self, x):
        # x: [batch, N, W, F]
        batch, N, W, feat_dim = x.shape

        # Flatten batch+zone dims to compute scores for each time step
        x_flat = x.view(batch * N, W, feat_dim)          # [B*N, W, F]
        scores  = self.score_net(x_flat)           # [B*N, W, 1]
        scores  = scores.squeeze(-1)               # [B*N, W]

        attn_weights = F.softmax(scores, dim=-1)   # [B*N, W]  ← sums to 1

        # Weighted sum: attend over the window dimension
        x_attended = torch.bmm(
            attn_weights.unsqueeze(1),             # [B*N, 1, W]
            x_flat                                 # [B*N, W, F]
        ).squeeze(1)                               # [B*N, F]

        out          = x_attended.view(batch, N, feat_dim)
        attn_weights = attn_weights.view(batch, N, W)

        return out, attn_weights


# ─── Feature-Aware Graph Convolution ─────────────────────────────────────────

class FeatureAwareGraphConv(nn.Module):

    def __init__(self, features_in: int, features_out: int):
        super().__init__()
        self.W = nn.Linear(features_in, features_out, bias=False)
        self.bias = nn.Parameter(torch.zeros(features_out))

    def forward(self, h, adj_norm):
        # h:        [batch, N, F_in]
        # adj_norm: [N, N]

        # Linear transform of node features
        support = self.W(h)                                # [batch, N, F_out]

        # Graph aggregation: each node collects from neighbours
        # adj_norm is [N, N], support is [batch, N, F_out]
        out = torch.matmul(adj_norm, support) + self.bias  # [batch, N, F_out]

        return F.relu(out), adj_norm   # return adj as spatial attention proxy


# ─── Full ST-GNN Model ────────────────────────────────────────────────────────

class ST_GNN(nn.Module):
    
    def __init__(self,
                 n_zones:    int = 15,
                 n_features: int = 6,
                 window:     int = 7,
                 hidden_dim: int = 64,
                 n_layers:   int = 2):
        super().__init__()

        self.n_zones    = n_zones
        self.n_features = n_features
        self.window     = window
        self.hidden_dim = hidden_dim

        # ── Layer 1: Temporal attention (compress window → single vector) ────
        self.temporal_attn = TemporalAttention(n_features, window)

        # ── Layer 2+: Stacked spatial graph convolutions ──────────────────────
        self.spatial_layers = nn.ModuleList()
        in_dim = n_features
        for _ in range(n_layers):
            self.spatial_layers.append(FeatureAwareGraphConv(in_dim, hidden_dim))
            in_dim = hidden_dim

        # ── Layer 3: Prediction head ──────────────────────────────────────────
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),           # one prediction per zone
        )

        # Store attention weights for explainability (filled during forward pass)
        self.last_temporal_weights = None   # [batch, N, W]
        self.last_spatial_weights  = None   # [N, N]

    def forward(self, x, adj_norm):
       
        # ── Step 1: Temporal attention ────────────────────────────────────────
        h, temp_weights = self.temporal_attn(x)
        # h: [batch, N, features]

        # Save for explainability
        self.last_temporal_weights = temp_weights.detach()

        # ── Step 2: Spatial graph convolutions ───────────────────────────────
        for layer in self.spatial_layers:
            h, spatial_w = layer(h, adj_norm)
            self.last_spatial_weights = spatial_w.detach()

        # ── Step 3: Predict one value per zone ───────────────────────────────
        out = self.predictor(h).squeeze(-1)   # [batch, N]

        return out

    def get_attention_weights(self):
        return {
            "temporal": self.last_temporal_weights,   # which past days matter
            "spatial":  self.last_spatial_weights,    # which zones influence which
        }


# ─── Normalised Adjacency (pre-compute once, reuse every forward pass) ────────

def normalise_adjacency(adj: "np.ndarray") -> torch.Tensor:
   
    import numpy as np

    A_tilde = adj + np.eye(adj.shape[0])                     # add self-loops
    D       = np.diag(A_tilde.sum(axis=1))                   # degree matrix
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-9))  # D^(-1/2)
    A_norm  = D_inv_sqrt @ A_tilde @ D_inv_sqrt

    return torch.FloatTensor(A_norm)
