"""
AMR Model Architecture — CNN-GRU-GNN
Drop-in replacements for torch_geometric components so that deployment
requires only plain PyTorch (no torch_geometric wheel needed).
The parameter names are kept identical so that existing .pth checkpoints
load without any key remapping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── lightweight replacements for torch_geometric ──────────────────────

class SimpleData:
    """Minimal stand-in for torch_geometric.data.Data."""
    def __init__(self, x=None, edge_index=None):
        self.x = x
        self.edge_index = edge_index


class GCNConv(nn.Module):
    """
    Drop-in replacement for torch_geometric.nn.GCNConv.
    Keeps the same parameter names (lin.weight, bias) so that
    state-dicts trained with the real GCNConv load unchanged.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, edge_index):
        num_nodes = x.size(0)

        # linear projection
        x = self.lin(x)

        # build adjacency (with self-loops) from edge_index
        adj = torch.zeros(num_nodes, num_nodes, device=x.device)
        adj[edge_index[0], edge_index[1]] = 1.0
        adj += torch.eye(num_nodes, device=x.device)

        # symmetric normalisation  D^{-½} A D^{-½}
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        D = torch.diag(deg_inv_sqrt)
        norm_adj = D @ adj @ D

        return norm_adj @ x + self.bias


def global_mean_pool(x, batch, size=None):
    """Drop-in replacement for torch_geometric.nn.global_mean_pool."""
    if size is None:
        size = int(batch.max().item()) + 1
    out = torch.zeros(size, x.size(1), device=x.device)
    count = torch.zeros(size, 1, device=x.device)
    idx = batch.unsqueeze(1).expand_as(x)
    out.scatter_add_(0, idx, x)
    count.scatter_add_(0, batch.unsqueeze(1),
                       torch.ones(batch.size(0), 1, device=x.device))
    return out / count.clamp(min=1)


# ── graph helpers ─────────────────────────────────────────────────────

def build_edge_index(n: int):
    edges = []
    for i in range(n - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    return torch.tensor(edges).t().contiguous()


def batch_edge_index(edge_index, B, N):
    return torch.cat([edge_index + i * N for i in range(B)], dim=1)


# ── model components ──────────────────────────────────────────────────

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

    def forward(self, x):
        return self.net(x)


class GNNModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = GCNConv(385, 128)
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, 64)
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, data, batch):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)


class CNN_GRU_GNN(nn.Module):
    """Full hybrid model: CNN → Bi-GRU → GCN graph classifier."""
    NUM_CLASSES = 11
    MODULATION_CLASSES = [
        '8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK',
        'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM',
    ]

    def __init__(self, num_classes: int = 11):
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=192,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.gnn = GNNModel(num_classes)

    def forward(self, x, edge_index):
        features = self.cnn(x)                   # (B, 128, 128)
        x = features.permute(0, 2, 1)            # (B, 128, 128)
        x, _ = self.gru(x)                       # (B, 128, 384)
        B, N, C = x.shape
        x = x.reshape(B * N, C)
        pos = torch.arange(N, device=x.device).float() / N
        pos = pos.unsqueeze(0).repeat(B, 1).reshape(-1, 1)
        x = torch.cat([x, pos], dim=1)           # (B*N, 385)
        edge = batch_edge_index(edge_index, B, N)
        data = SimpleData(x=x, edge_index=edge)
        batch_vec = torch.arange(B, device=x.device).repeat_interleave(N)
        return self.gnn(data, batch_vec)
