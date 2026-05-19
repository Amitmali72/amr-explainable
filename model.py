"""
CNN-GRU-GNN architecture for Automatic Modulation Recognition.

This implementation keeps deployment simple by replacing the small subset of
PyTorch Geometric used during training with plain PyTorch equivalents. Parameter
names match the training-time modules so existing checkpoints can be loaded with
``model.load_state_dict(torch.load(...))``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


MODULATION_CLASSES = [
    "8PSK",
    "AM-DSB",
    "AM-SSB",
    "BPSK",
    "CPFSK",
    "GFSK",
    "PAM4",
    "QAM16",
    "QAM64",
    "QPSK",
    "WBFM",
]


class SimpleData:
    """Minimal stand-in for torch_geometric.data.Data."""

    def __init__(self, x: torch.Tensor, edge_index: torch.Tensor):
        self.x = x
        self.edge_index = edge_index


class GCNConv(nn.Module):
    """Plain-PyTorch GCNConv compatible with PyTorch Geometric state dicts."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        x = self.lin(x)

        adjacency = torch.zeros(num_nodes, num_nodes, device=x.device, dtype=x.dtype)
        adjacency[edge_index[0], edge_index[1]] = 1.0
        adjacency += torch.eye(num_nodes, device=x.device, dtype=x.dtype)

        degree = adjacency.sum(dim=1)
        degree_inv_sqrt = degree.pow(-0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
        norm = degree_inv_sqrt[:, None] * adjacency * degree_inv_sqrt[None, :]
        return norm @ x + self.bias


def global_mean_pool(x: torch.Tensor, batch: torch.Tensor, size: int | None = None) -> torch.Tensor:
    """Mean-pool node embeddings into graph embeddings."""

    if size is None:
        size = int(batch.max().item()) + 1

    out = torch.zeros(size, x.size(1), device=x.device, dtype=x.dtype)
    count = torch.zeros(size, 1, device=x.device, dtype=x.dtype)
    index = batch.unsqueeze(1).expand_as(x)
    out.scatter_add_(0, index, x)
    count.scatter_add_(0, batch.unsqueeze(1), torch.ones(batch.size(0), 1, device=x.device, dtype=x.dtype))
    return out / count.clamp(min=1)


def build_edge_index(num_nodes: int = 128) -> torch.Tensor:
    """Build a bidirectional temporal chain graph for I/Q samples."""

    edges = []
    for idx in range(num_nodes - 1):
        edges.append([idx, idx + 1])
        edges.append([idx + 1, idx])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def batch_edge_index(edge_index: torch.Tensor, batch_size: int, num_nodes: int) -> torch.Tensor:
    """Offset a single graph edge_index for batched graph inference."""

    return torch.cat([edge_index + sample_idx * num_nodes for sample_idx in range(batch_size)], dim=1)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, data: SimpleData, batch: torch.Tensor) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)


class CNN_GRU_GNN(nn.Module):
    """Hybrid AMR classifier: CNN feature extractor, Bi-GRU, temporal GCN."""

    NUM_CLASSES = len(MODULATION_CLASSES)
    MODULATION_CLASSES = MODULATION_CLASSES

    def __init__(self, num_classes: int = NUM_CLASSES):
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

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)
        sequence = features.permute(0, 2, 1)
        sequence, _ = self.gru(sequence)

        batch_size, num_nodes, channels = sequence.shape
        nodes = sequence.reshape(batch_size * num_nodes, channels)
        position = torch.arange(num_nodes, device=x.device, dtype=nodes.dtype) / num_nodes
        position = position.unsqueeze(0).repeat(batch_size, 1).reshape(-1, 1)
        nodes = torch.cat([nodes, position], dim=1)

        batched_edges = batch_edge_index(edge_index, batch_size, num_nodes)
        graph = SimpleData(x=nodes, edge_index=batched_edges)
        batch_vector = torch.arange(batch_size, device=x.device).repeat_interleave(num_nodes)
        return self.gnn(graph, batch_vector)
