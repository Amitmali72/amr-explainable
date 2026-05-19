"""Gradient-based explainability for AMR signal classification."""

from __future__ import annotations

import numpy as np
import torch

from inference import normalize_signal


def compute_saliency(model, signal: np.ndarray, edge_index: torch.Tensor, device: torch.device) -> tuple[np.ndarray, int]:
    """Return normalized absolute input gradients for I and Q channels."""

    model.eval()
    model.zero_grad(set_to_none=True)

    normalized = normalize_signal(signal)
    input_tensor = torch.tensor(normalized, dtype=torch.float32, device=device).unsqueeze(0)
    input_tensor.requires_grad_(True)

    logits = model(input_tensor, edge_index)
    pred_idx = int(torch.argmax(logits, dim=1).item())
    logits[0, pred_idx].backward()

    gradients = input_tensor.grad.detach().abs().squeeze(0).cpu().numpy()
    saliency = gradients / (float(gradients.max()) + 1e-8)
    return saliency.astype(np.float32), pred_idx


def channel_importance(saliency: np.ndarray) -> dict[str, float]:
    """Aggregate saliency into I/Q feature-importance scores."""

    totals = np.sum(saliency, axis=1)
    denom = float(np.sum(totals)) + 1e-8
    return {
        "I channel": float(totals[0] / denom),
        "Q channel": float(totals[1] / denom),
    }
