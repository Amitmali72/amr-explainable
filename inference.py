"""Model loading, signal preparation, and AMR inference utilities."""

from __future__ import annotations

import json
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from adaptive import route_model
from model import CNN_GRU_GNN, MODULATION_CLASSES, build_edge_index


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample_signals"


@dataclass(frozen=True)
class SignalSample:
    signal: np.ndarray
    modulation: str
    snr: float
    source: str


@dataclass(frozen=True)
class PredictionResult:
    predicted_modulation: str
    confidence: float
    probabilities: dict[str, float]
    logits: np.ndarray
    active_model_key: str
    active_model_name: str
    routing_message: str


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Normalize a 2x128 I/Q sample to a stable inference range."""

    signal = np.asarray(signal, dtype=np.float32)
    if signal.shape != (2, 128):
        raise ValueError(f"Expected signal shape (2, 128), received {signal.shape}.")

    max_abs = float(np.max(np.abs(signal)))
    if max_abs < 1e-8:
        return signal
    return signal / max_abs


def parse_signal_json(payload: dict[str, Any], source: str = "uploaded JSON") -> SignalSample:
    """Parse the supported deployment JSON format."""

    required = {"modulation", "snr", "iSamples", "qSamples"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Missing required JSON fields: {', '.join(missing)}")

    i_samples = np.asarray(payload["iSamples"], dtype=np.float32)
    q_samples = np.asarray(payload["qSamples"], dtype=np.float32)
    if i_samples.size != 128 or q_samples.size != 128:
        raise ValueError("iSamples and qSamples must each contain exactly 128 values.")

    signal = np.stack([i_samples, q_samples], axis=0)
    return SignalSample(
        signal=normalize_signal(signal),
        modulation=str(payload["modulation"]),
        snr=float(payload["snr"]),
        source=source,
    )


def load_json_sample(path: Path) -> SignalSample:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return parse_signal_json(payload, source=path.name)


def load_available_samples() -> list[SignalSample]:
    """Load bundled JSON samples, with compatibility fallbacks for this repo."""

    samples: list[SignalSample] = []
    for folder in [SAMPLE_DIR, PROJECT_ROOT / "app" / "training"]:
        if folder.exists():
            for path in sorted(folder.glob("sample_*.json")):
                try:
                    samples.append(load_json_sample(path))
                except (ValueError, json.JSONDecodeError):
                    continue

    signals_path = PROJECT_ROOT / "deploy" / "sample_signals.npy"
    metadata_path = PROJECT_ROOT / "deploy" / "sample_metadata.pkl"
    if signals_path.exists() and metadata_path.exists():
        try:
            signals = np.load(signals_path)
            with metadata_path.open("rb") as handle:
                metadata = pickle.load(handle)
            for idx, meta in enumerate(metadata):
                samples.append(
                    SignalSample(
                        signal=normalize_signal(signals[idx]),
                        modulation=str(meta.get("mod", "unknown")),
                        snr=float(meta.get("snr", 0)),
                        source=f"RadioML demo sample {idx}",
                    )
                )
        except Exception:
            pass

    if samples:
        return samples
    return [generate_synthetic_signal("QPSK", snr=10, seed=7)]


def generate_synthetic_signal(modulation: str = "QPSK", snr: float = 10, seed: int | None = None) -> SignalSample:
    """Generate a deterministic 128-sample I/Q demo signal with AWGN."""

    rng = np.random.default_rng(seed)
    num_samples = 128
    modulation = modulation.upper()

    if modulation == "BPSK":
        symbols = rng.choice([-1 + 0j, 1 + 0j], size=num_samples)
    elif modulation == "QPSK":
        phase = rng.integers(0, 4, size=num_samples) * (math.pi / 2) + math.pi / 4
        symbols = np.exp(1j * phase)
    elif modulation == "8PSK":
        phase = rng.integers(0, 8, size=num_samples) * (math.pi / 4)
        symbols = np.exp(1j * phase)
    elif modulation == "QAM16":
        levels = np.array([-3, -1, 1, 3], dtype=np.float32)
        symbols = rng.choice(levels, num_samples) + 1j * rng.choice(levels, num_samples)
        symbols = symbols / np.sqrt(np.mean(np.abs(symbols) ** 2))
    elif modulation == "QAM64":
        levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7], dtype=np.float32)
        symbols = rng.choice(levels, num_samples) + 1j * rng.choice(levels, num_samples)
        symbols = symbols / np.sqrt(np.mean(np.abs(symbols) ** 2))
    else:
        t = np.linspace(0, 1, num_samples, endpoint=False)
        phase = 2 * np.pi * (6 * t + 0.8 * np.sin(2 * np.pi * 5 * t))
        symbols = np.exp(1j * phase)

    shaped = np.convolve(symbols, np.ones(3) / 3, mode="same")
    noisy = add_awgn(np.stack([shaped.real, shaped.imag]).astype(np.float32), snr)
    return SignalSample(signal=normalize_signal(noisy), modulation=modulation, snr=float(snr), source="synthetic generator")


def add_awgn(signal: np.ndarray, target_snr_db: float, seed: int | None = None) -> np.ndarray:
    """Apply additive white Gaussian noise to a 2x128 signal."""

    rng = np.random.default_rng(seed)
    signal = np.asarray(signal, dtype=np.float32)
    power = float(np.mean(signal**2))
    if power < 1e-10:
        return signal

    snr_linear = 10 ** (target_snr_db / 10)
    noise_power = power / snr_linear
    noise = rng.normal(0, np.sqrt(noise_power), size=signal.shape).astype(np.float32)
    return signal + noise


def load_amr_models(device_name: str = "cpu") -> tuple[dict[str, CNN_GRU_GNN], torch.device, torch.Tensor]:
    """Load both deployment checkpoints for CPU inference."""

    device = torch.device(device_name)
    model_paths = {
        "high_snr": MODEL_DIR / "best_model.pth",
        "robust": MODEL_DIR / "final_finetuned_model.pth",
    }
    missing = [str(path) for path in model_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model checkpoint(s): "
            + ", ".join(missing)
            + ". Copy trained .pth files into the models/ directory."
        )

    models: dict[str, CNN_GRU_GNN] = {}
    for key, path in model_paths.items():
        model = CNN_GRU_GNN(num_classes=len(MODULATION_CLASSES)).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models[key] = model

    edge_index = build_edge_index(128).to(device)
    return models, device, edge_index


def predict_modulation(
    signal: np.ndarray,
    snr: float,
    models: dict[str, CNN_GRU_GNN],
    device: torch.device,
    edge_index: torch.Tensor,
) -> PredictionResult:
    """Route to the correct AMR model and run batch-size-1 inference."""

    model_key, model_name, routing_message = route_model(snr)
    model = models[model_key]
    normalized = normalize_signal(signal)

    with torch.no_grad():
        tensor = torch.tensor(normalized, dtype=torch.float32, device=device).unsqueeze(0)
        logits = model(tensor, edge_index)
        probabilities = F.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

    pred_idx = int(np.argmax(probabilities))
    probability_map = {
        modulation: float(probabilities[idx]) for idx, modulation in enumerate(MODULATION_CLASSES)
    }
    return PredictionResult(
        predicted_modulation=MODULATION_CLASSES[pred_idx],
        confidence=float(probabilities[pred_idx]),
        probabilities=probability_map,
        logits=logits.squeeze(0).detach().cpu().numpy(),
        active_model_key=model_key,
        active_model_name=model_name,
        routing_message=routing_message,
    )


def random_sample(samples: list[SignalSample]) -> SignalSample:
    return random.choice(samples)
