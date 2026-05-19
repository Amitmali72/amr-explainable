"""Adaptive modulation and SNR-aware routing logic."""

from __future__ import annotations


SPECTRAL_EFFICIENCY = {
    "BPSK": 1.0,
    "QPSK": 2.0,
    "QAM16": 4.0,
    "QAM64": 6.0,
}


def route_model(snr: float) -> tuple[str, str, str]:
    """Select the best checkpoint for the current SNR regime."""

    if snr >= 0:
        return (
            "high_snr",
            "High-SNR Model",
            "Clean-channel specialist selected because the signal is at or above 0 dB.",
        )
    return (
        "robust",
        "Robust Fine-Tuned Model",
        "Noise-hardened model selected because the signal is below 0 dB.",
    )


def recommend_modulation(snr: float) -> dict[str, str | float]:
    """Recommend an adaptive modulation mode from current SNR."""

    if snr < 0:
        modulation = "BPSK"
        profile = "maximum robustness"
        explanation = "The channel is noise dominated, so binary phase states preserve link reliability."
    elif snr < 8:
        modulation = "QPSK"
        profile = "balanced reliability"
        explanation = "The signal is usable but still noisy, so QPSK doubles throughput while keeping phase spacing wide."
    elif snr < 15:
        modulation = "QAM16"
        profile = "high-throughput mode"
        explanation = "The channel has enough margin for amplitude and phase levels without jumping to the densest constellation."
    else:
        modulation = "QAM64"
        profile = "maximum spectral efficiency"
        explanation = "The clean channel can support dense constellation points and the highest throughput option."

    return {
        "modulation": modulation,
        "spectral_efficiency": SPECTRAL_EFFICIENCY[modulation],
        "profile": profile,
        "explanation": explanation,
    }
