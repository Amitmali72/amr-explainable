from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# from adaptive import recommend_modulation
# from inference import (
#     SignalSample,
#     generate_synthetic_signal,
#     load_amr_models,
#     load_available_samples,
#     normalize_signal,
#     parse_signal_json,
#     predict_modulation,
# )
# from model import MODULATION_CLASSES
# from saliency import channel_importance, compute_saliency


PAGE_TITLE = "Adaptive AMR Control System"
# PDF_CHAT_PATH = Path(__file__).resolve().parent / "app" / "explainability" / "pdf_chat.py"



st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="AMR",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --bg: #070912;
            --panel: rgba(17, 24, 47, 0.74);
            --panel-2: rgba(24, 32, 66, 0.62);
            --cyan: #35e7ff;
            --blue: #4c7dff;
            --violet: #a855f7;
            --magenta: #ff4fd8;
            --muted: #9aa9c7;
            --text: #edf4ff;
            --border: rgba(92, 133, 255, 0.26);
        }

        html, body, [class*="css"] {
            font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 20% 0%, rgba(76, 125, 255, 0.26), transparent 26rem),
                radial-gradient(circle at 85% 10%, rgba(168, 85, 247, 0.2), transparent 28rem),
                linear-gradient(135deg, #070912 0%, #0d1020 45%, #080914 100%);
            color: var(--text);
        }

        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }

        [data-testid="stSidebar"] {
            background: rgba(8, 11, 24, 0.92);
            border-right: 1px solid rgba(76, 125, 255, 0.25);
        }

        h1, h2, h3, h4, p, span, label {
            color: var(--text);
            letter-spacing: 0;
        }

        .hero {
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(80, 140, 255, 0.35);
            border-radius: 18px;
            padding: 28px 30px;
            margin-bottom: 20px;
            background:
                linear-gradient(135deg, rgba(23, 32, 66, 0.86), rgba(14, 18, 40, 0.72)),
                repeating-linear-gradient(90deg, rgba(53, 231, 255, 0.06) 0 1px, transparent 1px 48px);
            box-shadow: 0 0 40px rgba(53, 231, 255, 0.12), inset 0 0 52px rgba(168, 85, 247, 0.06);
            backdrop-filter: blur(14px);
        }

        .hero::after {
            content: "";
            position: absolute;
            inset: auto 0 0 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--cyan), var(--violet), transparent);
            box-shadow: 0 0 16px rgba(53, 231, 255, 0.9);
        }

        .hero-kicker {
            color: var(--cyan);
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            margin-bottom: 8px;
        }

        .hero h1 {
            font-size: clamp(2.25rem, 5vw, 4.9rem);
            line-height: 0.95;
            margin: 0 0 12px 0;
            max-width: 980px;
        }

        .hero-sub {
            max-width: 940px;
            color: #b8c7ef;
            font-size: 1.04rem;
            line-height: 1.7;
            margin: 0;
        }

        .glass-card {
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 18px;
            background: linear-gradient(145deg, rgba(18, 25, 54, 0.75), rgba(13, 18, 39, 0.58));
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.04);
            backdrop-filter: blur(16px);
            min-height: 100%;
        }

        .metric-card {
            border: 1px solid rgba(53, 231, 255, 0.24);
            border-radius: 14px;
            padding: 15px 16px;
            background: rgba(12, 17, 38, 0.72);
            box-shadow: inset 0 0 22px rgba(76, 125, 255, 0.06);
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.76rem;
            text-transform: uppercase;
            font-weight: 800;
            letter-spacing: 0.12em;
        }

        .metric-value {
            color: var(--text);
            font-size: 1.65rem;
            font-weight: 800;
            margin-top: 6px;
            line-height: 1.05;
        }

        .metric-note {
            color: #8da0c8;
            font-size: 0.84rem;
            margin-top: 8px;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            border: 1px solid rgba(53, 231, 255, 0.36);
            border-radius: 999px;
            padding: 7px 12px;
            color: #dffaff;
            background: rgba(53, 231, 255, 0.08);
            box-shadow: 0 0 18px rgba(53, 231, 255, 0.12);
            font-weight: 700;
            font-size: 0.86rem;
        }

        .pulse-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--cyan);
            box-shadow: 0 0 10px var(--cyan);
            animation: pulse 1.45s infinite ease-in-out;
        }

        @keyframes pulse {
            0%, 100% { transform: scale(0.9); opacity: 0.7; }
            50% { transform: scale(1.25); opacity: 1; }
        }

        .prediction {
            border: 1px solid rgba(255, 79, 216, 0.32);
            border-radius: 16px;
            padding: 22px;
            background: linear-gradient(145deg, rgba(78, 36, 140, 0.38), rgba(14, 18, 40, 0.78));
            box-shadow: 0 0 36px rgba(168, 85, 247, 0.18), inset 0 0 30px rgba(255, 79, 216, 0.05);
        }

        .prediction .value {
            font-size: clamp(2.3rem, 6vw, 4.4rem);
            font-weight: 900;
            line-height: 0.95;
            color: #ffffff;
            text-shadow: 0 0 22px rgba(255, 79, 216, 0.48);
        }

        .stButton > button, .stDownloadButton > button {
            border: 1px solid rgba(53, 231, 255, 0.38);
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            color: white;
            border-radius: 10px;
            font-weight: 800;
            min-height: 42px;
            box-shadow: 0 0 22px rgba(76, 125, 255, 0.2);
        }

        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: rgba(53, 231, 255, 0.72);
            box-shadow: 0 0 30px rgba(53, 231, 255, 0.28);
            transform: translateY(-1px);
        }

        div[data-testid="stMetricValue"] {
            color: var(--cyan);
            font-weight: 900;
        }

        .chat-bubble {
            border: 1px solid rgba(76, 125, 255, 0.22);
            border-radius: 12px;
            padding: 10px 12px;
            background: rgba(19, 27, 56, 0.72);
            color: #dce7ff;
            font-size: 0.9rem;
            line-height: 1.45;
            margin-bottom: 10px;
        }

        .small-muted {
            color: var(--muted);
            font-size: 0.86rem;
            line-height: 1.5;
        }

        hr {
            border-color: rgba(92, 133, 255, 0.2);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def neon_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def cached_models():
    return load_amr_models("cpu")


@st.cache_data(show_spinner=False)
def cached_samples() -> list[SignalSample]:
    return load_available_samples()


@st.cache_resource(show_spinner=False)
def cached_pdf_chat_backend(source_mtime: float):
    spec = importlib.util.spec_from_file_location("amr_pdf_chat", PDF_CHAT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load pdf_chat backend from {PDF_CHAT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def retrieve_answer(question: str) -> str:
    """Use the real app/explainability/pdf_chat.py hybrid RAG backend."""

    backend = cached_pdf_chat_backend(PDF_CHAT_PATH.stat().st_mtime)
    return backend.answer_question(question)


def plot_waveform(signal: np.ndarray, title: str) -> go.Figure:
    x = np.arange(signal.shape[1])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=signal[0],
            mode="lines",
            name="I channel",
            line=dict(color="#35e7ff", width=2.6),
            hovertemplate="t=%{x}<br>I=%{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=signal[1],
            mode="lines",
            name="Q channel",
            line=dict(color="#ff4fd8", width=2.6),
            hovertemplate="t=%{x}<br>Q=%{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,9,18,0.78)",
        height=360,
        margin=dict(l=24, r=18, t=54, b=34),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
        xaxis=dict(title="Time sample", gridcolor="rgba(76,125,255,0.12)", zeroline=False),
        yaxis=dict(title="Normalized amplitude", gridcolor="rgba(76,125,255,0.12)", zeroline=False),
    )
    return fig


def plot_constellation(signal: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=signal[0],
            y=signal[1],
            mode="markers",
            name="I/Q point",
            marker=dict(size=8, color="#35e7ff", opacity=0.78, line=dict(width=1, color="#ffffff")),
            hovertemplate="I=%{x:.4f}<br>Q=%{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="I/Q Constellation",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,9,18,0.78)",
        height=360,
        margin=dict(l=24, r=18, t=54, b=34),
        xaxis=dict(title="In-phase", gridcolor="rgba(76,125,255,0.12)", zerolinecolor="rgba(255,255,255,0.18)"),
        yaxis=dict(title="Quadrature", gridcolor="rgba(76,125,255,0.12)", zerolinecolor="rgba(255,255,255,0.18)", scaleanchor="x", scaleratio=1),
    )
    return fig


def plot_probabilities(probabilities: dict[str, float]) -> go.Figure:
    ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    labels = [item[0] for item in ordered]
    values = [item[1] for item in ordered]
    colors = ["#35e7ff" if idx == 0 else "#7c3aed" for idx in range(len(labels))]

    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker=dict(color=colors)))
    fig.update_layout(
        title="Class Probability Distribution",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,9,18,0.78)",
        height=430,
        margin=dict(l=18, r=20, t=54, b=34),
        xaxis=dict(title="Probability", range=[0, 1], gridcolor="rgba(76,125,255,0.12)"),
        yaxis=dict(autorange="reversed", title=""),
        showlegend=False,
    )
    return fig


def plot_snr_meter(snr: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=snr,
            number={"suffix": " dB", "font": {"size": 36, "color": "#edf4ff"}},
            gauge={
                "axis": {"range": [-20, 20], "tickcolor": "#9aa9c7"},
                "bar": {"color": "#35e7ff"},
                "bgcolor": "rgba(7,9,18,0.78)",
                "borderwidth": 1,
                "bordercolor": "rgba(76,125,255,0.3)",
                "steps": [
                    {"range": [-20, 0], "color": "rgba(255,79,216,0.18)"},
                    {"range": [0, 8], "color": "rgba(76,125,255,0.20)"},
                    {"range": [8, 15], "color": "rgba(53,231,255,0.18)"},
                    {"range": [15, 20], "color": "rgba(52,211,153,0.22)"},
                ],
                "threshold": {"line": {"color": "#ffffff", "width": 3}, "thickness": 0.75, "value": snr},
            },
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(l=12, r=12, t=20, b=8),
    )
    return fig


def plot_saliency_overlay(signal: np.ndarray, saliency: np.ndarray, channel_idx: int, channel_name: str, color: str) -> go.Figure:
    x = np.arange(signal.shape[1])
    channel = signal[channel_idx]
    sal = saliency[channel_idx]
    sal_scaled = (sal * (np.max(np.abs(channel)) + 1e-6)).astype(np.float32)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=channel,
            mode="lines",
            name=f"{channel_name} waveform",
            line=dict(color=color, width=2.4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=sal_scaled,
            mode="lines",
            name="saliency intensity",
            line=dict(color="rgba(255,255,255,0)", width=0),
            fill="tozeroy",
            fillcolor="rgba(255, 79, 216, 0.28)" if channel_idx == 1 else "rgba(53, 231, 255, 0.26)",
        )
    )
    fig.update_layout(
        title=f"{channel_name} Channel Saliency Overlay",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,9,18,0.78)",
        height=310,
        margin=dict(l=24, r=18, t=54, b=34),
        xaxis=dict(title="Time sample", gridcolor="rgba(76,125,255,0.12)", zeroline=False),
        yaxis=dict(title="Amplitude / importance", gridcolor="rgba(76,125,255,0.12)", zeroline=False),
    )
    return fig


def plot_feature_importance(importance: dict[str, float]) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=list(importance.keys()),
            y=list(importance.values()),
            marker=dict(color=["#35e7ff", "#ff4fd8"]),
            text=[f"{value:.1%}" for value in importance.values()],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="I/Q Feature Importance",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,9,18,0.78)",
        height=310,
        margin=dict(l=24, r=18, t=54, b=34),
        yaxis=dict(title="Relative saliency", range=[0, 1], gridcolor="rgba(76,125,255,0.12)"),
        xaxis=dict(title=""),
        showlegend=False,
    )
    return fig


def init_state(samples: list[SignalSample]) -> None:
    if "current_sample" not in st.session_state:
        st.session_state.current_sample = samples[0]
    if "assistant_history" not in st.session_state:
        st.session_state.assistant_history = [
            ("assistant", "Ask about AMR, SNR, I/Q waveforms, adaptive modulation, or saliency maps.")
        ]


# def render_sidebar() -> None:
#     with st.sidebar:

#         st.markdown("# RAG Communication Assistant")

#         st.markdown(
#             """
#             <div class='small-muted'>
#             Hybrid FAISS + BM25 retrieval using
#             <code>pdf_chat.py</code>.
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )

#         st.divider()

#         # CHAT HISTORY
#         for role, message in st.session_state.assistant_history:

#             with st.chat_message(role):

#                 st.markdown(message)

#         # CHAT INPUT
#         prompt = st.chat_input(
#             "Ask about AMR, SNR, RadioML..."
#         )

#         if prompt:

#             # USER MESSAGE
#             st.session_state.assistant_history.append(
#                 ("user", prompt)
#             )

#             with st.chat_message("user"):
#                 st.markdown(prompt)

#             # ASSISTANT RESPONSE
#             with st.chat_message("assistant"):

#                 with st.spinner("Searching knowledge base..."):

#                     try:
#                         answer = retrieve_answer(prompt)

#                     except Exception as exc:
#                         answer = f"RAG backend error:\n\n{exc}"

#                 # STREAM EFFECT
#                 response_placeholder = st.empty()

#                 full_response = ""

#                 for word in answer.split():

#                     full_response += word + " "

#                     response_placeholder.markdown(full_response + "▌")

#                 response_placeholder.markdown(full_response)

#             st.session_state.assistant_history.append(
#                 ("assistant", answer)
#             )


def resolve_input(samples: list[SignalSample]) -> tuple[SignalSample, float]:
    st.markdown("### Signal Input Panel")
    input_mode = st.radio("Input source", ["RadioML sample", "Upload JSON", "Synthetic signal"], horizontal=True)

    selected = st.session_state.current_sample

    if input_mode == "Upload JSON":
        uploaded = st.file_uploader("Upload I/Q JSON", type=["json"])
        if uploaded is not None:
            try:
                payload = json.loads(uploaded.getvalue().decode("utf-8"))
                selected = parse_signal_json(payload, source=uploaded.name)
                st.session_state.current_sample = selected
            except (ValueError, json.JSONDecodeError) as exc:
                st.error(f"Invalid signal JSON: {exc}")
    elif input_mode == "Synthetic signal":
        modulation = st.selectbox("Synthetic modulation", ["BPSK", "QPSK", "8PSK", "QAM16", "QAM64", "WBFM"], index=1)
        seed = st.number_input("Signal seed", min_value=0, max_value=9999, value=42, step=1)
        base_snr = st.slider("SNR", min_value=-20, max_value=20, value=int(selected.snr), step=1)
        selected = generate_synthetic_signal(modulation, float(base_snr), seed=int(seed))
        st.session_state.current_sample = selected
    else:
        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("Random RadioML sample", use_container_width=True):
                st.session_state.current_sample = random.choice(samples)
                selected = st.session_state.current_sample
        with col_b:
            options = [f"{idx}: {sample.modulation} at {sample.snr:g} dB ({sample.source})" for idx, sample in enumerate(samples)]
            chosen = st.selectbox("Bundled sample", range(len(samples)), format_func=lambda idx: options[idx])
            if st.button("Load selected sample", use_container_width=True):
                st.session_state.current_sample = samples[chosen]
                selected = st.session_state.current_sample

    target_snr = st.slider(
        "SNR slider",
        min_value=-20,
        max_value=20,
        value=int(round(float(selected.snr))),
        step=1,
        help="Controls model routing and adaptive modulation decisions.",
    )
    display_sample = SignalSample(
        signal=normalize_signal(selected.signal),
        modulation=selected.modulation,
        snr=float(target_snr),
        source=selected.source,
    )
    return display_sample, float(target_snr)


# def main() -> None:
#     inject_css()

#     samples = cached_samples()
#     init_state(samples)
#     #render_sidebar()

#     st.markdown(
#         """
#         <section class="hero">
#             <div class="hero-kicker">AI-powered wireless communication control system</div>
#             <h1>Adaptive Automatic Modulation Recognition</h1>
#             <p class="hero-sub">
#                 CNN-GRU-GNN inference, SNR-aware model routing, adaptive modulation recommendation,
#                 and gradient-based explainability for I/Q radio signals.
#             </p>
#         </section>
#         """,
#         unsafe_allow_html=True,
#     )

#     try:
#         models, device, edge_index = cached_models()
#         model_ready = True
#         model_error = ""
#     except Exception as exc:
#         models, device, edge_index = None, None, None
#         model_ready = False
#         model_error = str(exc)

#     left, right = st.columns([0.9, 1.3], gap="large")
#     with left:
#         sample, snr = resolve_input(samples)
#         recommendation = recommend_modulation(snr)

#         st.markdown("### System Status")
#         status_cols = st.columns(2)
#         with status_cols[0]:
#             neon_card("Input SNR", f"{snr:g} dB", sample.source)
#         with status_cols[1]:
#             neon_card("True Label", sample.modulation, "Available for demo samples")

#         st.plotly_chart(plot_snr_meter(snr), use_container_width=True)

#     with right:
#         route_key = "high_snr" if snr >= 0 else "robust"
#         route_name = "High-SNR Model" if route_key == "high_snr" else "Robust Fine-Tuned Model"
#         st.markdown("### Model Routing")
#         st.markdown(
#             f"""
#             <div class="glass-card">
#                 <span class="badge"><span class="pulse-dot"></span>{route_name}</span>
#                 <p class="small-muted" style="margin-top: 14px;">
#                     {'Clean-channel specialist selected because SNR is at or above 0 dB.' if snr >= 0 else 'Noise-hardened model selected because SNR is below 0 dB.'}
#                 </p>
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )

#         if not model_ready:
#             st.error(model_error)
#             st.stop()

#         result = predict_modulation(sample.signal, snr, models, device, edge_index)
#         active_model = models[result.active_model_key]
#         saliency, _ = compute_saliency(active_model, sample.signal, edge_index, device)
#         importance = channel_importance(saliency)

#         p_col1, p_col2 = st.columns([0.95, 1.05], gap="large")
#         with p_col1:
#             st.markdown(
#                 f"""
#                 <div class="prediction">
#                     <div class="metric-label">Predicted Modulation</div>
#                     <div class="value">{result.predicted_modulation}</div>
#                     <div class="metric-note">Confidence {result.confidence:.2%}</div>
#                 </div>
#                 """,
#                 unsafe_allow_html=True,
#             )
#         with p_col2:
#             rec_note = f"{recommendation['spectral_efficiency']:.1f} bits/s/Hz · {recommendation['profile']}"
#             neon_card("Adaptive Recommendation", str(recommendation["modulation"]), rec_note)
#             st.markdown(f"<div class='small-muted'>{recommendation['explanation']}</div>", unsafe_allow_html=True)

#     tab_wave, tab_prediction, tab_xai, tab_status = st.tabs(
#         ["Waveform Visualization", "Modulation Prediction", "Explainable AI", "System Status"]
#     )

#     with tab_wave:
#         c1, c2 = st.columns(2, gap="large")
#         with c1:
#             st.plotly_chart(plot_waveform(sample.signal, "I/Q Waveform"), use_container_width=True)
#         with c2:
#             st.plotly_chart(plot_constellation(sample.signal), use_container_width=True)

#     with tab_prediction:
#         c1, c2 = st.columns([1.15, 0.85], gap="large")
#         with c1:
#             st.plotly_chart(plot_probabilities(result.probabilities), use_container_width=True)
#         with c2:
#             top3 = sorted(result.probabilities.items(), key=lambda item: item[1], reverse=True)[:3]
#             st.markdown("### Top Classes")
#             for rank, (name, prob) in enumerate(top3, start=1):
#                 neon_card(f"Rank {rank}", name, f"{prob:.2%} posterior probability")

#     with tab_xai:
#         x1, x2 = st.columns(2, gap="large")
#         with x1:
#             st.plotly_chart(plot_saliency_overlay(sample.signal, saliency, 0, "I", "#35e7ff"), use_container_width=True)
#         with x2:
#             st.plotly_chart(plot_saliency_overlay(sample.signal, saliency, 1, "Q", "#ff4fd8"), use_container_width=True)

#         x3, x4 = st.columns([0.85, 1.15], gap="large")
#         with x3:
#             st.plotly_chart(plot_feature_importance(importance), use_container_width=True)
#         with x4:
#             st.markdown(
#                 f"""
#                 <div class="glass-card">
#                     <div class="metric-label">Saliency Readout</div>
#                     <div class="metric-value">{max(importance, key=importance.get)}</div>
#                     <div class="metric-note">
#                         The dominant channel contributed {max(importance.values()):.1%} of the normalized gradient energy.
#                         Peaks in the overlays mark waveform regions with strongest influence on the predicted class logit.
#                     </div>
#                 </div>
#                 """,
#                 unsafe_allow_html=True,
#             )

#     with tab_status:
#         s1, s2, s3, s4 = st.columns(4)
#         with s1:
#             neon_card("Processing Status", "Online", "CPU batch-size-1 inference")
#         with s2:
#             neon_card("Active Model", result.active_model_name, result.routing_message)
#         with s3:
#             neon_card("Confidence", f"{result.confidence:.2%}", "Softmax prediction confidence")
#         with s4:
#             neon_card("Classes", str(len(MODULATION_CLASSES)), "RadioML modulation labels")

#         st.markdown("### Deployment Checklist")
#         st.markdown(
#             """
#             <div class="glass-card">
#                 <div class="small-muted">
#                     Root entry point: <code>app.py</code><br>
#                     Model checkpoints: <code>models/best_model.pth</code>, <code>models/final_finetuned_model.pth</code><br>
#                     Local command: <code>streamlit run app.py</code><br>
#                     Cloud targets: Streamlit Cloud or HuggingFace Spaces using <code>requirements.txt</code>
#                 </div>
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )

def main():
    st.title("AMR Dashboard Running")
    st.success("Deployment successful")


if __name__ == "__main__":
    main()
