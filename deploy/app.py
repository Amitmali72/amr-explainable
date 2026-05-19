import streamlit as st
import torch
import numpy as np
import pickle
import plotly.graph_objects as go
from pathlib import Path

# Import model architecture
from model import CNN_GRU_GNN, build_edge_index

# App configuration
st.set_page_config(
    page_title="AMR Explainable AI",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
    }
    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif;
        color: #E0E0E0;
    }
    .stMetric-value {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #00E5FF !important;
    }
    .stMetric-label {
        font-size: 1rem !important;
        color: #A0A0A0 !important;
    }
    .stButton>button {
        background: linear-gradient(135deg, #00E5FF 0%, #007BFF 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 229, 255, 0.3);
    }
    .css-1d391kg {
        background-color: #1A1D24;
    }
    .card {
        background-color: #1A1D24;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(CNN_GRU_GNN.MODULATION_CLASSES)
    
    # Load Clean Model
    model_clean = CNN_GRU_GNN(num_classes=num_classes).to(device)
    model_clean.load_state_dict(torch.load(r"d:\amr-explainable\app\training\best_model.pth", map_location=device))
    model_clean.eval()
    
    # Load Robust Model
    model_robust = CNN_GRU_GNN(num_classes=num_classes).to(device)
    model_robust.load_state_dict(torch.load(r"d:\amr-explainable\app\training\final_finetuned_model.pth", map_location=device))
    model_robust.eval()
    
    edge_index = build_edge_index(128).to(device)
    
    return model_clean, model_robust, device, edge_index

@st.cache_data
def load_samples():
    metadata_path = r"d:\amr-explainable\deploy\sample_metadata.pkl"
    signals_path = r"d:\amr-explainable\deploy\sample_signals.npy"
    
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    signals = np.load(signals_path)
    
    return signals, metadata

def normalize_signal(signal):
    max_val = np.max(np.abs(signal))
    if max_val == 0:
        max_val = 1
    return signal / max_val

def get_saliency_map(model, signal, edge_index, device):
    model.train() # Needs to be in train mode for gradients
    signal_t = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(device)
    signal_t.requires_grad = True

    out = model(signal_t, edge_index)
    pred_class = out.argmax()

    out[0, pred_class].backward()

    saliency = signal_t.grad.abs().squeeze().cpu().numpy()
    saliency = saliency / (saliency.max() + 1e-8)

    model.eval()
    return saliency, pred_class.item()

def plot_signal(signal, title="I/Q Signal"):
    fig = go.Figure()
    x = np.arange(signal.shape[1])
    
    fig.add_trace(go.Scatter(x=x, y=signal[0], mode='lines', name='I Channel', line=dict(color='#00E5FF', width=2)))
    fig.add_trace(go.Scatter(x=x, y=signal[1], mode='lines', name='Q Channel', line=dict(color='#FF00E5', width=2)))
    
    fig.update_layout(
        title=title,
        template="plotly_dark",
        plot_bgcolor="#1A1D24",
        paper_bgcolor="#1A1D24",
        xaxis_title="Time Step",
        yaxis_title="Amplitude",
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_saliency(signal, saliency, channel_name, color_sig, color_sal):
    fig = go.Figure()
    x = np.arange(signal.shape[0])
    
    fig.add_trace(go.Scatter(x=x, y=signal, mode='lines', name=f'{channel_name} Channel', line=dict(color=color_sig, width=2)))
    
    # Fill under saliency curve
    fig.add_trace(go.Scatter(x=x, y=saliency, mode='lines', name='Saliency', 
                             line=dict(color=color_sal, width=0),
                             fill='tozeroy', fillcolor=f'rgba{color_sal[3:-1]}, 0.3)'))
    
    fig.update_layout(
        title=f"{channel_name} Channel Saliency",
        template="plotly_dark",
        plot_bgcolor="#1A1D24",
        paper_bgcolor="#1A1D24",
        xaxis_title="Time Step",
        yaxis_title="Amplitude / Importance",
        margin=dict(l=40, r=40, t=40, b=40),
        height=300
    )
    return fig

def main():
    st.markdown("<h1>📡 Automatic Modulation Recognition (AMR) & Explainability</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #A0A0A0; font-size: 1.1rem;'>Deploying CNN-GRU-GNN Models for Robust Radio Signal Classification</p>", unsafe_allow_html=True)
    
    # Load data and models
    with st.spinner("Loading models and samples..."):
        model_clean, model_robust, device, edge_index = load_models()
        signals, metadata = load_samples()
        
    st.sidebar.markdown("<h2>⚙️ Configuration</h2>", unsafe_allow_html=True)
    
    # Model Selection
    model_choice = st.sidebar.radio(
        "Select Model",
        ["Clean Model (best_model.pth)", "Robust Model (final_finetuned.pth)"],
        help="Choose the model to perform inference with."
    )
    selected_model = model_clean if "Clean" in model_choice else model_robust
    
    # Sample Selection
    st.sidebar.markdown("<h2>📶 Input Signal</h2>", unsafe_allow_html=True)
    sample_options = [f"Sample {i} (True: {m['mod']}, SNR: {m['snr']}dB)" for i, m in enumerate(metadata)]
    selected_sample_idx = st.sidebar.selectbox("Select a sample from the dataset", range(len(sample_options)), format_func=lambda x: sample_options[x])
    
    raw_signal = signals[selected_sample_idx]
    meta = metadata[selected_sample_idx]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric("True Modulation", str(meta['mod']))
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric("Signal-to-Noise Ratio (SNR)", f"{meta['snr']} dB")
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric("Selected Model", "Robust" if "Robust" in model_choice else "Clean")
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Plot Input Signal
    st.markdown("<h3>Input I/Q Signal</h3>", unsafe_allow_html=True)
    st.plotly_chart(plot_signal(raw_signal), use_container_width=True)
    
    st.markdown("<hr style='border-color: #333;'>", unsafe_allow_html=True)
    
    # Inference Action
    if st.button("🚀 Run Inference & Explainability"):
        with st.spinner("Analyzing signal..."):
            signal_norm = normalize_signal(raw_signal)
            
            saliency, pred_idx = get_saliency_map(selected_model, signal_norm, edge_index, device)
            pred_mod = CNN_GRU_GNN.MODULATION_CLASSES[pred_idx]
            
            st.markdown(f"<h2>🧠 Prediction Results</h2>", unsafe_allow_html=True)
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.success(f"**Predicted Modulation:** {pred_mod}")
            with res_col2:
                if str(pred_mod) == str(meta['mod']):
                    st.info("✅ Correct Prediction!")
                else:
                    st.error("❌ Incorrect Prediction")
            
            st.markdown("<h3>🔍 Explainability (Saliency Maps)</h3>", unsafe_allow_html=True)
            st.markdown("<p style='color: #A0A0A0;'>The shaded areas indicate which time steps of the signal had the most influence on the model's decision.</p>", unsafe_allow_html=True)
            
            sal_col1, sal_col2 = st.columns(2)
            
            # Use rgba strings for fill colors in Plotly
            with sal_col1:
                st.plotly_chart(plot_saliency(signal_norm[0], saliency[0], "I", "#00E5FF", "rgba(0, 229, 255, 1)"), use_container_width=True)
            
            with sal_col2:
                st.plotly_chart(plot_saliency(signal_norm[1], saliency[1], "Q", "#FF00E5", "rgba(255, 0, 229, 1)"), use_container_width=True)

if __name__ == "__main__":
    main()
