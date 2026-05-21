# 🌊 Adaptive Automatic Modulation Recognition (AMR)

> **AI-powered wireless communication control system** with explainable AI, adaptive modulation recommendation, and SNR-aware model routing for I/Q radio signals.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io)

---

## 🎯 Overview

This repository contains an **interactive Streamlit dashboard** and supporting Python modules for **Automatic Modulation Recognition (AMR)** using a sophisticated dual-model architecture. The system intelligently routes signals through either a **high-SNR CNN-GRU model** or a **robust fine-tuned model** based on signal noise characteristics, and provides **gradient-based explainability** via saliency maps.

### Key Features

✨ **Dual-Model Routing**
- CNN-GRU-GNN architecture with SNR-aware model selection
- High-SNR model for clean channels (SNR ≥ 0 dB)
- Robust fine-tuned model for noisy conditions (SNR < 0 dB)

🔍 **Explainable AI (XAI)**
- Gradient-based saliency analysis for I/Q waveforms
- Channel importance attribution
- Interpretable feature visualization

🛠️ **Adaptive Modulation**
- Real-time modulation recommendation based on channel SNR
- Spectral efficiency metrics and profile recommendations
- Dynamic adaptation to channel conditions

🤖 **RAG Assistant**
- Hybrid FAISS + BM25 retrieval system
- PDF knowledge base integration
- Optional Ollama and OpenAI API support

📊 **Interactive Dashboard**
- Real-time signal visualization
- I/Q constellation plots
- Probability distribution charts
- SNR gauge and system diagnostics

---

## 📁 Project Structure

```
amr-explainable/
├── app.py                              # Main Streamlit dashboard
├── model.py                            # CNN-GRU-GNN model definitions
├── inference.py                        # Signal processing and prediction logic
├── saliency.py                         # Gradient-based explainability module
├── adaptive.py                         # Adaptive modulation recommendation engine
├── requirements.txt                    # Python dependencies
│
├── app/
│   └── explainability/
│       ├── pdf_chat.py                 # RAG backend with FAISS + BM25
│       ├── radioml_knowledge_base.txt  # Domain knowledge reference
│       └── faiss_index/                # Persisted FAISS embeddings
│
├── models/
│   ├── best_model.pth                  # High-SNR CNN-GRU model
│   └── final_finetuned_model.pth       # Robust fine-tuned model
│
├── data/
│   └── sample_signals/                 # Test and demo signals
│
├── DL_ResearchPaper.pdf                # Technical reference paper
├── DL_project_ppt.pdf                  # Project presentation
├── AMR_DeepLearning.jpg                # Architecture diagram
└── README_DEPLOYMENT.md                # Deployment guide
```

---

## 🚀 Quick Start

### Local Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Amitmali72/amr-explainable.git
   cd amr-explainable
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate          # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the dashboard:**
   ```bash
   streamlit run app.py
   ```

5. **Access the application:**
   Open your browser and navigate to `http://localhost:8502`

---

## 📊 Usage Guide

### Input Modes

The dashboard supports three signal input methods:

#### 1. **RadioML Sample**
- Browse pre-loaded benchmark signals from RadioML dataset
- Adjust SNR dynamically with a slider
- Load random or specific modulation types

#### 2. **Upload JSON**
- Upload custom I/Q signal data in JSON format
- Expected format:
  ```json
  {
    "signal": [[I_samples], [Q_samples]],
    "modulation": "QPSK",
    "snr": 5.0
  }
  ```

#### 3. **Synthetic Signal**
- Generate synthetic test signals on-the-fly
- Select modulation type, SNR, and random seed
- Supported modulations: BPSK, QPSK, 8PSK, QAM16, QAM64, WBFM

### Dashboard Tabs

| Tab | Description |
|-----|-------------|
| **Waveform Visualization** | I/Q time-domain waveform and constellation diagram |
| **Modulation Prediction** | Class probability distribution and top-3 predictions |
| **Explainable AI** | Saliency overlays for I and Q channels, feature importance |
| **System Status** | Processing status, active model, confidence, and deployment info |

### Model Routing

The system automatically selects the optimal model based on SNR:

```
SNR < 0 dB  →  Robust Fine-Tuned Model (noise hardened)
SNR ≥ 0 dB  →  High-SNR CNN-GRU Model (clean channel specialist)
```

---

## 🤖 RAG Assistant

The sidebar assistant uses a hybrid retrieval system to answer domain-specific questions.

### Configuration

**Default (Local Mode)** — No API keys required:
```bash
set AMR_RAG_PROVIDER=local
streamlit run app.py
```

**OpenAI-Compatible API:**
```bash
set AMR_RAG_PROVIDER=api
set AMR_API_BASE_URL=https://your-provider.example/openai/v1
set AMR_API_KEY=your_api_key
set AMR_API_MODEL=your_model_name
streamlit run app.py
```

**Ollama (Local LLM):**
```bash
# First, start Ollama
ollama pull mistral:latest
ollama serve

# Then, in another terminal
set AMR_RAG_PROVIDER=ollama
set OLLAMA_URL=http://127.0.0.1:11434/api/chat
set AMR_OLLAMA_MODEL=mistral:latest
set AMR_OLLAMA_NUM_GPU=20
set AMR_OLLAMA_NUM_CTX=2048
streamlit run app.py
```

---

## 📦 Dependencies

### Core Requirements
- **Streamlit** (≥1.32) — Interactive web dashboard
- **PyTorch** (≥2.2) — Deep learning framework
- **NumPy** (≥1.26) — Numerical computing
- **Plotly** (≥5.18) — Interactive visualization

### NLP & RAG
- **LangChain Community** (≥0.4) — RAG orchestration
- **FAISS** (≥1.8) — Dense semantic search
- **rank-bm25** (≥0.2) — Sparse keyword search
- **sentence-transformers** (≥2.7) — Text embeddings

### Optional
- **Ollama** — Local LLM inference
- **Groq** — Fast inference API

For the complete list, see [`requirements.txt`](requirements.txt).

---

## 🎓 Technical Details

### Model Architecture

**CNN-GRU-GNN Dual-Model System:**

1. **CNN Frontend:** Feature extraction from I/Q waveforms
2. **GRU Middle:** Temporal dependency modeling
3. **GNN Fusion:** Graph-based multi-model routing
4. **Dual Routing:** SNR-conditioned model selection

### Saliency Analysis

The explainability module computes gradient-based saliency maps:

```python
saliency, _ = compute_saliency(model, signal, edge_index, device)
importance = channel_importance(saliency)  # I/Q attribution
```

- Highlights waveform regions most influential to the prediction
- Scales with gradient magnitude for interpretability
- Supports both I and Q channels separately

### Supported Modulation Classes

- **Analog:** WBFM
- **Digital PSK:** BPSK, QPSK, 8PSK
- **Digital QAM:** QAM16, QAM64

---

## 🌐 Deployment

### Streamlit Cloud

1. Push repository to GitHub
2. Create new Streamlit Cloud app
3. Set entry point to `app.py`
4. Ensure `.pth` files are in `models/` directory
5. Deploy

### HuggingFace Spaces

1. Create new Space with Streamlit SDK
2. Upload repository contents
3. Keep `app.py` at root with `requirements.txt`
4. Model checkpoints in `models/` directory
5. Space auto-runs the app

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8502
CMD ["streamlit", "run", "app.py"]
```

See [`README_DEPLOYMENT.md`](README_DEPLOYMENT.md) for detailed deployment instructions.

---

## 📖 Research & Documentation

- 📄 **Research Paper:** [`DL_ResearchPaper.pdf`](DL_ResearchPaper.pdf)
- 🎤 **Project Presentation:** [`DL_project_ppt.pdf`](DL_project_ppt.pdf)
- 🖼️ **Architecture Diagram:** [`AMR_DeepLearning.jpg`](AMR_DeepLearning.jpg)

---

## 🔧 Core Modules

### `app.py`
Main Streamlit dashboard with UI components, signal input modes, visualization, and assistant integration.

### `model.py`
Defines CNN-GRU-GNN architecture and modulation class mappings.

### `inference.py`
Signal preprocessing, model loading, prediction pipeline, and signal generation utilities.

### `saliency.py`
Gradient-based explainability: saliency computation and channel importance attribution.

### `adaptive.py`
Adaptive modulation recommendation engine with SNR-based routing logic.

### `app/explainability/pdf_chat.py`
RAG backend with FAISS + BM25 hybrid retrieval and optional LLM integration.

---

## 📊 Performance Metrics

| Model | SNR Range | Target Scenario | Specialization |
|-------|-----------|-----------------|-----------------|
| **High-SNR CNN-GRU** | ≥ 0 dB | Clean channels, low noise | Spectral efficiency |
| **Robust Fine-Tuned** | < 0 dB | Noisy channels, interference | Reliability |

---

## 🛠️ Development

### Running Tests

```bash
# Test signal generation
python -c "from inference import generate_synthetic_signal; sig = generate_synthetic_signal('QPSK', 5.0); print(sig)"

# Test model loading
python -c "from inference import load_amr_models; models, device, edge_index = load_amr_models('cpu')"
```

### Adding New Modulation Classes

1. Update `MODULATION_CLASSES` in `model.py`
2. Retrain models on new class labels
3. Update knowledge base in `app/explainability/radioml_knowledge_base.txt`

---

## 📝 License

This project is licensed under the **MIT License** — see the LICENSE file for details.

---

## 👨‍💻 Author

**Amit Mali** | [GitHub](https://github.com/Amitmali72) | [LinkedIn](https://linkedin.com/in/amitmali)

---

## 🙏 Acknowledgments

- RadioML dataset for benchmark signals
- Streamlit for the interactive framework
- PyTorch community for deep learning tools
- LangChain for RAG orchestration

---

## 💬 Support & Feedback

For issues, questions, or suggestions:
- Open a [GitHub Issue](https://github.com/Amitmali72/amr-explainable/issues)
- Check the [Deployment Guide](README_DEPLOYMENT.md)
- Review the [Research Paper](DL_ResearchPaper.pdf)

---

<div align="center">

**Built with ❤️ for wireless AI and explainability**

⭐ If you find this project useful, please consider giving it a star!

</div>
