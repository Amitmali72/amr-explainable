# Adaptive AMR Streamlit Deployment

This project is a deployment-ready Streamlit dashboard for AI-powered Adaptive
Automatic Modulation Recognition using a CNN-GRU-GNN two-model routing system.

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The local Streamlit config binds to `localhost:8502` to avoid Windows socket
permission conflicts on the default `8501` port.

## RAG Assistant

The sidebar assistant uses the project RAG file at
`app/explainability/pdf_chat.py` and retrieves from
`app/explainability/radioml_knowledge_base.txt`.

By default it uses a local extractive answer engine, so it works without Ollama
and is deployable on Streamlit Cloud or HuggingFace Spaces. Retrieval remains
research-grade: FAISS dense search + BM25 sparse search + Reciprocal Rank
Fusion. The FAISS index is persisted under `app/explainability/faiss_index/`
after the first build.

Default mode:

```bash
set AMR_RAG_PROVIDER=local
```

No API key and no Ollama server are required in this mode.

To use a free-tier or hosted OpenAI-compatible API, set these variables before
running Streamlit:

```bash
set AMR_RAG_PROVIDER=api
set AMR_API_BASE_URL=https://your-provider.example/openai/v1
set AMR_API_KEY=your_api_key
set AMR_API_MODEL=your_model_name
```

To use Ollama locally, set:

```bash
set AMR_RAG_PROVIDER=ollama
set OLLAMA_URL=http://127.0.0.1:11434/api/chat
set AMR_OLLAMA_MODEL=mistral:latest
set AMR_OLLAMA_NUM_GPU=20
set AMR_OLLAMA_NUM_CTX=2048
```

Then make sure Ollama is running:

```bash
ollama pull mistral:latest
ollama serve
```

If Ollama fails to load the model, the app falls back to the local RAG answer
engine instead of failing.

Then run the dashboard:

```bash
streamlit run app.py
```

On Windows, use a fresh virtual environment if an older global Torch install
fails to load `fbgemm.dll`. The included `intel-openmp` requirement supplies the
runtime dependency commonly needed by PyTorch CPU/CUDA wheels on Windows.

## Required Structure

```text
project/
├── app.py
├── model.py
├── inference.py
├── saliency.py
├── adaptive.py
├── requirements.txt
├── app/
│   └── explainability/
│       ├── pdf_chat.py
│       ├── radioml_knowledge_base.txt
│       └── faiss_index/
├── models/
│   ├── best_model.pth
│   └── final_finetuned_model.pth
└── data/
    └── sample_signals/
```

## Streamlit Cloud

1. Push the repository to GitHub.
2. Create a new Streamlit Cloud app from the repository.
3. Set the app entry point to `app.py`.
4. Ensure both `.pth` files are committed under `models/`.
5. Deploy.

## HuggingFace Spaces

1. Create a new Space with the Streamlit SDK.
2. Upload the repository contents.
3. Keep `app.py` at the root and include `requirements.txt`.
4. Ensure model checkpoints are in `models/`.
5. The Space will run the Streamlit app automatically.
