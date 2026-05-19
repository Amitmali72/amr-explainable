"""
Production-Grade Hybrid RAG for RadioML Knowledge Base
======================================================

Features:
- Semantic section-aware chunking
- Hybrid Retrieval (FAISS + BM25)
- Reciprocal Rank Fusion (RRF)
- Cross-Encoder Reranking
- Persistent FAISS index
- Metadata-aware retrieval
- Context-grounded answering
- Ollama/OpenAI/local support
- Proper citations
- Clean architecture

Install:
pip install \
langchain \
langchain-community \
langchain-huggingface \
sentence-transformers \
faiss-cpu \
rank-bm25 \
transformers \
torch \
requests

Optional:
pip install faiss-gpu
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import requests

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

KNOWLEDGE_BASE_PATH = BASE_DIR / "radioml_knowledge_base.txt"

FAISS_INDEX_DIR = BASE_DIR / "faiss_index"
FAISS_META_PATH = FAISS_INDEX_DIR / "index_meta.json"

# Better embedding model for technical RAG
EMBEDDING_MODEL = os.getenv(
    "AMR_EMBEDDING_MODEL",
    "BAAI/bge-large-en-v1.5"
)

# Retrieval
TOP_K = int(os.getenv("AMR_TOP_K", "5"))
INITIAL_RETRIEVAL_K = int(os.getenv("AMR_INITIAL_K", "12"))

# RAG Provider
RAG_PROVIDER = "ollama"

# Ollama
OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://127.0.0.1:11434/api/chat"
)

OLLAMA_MODEL = os.getenv(
    "AMR_OLLAMA_MODEL",
    "mistral:latest"
)

OLLAMA_TIMEOUT = int(os.getenv("AMR_OLLAMA_TIMEOUT", "180"))

# OpenAI Compatible API
API_BASE_URL = os.getenv("AMR_API_BASE_URL", "").rstrip("/")
API_KEY = os.getenv("AMR_API_KEY", "")
API_MODEL = os.getenv("AMR_API_MODEL", "")

# System Prompt
SYSTEM_PROMPT = """
You are an intelligent AI assistant specialized in:

- Wireless Communication
- Signal Processing
- RadioML
- Automatic Modulation Recognition
- Deep Learning

Your task:
- Read retrieved knowledge carefully
- Understand the context
- Generate a clean human-like answer
- Explain concepts conversationally
- Be concise unless detailed explanation requested
- NEVER dump raw context
- NEVER copy huge chunks
- Summarize intelligently
- Answer like a smart AI assistant

If information is missing:
Say:
"I could not find that in my knowledge base."

Do not mention retrieval, chunks, sources, or context unless asked.
"""

# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class HybridRAGIndex:
    documents: list[Document]
    db: FAISS
    bm25: BM25Okapi


@dataclass
class RAGResponse:
    answer: str
    provider: str
    sources: list[dict]


# =============================================================================
# TOKENIZATION
# =============================================================================


def tokenize(text: str) -> list[str]:
    return re.findall(
        r"[a-zA-Z0-9][a-zA-Z0-9+\-_/]*",
        text.lower()
    )


# =============================================================================
# QUERY EXPANSION
# =============================================================================


def expand_query(query: str) -> str:
    """
    Simple query expansion for better recall.
    """

    q = query.lower()

    expansions = []

    if "qam64" in q:
        expansions.extend([
            "64QAM",
            "high order modulation",
            "dense constellation"
        ])

    if "snr" in q:
        expansions.extend([
            "signal to noise ratio",
            "noise power",
            "channel quality"
        ])

    if "ber" in q:
        expansions.extend([
            "bit error rate",
            "communication errors"
        ])

    if "qpsk" in q:
        expansions.extend([
            "phase shift keying",
            "4 phase modulation"
        ])

    return query + " " + " ".join(expansions)


# =============================================================================
# SEMANTIC SECTION CHUNKING
# =============================================================================


def load_documents() -> list[Document]:

    if not KNOWLEDGE_BASE_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge base not found: {KNOWLEDGE_BASE_PATH}"
        )

    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    pattern = r"(SECTION\s+\d+:[\s\S]*?)(?=SECTION\s+\d+:|$)"

    sections = re.findall(pattern, text)

    documents = []

    for section in sections:

        title_match = re.search(
            r"SECTION\s+\d+:\s*(.+)",
            section
        )

        title = (
            title_match.group(1).strip()
            if title_match
            else "Unknown"
        )

        topic = title.lower()

        documents.append(
            Document(
                page_content=section.strip(),
                metadata={
                    "section_title": title,
                    "topic": topic,
                    "source": "RadioML Knowledge Base"
                }
            )
        )

    return documents


# =============================================================================
# EMBEDDINGS
# =============================================================================


@lru_cache(maxsize=1)
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


# =============================================================================
# RERANKER
# =============================================================================


@lru_cache(maxsize=1)
def get_reranker():

    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )


# =============================================================================
# INDEX METADATA
# =============================================================================


def index_metadata() -> dict[str, Any]:

    return {
        "embedding_model": EMBEDDING_MODEL,
        "knowledge_base_mtime":
            KNOWLEDGE_BASE_PATH.stat().st_mtime,
    }


def is_saved_index_fresh() -> bool:

    if not FAISS_INDEX_DIR.exists():
        return False

    if not FAISS_META_PATH.exists():
        return False

    try:
        saved = json.loads(
            FAISS_META_PATH.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError:
        return False

    return saved == index_metadata()


# =============================================================================
# LOAD / BUILD FAISS
# =============================================================================


def load_or_build_faiss(
    documents: list[Document]
) -> FAISS:

    embeddings = get_embeddings()

    if is_saved_index_fresh():

        return FAISS.load_local(
            str(FAISS_INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True
        )

    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    db = FAISS.from_documents(
        documents,
        embeddings
    )

    db.save_local(str(FAISS_INDEX_DIR))

    FAISS_META_PATH.write_text(
        json.dumps(index_metadata(), indent=2),
        encoding="utf-8"
    )

    return db


# =============================================================================
# BUILD INDEX
# =============================================================================


@lru_cache(maxsize=1)
def build_rag_index() -> HybridRAGIndex:

    documents = load_documents()

    db = load_or_build_faiss(documents)

    tokenized_docs = [
        tokenize(doc.page_content)
        for doc in documents
    ]

    bm25 = BM25Okapi(tokenized_docs)

    return HybridRAGIndex(
        documents=documents,
        db=db,
        bm25=bm25
    )


# =============================================================================
# RERANK
# =============================================================================


def rerank_documents(
    query: str,
    docs: list[Document]
) -> list[Document]:

    reranker = get_reranker()

    pairs = [
        [query, doc.page_content]
        for doc in docs
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )

    return [doc for _, doc in ranked]


# =============================================================================
# HYBRID SEARCH
# =============================================================================


def hybrid_search(
    query: str,
    k: int = TOP_K
) -> list[Document]:

    expanded_query = expand_query(query)

    index = build_rag_index()

    # Dense Retrieval
    dense_docs = index.db.similarity_search(
        expanded_query,
        k=INITIAL_RETRIEVAL_K
    )

    # Sparse Retrieval
    bm25_scores = index.bm25.get_scores(
        tokenize(expanded_query)
    )

    top_sparse_indices = np.argsort(
        bm25_scores
    )[-INITIAL_RETRIEVAL_K:][::-1]

    sparse_docs = [
        index.documents[i]
        for i in top_sparse_indices
    ]

    # Reciprocal Rank Fusion
    rrf_k = 60

    doc_scores = {}
    doc_map = {}

    for rank, doc in enumerate(dense_docs):

        key = doc.page_content

        doc_scores[key] = (
            doc_scores.get(key, 0)
            + 1 / (rrf_k + rank + 1)
        )

        doc_map[key] = doc

    for rank, doc in enumerate(sparse_docs):

        key = doc.page_content

        doc_scores[key] = (
            doc_scores.get(key, 0)
            + 1 / (rrf_k + rank + 1)
        )

        doc_map[key] = doc

    retrieved = sorted(
        doc_scores,
        key=lambda x: doc_scores[x],
        reverse=True
    )

    retrieved_docs = [
        doc_map[x]
        for x in retrieved[:INITIAL_RETRIEVAL_K]
    ]

    # Cross Encoder Reranking
    reranked_docs = rerank_documents(
        query,
        retrieved_docs
    )

    return reranked_docs[:k]


# =============================================================================
# CONTEXT FORMATTER
# =============================================================================


def build_context(
    docs: list[Document]
) -> str:

    contexts = []

    for idx, doc in enumerate(docs, start=1):

        title = doc.metadata.get(
            "section_title",
            "Unknown"
        )

        content = doc.page_content

        # Clean
        content = re.sub(r"=+", "", content)
        content = re.sub(r"\s+", " ", content)

        # Keep only first useful 1200 chars
        content = content[:1200]

        contexts.append(
            f"""
        SOURCE {idx}
        SECTION: {title}

        {content}
        """
        )

    return "\n\n".join(contexts)
# =============================================================================
# CITATIONS
# =============================================================================


def build_sources(
    docs: list[Document]
) -> list[dict]:

    sources = []

    for doc in docs:

        sources.append({
            "section":
                doc.metadata.get(
                    "section_title",
                    "Unknown"
                ),

            "preview":
                doc.page_content[:250]
        })

    return sources


# =============================================================================
# LOCAL ANSWER
# =============================================================================


def local_answer(
    query: str,
    docs: list[Document]
    ) -> RAGResponse:

    if not docs:
        return RAGResponse(
            answer="I could not find that in my knowledge base.",
            provider="local",
            sources=[]
        )

    top_doc = docs[0]

    text = top_doc.page_content

    text = re.sub(r"=+", "", text)
    text = re.sub(r"\s+", " ", text)

    sentences = re.split(r'(?<=[.!?])\s+', text)

    useful = []

    for sentence in sentences:

        sentence = sentence.strip()

        if len(sentence) < 30:
            continue

        useful.append(sentence)

        if len(useful) >= 6:
            break

    answer = "\n\n".join(
        [f"- {s}" for s in useful]
    )

    return RAGResponse(
        answer=answer,
        provider="local",
        sources=[]
    )


# =============================================================================
# OLLAMA GENERATION
# =============================================================================


def ollama_answer(
    query: str,
    docs: list[Document]
) -> RAGResponse:

    context = build_context(docs)

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content":
                        f"""
                    Context:
                    {context}

                    Question:
                    {query}
                    """
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9
            }
        },
        timeout=OLLAMA_TIMEOUT
    )

    if not response.ok:
        raise RuntimeError(
            response.text
        )

    payload = response.json()

    answer = payload["message"]["content"]

    return RAGResponse(
        answer=answer,
        provider="ollama",
        sources=build_sources(docs)
    )


# =============================================================================
# OPENAI COMPATIBLE API
# =============================================================================


def api_answer(
    query: str,
    docs: list[Document]
    ) -> RAGResponse:

    if not API_BASE_URL:
        raise RuntimeError(
            "API URL missing"
        )

    context = build_context(docs)

    response = requests.post(
        f"{API_BASE_URL}/chat/completions",
        headers={
            "Authorization":
                f"Bearer {API_KEY}",

            "Content-Type":
                "application/json"
        },
        json={
            "model": API_MODEL,

            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content":
                        f"""
                    Context:
                    {context}

                    Question:
                    {query}
                    """
                }
            ],

            "temperature": 0.1,
            "top_p": 0.9
        },
        timeout=120
    )

    if not response.ok:
        raise RuntimeError(
            response.text
        )

    payload = response.json()

    answer = (
        payload["choices"][0]
        ["message"]["content"]
    )

    return RAGResponse(
        answer=answer,
        provider="api",
        sources=build_sources(docs)
    )


# =============================================================================
# MAIN ENTRY
# =============================================================================


def answer_question_with_metadata(
    query: str,
    k: int = TOP_K
    ) -> RAGResponse:

    query = query.strip()

    if not query:

        return RAGResponse(
            answer="Ask a question.",
            provider="local",
            sources=[]
        )

    docs = hybrid_search(query, k=k)

    try:

        if RAG_PROVIDER == "ollama":
            return ollama_answer(query, docs)

        elif RAG_PROVIDER in {
            "api",
            "openai"
        }:
            return api_answer(query, docs)

        else:
            return local_answer(query, docs)

    except Exception as e:

        fallback = local_answer(query, docs)

        fallback.answer = (
            f"Generator failed: {str(e)}\n\n"
            + fallback.answer
        )

        return fallback


# =============================================================================
# SIMPLE API
# =============================================================================


def answer_question(query: str) -> str:

    response = answer_question_with_metadata(query)

    return response.answer


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":

    while True:

        query = input("\nQuestion: ")

        if query.lower() in {"exit", "quit"}:
            break

        result = answer_question(query)

        print("\n")
        print(result)
        print("\n")