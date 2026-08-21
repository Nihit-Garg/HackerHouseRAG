# HackerHouseRAG — Voice-Enabled RAG Pipeline

A production-quality, voice-enabled Retrieval-Augmented Generation (RAG) system built for a hackathon. Ask questions by voice or text and receive answers grounded strictly in the retrieved corpus — no hallucination. Includes a FastAPI backend and a React frontend (`frontend/`) for asking questions and reviewing answers in the browser.

---

## Architecture Overview

Audio or text input flows through a LangGraph state graph with three sequential nodes: **STT** (Sarvam AI) transcribes the audio query, **Retrieval** runs hybrid dense (FAISS + BGE embeddings) and sparse (BM25) search with Reciprocal Rank Fusion to find the most relevant passages, and **Generation** passes the retrieved context to a local Ollama LLM (qwen2.5:7b) with a strict grounding prompt that prohibits outside-knowledge answers. A FastAPI server exposes the pipeline as a REST API (`/query/text`, `/query/audio`), and a CLI script (`scripts/run_query.py`) provides quick local testing. All stage timings are captured in the response dict for Day-2 benchmarking.

---

## Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Orchestration | LangGraph | State graph with typed nodes; easy to extend to agentic flows |
| API | FastAPI + Uvicorn | Async, auto Swagger docs, multipart file upload |
| Embedding model | `BAAI/bge-small-en-v1.5` | 33M params, 384-dim, strong retrieval benchmarks |
| Vector store | FAISS flat (IndexFlatIP) | Exact cosine search; flat is correct at ≤10k chunks |
| Sparse retrieval | rank_bm25 (BM25Okapi) | Keyword recall complement to dense search |
| Fusion | Reciprocal Rank Fusion (RRF) | Simple, effective, parameter-light hybrid ranking |
| LLM | Ollama `qwen2.5:7b` | Strong instruction following; fits 8GB VRAM |
| STT | Sarvam AI batch API | Indian-language-aware, English supported |
| Chunking | tiktoken + nltk | Real token-based chunking, not `.split()` |
| Frontend | React 19 + Vite | Fast dev server, no server-side rendering needed for a single-user tool |
| Frontend routing | react-router-dom | Four static routes: About, Ask, History, Status |

---

## Project Structure

```
HackerHouseRAG/
├── data/
│   ├── raw/              # (empty — dataset downloaded at runtime)
│   ├── processed/        # chunks.json saved after build_index
│   └── index/            # faiss.index, faiss_meta.json, bm25.pkl
├── src/
│   ├── config.py         # All constants + env vars
│   ├── ingestion.py      # Load + clean MSMARCO-XI
│   ├── chunking.py       # 3 chunking strategies
│   ├── embeddings.py     # BGE embedding wrapper
│   ├── vectorstore.py    # FAISS + BM25 build/load/query
│   ├── retrieval.py      # Hybrid retrieval + RRF fusion
│   ├── stt.py            # Sarvam STT wrapper
│   ├── generation.py     # Ollama LLM grounded generation
│   ├── graph.py          # LangGraph pipeline (RAGPipeline class)
│   └── api.py            # FastAPI app
├── tests/
│   ├── conftest.py
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   └── test_pipeline.py
├── scripts/
│   ├── build_index.py    # One-time index build
│   └── run_query.py      # CLI query runner
├── frontend/              # React + Vite web UI (see Frontend section below)
│   ├── src/
│   │   ├── pages/         # AboutPage, AskPage, HistoryPage, StatusPage
│   │   ├── components/    # Sidebar, AnswerCard, MessageBubble, etc.
│   │   ├── hooks/         # useHistory, useSystemStatus
│   │   ├── utils/         # formatting + guardrail label helpers
│   │   └── api.js         # fetch wrapper for the FastAPI backend
│   └── README.md
├── .env.example
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Prerequisites

```bash
# Python 3.10+
python --version

# Install Ollama: https://ollama.com/download
ollama serve  # start the Ollama server (run in background)

# Pull the LLM
ollama pull qwen2.5:7b
```

### 2. Clone & Install

```bash
git clone <your-repo-url>
cd HackerHouseRAG

pip install -r requirements.txt

# NLTK data for sentence chunking
python -c "import nltk; nltk.download('punkt_tab')"
```

### 3. Configure API Keys

```bash
cp .env.example .env
# Edit .env and set your SARVAM_API_KEY
```

### 4. Build the Index (one-time)

```bash
python scripts/build_index.py
```

This downloads ~5000 English passages from MSMARCO-XI, chunks them, embeds them, and saves FAISS + BM25 indexes to `data/index/`. Takes ~5–10 minutes on first run (embedding model download included).

Options:
```bash
python scripts/build_index.py --strategy sentence --chunk-size 128
python scripts/build_index.py --force   # rebuild from scratch
```

### 5. Start the API Server

```bash
cd src
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI: http://localhost:8000/docs

### 6. Start the Frontend

```bash
cd frontend
npm install
cp .env.example .env   # points VITE_API_BASE_URL at the backend, defaults to localhost:8000
npm run dev
```

Open http://localhost:5173. The backend must already be running (step 5) — see the [Frontend](#frontend) section below for what each page does.

### 7. Run a Query

**CLI (text):**
```bash
python scripts/run_query.py --text "What is machine learning?"
```

**CLI (audio):**
```bash
python scripts/run_query.py --audio path/to/query.wav
```

**API (curl):**
```bash
# Text query
curl -X POST http://localhost:8000/query/text \
  -H "Content-Type: application/json" \
  -d '{"query": "What is deep learning?"}'

# Audio upload
curl -X POST http://localhost:8000/query/audio \
  -F "file=@query.wav"
```

---

## Chunking Strategy Rationale

Three strategies are implemented, each suited to different scenarios:

| Strategy | When to Use | Trade-offs |
|----------|-------------|------------|
| **Fixed-size with overlap** (`fixed`) | Fast baseline, uniform chunk distribution | May split sentences mid-thought; overlap adds redundancy |
| **Sentence/semantic** (`sentence`) | When sentence boundary integrity matters (QA, citation) | Chunks vary in size; very long sentences may exceed budget |
| **Metadata-aware** (`metadata`) | Default for RAG with citation needs | Same as fixed but every chunk carries full provenance metadata |

The metadata-aware strategy is the **default** because it is identical in splitting behaviour to fixed-size but ensures every chunk carries `{passage_id, chunk_id, source_query, position_in_doc, strategy_used}` — required for Day-2 filtering and citation features.

All three strategies can be run independently and benchmarked by building separate indexes:
```bash
python scripts/build_index.py --strategy fixed
python scripts/build_index.py --strategy sentence
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| GET | `/index/status` | Check if indexes are built |
| POST | `/query/text` | Text query → RAG answer |
| POST | `/query/audio` | Audio file → STT → RAG answer |

### Response Schema

```json
{
  "query": "path/to/audio.wav or text",
  "transcribed_text": "what is machine learning?",
  "retrieved_chunks": [
    {
      "text": "...",
      "metadata": {"passage_id": "...", "chunk_id": "...", ...},
      "dense_score": 0.87,
      "bm25_score": 4.2,
      "rrf_score": 0.031
    }
  ],
  "answer": "Machine learning is...",
  "stage_timings": {
    "stt_ms": 420.0,
    "retrieval_ms": 38.5,
    "generation_ms": 2100.0,
    "total_ms": 2560.0
  },
  "errors": []
}
```

---

## Frontend

A React + Vite single-page app in `frontend/` (own [README](frontend/README.md)) that talks to the FastAPI backend over HTTP — nothing is server-rendered, and no data is persisted server-side. Four pages, reachable from the sidebar:

| Page | Route | What it does |
|------|-------|--------------|
| **About** | `/about` | Plain-language explanation of what the assistant does and doesn't do |
| **Ask** | `/ask` | Text box, mic recording, or audio file upload → sends to `/query/text` or `/query/audio`, renders the answer with its sources and a verified/refused status |
| **History** | `/history` | Every past question and answer from this browser, read from `localStorage` (the backend keeps no history, so neither does this page beyond the device it ran on) |
| **Status** | `/status` | Live `/health` and `/index/status` checks, knowledge-base size, and the timing breakdown (STT / retrieval / generation) of the last query |

Answers are shown differently depending on outcome — a normal grounded answer gets a "Verified Response" badge with the model's self-reported confidence; anything caught by one of the four guardrails (unsafe input, off-topic, weak retrieval, or ungrounded generation) gets a distinct refusal card instead, using the `guardrail_triggered` / `guardrail_detail` fields on the `/query/*` response.

---

## Running Tests

```bash
pytest
```

Tests do **not** require a running Ollama server or Sarvam API key — STT and LLM are mocked.

---

## Known Limitations

1. **FAISS flat index**: Exact search is correct at 5–10k chunks but will not scale beyond ~100k without switching to an IVF/HNSW index (controlled by `FAISS_INDEX_TYPE` config).
2. **Sarvam STT endpoint**: API field names are flagged with `# TODO` comments in `stt.py` — verify against current Sarvam docs before production use.
3. **English-only**: `LANGUAGE_FILTER=en` is set by default. Set `LANGUAGE_FILTER=` (empty) in `.env` to load all languages.
4. **Ollama cold start**: First query after server start may be slow (~5s) as qwen2.5:7b loads into VRAM.
5. **No streaming**: Generation is synchronous (batch). Streaming via Ollama's SSE API can be added to the FastAPI endpoint for Day-2.
6. **Single-node**: No distributed retrieval or load balancing — intended for single-machine hackathon use.
