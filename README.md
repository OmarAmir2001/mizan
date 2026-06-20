# ⚖️ Mizan — Egyptian Labor Law Intelligence Agent

> An agentic RAG system for searching, understanding, and getting multi-step explanations of Egyptian labor laws — with a real CRAG loop, persistent user memory, and a live deployed demo.

🔗 **Live Demo:** [huggingface.co/spaces/OmarAmir2001/mizan](https://huggingface.co/spaces/OmarAmir2001/mizan)

---

## What is Mizan?

Mizan (ميزان, Arabic for "scale/balance") is an AI-powered legal assistant built on LangGraph that answers questions about Egyptian labor law. Unlike a simple RAG chatbot, Mizan uses a **Corrective RAG (CRAG) loop** that grades retrieved documents, rewrites queries when retrieval quality is poor, and retries — ensuring the LLM only generates answers from genuinely relevant legal text.

Mizan also builds a **persistent memory profile** for each user across sessions, learning their name, profession, preferences, and past topics using Trustcall for safe, patch-based memory updates.

---

## Architecture

```
User Query
    │
    ▼
load_profile          ← reads user memory from LangGraph Store
    │
    ▼
retrieve_node         ← embeds query, searches ChromaDB (multilingual-e5-large)
    │
    ▼
grader                ← LLM grades each chunk: relevant (1.0) or not (0.0)
    │
    ├── poor quality → rewrite_node → retrieve_node (retry loop, max 2 attempts)
    │
    └── good quality → generate_node ← uses user profile + instructions in system prompt
                            │
                            ▼
                       save_profile  ← Trustcall extracts and patches user memory
                            │
                            ▼
                          END
```

---

## Key Features

- **Corrective RAG (CRAG)** — grades retrieved chunks, rewrites queries when retrieval fails, retries up to 2 times before falling back to best available context
- **Multilingual Embeddings** — uses `intfloat/multilingual-e5-large` for Arabic + English semantic search
- **Long-term User Memory** — remembers name, profession, language preference, and last topic across sessions using LangGraph Store + Trustcall
- **Self-improving Instructions** — Mizan updates its own behavioral instructions per user based on feedback (procedural memory)
- **Streaming** — token-by-token streaming via LangGraph's stream API
- **Arabic + English** — auto-detects query language and responds accordingly
- **LangGraph Studio** — fully compatible with LangGraph dev server for visual debugging

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent Framework | LangGraph |
| LLM | Groq — llama-3.3-70b-versatile |
| Embeddings | intfloat/multilingual-e5-large |
| Vector Store | ChromaDB |
| Memory (short-term) | LangGraph MemorySaver |
| Memory (long-term) | LangGraph Store + Trustcall |
| PDF Parsing | PyMuPDF |
| UI | Gradio |
| Package Management | uv |
| Deployment | HuggingFace Spaces |

---

## Project Structure

```
mizan/
├── app.py                  # Gradio UI — entry point
├── model/
│   ├── graph.py             # LangGraph graph — nodes, edges, state, compilation
│   ├── memory.py            # Long-term memory — UserProfile, Trustcall extractors
│   └── ingest.py            # PDF ingestion — chunking, embedding, ChromaDB storage
├── chroma_db/               # Persisted ChromaDB vector store
├── docs/
│   └── labor_law.pdf        # Egyptian Labor Law source document
├── langgraph.json           # LangGraph deployment config
├── pyproject.toml           # Project metadata + dependencies (uv)
├── uv.lock                  # Locked dependency versions
└── .env                     # API keys (not committed)
```

---

## How It Works

### 1. Ingestion (`ingest.py`)
The Egyptian Labor Law PDF is parsed with PyMuPDF, chunked with `RecursiveCharacterTextSplitter` (512 tokens, 64 overlap), embedded with `multilingual-e5-large`, and stored in ChromaDB with persistent storage.

### 2. CRAG Loop (`graph.py`)

**retrieve_node** — embeds the query (or rewritten query on retry) using the same embedding model, queries ChromaDB for top 5 chunks.

**grader** — for each retrieved chunk, asks the LLM: "Is this chunk relevant to the question?" Returns 1.0 (yes) or 0.0 (no).

**route_after_grading** — conditional edge: if any chunk scores ≥ 0.5 OR attempts ≥ 2, go to generate. Otherwise go to rewrite.

**rewrite_node** — uses structured output (`RewrittenQuery`) to generate a more specific, legally-precise version of the original query. Increments attempt counter.

**generate_node** — combines retrieved chunks into context, injects user profile and behavioral instructions into system prompt, generates final answer.

### 3. Memory (`memory.py`)

**load_profile** — at the start of each session, searches LangGraph Store for the user's profile and behavioral instructions by `user_id`. Injects them into state.

**save_profile** — after each answer, Trustcall extracts updated profile facts and behavioral preferences from the conversation, patches the existing memory (never overwrites), and saves back to the Store.

---

## Running Locally

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Clone the repo
git clone https://github.com/OmarAmir2001/mizan.git
cd mizan

# Install uv if you don't have it
# Windows (PowerShell):
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies (creates .venv automatically, reads pyproject.toml + uv.lock)
uv sync

# Set up environment variables
cp .env.example .env
# Add your GROQ_API_KEY and LANGSMITH_API_KEY to .env

# Run the Gradio UI
uv run app.py
```

---

## LangGraph Dev Server

```bash
uv add "langgraph-cli[inmem]"
uv run langgraph dev
```

Opens LangGraph Studio at `https://smith.langchain.com/studio` for visual graph debugging and testing.

---

## Skills Demonstrated

- **Agentic RAG** with corrective loop and query rewriting
- **LangGraph** state management, conditional edges, graph compilation
- **Long-term memory** with LangGraph Store, Trustcall, namespaced user profiles
- **Multilingual NLP** — Arabic/English detection and response
- **Vector search** with ChromaDB and sentence-transformers
- **Streaming** with LangGraph stream API
- **Gradio UI** with chat history, user ID management, session handling
- **Modern Python tooling** — uv for fast, reproducible dependency management
- **HuggingFace Spaces** deployment

---

## Environment Variables

```
GROQ_API_KEY=           # Required — Groq API key
LANGSMITH_API_KEY=      # Optional — for LangSmith tracing
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=mizan
```

---

## Roadmap

- [ ] Switch from ChromaDB to Qdrant for production vector store
- [ ] Add article-level source citations instead of chunk IDs
- [ ] Multi-turn conversation with full context awareness
- [ ] Support for additional Egyptian law domains (civil, criminal, commercial)
- [ ] Arabic UI mode with full RTL support

---

## License

MIT

---

*Built as part of an AI Engineering portfolio. Other projects: Customer Support Agent, Research & Report Generator, AI Code Reviewer.*