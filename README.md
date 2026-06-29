# RAG Chatbot Backend

Backend for a Retrieval Augmented Generation (RAG) chatbot. You give it your own documents (PDF, Word, CSV, plain text), and it answers questions about them in natural language. Relevant pieces of your documents are retrieved and fed to a language model so answers stay grounded in your content.

It runs in two modes. RAG mode answers strictly from your documents. Agent mode gives the model a set of tools (document search, web search, calculator, current date) and lets it decide what to use.

## Table of contents

1. Overview
2. Tech stack
3. Prerequisites
4. Installation
5. Configuration
6. Running it
7. Chat modes
8. API reference
9. Project structure
10. Modules

## 1. Overview

How a question becomes an answer in RAG mode:

```
Documents (PDF, DOCX, CSV, TXT)
   -> loader.py    reads raw text from each file
   -> chunker.py   splits text into overlapping chunks
   -> embedder.py  turns each chunk into a vector
   -> retriever.py stores the vectors in ChromaDB

At question time:
   question -> embedder.py -> retriever.py finds the closest chunks
            -> rag.py builds a prompt with those chunks as context
            -> llm.py streams the answer from the model
```

Agent mode replaces the middle step: the model receives the question plus the tool definitions and chooses which tools to call before answering. That logic lives in `agent.py` and `tools.py`.

## 2. Tech stack

- **Python 3.10 or newer.** Uses modern type hints like `list[dict]` and `str | None`.
- **FastAPI and Uvicorn.** The web framework and the server that runs it.
- **ChromaDB.** Local on-disk vector database. Runs inside the Python process.
- **sentence-transformers.** Produces embeddings. Default model `all-MiniLM-L6-v2`, runs on CPU.
- **Groq.** Hosts the language model. Default `qwen/qwen3-32b`. Needs a free Groq API key.
- **PyMuPDF, python-docx, pandas.** Read PDF, Word, and CSV files.
- **ddgs.** DuckDuckGo search, used by the web search tool.

## 3. Prerequisites

- Python 3.10 or newer on your PATH. Check with `python --version`.
- pip (comes with Python).
- Around 1 GB of free disk space for PyTorch and the transformer libraries.
- A Groq API key, free from the Groq console.
- An internet connection on first run so the embedding model can download and cache. It runs offline after that.

## 4. Installation

From the `backend` folder:

```bash
# 1. create and activate a virtual environment
python -m venv venv
venv\Scripts\Activate.ps1        # Windows (PowerShell)
# source venv/bin/activate       # macOS / Linux

# 2. install dependencies (pulls in PyTorch, takes a few minutes)
pip install -r requirements.txt

# 3. create your .env file (see next section)
```

## 5. Configuration

The API key is read from a `.env` file in the `backend` folder. It is not committed (it is in `.gitignore`). Create it with one line:

```
GROQ_API_KEY=your_groq_api_key_here
```

Everything else lives in `app/config.py`, which loads the `.env` and creates the `data`, `data/chroma_db`, and `documents` folders on first import. Values worth tuning:

- `CHUNK_SIZE` (`500`) and `OVERLAP` (`100`): chunk length and how much neighbouring chunks share. Overlap must be smaller than chunk size.
- `EMBED_MODEL` (`all-MiniLM-L6-v2`): the embedding model. Changing it after indexing means you must reindex.
- `MODEL` (`qwen/qwen3-32b`), `TEMPERATURE` (`0.3`), `MAX_TOKENS` (`1024`): language model settings.
- `SIMILARITY_THRESHOLD` (`0.4`) and `TOP_K` (`5`): how strict retrieval is and how many chunks it pulls. If the bot too often says it has no information, lower the threshold or raise top_k.

## 6. Running it

Everything launches through `main.py`.

```bash
python main.py            # interactive CLI chat (RAG mode)
python main.py --index    # index new or changed files in documents/
python main.py --server   # start the HTTP server on 127.0.0.1:8000
```

Index first so there is something to query. In the CLI, type `quit`, `clear`, or `stats`; anything else is a question. With `--server` running, interactive API docs are at `/docs` and `/redoc`. You can also run the server directly with `uvicorn server:app --reload`.

## 7. Chat modes

**RAG mode** retrieves relevant chunks every turn and tells the model to answer only from them. If nothing clears the similarity threshold, it replies that it has no such information in the documents. Use it for grounded answers limited to your own content.

**Agent mode** hands the model four tools and lets it decide: document search, web search through DuckDuckGo, a safe calculator, and current date. It can chain calls, capped at five steps. Use it for a more general assistant that can reach beyond your documents.

## 8. API reference

Base URL: `http://127.0.0.1:8000`. Chat responses stream as `text/plain`.

**Chat**
- `POST /api/chat/` RAG chat. Body `{ "message": "...", "session_id": "optional" }`. A new session id is returned in the `X-Session-Id` header and source files in `X-Sources`.
- `POST /api/chat/agent` Agent chat. Same body shape.
- `POST /api/chat/session` Create an empty session.
- `DELETE /api/chat/session/{id}` Clear a session.
- `GET /api/chat/session/{id}/history` Return a session's history.

**Documents**
- `GET /api/documents/` List indexed documents with chunk counts.
- `POST /api/documents/upload` Multipart `file` upload. Saved and indexed immediately. Unsupported types return 415.
- `DELETE /api/documents/{filename}` Remove the file and its chunks.
- `GET /api/documents/{filename}/status` Whether it exists, is indexed, and its chunk count.

**Admin**
- `GET /api/admin/health` Liveness plus chunk count.
- `GET /api/admin/stats` Database stats and current settings.
- `POST /api/admin/reindex` Rescan and index new or changed files.
- `PATCH /api/admin/settings` Update `similarity_threshold`, `top_k`, `chunk_size`, or `overlap` live, in memory until restart.

## 9. Project structure

```
backend/
  main.py            Entry point. Picks CLI, index, or server mode.
  server.py          FastAPI app, CORS, and router wiring.
  requirements.txt   Pinned Python dependencies.
  .env               Your Groq API key (you create this; not committed).
  .gitignore

  app/               Core engine. No web code.
    config.py        Settings and paths.
    loader.py        Reads text from PDF, DOCX, CSV, TXT.
    chunker.py       Splits text into overlapping chunks.
    embedder.py      Turns text into embedding vectors.
    retriever.py     ChromaDB store, search, and stats.
    indexer.py       Load, chunk, embed, store, with a registry.
    llm.py           Groq chat and streaming.
    rag.py           RAG pipeline and RAGSession.
    tools.py         The four agent tools and their schemas.
    agent.py         Agent tool loop and AgentSession.

  api/               Web layer. Thin wrappers over app/.
    chat.py          RAG and agent chat, with sessions.
    documents.py     Upload, list, delete, status.
    admin.py         Health, stats, reindex, live settings.

  documents/         Source files. knowledge.txt ships as a sample.
  data/              Generated at runtime. Not committed.
```

The `app` folder is the engine and knows nothing about HTTP. The `api` folder receives requests and calls into it. The same engine works from the CLI, the server, or your own scripts.

## 10. Modules

**config.py.** Single source of truth for settings and paths. Loads `.env` and bootstraps the required directories.

**loader.py.** Reads four file types into plain text. `load_document` picks the reader from the extension; supported types are in `SUPPORTED_FORMATS`.

**chunker.py.** `chunk_text` slices text into target-size pieces with overlap, breaking on spaces, with a guard that always moves forward.

**embedder.py.** Loads the sentence-transformer model once and reuses it. `embed` handles a list of texts; `embed_one` handles a single query.

**retriever.py.** Owns the persistent ChromaDB collection (cosine similarity). `search` embeds the query, takes the top matches, scores them 0 to 1, drops weak ones, and returns the rest best first. `get_stats` reports the chunk count.

**indexer.py.** `index_document` runs load, chunk, embed, store for one file, clearing old chunks first so re-indexing replaces stale content. `index_folder` does this for every supported file, skipping unchanged ones via a JSON registry. `delete_document` removes a file's chunks.

**llm.py.** Thin layer over Groq. Holds the system prompt and exposes `stream_chat`, which yields the answer in pieces.

**rag.py.** `build_rag_prompt` wraps retrieved chunks as context and instructs the model to answer only from them. `RAGSession` is one conversation, keeping its own history and listing source files per answer.

**tools.py.** The four agent tools as plain functions returning strings, plus `TOOL_FUNCTIONS` (name to function) and `TOOL_SCHEMAS` (definitions sent to the model). The calculator uses a whitelist, not raw `eval`.

**agent.py.** Runs the tool loop: send the conversation and tool definitions, run any requested tools, append results, repeat until the model returns a plain answer, capped at five iterations. `AgentSession` mirrors `RAGSession` so the API treats both modes the same.

---

If anything here does not match the code, trust the code and update this file.
